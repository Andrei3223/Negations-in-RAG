"""Swap the instruction prefix of an already-built parquet dataset.

The prompt templates in qa_search_train_merge.make_prefix() differ only in the
instruction text that precedes "Question: {question}". So a dataset built with
one template can be converted to another by replacing that prefix -- no need to
regenerate from the source corpus.

That matters because the negation splits in this repo (nevir-with-answers,
negations-free-*, negation-controlled-*) have no build script here; they were
produced elsewhere and copied in. Rebuilding via qa_search_train_merge.py would
silently drop them.

    python scripts/data_process/retemplate.py \
        --src data/combined_nq_hotpot_nevir_negfree_upsampled \
        --dst data/combined_nq_hotpot_nevir_negfree_upsampled_decomposed

Refuses to run if any row does not carry the expected source prefix exactly,
rather than writing a half-converted dataset.
"""

import argparse
import importlib.util
import os
import sys
import glob

import pandas as pd


def _load_make_prefix():
    """Import make_prefix from the merge script so templates cannot drift apart."""
    path = os.path.join(os.path.dirname(__file__), 'qa_search_train_merge.py')
    spec = importlib.util.spec_from_file_location('_merge', path)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ['retemplate']
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass  # the module runs argparse under __main__; we only want make_prefix
    finally:
        sys.argv = argv
    return mod.make_prefix


def prefix_for(make_prefix, template_type):
    """The instruction text that precedes the question, for one template."""
    sentinel = '<<QUESTION>>'
    rendered = make_prefix({'question': sentinel}, template_type)
    head, sep, _ = rendered.partition(f'Question: {sentinel}')
    if not sep:
        raise RuntimeError(f'template {template_type!r} does not end in "Question: ..."')
    return head


def convert_frame(df, old_prefix, new_prefix, label='<frame>'):
    """Return a copy of `df` with the prompt prefix swapped.

    Raises rather than converting part of the data: a silently half-converted
    dataset would train without any visible error.
    """
    df = df.copy()
    texts = [row[0]['content'] for row in df['prompt']]
    bad = sum(1 for t in texts if not t.startswith(old_prefix))
    if bad:
        raise SystemExit(
            f'{label}: {bad} of {len(texts)} rows do not start with the expected '
            f'prefix. Wrong source template, or this data was built differently. '
            f'Nothing written.'
        )

    df['prompt'] = [
        [{'role': row[0]['role'], 'content': new_prefix + text[len(old_prefix):]}]
        for row, text in zip(df['prompt'], texts)
    ]
    return df


def convert_file(src, dst, old_prefix, new_prefix):
    df = convert_frame(pd.read_parquet(src), old_prefix, new_prefix, label=src)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    df.to_parquet(dst)
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='directory of .parquet files to convert')
    ap.add_argument('--dst', required=True, help='directory to write converted copies to')
    ap.add_argument('--from', dest='src_tmpl', default='base')
    ap.add_argument('--to', dest='dst_tmpl', default='decomposed')
    args = ap.parse_args()

    make_prefix = _load_make_prefix()
    old_prefix = prefix_for(make_prefix, args.src_tmpl)
    new_prefix = prefix_for(make_prefix, args.dst_tmpl)
    print(f'{args.src_tmpl} prefix: {len(old_prefix)} chars')
    print(f'{args.dst_tmpl} prefix: {len(new_prefix)} chars')

    files = sorted(glob.glob(os.path.join(args.src, '*.parquet')))
    if not files:
        raise SystemExit(f'no .parquet files under {args.src}')

    for src in files:
        dst = os.path.join(args.dst, os.path.basename(src))
        n = convert_file(src, dst, old_prefix, new_prefix)
        print(f'  {os.path.basename(src):<16} {n:>8,} rows -> {dst}')


if __name__ == '__main__':
    main()
