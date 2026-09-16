"""Build a negation-only training set rendered with the SubSearch template.

Two things motivate this:

1. Every parquet in data/ was built with Search-R1's *base* prompt, which never
   mentions sub-query decomposition. Across the completed 200-step run
   (logs/subsearch-grpo-339350.out) 0 of 77,150 <search> blocks contained '##'
   and 1 of 38,024 decomposition-scoring events had more than one sub-query, so
   SubSearch's central mechanism has never actually fired.
2. The full mix is 189k rows and takes days per run. A negation-only set is
   ~1.8k rows, small enough to iterate on in hours.

Rows are taken from exactly the ones the big training set used -- verified
identical, not merely equal in count:

    nevir-with-answers                   train_clean.parquet / validation_clean.parquet
    negations-free-sentence-with-answer  train.parquet       / validation.parquet

The _upsampled build is exactly 11x duplication of these same unique rows, so
taking each once loses no content.

    python scripts/data_process/build_negation_only.py

Prints a before/after prompt example and a stats table when it finishes.
"""

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retemplate  # noqa: E402  -- single source of truth for the templates

NEGATION_SOURCES = ['nevir-with-answers', 'negations-free-sentence-with-answer']

# Held-out test rows. The combined build has no test split, so these come
# straight from the source directories; both were checked to share the combined
# schema and to have zero overlap with the train or val rows selected below.
TEST_FILES = [
    'data/negations_with_answer_rag/nevir-with-answers/test_clean.parquet',
    'data/negations_with_answer_rag/negations-free-sentence-with-answer/test_clean.parquet',
]

NEG_MARKERS = re.compile(
    r"\b(not|n't|no|never|none|nobody|nothing|without|except|unless|neither|nor"
    r"|cannot|didn|doesn|isn|aren|wasn|weren|won|don)\b", re.I)


def question_of(prompt_cell):
    return prompt_cell[0]['content'].split('Question: ', 1)[-1].strip()


def print_example(before_df, after_df):
    print('\n' + '=' * 78)
    print('HOW THE INPUT DATA DIFFERS')
    print('=' * 78)
    b = before_df.prompt.iloc[0][0]['content']
    a = after_df.prompt.iloc[0][0]['content']
    row = before_df.iloc[0]
    print(f"\ndata_source : {row.data_source}")
    print(f"question    : {question_of(row.prompt)}")
    print(f"gold answer : {list(row.reward_model['ground_truth']['target'])}")
    print(f"\nOnly the `prompt` field changes. Everything else -- data_source, ability,")
    print(f"reward_model.ground_truth, extra_info -- is copied through untouched.\n")
    for name, text in (('BEFORE (base / Search-R1)', b), ('AFTER (decomposed / SubSearch)', a)):
        print('-' * 78)
        print(f'{name}   [{len(text)} chars]')
        print('-' * 78)
        print(text.strip())
        print()


def print_stats(splits):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
        enc = lambda s: len(tok.encode(s))
    except Exception as exc:
        print(f'\n[stats] tokenizer unavailable ({exc}); reporting whitespace words')
        enc = lambda s: len(s.split())

    print('\n' + '=' * 78)
    print('DATASET STATS')
    print('=' * 78)
    hdr = (f"\n{'split':<6} {'data_source':<38} {'rows':>6} {'q_tok':>7} "
           f"{'q_max':>6} {'n_ans':>6} {'ans_tok':>8} {'%neg':>7}")
    print(hdr)
    print('-' * len(hdr))
    for name, df in splits.items():
        for ds in NEGATION_SOURCES:
            s = df[df.data_source == ds]
            if not len(s):
                continue
            q = s.prompt.apply(question_of)
            qt = q.apply(enc)
            ans = s.reward_model.apply(lambda r: list(r['ground_truth']['target']))
            n_ans = ans.apply(len)
            a_tok = ans.apply(lambda a: sum(enc(str(x)) for x in a) / len(a) if len(a) else 0)
            pct_neg = q.apply(lambda x: bool(NEG_MARKERS.search(x))).mean() * 100
            print(f'{name:<6} {ds:<38} {len(s):>6,} {qt.mean():>7.1f} '
                  f'{qt.max():>6} {n_ans.mean():>6.2f} {a_tok.mean():>8.1f} {pct_neg:>6.1f}%')
        tot = df[df.data_source.isin(NEGATION_SOURCES)]
        print(f'{name:<6} {"TOTAL":<38} {len(tot):>6,}')

    full = pd.concat(splits.values())
    p_tok = full.prompt.apply(lambda p: enc(p[0]['content']))
    print(f'\nfull prompt length (decomposed template): mean {p_tok.mean():.0f} '
          f'/ max {p_tok.max()} tokens, over {len(full):,} rows')

    print('\nStructure: both sources are contrastive pairs -- adjacent rows are the')
    print('same question with and without a negation, carrying different gold answers.')
    tr = splits['train']
    s = tr[tr.data_source == 'nevir-with-answers'].reset_index(drop=True)
    for i in (0, 1):
        print(f"    Q: {question_of(s.prompt.iloc[i])}")
        print(f"    A: {list(s.reward_model.iloc[i]['ground_truth']['target'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--combined', default='data/combined_nq_hotpot_nevir_negfree',
                    help='non-upsampled combined build to filter train/val from')
    ap.add_argument('--dst', default='data/negation_only_decomposed')
    ap.add_argument('--to', dest='dst_tmpl', default='decomposed')
    ap.add_argument('--from', dest='src_tmpl', default='base')
    args = ap.parse_args()

    make_prefix = retemplate._load_make_prefix()
    old_prefix = retemplate.prefix_for(make_prefix, args.src_tmpl)
    new_prefix = retemplate.prefix_for(make_prefix, args.dst_tmpl)
    print(f'{args.src_tmpl} prefix: {len(old_prefix)} chars')
    print(f'{args.dst_tmpl} prefix: {len(new_prefix)} chars  '
          f'(+{len(new_prefix) - len(old_prefix)})\n')

    # train / val: the exact rows the big training set used.
    raw = {}
    for split, fname in (('train', 'train.parquet'), ('val', 'val.parquet')):
        df = pd.read_parquet(os.path.join(args.combined, fname))
        raw[split] = df[df.data_source.isin(NEGATION_SOURCES)].reset_index(drop=True)

    # test: held out, never seen by the big run.
    raw['test'] = pd.concat([pd.read_parquet(f) for f in TEST_FILES],
                            ignore_index=True)

    os.makedirs(args.dst, exist_ok=True)
    out = {}
    for split, df in raw.items():
        conv = retemplate.convert_frame(df, old_prefix, new_prefix, label=split)
        path = os.path.join(args.dst, f'{split}.parquet')
        conv.to_parquet(path)
        out[split] = conv
        counts = ', '.join(f'{k}={v:,}' for k, v in conv.data_source.value_counts().items())
        print(f'  {split:<6} {len(conv):>6,} rows -> {path}   ({counts})')

    print_example(raw['train'], out['train'])
    print_stats(out)


if __name__ == '__main__':
    main()
