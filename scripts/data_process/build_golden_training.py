"""
Build the golden-document training sets: negation rows answered from fixed
documents (no retrieval), optionally mixed with live-retrieval NQ / HotpotQA.

Every negation row carries the golden-doc columns written by
negations_golden_docs.py (gold_doc, hard_neg_doc, rand_neg_doc, doc_order,
doc_order3). With retriever.mode=golden_hn_rn each <search> on such a row returns
gold + the pair's other document + an unrelated document, in the row's fixed
doc_order3. NQ / HotpotQA rows carry '' in those columns and, with
retriever.live_fallback=true, are searched live -- so one batch mixes both.

Setups (clean negation files only, see docs/datasets.md):

  nevir_nfs              NevIR + negations-free-sentence
  nevir_nfwa             NevIR + all clean negations-free (the sentence subset is
                         already inside it, so it is not added again)
  nevir_nfwa_nq_hotpot   nevir_nfwa + as many NQ rows and as many HotpotQA rows as
                         there are negation rows (train); VAL_QA_ROWS each in val

Validation rows are golden too (live transfer is measured with run/eval.py).
Test sets stay the existing test_clean*_docs.parquet files.

Usage (run once per setup and template; negations_golden_docs.py --split all first):
  python scripts/data_process/build_golden_training.py --setup nevir_nfs
  python scripts/data_process/build_golden_training.py --setup nevir_nfs --template decomposed
Output: data/golden_<setup>[_decomposed]/{train,val}.parquet
"""

import argparse
import collections
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retemplate  # noqa: E402  -- single source of truth for the templates

RAG = 'data/negations_with_answer_rag'
DOCS = {  # (train, val) golden-doc files per negation source
    'nevir-with-answers': (f'{RAG}/nevir-with-answers/train_clean_docs.parquet',
                           f'{RAG}/nevir-with-answers/validation_clean_docs.parquet'),
    'negations-free-with-answer': (f'{RAG}/negations-free-with-answer/train_clean2_docs.parquet',
                                   f'{RAG}/negations-free-with-answer/validation_clean2_docs.parquet'),
    'negations-free-sentence-with-answer': (
        f'{RAG}/negations-free-sentence-with-answer/train_docs.parquet',
        f'{RAG}/negations-free-sentence-with-answer/validation_docs.parquet'),
}
TEST_DOCS = [f'{RAG}/nevir-with-answers/test_clean_docs.parquet',
             f'{RAG}/negations-free-with-answer/test_clean2_docs.parquet',
             f'{RAG}/negations-free-sentence-with-answer/test_clean_docs.parquet']
SETUPS = {
    'nevir_nfs': (['nevir-with-answers', 'negations-free-sentence-with-answer'], False),
    'nevir_nfwa': (['nevir-with-answers', 'negations-free-with-answer'], False),
    'nevir_nfwa_nq_hotpot': (['nevir-with-answers', 'negations-free-with-answer'], True),
}
COMBINED = 'data/combined_nq_hotpot_nevir_negfree'   # base-template NQ / HotpotQA
QA_SOURCES = ['nq', 'hotpotqa']
VAL_QA_ROWS = 250

TRAINER_COLS = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
DOC_COLS = ['gold_doc', 'hard_neg_doc', 'rand_neg_doc', 'doc_order', 'doc_order3']
EXTRA_KEYS = ['split', 'index', 'pair_id', 'query_role', 'negation_type']


def unify_extra(ei):
    """One extra_info struct for every source, so the parquet schema is uniform."""
    ei = ei if isinstance(ei, dict) else {}
    return {k: ei.get(k) for k in EXTRA_KEYS}


def question(row):
    return row['prompt'][0]['content'].rsplit('Question:', 1)[-1].strip().lower()


def load_negation(sources, split_idx):
    frames = []
    for src in sources:
        df = pd.read_parquet(DOCS[src][split_idx])
        if set(df.data_source) != {src}:
            raise ValueError(f'{DOCS[src][split_idx]}: data_source {set(df.data_source)} != {src}')
        missing = [c for c in DOC_COLS if c not in df.columns]
        if missing:
            raise ValueError(f'{DOCS[src][split_idx]} lacks {missing}; rerun negations_golden_docs.py --split all')
        frames.append(df[TRAINER_COLS + DOC_COLS])
    return pd.concat(frames, ignore_index=True)


def sample_qa(path, n_each, seed):
    df = pd.read_parquet(path)
    frames = []
    for src in QA_SOURCES:
        pool = df[df.data_source == src]
        if len(pool) < n_each:
            raise ValueError(f'{path}: only {len(pool)} {src} rows, need {n_each}')
        frames.append(pool.sample(n=n_each, random_state=seed)[TRAINER_COLS])
    out = pd.concat(frames, ignore_index=True)
    for c in DOC_COLS:
        out[c] = ''   # no golden docs: searched live via retriever.live_fallback
    return out


def check_pairs(df, label):
    roles = collections.defaultdict(set)
    for ds, ei in zip(df.data_source, df.extra_info):
        if ei.get('pair_id') is not None:
            roles[(ds, str(ei['pair_id']))].add(ei['query_role'])
    broken = [k for k, v in roles.items() if v != {'q1', 'q2'}]
    if broken:
        raise ValueError(f'{label}: {len(broken)} incomplete pairs, e.g. {broken[:3]}')
    return len(roles)


def write(df, path):
    tmp = f'{path}.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)   # atomic: a run may be reading the old file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setup', required=True, choices=sorted(SETUPS))
    ap.add_argument('--template', default='base', choices=['base', 'decomposed'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dst', default=None, help='default: data/golden_<setup>[_decomposed]')
    args = ap.parse_args()
    dst = args.dst or f'data/golden_{args.setup}' + ('' if args.template == 'base' else '_decomposed')
    sources, with_qa = SETUPS[args.setup]

    train, val = load_negation(sources, 0), load_negation(sources, 1)
    n_neg = len(train)
    if with_qa:
        train = pd.concat([train, sample_qa(f'{COMBINED}/train.parquet', n_neg, args.seed)],
                          ignore_index=True)
        val = pd.concat([val, sample_qa(f'{COMBINED}/val.parquet', VAL_QA_ROWS, args.seed)],
                        ignore_index=True)
    # Train shuffled; val kept in order so the two halves of a pair stay adjacent.
    train = train.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    for df in (train, val):
        df['extra_info'] = [unify_extra(ei) for ei in df['extra_info']]

    # No test question may appear in train or val. Checked by question text:
    # negations-free pair_ids are per-split row indices, so ids collide by design.
    test_q = set()
    for f in TEST_DOCS:
        test_q |= {question(r) for _, r in pd.read_parquet(f).iterrows()}
    # drop_conflicting_pairs.py removes these at the source (val repeating train or
    # test questions, conflicting labels); refuse to build if any came back.
    train_q = {question(r) for _, r in train.iterrows()}
    for name, df, banned in (('train', train, test_q), ('val', val, test_q | train_q)):
        leak = sum(question(r) in banned for _, r in df.iterrows())
        if leak:
            raise ValueError(f'{name}: {leak} rows repeat a test{" or train" if name == "val" else ""} '
                             f'question; run scripts/data_process/drop_conflicting_pairs.py')

    if args.template != 'base':
        make_prefix = retemplate._load_make_prefix()
        old = retemplate.prefix_for(make_prefix, 'base')
        new = retemplate.prefix_for(make_prefix, args.template)
        train = retemplate.convert_frame(train, old, new, label='train')
        val = retemplate.convert_frame(val, old, new, label='val')

    os.makedirs(dst, exist_ok=True)
    print(f'== {args.setup} ({args.template}) -> {dst}')
    for name, df in (('train', train), ('val', val)):
        pairs = check_pairs(df, name)
        write(df, os.path.join(dst, f'{name}.parquet'))
        counts = ', '.join(f'{k}={v:,}' for k, v in df.data_source.value_counts().items())
        golden = int((df.gold_doc != '').sum())
        print(f'   {name:<5} {len(df):>6,} rows ({golden:,} golden, {len(df) - golden:,} live), '
              f'{pairs} complete pairs   [{counts}]')

    ex = train[train.gold_doc != ''].iloc[0]
    print(f'\n   example golden row ({ex.data_source}, doc_order3={ex.doc_order3}):')
    print(f'   Q: {question(ex)}')
    print(f'   A: {list(ex.reward_model["ground_truth"]["target"])}')
    for k in ('gold_doc', 'hard_neg_doc', 'rand_neg_doc'):
        print(f'   {k}: {ex[k][:150]!r}')


if __name__ == '__main__':
    main()
