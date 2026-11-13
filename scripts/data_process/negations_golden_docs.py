"""
Attach golden, hard-negative and random-negative documents to a negation
parquet, for the golden-document retriever modes
(retriever.mode = golden | golden_hn | golden_rn | golden_hn_rn), in eval
(eval/retriever=golden*) and in training (build_golden_training.py reads the
train/validation outputs).

The base parquet (built by negations_with_answer_rag.py / create_clean2.py) has
no documents. They are recovered from the raw source via pair_id -- the raw row
index for the negations-free sets, the raw `id` column for nevir:

  q1 record: gold_doc = doc1, hard_neg_doc = doc2
  q2 record: gold_doc = doc2, hard_neg_doc = doc1

The two halves of a pair are minimal pairs, so the other half's document is
topically identical and differs exactly on the negation -- the hardest
distractor available.

rand_neg_doc is the opposite control: a document from a DIFFERENT topic (or, for
nevir, which has no topic, simply another pair), so it is unrelated rather than
minimally different. golden_hn vs golden_rn isolates the cost of the minimal-pair
distractor; golden_rn vs golden isolates the cost of a second document at all.

Adds five scalar string columns (scalars, not one list column: collate_fn's
np.array(dtype=object) gives a list column a data-dependent shape):

  gold_doc      '"<title>"\\n<doc>'   same convention as build_custom_corpus.py,
  hard_neg_doc  '"<title>"\\n<doc>'   which _passages2string splits on the first \\n
  rand_neg_doc  '"<title>"\\n<doc>'   carries the RANDOM doc's own title
  doc_order     'gold,neg' | 'neg,gold'   gold's slot, shared by golden_hn and golden_rn
  doc_order3    permutation of 'gold,neg,rand'   slot order for golden_hn_rn (3 docs)

The title is the row's `topic` where the dataset has one; nevir has none, so its
docs get the constant title "Document" -- a title LINE is mandatory either way,
since _passages2string treats the first line as the title and drops it.

doc_order and the random draw are fixed here, not at runtime, so a model that
searches twice sees the same context both times and the exact context is
inspectable in the parquet. Both are pure functions of pair_id + query_role, so
re-running this script reproduces the same file.

The random-negative pool is the split's own file, so train/validation docs never
draw a test document (and vice versa).

Usage:
  python scripts/data_process/negations_golden_docs.py --dataset negations-free-with-answer
  python scripts/data_process/negations_golden_docs.py --dataset all                 # test
  python scripts/data_process/negations_golden_docs.py --dataset all --split train
  python scripts/data_process/negations_golden_docs.py --dataset all --split all
"""

import argparse
import collections
import hashlib
import os

import pandas as pd

ORDER_SALT = 'golden-doc-order-v1'
ORDER3_SALT = 'golden-doc-order3-v1'
ORDERS3 = ['gold,neg,rand', 'gold,rand,neg', 'neg,gold,rand',
           'neg,rand,gold', 'rand,gold,neg', 'rand,neg,gold']
RAND_SALT = 'random-negative-v1'
POS = {'q1': 'doc1', 'q2': 'doc2'}
NEG = {'q1': 'doc2', 'q2': 'doc1'}
ANSWER = {'q1': 'answer1', 'q2': 'answer2'}

RAG_DIR = 'data/negations_with_answer_rag'
SRC_DIR = os.environ.get('RAW_DATA_DIR', '/home/16207041/data')

SPLITS = ('train', 'validation', 'test')

# base / raw: per split -- the clean base file each experiment uses, and the raw
# source its pair_ids index into.
# key_col: raw column holding pair_id, or None when pair_id is the raw row index.
# title_col: raw column used as the doc title, or None -> constant "Document".
DATASETS = {
    'negations-free-with-answer': {
        'base': {s: f'{RAG_DIR}/negations-free-with-answer/{s}_clean2.parquet' for s in SPLITS},
        'raw': {s: f'{SRC_DIR}/negations_free_with_answer/{s}.parquet' for s in SPLITS},
        'key_col': None,
        'title_col': 'topic',
    },
    'negations-free-sentence-with-answer': {
        # train/validation have no _clean file: the whole subset is built from _clean2.
        'base': {'train': f'{RAG_DIR}/negations-free-sentence-with-answer/train.parquet',
                 'validation': f'{RAG_DIR}/negations-free-sentence-with-answer/validation.parquet',
                 'test': f'{RAG_DIR}/negations-free-sentence-with-answer/test_clean.parquet'},
        'raw': {s: f'{SRC_DIR}/negations_free_sentence_with_answer/{s}.parquet' for s in SPLITS},
        'key_col': None,
        'title_col': 'topic',
    },
    'nevir-with-answers': {
        'base': {s: f'{RAG_DIR}/nevir-with-answers/{s}_clean.parquet' for s in SPLITS},
        'raw': {s: f'{SRC_DIR}/nevir_with_answers/{s}_clean.parquet' for s in SPLITS},
        'key_col': 'id',
        'title_col': None,   # no topic column; see module docstring
    },
}
CONST_TITLE = 'Document'


def doc_order(pair_id: str, query_role: str) -> str:
    # hashlib, not hash(): str hashing is randomized per process by PYTHONHASHSEED.
    h = hashlib.sha256(f'{ORDER_SALT}|{pair_id}|{query_role}'.encode()).digest()
    return 'gold,neg' if h[0] % 2 == 0 else 'neg,gold'


def doc_order3(pair_id: str, query_role: str) -> str:
    """Slot order of gold / hard negative / random negative for golden_hn_rn."""
    h = hashlib.sha256(f'{ORDER3_SALT}|{pair_id}|{query_role}'.encode()).digest()
    return ORDERS3[int.from_bytes(h[:8], 'big') % len(ORDERS3)]


def pick_random(pair_id: str, query_role: str, candidates: list):
    """One candidate (pair_id, doc column), chosen deterministically for this row."""
    h = hashlib.sha256(f'{RAND_SALT}|{pair_id}|{query_role}'.encode()).digest()
    return candidates[int.from_bytes(h[:8], 'big') % len(candidates)]


def as_contents(title, text) -> str:
    return f'"{str(title).strip()}"\n{str(text).strip()}'


def build(name, base_path, raw_path, out, key_col, title_col):
    df = pd.read_parquet(base_path)
    raw = pd.read_parquet(raw_path)
    base_cols = list(df.columns)

    # A missing answer survives _clean2 as the string 'nan' (negations-free train
    # pair 59: raw answer2 is None). Training on target 'nan' teaches nothing, and a
    # lone half breaks the pair, so drop the whole pair and say so.
    def _missing(rm):
        t = str(rm['ground_truth']['target'][0]).strip().lower()
        return t in ('', 'nan', 'none')
    bad = {str(ei['pair_id']) for ei, rm in zip(df['extra_info'], df['reward_model']) if _missing(rm)}
    if bad:
        keep = [str(ei['pair_id']) not in bad for ei in df['extra_info']]
        print(f'   {name}: dropping {len(bad)} pair(s) with a missing answer: {sorted(bad)}')
        df = df[keep].reset_index(drop=True)

    # pair_id -> raw row. Numeric sort when pair_id is the row index, so the pool
    # keeps the natural order (string sort would put '10' before '2' and change
    # every random draw).
    if key_col is None:
        rows = {str(i): raw.iloc[i] for i in range(len(raw))}
        keys = sorted({str(ei['pair_id']) for ei in df['extra_info']}, key=int)
    else:
        rows = {str(k): raw.iloc[i] for i, k in enumerate(raw[key_col])}
        keys = sorted({str(ei['pair_id']) for ei in df['extra_info']})
    missing = [k for k in keys if k not in rows]
    if missing:
        raise ValueError(f'{name}: {len(missing)} pair_ids absent from {raw_path}, e.g. {missing[:3]}')

    def title_of(row):
        return CONST_TITLE if title_col is None else row[title_col]

    # Random-negative pool: doc1 and doc2 of every pair this eval file contains.
    pool = [(k, col) for k in keys for col in ('doc1', 'doc2') if str(rows[k][col]).strip()]

    gold, hneg, rneg, order, order3, rand_src = [], [], [], [], [], []
    for ei, rm in zip(df['extra_info'], df['reward_model']):
        pair_id, role = str(ei['pair_id']), ei['query_role']
        r = rows[pair_id]
        # Join proof: the raw row's answer must be this record's ground truth.
        target = str(rm['ground_truth']['target'][0]).strip()
        if str(r[ANSWER[role]]).strip() != target:
            raise ValueError(f'{name} pair_id={pair_id} role={role}: raw {ANSWER[role]}={r[ANSWER[role]]!r} != target {target!r}')
        for col in (POS[role], NEG[role]):
            if pd.isna(r[col]) or not str(r[col]).strip():
                raise ValueError(f'{name} pair_id={pair_id}: empty {col}')
        gold.append(as_contents(title_of(r), r[POS[role]]))
        hneg.append(as_contents(title_of(r), r[NEG[role]]))
        order.append(doc_order(pair_id, role))
        order3.append(doc_order3(pair_id, role))

        # Unrelated distractor. Where there are topics, excluding the topic (not
        # just the pair) keeps the doc off-subject and also rules out this row's
        # own two docs. nevir has no topics, so exclude the pair itself.
        if title_col is None:
            cands = [c for c in pool if c[0] != pair_id]
        else:
            topic = str(r[title_col]).strip()
            cands = [c for c in pool if str(rows[c[0]][title_col]).strip() != topic]
        if not cands:
            raise ValueError(f'{name} pair_id={pair_id}: no unrelated candidate')
        src_key, src_col = pick_random(pair_id, role, cands)
        # NevIR reuses one document pair across several question pairs (ids 10-1,
        # 10-3, ...), so "another pair" can hand back this row's own gold or hard
        # negative. Redraw only then, from candidates with different text: rows
        # without a collision keep their original draw (the test files built before
        # this check had none and are unchanged).
        own = {str(r[POS[role]]).strip(), str(r[NEG[role]]).strip()}
        if str(rows[src_key][src_col]).strip() in own:
            cands = [c for c in cands if str(rows[c[0]][c[1]]).strip() not in own]
            if not cands:
                raise ValueError(f'{name} pair_id={pair_id}: no candidate with different text')
            src_key, src_col = pick_random(pair_id, role, cands)
        src = rows[src_key]
        rneg.append(as_contents(title_of(src), src[src_col]))
        rand_src.append((src_key, src_col))

    df['gold_doc'], df['hard_neg_doc'] = gold, hneg
    df['rand_neg_doc'], df['doc_order'] = rneg, order
    df['doc_order3'] = order3
    assert list(df.columns) == base_cols + ['gold_doc', 'hard_neg_doc', 'rand_neg_doc', 'doc_order', 'doc_order3']
    for i, (g, h, rn) in enumerate(zip(gold, hneg, rneg)):
        assert rn not in (g, h) and rn.strip(), f'{name} row {i}: random doc collides with gold/hard'

    roles = [ei['query_role'] for ei in df['extra_info']]
    reuse = collections.Counter(rand_src)
    print(f'== {name}')
    print(f'   rows: {len(df)}  pairs: {len(keys)}')
    print(f'   doc_order: {dict(collections.Counter(order))}')
    print(f'   doc_order by role: {dict(collections.Counter(zip(roles, order)))}')
    print(f'   doc_order3: {dict(sorted(collections.Counter(order3).items()))}')
    print(f'   random negatives: pool {len(pool)} docs, {len(reuse)} distinct drawn, '
          f'max reuse {max(reuse.values())}')

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    # Atomic: a run may be reading the old file right now.
    tmp = f'{out}.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, out)
    print(f'   wrote {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='negations-free-with-answer',
                        choices=sorted(DATASETS) + ['all'])
    parser.add_argument('--split', default='test', choices=list(SPLITS) + ['all'])
    parser.add_argument('--base', default=None, help='override the registry base parquet')
    parser.add_argument('--raw', default=None, help='override the registry raw parquet')
    parser.add_argument('--out', default=None, help='default: <base>_docs.parquet')
    args = parser.parse_args()

    names = sorted(DATASETS) if args.dataset == 'all' else [args.dataset]
    splits = list(SPLITS) if args.split == 'all' else [args.split]
    if (len(names) > 1 or len(splits) > 1) and (args.base or args.raw or args.out):
        parser.error('--base/--raw/--out only apply to a single --dataset and --split')
    for name in names:
        cfg = DATASETS[name]
        for split in splits:
            base = args.base or cfg['base'][split]
            build(f'{name} [{split}]', base, args.raw or cfg['raw'][split],
                  args.out or base.replace('.parquet', '_docs.parquet'),
                  cfg['key_col'], cfg['title_col'])


if __name__ == '__main__':
    main()
