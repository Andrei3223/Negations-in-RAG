"""Build the MS MARCO replay data for retriever fine-tuning from the shared copy of
MS MARCO v1 passage (no download). Settings: conf/marco/msmarco_v1.yaml.

    python -m retriever_finetune.prepare_marco [hydra overrides, e.g. marco.num_train=891]

Writes <marco.data_dir>/:
  train.parquet  num_train rows {qid, query, pos_text, neg_text}: the first triple of
                 each new query in the line-shuffled train triples (a random sample);
                 neg_text is that triple's BM25 negative.
  val.parquet    the 500 queries of the held-out train-eval triples {qid, query,
                 pos_texts[], neg_texts[]}: all their positives plus val_negs_per_query
                 BM25 negatives sampled with `seed`. None of these queries are in the
                 train triples.
  meta.json      sources, seed and counts.
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from hydra import compose, initialize

TRAIN_TRIPLES = 'msmarco-qidpidtriples.rnd-shuf.train.tsv'
EVAL_TRIPLES = 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv'
QUERIES = 'queries.train.tsv'
COLLECTION = 'collection.tsv'


def read_val(path, negs_per_query, rng):
    pos, neg = defaultdict(set), defaultdict(set)
    with open(path) as f:
        for line in f:
            qid, p, n = line.rstrip('\n').split('\t')
            pos[qid].add(p)
            neg[qid].add(n)
    val = {}
    for qid in sorted(pos):
        candidates = sorted(neg[qid] - pos[qid])   # a BM25 negative can be a labelled positive
        val[qid] = (sorted(pos[qid]), rng.sample(candidates, min(negs_per_query, len(candidates))))
    return val


def read_train(path, num, exclude):
    train, seen = [], set()
    with open(path) as f:
        for line in f:
            qid, p, n = line.rstrip('\n').split('\t')
            if qid in seen or qid in exclude:
                continue
            seen.add(qid)
            train.append((qid, p, n))
            if len(train) == num:
                break
    assert len(train) == num, f'only {len(train)} distinct queries in {path}'
    return train


def read_ids(path, wanted):
    """{id: text} for the wanted ids of a `id \\t text` file, stopping once all are found."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            i, text = line.rstrip('\n').split('\t', 1)
            if i in wanted:
                out[i] = text
                if len(out) == len(wanted):
                    break
    missing = wanted - out.keys()
    assert not missing, f'{len(missing)} ids missing from {path}, e.g. {sorted(missing)[:5]}'
    return out


def main():
    with initialize(config_path='conf', version_base='1.3'):
        cfg = compose('config', overrides=sys.argv[1:]).marco
    src, out = Path(cfg.source_dir), Path(cfg.data_dir)
    t0 = time.time()

    val = read_val(src / EVAL_TRIPLES, cfg.val_negs_per_query, random.Random(cfg.seed))
    train = read_train(src / TRAIN_TRIPLES, cfg.num_train, exclude=set(val))
    assert not {q for q, _, _ in train} & set(val), 'train and val share queries'

    qids = {q for q, _, _ in train} | set(val)
    pids = {p for _, p, n in train} | {n for _, p, n in train}
    for pos, neg in val.values():
        pids |= set(pos) | set(neg)
    print(f'{len(train)} train + {len(val)} val queries, {len(pids)} passages to fetch')
    queries = read_ids(src / QUERIES, qids)
    passages = read_ids(src / COLLECTION, pids)
    print(f'read texts in {time.time() - t0:.0f}s')

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'qid': q, 'query': queries[q], 'pos_text': passages[p], 'neg_text': passages[n]}
                  for q, p, n in train]).to_parquet(out / 'train.parquet', index=False)
    pd.DataFrame([{'qid': q, 'query': queries[q],
                   'pos_texts': [passages[p] for p in pos], 'neg_texts': [passages[n] for n in neg]}
                  for q, (pos, neg) in val.items()]).to_parquet(out / 'val.parquet', index=False)
    meta = {
        'source_dir': str(src), 'train_triples': TRAIN_TRIPLES, 'val_triples': EVAL_TRIPLES,
        'seed': cfg.seed, 'num_train': len(train), 'num_val': len(val),
        'val_positives': sum(len(p) for p, _ in val.values()),
        'val_negatives': sum(len(n) for _, n in val.values()),
        'val_unique_passages': len({passages[p] for pos, neg in val.values() for p in pos + neg}),
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
