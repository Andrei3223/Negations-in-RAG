"""Build a SPLADE sparse index (see splade.py) over a jsonl corpus with a "contents" field.

Rows are numbered in the order retrieval_server.load_corpus reads the corpus, so the
row ids the server returns map to the same passages. The corpus is cut into shards of
--shard_size rows; worker w of N encodes every shard s with s % N == w, so N processes
(one per GPU, see run/build_splade_index.job) split the work. Finished shards are
skipped on restart. --finalize checks every shard and writes meta.json.

    python subsearch/search/splade_index_builder.py --model naver/splade-v3 \
        --corpus_path data/custom_index/corpus.jsonl --index_dir data/custom_index_splade_v3
    python subsearch/search/splade_index_builder.py ... --finalize
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from splade import SpladeModel, sparsify, write_shard  # noqa: E402


def load_corpus(corpus_path):
    # Same call as retrieval_server.load_corpus (not imported: that module pulls in faiss).
    import datasets
    return datasets.load_dataset('json', data_files=corpus_path, split='train', num_proc=4)


def build_shard(model, texts, args):
    """COO triples of one shard. Batches are length-sorted to cut padding."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    rows, cols, vals = [], [], []
    for b in range(0, len(order), args.batch_size):
        idx = order[b:b + args.batch_size]
        dense = model.encode([texts[i] for i in idx])
        r, c, v = sparsify(dense, torch.tensor(idx), args.max_terms)
        rows.append(r); cols.append(c); vals.append(v)
    return torch.cat(rows), torch.cat(cols), torch.cat(vals)


def finalize(args, n_docs):
    n_shards = math.ceil(n_docs / args.shard_size)
    names, nnz, covered = [], 0, 0
    for s in range(n_shards):
        side = Path(args.index_dir) / f'shard_{s:05d}.json'
        assert side.exists(), f'shard {s} missing: rerun the builder workers'
        info = json.loads(side.read_text())
        assert info['start'] == covered, f'shard {s} starts at {info["start"]}, expected {covered}'
        covered += info['n_rows']
        nnz += info['nnz']
        names.append(f'shard_{s:05d}.pt')
    assert covered == n_docs, f'shards cover {covered} rows of {n_docs}'
    from transformers import AutoConfig
    meta = {'model': args.model, 'corpus_path': str(Path(args.corpus_path).resolve()),
            'n_docs': n_docs, 'vocab_size': AutoConfig.from_pretrained(args.model).vocab_size,
            'max_length': args.max_length, 'max_terms': args.max_terms,
            'shard_size': args.shard_size, 'nnz': nnz, 'avg_terms_per_doc': nnz / n_docs,
            'shards': names}
    (Path(args.index_dir) / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'[finalize] {n_docs} docs, {n_shards} shards, {nnz} weights '
          f'({nnz / n_docs:.1f} per doc, ~{nnz * 8 / 2**30:.1f} GiB on GPU as fp32+int32)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='naver/splade-v3')
    p.add_argument('--corpus_path', required=True)
    p.add_argument('--index_dir', required=True)
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--max_terms', type=int, default=256, help='weights kept per passage (0 = all)')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--shard_size', type=int, default=500_000, help='rows per shard')
    p.add_argument('--worker_id', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--limit', type=int, default=0, help='only the first N rows (sizing runs)')
    p.add_argument('--finalize', action='store_true', help='check the shards and write meta.json')
    args = p.parse_args()
    Path(args.index_dir).mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.corpus_path)
    n_docs = min(len(corpus), args.limit) if args.limit else len(corpus)
    if args.finalize:
        finalize(args, n_docs)
        return

    model = SpladeModel(args.model, 'cuda', args.max_length)
    n_shards = math.ceil(n_docs / args.shard_size)
    mine = [s for s in range(n_shards) if s % args.num_workers == args.worker_id]
    print(f'[worker {args.worker_id}/{args.num_workers}] {n_docs} docs, {n_shards} shards, '
          f'mine: {len(mine)}', flush=True)
    for s in mine:
        if (Path(args.index_dir) / f'shard_{s:05d}.json').exists():
            print(f'[worker {args.worker_id}] shard {s} done already, skipping', flush=True)
            continue
        start, end = s * args.shard_size, min((s + 1) * args.shard_size, n_docs)
        t0 = time.time()
        texts = corpus[start:end]['contents']
        rows, cols, vals = build_shard(model, texts, args)
        nnz = write_shard(args.index_dir, s, start, end - start, rows, cols, vals)
        dt = time.time() - t0
        print(f'[worker {args.worker_id}] shard {s}: rows {start}-{end}, {nnz / (end - start):.1f} '
              f'terms/doc, {dt:.0f}s ({(end - start) / dt:.0f} docs/s)', flush=True)


if __name__ == '__main__':
    main()
