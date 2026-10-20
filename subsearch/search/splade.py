"""SPLADE encoding and a GPU sparse-matrix index.

One definition of the representation, shared by retrieval_server.py (queries),
splade_index_builder.py (passages) and retriever_finetune (retriever-only eval):

    w_j = max_i log(1 + relu(logit_ij)) over the text's tokens i, j in the vocabulary

A text becomes a vocabulary-sized vector (30,522 for BERT) that is mostly zero, and
score(q, d) = w(q) . w(d). SPLADE-v3 uses no query/passage prefixes.

The index is a list of CSR shards (rows = passages in corpus order, columns =
vocabulary) kept on the GPUs; a search is shard @ queries.T per shard, then a top-k
merge. Exact (brute force over every stored weight), like the Flat FAISS index the
e5 retriever uses, so no inverted-index software (Java / pyserini) is needed.

Index directory layout:
    meta.json            model, max_length, max_terms, n_docs, vocab_size, nnz, shards
    shard_00000.pt       {'start': first row, 'crow': int64, 'col': int32, 'val': fp16}
    shard_00000.json     {'start', 'n_rows', 'nnz'}: marks the shard as finished
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SpladeModel:
    """SPLADE encoder: texts -> dense (B, vocab) float32 weight vectors."""

    def __init__(self, path, device='cuda', max_length=256, bf16=True):
        self.device = torch.device(device)
        self.max_length = max_length
        # Pre-Ampere GPUs (the cluster's Titan X nodes) have no bf16 GEMMs: fp32 there.
        self.bf16 = bf16 and self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(path).to(self.device).eval()
        self.vocab_size = self.model.config.vocab_size

    @torch.no_grad()
    def encode(self, texts, max_length=None):
        batch = self.tokenizer(list(texts), max_length=max_length or self.max_length,
                               padding=True, truncation=True, return_tensors='pt').to(self.device)
        dtype = torch.bfloat16 if self.bf16 else torch.float32
        with torch.autocast(self.device.type, dtype=dtype, enabled=self.bf16):
            logits = self.model(**batch).logits                     # (B, L, V)
        return splade_pool(logits, batch['attention_mask'])


def splade_pool(logits, attention_mask):
    """(B, L, V) MLM logits -> (B, V) SPLADE weights, max_i log(1 + relu(logit_i)).
    log1p(relu(.)) is monotonic, so the max over tokens is taken on the raw logits
    first: (B, V) instead of a (B, L, V) float32 tensor. Same values and gradients
    (the gradient reaches the arg-max token either way)."""
    logits = logits.masked_fill(attention_mask[..., None] == 0, float('-inf'))
    return torch.log1p(torch.relu(logits.max(dim=1).values.float()))


def flops(weights):
    """FLOPS regulariser (SPLADE): sum over vocabulary terms of the squared mean
    weight in the batch. Penalises terms active in many texts, which is what makes an
    inverted index slow; rare, specific terms cost little."""
    return (weights.mean(dim=0) ** 2).sum()


def sparsify(dense, rows, max_terms=None):
    """Nonzero weights of a (B, V) matrix as COO triples on the CPU, keeping at most
    max_terms per row. `rows` (B,) gives each row's id within its shard."""
    if max_terms and max_terms < dense.size(1):
        top = dense.topk(max_terms, dim=1)
        dense = torch.zeros_like(dense).scatter_(1, top.indices, top.values)
    nz = dense.nonzero(as_tuple=False)
    return (rows.to(dense.device)[nz[:, 0]].cpu(), nz[:, 1].to(torch.int32).cpu(),
            dense[nz[:, 0], nz[:, 1]].half().cpu())


def write_shard(index_dir, shard_id, start, n_rows, rows, cols, vals):
    """Write rows [start, start + n_rows) of the corpus as one CSR shard, from COO
    triples in any order (the builder encodes length-sorted batches). Returns nnz."""
    key = rows.long() * 2**20 + cols.long()          # vocab < 2^20: sort by row, then col
    order = key.argsort()
    rows, cols, vals = rows[order], cols[order], vals[order]
    crow = torch.zeros(n_rows + 1, dtype=torch.int64)
    crow[1:] = torch.bincount(rows.long(), minlength=n_rows).cumsum(0)
    path = Path(index_dir) / f'shard_{shard_id:05d}.pt'
    tmp = path.with_name(path.name + '.tmp')
    torch.save({'start': start, 'crow': crow, 'col': cols, 'val': vals}, tmp)
    os.replace(tmp, path)
    (Path(index_dir) / f'shard_{shard_id:05d}.json').write_text(
        json.dumps({'start': start, 'n_rows': n_rows, 'nnz': int(vals.numel())}))
    return int(vals.numel())


class SparseIndex:
    """CSR shards spread round-robin over `devices`; exact top-k by dot product."""

    def __init__(self, index_dir, devices=None, log=print):
        self.dir = Path(index_dir)
        self.meta = json.loads((self.dir / 'meta.json').read_text())
        self.vocab_size = self.meta['vocab_size']
        if devices is None:
            n = torch.cuda.device_count()
            devices = [f'cuda:{i}' for i in range(n)] if n else ['cpu']
        self.shards = []   # (start, n_rows, csr tensor)
        for i, name in enumerate(self.meta['shards']):
            s = torch.load(self.dir / name, map_location='cpu', weights_only=True)
            dev = torch.device(devices[i % len(devices)])
            n_rows = s['crow'].numel() - 1
            # torch's CSR needs crow and col of one dtype; int32 while nnz fits.
            idx_dtype = torch.int32 if s['col'].numel() < 2**31 else torch.int64
            csr = torch.sparse_csr_tensor(s['crow'].to(idx_dtype), s['col'].to(idx_dtype),
                                          s['val'].float(), size=(n_rows, self.vocab_size),
                                          device=dev)
            self.shards.append((s['start'], n_rows, csr))
            del s
        self.n_docs = sum(n for _, n, _ in self.shards)
        assert self.n_docs == self.meta['n_docs'], f'{self.n_docs} rows in shards, meta says {self.meta["n_docs"]}'
        log(f'[splade] loaded {len(self.shards)} shards, {self.n_docs} docs, '
            f'{self.meta["nnz"]} weights on {sorted(set(devices))}')

    @torch.no_grad()
    def search(self, q_dense, k):
        """q_dense: (B, vocab). Returns (scores (B, k), row ids (B, k)) on the CPU."""
        best_s, best_i = [], []
        for start, n_rows, csr in self.shards:
            s = torch.sparse.mm(csr, q_dense.to(csr.device, torch.float32).T)   # (n_rows, B)
            top = s.topk(min(k, n_rows), dim=0)
            best_s.append(top.values.T.cpu())
            best_i.append(top.indices.T.cpu() + start)
        scores, idx = torch.cat(best_s, 1), torch.cat(best_i, 1)
        top = scores.topk(min(k, scores.size(1)), dim=1)
        return top.values, idx.gather(1, top.indices)
