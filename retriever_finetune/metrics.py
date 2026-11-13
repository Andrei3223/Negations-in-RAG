"""Ranking metrics: for negation pairs, in-batch and pool nDCG / recall / MRR and the
NevIR pairwise accuracy; for MS MARCO, multi-positive nDCG / recall / MRR@10."""
import math
import random
from collections import defaultdict

import torch

from .data import build_batch


def ranking_metrics(scores, own, partner, ks):
    """Per-query metrics of own-doc retrieval. scores (Q, N), own/partner (Q,).

    nDCG@k: binary relevance (own doc = 1). nDCG@k_graded: own doc 2, partner 1,
    gain 2^rel - 1, i.e. the own > partner > rest order the loss trains for.
    Ties count in the query's favour.
    """
    s_own = scores.gather(1, own[:, None])
    s_par = scores.gather(1, partner[:, None])
    r_own = (scores > s_own).sum(1) + 1
    r_par = (scores > s_par).sum(1) + 1
    disc_own = 1.0 / torch.log2(r_own.float() + 1)
    disc_par = 1.0 / torch.log2(r_par.float() + 1)
    out = {'mrr': 1.0 / r_own.float(),
           'query_acc': (s_own > s_par).squeeze(1).float()}
    for k in ks:
        out[f'recall@{k}'] = (r_own <= k).float()
        out[f'ndcg@{k}'] = disc_own * (r_own <= k)
        ideal = 3.0 + (1.0 / math.log2(3) if k >= 2 else 0.0)
        out[f'ndcg@{k}_graded'] = (3.0 * disc_own * (r_own <= k) + disc_par * (r_par <= k)) / ideal
    return out


class Evaluator:
    """Fixed evaluation of one split.

    pool_*:    each query ranks every unique document of the split.
    inbatch_*: each query ranks the documents of its own batch of `batch_pairs`
               pairs. The batches are a fixed shuffle, identical across calls.
    pairwise_acc: a pair counts only if q1 prefers doc1 over doc2 AND q2 prefers
               doc2 over doc1 (NevIR); query_acc is the per-query version.

    With `corpus` (a list of texts), the pool is that corpus instead of the split's
    docs: e.g. the 3,346-doc custom corpus the agent's custom_* retrievers search,
    which holds doc1/doc2 of every split. Corpus entries are matched to pair docs by
    body text, since the corpus titles NevIR docs "nevir_<id>" where the parquet
    files use "Document".
    """

    def __init__(self, pairs, cfg_eval, corpus=None):
        self.pairs = pairs
        self.ks = list(cfg_eval.ks)
        self.encode_batch_size = cfg_eval.encode_batch_size
        self.queries = [q for p in pairs for q in (p.q1, p.q2)]
        self.sources = [p.source for p in pairs for _ in range(2)]
        pool = build_batch(pairs)
        self.docs, self.own, self.partner = pool.docs, pool.pos, pool.partner
        column = {t: j for j, t in enumerate(self.docs)}
        if corpus is not None:
            def body(t):
                return t.split('\n', 1)[-1].strip()
            in_corpus = {}
            for j, t in enumerate(corpus):
                in_corpus.setdefault(body(t), j)
            missing = [d for d in self.docs if body(d) not in in_corpus]
            assert not missing, f'{len(missing)} pair docs not in the corpus, e.g. {missing[0][:100]!r}'
            column = {t: in_corpus[body(t)] for t in self.docs}
            remap = torch.tensor([column[t] for t in self.docs])
            self.own, self.partner = remap[self.own], remap[self.partner]
            self.docs = list(corpus)

        order = list(range(len(pairs)))
        random.Random(cfg_eval.shuffle_seed).shuffle(order)
        self.batches = []   # (query rows, pool columns, local own, local partner)
        for start in range(0, len(order), cfg_eval.batch_pairs):
            idx = order[start:start + cfg_eval.batch_pairs]
            b = build_batch([pairs[i] for i in idx])
            rows = torch.tensor([2 * i + r for i in idx for r in (0, 1)])
            cols = torch.tensor([column[t] for t in b.docs])
            self.batches.append((rows, cols, b.pos, b.partner))

    @torch.no_grad()
    def evaluate(self, encoder, bf16):
        q = encoder.embed_all(self.queries, True, self.encode_batch_size, bf16).cpu()
        d = encoder.embed_all(self.docs, False, self.encode_batch_size, bf16).cpu()
        scores = q @ d.T

        per_query = {f'pool_{k}': v for k, v in
                     ranking_metrics(scores, self.own, self.partner, self.ks).items()}
        query_acc = per_query.pop('pool_query_acc')   # candidate-independent
        inbatch = defaultdict(lambda: torch.zeros(len(self.queries)))
        for rows, cols, own, partner in self.batches:
            sub = scores[rows][:, cols]
            for k, v in ranking_metrics(sub, own, partner, self.ks).items():
                if k != 'query_acc':
                    inbatch[f'inbatch_{k}'][rows] = v
        per_query.update(inbatch)
        per_query['query_acc'] = query_acc
        pair_acc = query_acc.view(-1, 2).min(dim=1).values   # both halves right

        sources = sorted(set(self.sources))
        out = {}
        for name in ['all'] + sources:
            qm = torch.tensor([name in ('all', s) for s in self.sources])
            pm = qm.view(-1, 2)[:, 0]
            for k, v in per_query.items():
                out[f'{name}/{k}'] = v[qm].mean().item()
            out[f'{name}/pairwise_acc'] = pair_acc[pm].mean().item()
            out[f'{name}/n_pairs'] = int(pm.sum())
        if getattr(encoder, 'sparse', False):   # SPLADE: index size / search cost
            out['all/terms_per_query'] = (q > 0).sum(1).float().mean().item()
            out['all/terms_per_doc'] = (d > 0).sum(1).float().mean().item()
        return out


def multi_positive_metrics(scores, rel, ks):
    """Per-query metrics with any number of labelled positives. scores, rel: (Q, N),
    rel boolean. nDCG@k is binary, IDCG over min(k, #positives); recall@k is the
    fraction of a query's positives in the top k; mrr@10 is MS MARCO's metric."""
    order = scores.argsort(dim=1, descending=True)
    hits = rel.gather(1, order).float()                       # (Q, N) in rank order
    n_pos = rel.sum(1).float()
    disc = 1.0 / torch.log2(torch.arange(scores.size(1), dtype=torch.float) + 2)
    out = {}
    for k in ks:
        dcg = (hits[:, :k] * disc[:k]).sum(1)
        ideal = torch.stack([disc[:min(k, int(n))].sum() for n in n_pos])
        out[f'ndcg@{k}'] = dcg / ideal
        out[f'recall@{k}'] = hits[:, :k].sum(1) / n_pos
    first = hits[:, :10].argmax(1)                            # 0 when no hit; masked below
    out['mrr@10'] = (hits[:, :10].sum(1) > 0).float() / (first.float() + 1)
    return out


class MarcoEvaluator:
    """MS MARCO val (prepare_marco.py): 500 held-out queries, their labelled positives
    and sampled BM25 negatives.

    pool_*:    each query ranks every val passage (~10.5k).
    inbatch_*: fixed batches of `2 * batch_pairs` queries rank the batch's positives plus
               each query's first BM25 negative, as in a training step.
    Labels are sparse (unlabelled positives exist), so values are pessimistic; compare
    against the step-0 baseline.
    """

    def __init__(self, rows, cfg_eval):
        self.ks = list(cfg_eval.ks)
        self.encode_batch_size = cfg_eval.encode_batch_size
        self.queries = [r.query for r in rows]
        self.docs, column = [], {}
        for r in rows:
            for t in r.positives + r.negatives:
                if t not in column:
                    column[t] = len(self.docs)
                    self.docs.append(t)
        self.rel = torch.zeros(len(rows), len(self.docs), dtype=torch.bool)
        for i, r in enumerate(rows):
            self.rel[i, [column[t] for t in r.positives]] = True

        order = list(range(len(rows)))
        random.Random(cfg_eval.shuffle_seed).shuffle(order)
        size = 2 * cfg_eval.batch_pairs
        self.batches = []   # (query rows, pool columns)
        for start in range(0, len(order), size):
            idx = order[start:start + size]
            cols = sorted({column[t] for i in idx for t in rows[i].positives + rows[i].negatives[:1]})
            self.batches.append((torch.tensor(idx), torch.tensor(cols)))

    @torch.no_grad()
    def evaluate(self, encoder, bf16):
        q = encoder.embed_all(self.queries, True, self.encode_batch_size, bf16).cpu()
        d = encoder.embed_all(self.docs, False, self.encode_batch_size, bf16).cpu()
        scores = q @ d.T
        per_query = {f'pool_{k}': v for k, v in multi_positive_metrics(scores, self.rel, self.ks).items()}
        inbatch = defaultdict(lambda: torch.zeros(len(self.queries)))
        for rows, cols in self.batches:
            sub = multi_positive_metrics(scores[rows][:, cols], self.rel[rows][:, cols], self.ks)
            for k, v in sub.items():
                inbatch[f'inbatch_{k}'][rows] = v
        per_query.update(inbatch)
        out = {f'marco/{k}': v.mean().item() for k, v in per_query.items()}
        out['marco/n_queries'] = len(self.queries)
        return out
