"""Negation minimal pairs (plus optional MS MARCO replay) for retriever training:
loading, mined negatives, batches.

Documents are identified by text throughout. The same text can belong to several
pairs (NevIR pairs with the same id prefix share a document; negations-free reuses
passages), and a batch holds each text once, so a shared document is one
candidate with one embedding rather than a false negative of itself.
"""
import random
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
import torch

SHORT_SOURCE = {
    'nevir-with-answers': 'nevir',
    'negations-free-sentence-with-answer': 'nfs',
}


@dataclass
class Pair:
    source: str     # short name: nevir / nfs
    pair_id: str
    family: str     # pairs whose documents are copies of one passage
    q1: str
    q2: str
    doc1: str       # q1's own document
    doc2: str       # q2's own document


@dataclass
class MarcoQuery:
    query: str
    positives: list     # train: [pos_text]; val: all labelled positives
    negatives: list     # BM25 negatives: train one, val val_negs_per_query


def load_marco(path):
    """MS MARCO rows written by prepare_marco.py (train or val parquet)."""
    df = pd.read_parquet(path)
    if 'pos_text' in df:
        return [MarcoQuery(r.query, [r.pos_text], [r.neg_text]) for r in df.itertuples(index=False)]
    return [MarcoQuery(r.query, list(r.pos_texts), list(r.neg_texts)) for r in df.itertuples(index=False)]


def question_of(prompt):
    # Same rule as scripts/data_process/drop_conflicting_pairs.py.
    return prompt[0]['content'].rsplit('Question:', 1)[-1].strip()


def family_of(source, pair_id):
    # NevIR ids are '<passage>-<variant>'. Two variants of one passage (139 of the
    # 335 train prefixes) share one document, and their other documents are
    # edits of the same text: near-copies, not negatives.
    if source == 'nevir':
        return f'{source}:{pair_id.rsplit("-", 1)[0]}'
    return f'{source}:{pair_id}'


def load_pairs(files):
    """Pairs from *_docs.parquet files, joined on (data_source, pair_id)."""
    halves = defaultdict(dict)
    for path in files:
        df = pd.read_parquet(path)
        for row in df.itertuples(index=False):
            source = SHORT_SOURCE.get(row.data_source, row.data_source)
            info = row.extra_info
            key = (source, str(info['pair_id']))
            role = info['query_role']
            assert role not in halves[key], f'{path}: two {role} rows for pair {key}'
            halves[key][role] = (question_of(row.prompt), row.gold_doc, row.hard_neg_doc)

    pairs = []
    for (source, pair_id), h in halves.items():
        assert set(h) == {'q1', 'q2'}, f'incomplete pair {source}/{pair_id}: {sorted(h)}'
        (q1, doc1, other1), (q2, doc2, other2) = h['q1'], h['q2']
        assert doc1 == other2 and doc2 == other1, f'gold/hard_neg docs disagree in pair {source}/{pair_id}'
        assert doc1 != doc2, f'identical documents in pair {source}/{pair_id}'
        pairs.append(Pair(source, pair_id, family_of(source, pair_id), q1, q2, doc1, doc2))
    return pairs


def check_no_leakage(train, val, test):
    """No train question in val/test and no val question in test (the
    build_golden_training.py rule, lowercased)."""
    def questions(pairs):
        return {q.lower() for p in pairs for q in (p.q1, p.q2)}
    tr, va, te = questions(train), questions(val), questions(test)
    assert not tr & va, f'{len(tr & va)} train questions in val, e.g. {sorted(tr & va)[:3]}'
    assert not tr & te, f'{len(tr & te)} train questions in test, e.g. {sorted(tr & te)[:3]}'
    assert not va & te, f'{len(va & te)} val questions in test, e.g. {sorted(va & te)[:3]}'


def families_by_text(pairs):
    """Text -> families whose pairs contain it."""
    out = defaultdict(set)
    for p in pairs:
        out[p.doc1].add(p.family)
        out[p.doc2].add(p.family)
    return out


@torch.no_grad()
def mine_negatives(pairs, encoder, top_k, batch_size, bf16):
    """Per query (order q1, q2 of pairs[0], q1, q2 of pairs[1], ...), the top_k train
    documents under `encoder`, excluding every document of the query's family."""
    text_families = families_by_text(pairs)
    texts = list(text_families)
    d = encoder.embed_all(texts, is_query=False, batch_size=batch_size, bf16=bf16)
    q = encoder.embed_all([x for p in pairs for x in (p.q1, p.q2)], is_query=True,
                          batch_size=batch_size, bf16=bf16)
    scores = q @ d.T
    family_columns = defaultdict(list)
    for j, t in enumerate(texts):
        for fam in text_families[t]:
            family_columns[fam].append(j)
    for i, p in enumerate(pairs):
        scores[2 * i:2 * i + 2, family_columns[p.family]] = float('-inf')
    top = scores.topk(top_k, dim=1).indices.tolist()
    return [[texts[j] for j in row] for row in top]


@dataclass
class Batch:
    queries: list       # Q = 2P negation queries (q1, q2 of each pair), then M MARCO queries
    docs: list          # N unique candidate texts: pair docs, mined, MARCO pos + BM25 neg
    pos: torch.Tensor   # (Q,) own doc column
    partner: torch.Tensor  # (Q,) partner doc column; -1 for MARCO rows
    mask: torch.Tensor  # (Q, N) True = same-family near-copy, not a candidate
    sources: list       # Q: nevir / nfs / marco
    n_neg: int          # rows [0, n_neg) are negation queries, the rest MARCO


def build_batch(pairs, mined=None, num_mined=0, rng=None, text_families=None,
                marco=None, marco_hard=1):
    """Candidates for a batch of pairs. `mined` holds each pair's (q1 pool, q2 pool);
    `text_families` enables the same-family mask (training only: evaluation ranks
    against everything, as the real index does).

    `marco` (MarcoQuery list) appends MARCO queries after the negation queries. Every
    query ranks the same shared candidate set, so MARCO passages are extra negatives
    for negation queries and negation documents are extra negatives for MARCO ones.
    Each MARCO query adds its positive and up to `marco_hard` BM25 negatives."""
    docs, column = [], {}

    def add(text):
        if text not in column:
            column[text] = len(docs)
            docs.append(text)
        return column[text]

    queries, pos, partner, families, sources = [], [], [], [], []
    for p in pairs:
        c1, c2 = add(p.doc1), add(p.doc2)
        queries += [p.q1, p.q2]
        pos += [c1, c2]
        partner += [c2, c1]
        families += [p.family, p.family]
        sources += [p.source, p.source]
    if num_mined:
        for pools in mined:
            for pool in pools:
                for text in rng.sample(pool, num_mined):
                    add(text)
    n_neg = len(queries)
    for m in marco or []:
        queries.append(m.query)
        pos.append(add(m.positives[0]))
        partner.append(-1)
        sources.append('marco')
        for text in m.negatives[:marco_hard]:
            add(text)

    mask = torch.zeros(len(queries), len(docs), dtype=torch.bool)
    if text_families is not None:
        for i, fam in enumerate(families):     # negation rows only
            for j, text in enumerate(docs):
                if fam in text_families.get(text, ()) and j != pos[i] and j != partner[i]:
                    mask[i, j] = True
    return Batch(queries, docs, torch.tensor(pos), torch.tensor(partner), mask, sources, n_neg)


def train_batches(pairs, batch_pairs, rng, mined=None, num_mined=0, text_families=None,
                  marco=None, marco_hard=1):
    """One epoch of shuffled batches. `mined` is mine_negatives' output for `pairs`.

    MARCO queries are shuffled and split as evenly as possible over the epoch's
    batches, so both sets are used exactly once per epoch (1:1 when len(marco) equals
    the number of negation queries)."""
    order = list(range(len(pairs)))
    rng.shuffle(order)
    starts = range(0, len(order), batch_pairs)
    chunks = [None] * len(starts)
    if marco:
        m_order = list(range(len(marco)))
        rng.shuffle(m_order)
        n = len(starts)
        bounds = [round(k * len(m_order) / n) for k in range(n + 1)]
        chunks = [[marco[i] for i in m_order[bounds[k]:bounds[k + 1]]] for k in range(n)]
    for start, chunk in zip(starts, chunks):
        idx = order[start:start + batch_pairs]
        pools = [(mined[2 * i], mined[2 * i + 1]) for i in idx] if num_mined else None
        yield build_batch([pairs[i] for i in idx], pools, num_mined, rng, text_families,
                          chunk, marco_hard)
