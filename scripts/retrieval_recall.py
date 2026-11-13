#!/usr/bin/env python3
"""Retrieval-only check on the custom doc1/doc2 corpus: for each test question, where
do its gold document and the other half of its pair rank?

Uses the same conventions as subsearch/search/retrieval_server.py and index_builder.py
("query: " / "passage: " prefixes, mean pooling, L2 norm, inner product), encoding the
corpus on the fly, so it needs no prebuilt index and runs on CPU.

Usage:
  python3 scripts/retrieval_recall.py intfloat/e5-base-v2 <fine-tuned checkpoint dir> ...
"""
import json
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

CORPUS = 'data/custom_index/corpus.jsonl'
TEST = {
    'nfs': 'data/negations_with_answer_rag/negations-free-sentence-with-answer/test_clean_docs.parquet',
    'nevir': 'data/negations_with_answer_rag/nevir-with-answers/test_clean_docs.parquet',
}
KS = (1, 2, 3, 10)


def norm(s):
    return ' '.join(str(s).split())


@torch.no_grad()
def encode(model, tok, texts, prefix, max_length, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = tok([prefix + t for t in texts[i:i + bs]], max_length=max_length, padding=True,
                truncation=True, return_tensors='pt')
        h = model(**b).last_hidden_state
        m = b['attention_mask'].unsqueeze(-1).float()
        out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1))
    return torch.cat(out)


def main(models):
    torch.set_num_threads(32)
    docs = [json.loads(l)['contents'] for l in open(CORPUS)]
    bodies = [norm(d.split('\n', 1)[1] if '\n' in d else d) for d in docs]
    for path in models:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModel.from_pretrained(path).eval()
        D = encode(model, tok, docs, 'passage: ', 256)
        print(f'\n{path}')
        for ds, f in TEST.items():
            t = pd.read_parquet(f)
            qs = [p[0]['content'].split('Question:', 1)[1].strip() for p in t.prompt]
            Q = encode(model, tok, qs, 'query: ', 256)
            order = (Q @ D.T).argsort(dim=1, descending=True)[:, :max(KS)].tolist()
            gold = [norm(g.split('\n', 1)[1]) for g in t.gold_doc]
            other = [norm(h.split('\n', 1)[1]) for h in t.hard_neg_doc]
            rows = []
            for o, g, h in zip(order, gold, other):
                ranked = [bodies[i] for i in o]
                rows.append({f'gold@{k}': g in ranked[:k] for k in KS}
                            | {f'other@{k}': h in ranked[:k] for k in KS}
                            | {'gold_above_other': (ranked.index(g) if g in ranked else 99)
                               < (ranked.index(h) if h in ranked else 99)})
            r = pd.DataFrame(rows).mean()
            print(f'  {ds:5} n={len(t):4}  ' + '  '.join(f'gold@{k} {r[f"gold@{k}"]:.3f}' for k in KS)
                  + f'  | other half @1 {r["other@1"]:.3f} @2 {r["other@2"]:.3f}'
                  + f'  | gold ranked above other half {r["gold_above_other"]:.3f}')


if __name__ == '__main__':
    main(sys.argv[1:] or ['intfloat/e5-base-v2'])
