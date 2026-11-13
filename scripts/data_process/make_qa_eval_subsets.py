#!/usr/bin/env python3
"""Fixed 1,000-question NQ / HotpotQA eval subsets, for checking whether negation
fine-tuning cost general open-domain QA (eval/dataset=nq_1k | hotpotqa_1k).

  NQ        1,000 of NQ test (data/nq_search/test.parquet, 3,610)
  HotpotQA  1,000 of HotpotQA dev (the hotpotqa rows of data/multihop_search/test.parquet,
            7,405), excluding the 250 questions exp3's training used for validation

Rows are copied unchanged (Search-R1 prompt, data_source nq/hotpotqa, reward_model),
sampled with a fixed seed, so rerunning reproduces the same files. Neither subset may
share a question with exp3's training or validation data.

Usage:
  python3 scripts/data_process/make_qa_eval_subsets.py
"""
import os

import pandas as pd

N = 1000
SEED = 42
OUT_DIR = 'data/qa_eval'
EXP3_DIR = 'data/golden_nevir_nfwa_nq_hotpot'   # exp3's training data (train + val)


def questions(df):
    return [p[0]['content'].split('Question:', 1)[1].strip().lower() for p in df.prompt]


def main():
    nq = pd.read_parquet('data/nq_search/test.parquet')
    multi = pd.read_parquet('data/multihop_search/test.parquet')
    hotpot = multi[multi.data_source == 'hotpotqa'].reset_index(drop=True)

    seen = set()
    for split in ('train', 'val'):
        seen |= set(questions(pd.read_parquet(f'{EXP3_DIR}/{split}.parquet', columns=['prompt'])))

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, df in (('nq_test_1k', nq), ('hotpotqa_dev_1k', hotpot)):
        pool = df[[q not in seen for q in questions(df)]]
        sub = pool.sample(n=N, random_state=SEED).reset_index(drop=True)
        qs = questions(sub)
        assert len(sub) == N and len(set(qs)) == N, f'{name}: duplicate questions'
        assert not set(qs) & seen, f'{name}: overlaps exp3 training data'
        assert sub.data_source.nunique() == 1
        out = f'{OUT_DIR}/{name}.parquet'
        sub.to_parquet(out, index=False)
        print(f'{name}: {len(df)} source rows, {len(df) - len(pool)} excluded (in exp3 train/val), '
              f'sampled {len(sub)} -> {out}  [{sub.data_source.iloc[0]}]')


if __name__ == '__main__':
    main()
