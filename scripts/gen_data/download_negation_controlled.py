"""
Download RoxanaMaria/gpt4o-negation-controlled and save all splits as parquet.

Output columns: topic, q1, doc1, q2, doc2, negation
"""

import os
import argparse
import datasets
import pandas as pd

HF_REPO = 'RoxanaMaria/gpt4o-negation-free'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/negation_controlled')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    print(f"Loading {HF_REPO} ...")
    raw = datasets.load_dataset(HF_REPO)

    splits = list(raw.keys())
    print(f"  Splits: {splits}")
    print(f"  Columns: {raw[splits[0]].column_names}")

    os.makedirs(args.local_dir, exist_ok=True)

    for split_name in splits:
        df = raw[split_name].to_pandas()
        out_path = os.path.join(args.local_dir, f'{split_name}.parquet')
        df.to_parquet(out_path, index=False)
        print(f"  [{split_name}] {len(df)} rows → {out_path}")

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print(f"  Copied to HDFS: {args.hdfs_dir}")
