"""
Upsample the negation minority sources (nevir + negations-free-sentence) in the
combined train.parquet by row duplication, so they make up a meaningful share of
the training mix instead of ~1%.

The QA majority (nq, hotpotqa) is left as-is; only the minority rows are replicated.
Output is written to a NEW directory so the original combined set is untouched.
val.parquet is copied through unchanged (you evaluate on the real distribution).

Usage:
  cd /home/16207041/Negations-in-RAG

  # target a minority fraction (default 0.10 == 10%):
  python3 scripts/data_process/upsample_minority.py --target_frac 0.10

  # or set the multiplier explicitly:
  python3 scripts/data_process/upsample_minority.py --mult 11
"""

import os
import argparse
import math
import shutil
import pandas as pd

SRC_DIR = "data/combined_nq_hotpot_nevir_negfree"
MINORITY = {"nevir-with-answers", "negations-free-sentence-with-answer"}
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default=SRC_DIR)
    ap.add_argument("--out_dir", default=None,
                    help="default: <src_dir>_upsampled")
    ap.add_argument("--target_frac", type=float, default=0.10,
                    help="desired minority fraction of the final train set")
    ap.add_argument("--mult", type=int, default=None,
                    help="explicit duplication factor; overrides --target_frac")
    args = ap.parse_args()

    out_dir = args.out_dir or (args.src_dir.rstrip("/") + "_upsampled")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(os.path.join(args.src_dir, "train.parquet"))
    is_min = df["data_source"].isin(MINORITY)
    minority = df[is_min]
    majority = df[~is_min]
    n_min, n_maj = len(minority), len(majority)
    print(f"train: {len(df)} rows  (minority={n_min}, majority={n_maj})")

    # mult to reach target fraction f:  f = mult*n_min / (mult*n_min + n_maj)
    if args.mult is not None:
        mult = args.mult
    else:
        f = args.target_frac
        mult = max(1, math.ceil((f * n_maj) / ((1 - f) * n_min)))
    print(f"duplication factor = {mult}x")

    # replicate the minority rows `mult` times, keep majority once
    upsampled = pd.concat([majority] + [minority] * mult, ignore_index=True)
    # shuffle so duplicates are spread across the epoch
    upsampled = upsampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    new_min = len(minority) * mult
    frac = new_min / len(upsampled)
    print(f"out: {len(upsampled)} rows  (minority={new_min} -> {frac:.1%})")

    upsampled.to_parquet(os.path.join(out_dir, "train.parquet"))
    shutil.copy(os.path.join(args.src_dir, "val.parquet"),
                os.path.join(out_dir, "val.parquet"))
    print(f"wrote {out_dir}/train.parquet  (+ val.parquet copied unchanged)")


if __name__ == "__main__":
    main()
