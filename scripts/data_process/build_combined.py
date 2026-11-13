"""
Build combined train/val parquet files from four sources:

  TRAIN = NQ (local) + HotpotQA (HF) + nevir clean + negations-free-sentence
  VAL   = NQ dev (HF, capped) + HotpotQA dev (HF, capped)
          + nevir clean validation + negations-free-sentence validation

All rows are normalized to the 5 columns the trainer reads:
  data_source, prompt, ability, reward_model, extra_info
extra_info is unified to a single struct so the parquet schema is consistent.

Usage:
  cd $REPO_ROOT
  python3 scripts/data_process/build_combined.py \
      --out_dir data/combined_nq_hotpot_nevir_negfree \
      --val_qa_cap 500
"""

import os
import argparse
import pandas as pd
import datasets

# ---- source locations -------------------------------------------------------
NQ_TRAIN_LOCAL = "data/nq_search/train.parquet"          # reuse existing (79,168)
NEVIR_TRAIN    = "data/negations_with_answer_rag/nevir-with-answers/train_clean.parquet"
NEVIR_VAL      = "data/negations_with_answer_rag/nevir-with-answers/validation_clean.parquet"
NEGFREE_TRAIN  = "data/negations_with_answer_rag/negations-free-sentence-with-answer/train.parquet"
NEGFREE_VAL    = "data/negations_with_answer_rag/negations-free-sentence-with-answer/validation.parquet"

HF_REPO        = "RUC-NLPIR/FlashRAG_datasets"
HOTPOT_CONFIG  = "hotpotqa"
NQ_CONFIG      = "nq"
SEED           = 42

# Common columns kept in the output
COLS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
EXTRA_KEYS = ["split", "index", "pair_id", "query_role", "negation_type"]


# ---- prompt template (matches qa_search_train_merge.py) ---------------------
def make_prefix(question: str) -> str:
    return (
        f"Answer the given question. "
        f"You must conduct reasoning inside <think> and </think> first every time you get new information. "
        f"After reasoning, if you find you lack some knowledge, you can call a search engine by "
        f"<search> query </search> and it will return the top searched results between <information> and "
        f"</information>. You can search as many times as your want. If you find no further external "
        f"knowledge needed, you can directly provide the answer inside <answer> and </answer>, without "
        f"detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
    )


def unify_extra(src: dict, split: str, index: int) -> dict:
    """Force a single, consistent extra_info struct across all sources."""
    out = {k: None for k in EXTRA_KEYS}
    if src:
        for k in EXTRA_KEYS:
            if k in src and src[k] is not None:
                out[k] = src[k]
    out["split"] = split
    out["index"] = int(index)
    # normalize types so pyarrow infers a stable schema
    out["pair_id"] = None if out["pair_id"] is None else str(out["pair_id"])
    out["query_role"] = None if out["query_role"] is None else str(out["query_role"])
    out["negation_type"] = None if out["negation_type"] is None else int(out["negation_type"])
    return out


def to_target_list(target) -> list:
    """reward_model.ground_truth.target -> plain list[str]."""
    if target is None:
        return []
    try:
        return [str(t) for t in list(target)]
    except TypeError:
        return [str(target)]


# ---- normalizers ------------------------------------------------------------
def norm_existing(df: pd.DataFrame, split: str) -> list:
    """Normalize an already-processed parquet (negation sets, local NQ)."""
    records = []
    for i, row in enumerate(df.to_dict("records")):
        prompt = row["prompt"]
        content = prompt[0]["content"] if len(prompt) else ""
        gt = row["reward_model"]["ground_truth"]
        records.append({
            "data_source": row["data_source"],
            "prompt": [{"role": "user", "content": content}],
            "ability": row.get("ability", "fact-reasoning"),
            "reward_model": {"style": "rule",
                             "ground_truth": {"target": to_target_list(gt.get("target"))}},
            "extra_info": unify_extra(row.get("extra_info"), split, i),
        })
    return records


def norm_hf_qa(hf_split, data_source: str, split: str, cap: int | None = None) -> list:
    """Build records from a raw FlashRAG QA split (question + golden_answers)."""
    if cap is not None and len(hf_split) > cap:
        hf_split = hf_split.shuffle(seed=SEED).select(range(cap))
    records = []
    for i, ex in enumerate(hf_split):
        q = ex["question"].strip()
        if not q.endswith("?"):
            q += "?"
        records.append({
            "data_source": data_source,
            "prompt": [{"role": "user", "content": make_prefix(q)}],
            "ability": "fact-reasoning",
            "reward_model": {"style": "rule",
                             "ground_truth": {"target": to_target_list(ex["golden_answers"])}},
            "extra_info": unify_extra(None, split, i),
        })
    return records


def hf_split(config: str, names: list[str]):
    """Load the first available split among `names` for a FlashRAG config."""
    ds = datasets.load_dataset(HF_REPO, config)
    for n in names:
        if n in ds:
            return ds[n]
    raise KeyError(f"None of {names} found for {config}; available: {list(ds.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/combined_nq_hotpot_nevir_negfree")
    ap.add_argument("--val_qa_cap", type=int, default=500,
                    help="Max NQ / Hotpot dev rows in val (per source). Use 0 to exclude QA from val.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------- TRAIN ----------------
    print("Building TRAIN ...")
    train = []
    train += norm_existing(pd.read_parquet(NQ_TRAIN_LOCAL), "train")          # NQ (local)
    train += norm_hf_qa(hf_split(HOTPOT_CONFIG, ["train"]),
                        "hotpotqa", "train")                                  # HotpotQA (HF)
    train += norm_existing(pd.read_parquet(NEVIR_TRAIN), "train")             # nevir clean
    train += norm_existing(pd.read_parquet(NEGFREE_TRAIN), "train")           # neg-free-sentence
    pd.DataFrame(train, columns=COLS).to_parquet(os.path.join(args.out_dir, "train.parquet"))
    print(f"  train.parquet: {len(train)} rows")

    # ---------------- VAL ----------------
    print("Building VAL ...")
    val = []
    if args.val_qa_cap > 0:
        val += norm_hf_qa(hf_split(NQ_CONFIG, ["dev", "validation", "test"]),
                          "nq", "validation", cap=args.val_qa_cap)
        val += norm_hf_qa(hf_split(HOTPOT_CONFIG, ["dev", "validation"]),
                          "hotpotqa", "validation", cap=args.val_qa_cap)
    val += norm_existing(pd.read_parquet(NEVIR_VAL), "validation")            # nevir clean
    val += norm_existing(pd.read_parquet(NEGFREE_VAL), "validation")          # neg-free-sentence
    pd.DataFrame(val, columns=COLS).to_parquet(os.path.join(args.out_dir, "val.parquet"))
    print(f"  val.parquet:   {len(val)} rows")


if __name__ == "__main__":
    main()
