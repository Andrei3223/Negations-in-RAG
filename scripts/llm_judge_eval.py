"""
LLM-as-judge evaluation for paired negation datasets.

Mirrors bertscore_eval.py's structure and metrics, replacing BERTScore F1 with
binary LLM correctness judgments (True / False).

Prompt source: ChatR1 paper, Table 9.
Calling pattern: scripts/gen_data/generate_answers.py.

Metrics per (model, dataset) combination:
  All           : fraction of all records judged correct
  pos (positive): fraction of the role mapped to 'positive' judged correct
  neg (negated) : fraction of the role mapped to 'negated' judged correct
  Per-instance  : fraction of pairs where BOTH pos AND neg are judged correct

Usage:
  python scripts/llm_judge_eval.py [--input run/extracted_pairs.csv]
                                   [--model gpt-4.1-mini]
                                   [--batch_size 32]
                                   [--env run/.env]
                                   [--models em-grpo,subsearch]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

from eval_utils import accumulate_pair_scores, pair_coverage, per_instance_score
from llm_client import call_with_retry, make_client

SWAP_ROLES = {
    "negations_free_with_answer": True,
    "negations_free_sentence_with_answer": True,
    "negation_controlled_with_answers": True,
    "nevir_with_answers": False,
}

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = BASE_DIR / "run" / "extracted_pairs.csv"
DEFAULT_ENV = BASE_DIR / "run" / ".env"

JUDGE_PROMPT = """\
Given a Question and its Golden Answer, verify whether the Predicted Answer is \
correct. The prediction is correct if it fully aligns with the meaning and key \
information of the Golden Answer. Respond with True if the prediction is correct \
and False otherwise.

Question: {question}
Golden Answer: {golden_answer}
Predicted Answer: {predicted_answer}"""


def call_judge(client, model: str, question: str, golden_answer: str, predicted_answer: str) -> bool:
    """Judge a single prediction. Returns True if correct, False otherwise."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        golden_answer=golden_answer,
        predicted_answer=predicted_answer,
    )
    result = call_with_retry(client, model, [{"role": "user", "content": prompt}], max_tokens=5)
    return (result or "").lower() == "true"


def judge_best_of(
    client: AzureOpenAI,
    model: str,
    questions: list[str],
    hypotheses: list[str],
    refs_per_item: list[list[str]],
    batch_size: int = 32,
) -> list[bool]:
    """Judge each prediction against all its golden answers in parallel.
    A prediction is correct if ANY golden answer yields True (mirrors BERTScore max-F1)."""
    # Build flat list of (item_index, golden_answer) tasks
    tasks: list[tuple[int, str]] = []
    item_slices: list[tuple[int, int]] = []

    for idx, refs in enumerate(refs_per_item):
        start = len(tasks)
        for ref in refs:
            tasks.append((idx, ref))
        item_slices.append((start, len(tasks)))

    verdicts = [False] * len(tasks)

    for chunk_start in range(0, len(tasks), batch_size):
        chunk = tasks[chunk_start: chunk_start + batch_size]
        with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            futures = {
                pool.submit(
                    call_judge,
                    client,
                    model,
                    questions[item_idx],
                    golden,
                    hypotheses[item_idx],
                ): chunk_start + i
                for i, (item_idx, golden) in enumerate(chunk)
            }
            for fut in as_completed(futures):
                verdicts[futures[fut]] = fut.result()

    # OR across all golden answers per item
    results = []
    for start, end in item_slices:
        results.append(any(verdicts[start:end]))
    return results


def accuracy(flags: list[bool]) -> float:
    if not flags:
        return float("nan")
    return round(sum(flags) / len(flags), 4)


def merge_csv(new_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Upsert new_df into an existing CSV by (model, dataset): rows for any
    (model, dataset) just recomputed replace the old ones, the rest are kept."""
    if path.exists():
        old = pd.read_csv(path)
        keys = set(zip(new_df["model"], new_df["dataset"]))
        old = old[~old.apply(lambda r: (r["model"], r["dataset"]) in keys, axis=1)]
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = merged.sort_values(["dataset", "model"]).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_CSV))
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Azure OpenAI deployment name for the judge (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of parallel judge calls per batch (default: 32)",
    )
    parser.add_argument(
        "--env",
        default=str(DEFAULT_ENV),
        help="Path to .env file with Azure credentials",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model names to evaluate (default: all)",
    )
    args = parser.parse_args()

    load_dotenv(args.env)

    client = make_client()

    df = pd.read_csv(args.input)
    if args.models:
        keep = [m.strip() for m in args.models.split(",")]
        df = df[df["model"].isin(keep)].reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Judge model: {args.model}")
    print()

    all_judged: list[pd.DataFrame] = []
    groups = df.groupby(["model", "dataset"])

    rows = []
    for (model, dataset), gdf in groups:
        print(f"{'=' * 60}")
        print(f"LLM judge for: model={model}  dataset={dataset}")
        print(f"  Total records: {len(gdf)}")

        questions = gdf["question"].fillna("").tolist()
        hyps = gdf["extracted_answer"].fillna("").tolist()
        refs_per_item = [json.loads(r) for r in gdf["golden_answers_json"]]

        correct = judge_best_of(
            client, args.model, questions, hyps, refs_per_item, args.batch_size
        )

        gdf = gdf.copy().reset_index(drop=True)
        gdf["llm_correct"] = correct
        gdf["judge_model"] = args.model
        all_judged.append(gdf)

        swap = SWAP_ROLES.get(dataset, False)
        pos_role, neg_role = ("q2", "q1") if swap else ("q1", "q2")

        correct_pos = gdf.loc[gdf["query_role"] == pos_role, "llm_correct"].tolist()
        correct_neg = gdf.loc[gdf["query_role"] == neg_role, "llm_correct"].tolist()

        acc_all = accuracy(correct)
        acc_pos = accuracy(correct_pos)
        acc_neg = accuracy(correct_neg)

        pair_scores = accumulate_pair_scores(gdf, "llm_correct")
        pi_score, pi_n = per_instance_score(pair_scores)
        total_pair_ids, complete_pairs, pair_frac = pair_coverage(pair_scores)

        print(f"  All  : {acc_all:.4f} (n={len(correct)})")
        print(f"  pos  : {acc_pos:.4f} (n={len(correct_pos)})")
        print(f"  neg  : {acc_neg:.4f} (n={len(correct_neg)})")
        print(f"  Pairs: {complete_pairs}/{total_pair_ids} complete ({pair_frac:.1%})")
        print(f"  Per-instance: {pi_score:.4f} (n={pi_n})")

        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "All_Judge": acc_all,
                "All_n": len(correct),
                "pos_Judge": acc_pos,
                "pos_n": len(correct_pos),
                "neg_Judge": acc_neg,
                "neg_n": len(correct_neg),
                "total_pair_ids": total_pair_ids,
                "complete_pairs": complete_pairs,
                "pair_fraction": round(pair_frac, 4),
                "per_instance": pi_score,
                "per_instance_n_pairs": pi_n,
                "judge_model": args.model,
            }
        )

    results_df = pd.DataFrame(rows)

    print(f"\n{'=' * 80}")
    print(f"LLM-as-Judge Summary  (model: {args.model})")
    print(f"{'=' * 80}")
    header = (
        f"{'Model':<20} {'Dataset':<34} {'All':>8} {'pos':>8} {'neg':>8}"
        f" {'Pairs':>14} {'Per-inst':>10}"
    )
    print(header)
    print("-" * len(header))
    for _, r in results_df.iterrows():
        pair_str = f"{int(r['complete_pairs'])}/{int(r['total_pair_ids'])} ({r['pair_fraction']:.1%})"
        print(
            f"{r['model']:<20} {r['dataset']:<34} "
            f"{r['All_Judge']:>8.4f} {r['pos_Judge']:>8.4f} {r['neg_Judge']:>8.4f}"
            f" {pair_str:>14} {r['per_instance']:>10.4f}"
        )

    out = BASE_DIR / "run" / "llm_judge_results.csv"
    merge_csv(results_df, out)
    print(f"\nSaved aggregate (merged) → {out}")

    if all_judged:
        details_df = pd.concat(all_judged, ignore_index=True)
        details_out = BASE_DIR / "run" / "llm_judge_details.csv"
        merge_csv(details_df, details_out)
        print(f"Saved per-row judgments (merged) → {details_out}")


if __name__ == "__main__":
    main()
