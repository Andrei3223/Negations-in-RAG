"""
Score the 3x-repeated nevir / negations_free_sentence eval runs and report
per-experiment mean +/- std for EM, token-F1, BERTScore (+ per-instance BERT).

Reuses the existing, validated pipeline:
  - extraction      : scripts/extract_pairs.py  (parse_log, build_lookup, match)
  - metric funcs    : verl qa_em (em_check, token_f1_score) + bert_score
  - aggregation     : scripts/eval_utils.py     (accumulate_pair_scores, per_instance_score)

Each run log is scored INDEPENDENTLY (one row per run) so we can take a real
std across the 3 repetitions. Incomplete runs are skipped (n may be < 3).

Outputs:
  run/eval_results_3x_runs.csv   (one row per run)
  run/eval_results_3x_agg.csv    (one row per experiment: mean + std)
"""
import glob
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean, pstdev, stdev

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))
sys.path.insert(0, str(BASE_DIR))

from extract_pairs import (
    parse_log, build_lookup, valid_pair_ids_from_eval_parquet, match_records,
)
from bertscore_eval import compute_bertscore_best_of, SWAP_ROLES
from eval_utils import accumulate_pair_scores, per_instance_score, pair_coverage
from verl.utils.reward_score.qa_em import em_check, token_f1_score

THRESHOLD = 0.5

# dataset tag in the log filename -> extraction config
DS = {
    "nevir": {
        "dataset": "nevir_with_answers",
        "eval_parquet": "data/negations_with_answer_rag/nevir-with-answers/test_clean.parquet",
        "src_split": "test_clean",
        "expected": 2600,
    },
    "nfs": {
        "dataset": "negations_free_sentence_with_answer",
        "eval_parquet": "data/negations_with_answer_rag/negations-free-sentence-with-answer/test_clean.parquet",
        "src_split": "test",
        "expected": 400,
    },
    # negations_free_with_answer_docs (test_clean2 + golden docs), used with the
    # golden / golden_hn retrievers. expected = 2 x 588 rows, as for the others
    # (2 records per evaluated row); not yet confirmed against a finished log.
    "nfwa": {
        "dataset": "negations_free_with_answer",
        "eval_parquet": "data/negations_with_answer_rag/negations-free-with-answer/test_clean2_docs.parquet",
        "src_split": "test_clean2",
        "expected": 1176,
    },
}

# Run name: <ds>-<model>-<retr>-r<rep>-<jobid>, one folder per run:
#   eval_logs/<name>/eval.log   (written by run/eval_negations_with_answers.sh)
RUN_RE = re.compile(r"^(nevir|nfs|nfwa)-(searchr1|subsearch)-(wiki|wiki_5|custom|custom_2|custom_x2|running|golden|golden1|golden_hn|golden_rn)-r(\d+)-(\d+)$")


def _log_paths():
    for path in sorted(glob.glob(str(BASE_DIR / "eval_logs" / "*" / "eval.log"))):
        yield os.path.basename(os.path.dirname(path)), path


def discover_runs():
    runs = []
    for name, path in _log_paths():
        m = RUN_RE.match(name)
        if not m:
            continue
        ds_tag, model, retr, rep, jobid = m.groups()
        runs.append({
            "path": path, "ds_tag": ds_tag, "model": model, "retr": retr,
            "rep": int(rep), "jobid": jobid,
            "label": f"{ds_tag}-{model}-{retr}-r{rep}",
        })
    return runs


def extract_run(run):
    """Return list of record dicts for one run, or None if incomplete."""
    cfg = DS[run["ds_tag"]]
    raw = parse_log(run["path"])
    if len(raw) < cfg["expected"]:
        print(f"  SKIP {run['label']}: {len(raw)}/{cfg['expected']} records (incomplete)")
        return None
    valid_ids = valid_pair_ids_from_eval_parquet(cfg["eval_parquet"])
    lookup = build_lookup(cfg["dataset"], valid_ids, cfg["src_split"])
    matched = match_records(raw, lookup)
    rows = []
    for r in matched:
        if r["query_role"] not in ("q1", "q2"):
            continue
        rows.append({
            "label": run["label"], "ds_tag": run["ds_tag"], "dataset": cfg["dataset"],
            "model": run["model"], "retr": run["retr"], "rep": run["rep"],
            "query_role": r["query_role"], "pair_id": r["pair_id"],
            "extracted_answer": r["extracted_answer"] or "",
            "golden": r["golden_answers"],
        })
    return rows


def metrics_for_run(gdf, dataset):
    """Replicate bertscore_eval.main()'s per-(model,dataset) computation."""
    swap = SWAP_ROLES.get(dataset, False)
    pos_role, neg_role = ("q2", "q1") if swap else ("q1", "q2")

    def split_mean(col, role):
        v = gdf.loc[gdf["query_role"] == role, col].tolist()
        return mean(v) if v else float("nan")

    bs_pairs = accumulate_pair_scores(gdf, "bs_f1")
    em_pairs = accumulate_pair_scores(gdf, "em")
    total_ids, complete, pair_frac = pair_coverage(bs_pairs)
    bs_pi, _ = per_instance_score(bs_pairs, THRESHOLD)
    em_pi, _ = per_instance_score(em_pairs, threshold=1.0)

    return {
        "EM_all": mean(gdf["em"]),       "EM_pos": split_mean("em", pos_role),       "EM_neg": split_mean("em", neg_role),
        "tokF1_all": mean(gdf["tok_f1"]),"tokF1_pos": split_mean("tok_f1", pos_role), "tokF1_neg": split_mean("tok_f1", neg_role),
        "BERT_all": mean(gdf["bs_f1"]),  "BERT_pos": split_mean("bs_f1", pos_role),  "BERT_neg": split_mean("bs_f1", neg_role),
        "PI_BERT": bs_pi, "PI_EM": em_pi,
        # rows with no <answer> tag or an empty one (logged as "None" / "")
        "N_EMPTY": int(gdf["empty"].sum()),
        "EMPTY_all": mean(gdf["empty"]), "EMPTY_pos": split_mean("empty", pos_role), "EMPTY_neg": split_mean("empty", neg_role),
        "complete_pairs": complete, "total_pairs": total_ids,
    }


def main():
    runs = discover_runs()
    print(f"Discovered {len(runs)} run logs")

    all_rows = []
    for run in runs:
        rows = extract_run(run)
        if rows:
            all_rows.extend(rows)
            print(f"  OK   {run['label']}: {len(rows)} records")

    df = pd.DataFrame(all_rows)

    # --- compute per-record EM / tokF1 / BERTScore (single BERT pass) ---
    hyps = df["extracted_answer"].fillna("").tolist()
    refs = df["golden"].tolist()
    print(f"\nComputing BERTScore over {len(df)} records (single pass)...")
    df["bs_f1"] = compute_bertscore_best_of(hyps, refs)
    df["em"] = [em_check(h, r) for h, r in zip(hyps, refs)]
    df["tok_f1"] = [token_f1_score(h, r) for h, r in zip(hyps, refs)]
    # Empty = no usable answer. Logs written before commit 079b7b0 (2026-09-16)
    # record such rows as "Beijing": the old extractor read the prompt's example
    # "<answer> Beijing </answer>" as model output. No gold answer in these
    # datasets contains Beijing, so it only ever marks a missing answer here.
    df["empty"] = [1.0 if str(h).strip() in ("", "None", "Beijing") else 0.0 for h in hyps]

    METRICS = ["EM_all", "EM_pos", "EM_neg", "tokF1_all", "tokF1_pos", "tokF1_neg",
               "BERT_all", "BERT_pos", "BERT_neg", "PI_BERT", "PI_EM",
               "N_EMPTY", "EMPTY_all", "EMPTY_pos", "EMPTY_neg"]

    # --- per-run metrics ---
    per_run = []
    for label, gdf in df.groupby("label"):
        meta = gdf.iloc[0]
        m = metrics_for_run(gdf.reset_index(drop=True), meta["dataset"])
        per_run.append({
            "label": label, "ds_tag": meta["ds_tag"], "dataset": meta["dataset"],
            "model": meta["model"], "retr": meta["retr"], "rep": int(meta["rep"]),
            **m,
        })
    run_df = pd.DataFrame(per_run).sort_values(["ds_tag", "model", "retr", "rep"])
    run_out = BASE_DIR / "run" / "eval_results_3x_runs.csv"
    run_df.to_csv(run_out, index=False)
    print(f"\nSaved per-run metrics -> {run_out}")

    # --- aggregate mean +/- std across reps ---
    agg = []
    for (ds_tag, dataset, model, retr), g in run_df.groupby(["ds_tag", "dataset", "model", "retr"]):
        row = {"ds_tag": ds_tag, "dataset": dataset, "model": model, "retr": retr, "n_reps": len(g)}
        for mcol in METRICS:
            vals = g[mcol].dropna().tolist()
            row[f"{mcol}_mean"] = mean(vals) if vals else float("nan")
            row[f"{mcol}_std"] = (stdev(vals) if len(vals) > 1 else 0.0)
        agg.append(row)
    agg_df = pd.DataFrame(agg).sort_values(["ds_tag", "model", "retr"])
    agg_out = BASE_DIR / "run" / "eval_results_3x_agg.csv"
    agg_df.to_csv(agg_out, index=False)
    print(f"Saved aggregated metrics -> {agg_out}")

    # --- markdown ---
    def fmt(m, s):
        return f"{m:.3f} ± {s:.3f}"
    print("\n\n### Per-experiment metrics (mean ± std over reps)\n")
    print("| Dataset | Model | Retr. | n | EM all | EM pos | EM neg | tokF1 all | tokF1 pos | tokF1 neg | BERT all | BERT pos | BERT neg | PI-BERT |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in agg_df.iterrows():
        ds = "nevir" if r["ds_tag"] == "nevir" else "neg-free-sent"
        print(
            f"| {ds} | {r['model']} | {r['retr']} | {int(r['n_reps'])} | "
            f"{fmt(r['EM_all_mean'], r['EM_all_std'])} | {fmt(r['EM_pos_mean'], r['EM_pos_std'])} | {fmt(r['EM_neg_mean'], r['EM_neg_std'])} | "
            f"{fmt(r['tokF1_all_mean'], r['tokF1_all_std'])} | {fmt(r['tokF1_pos_mean'], r['tokF1_pos_std'])} | {fmt(r['tokF1_neg_mean'], r['tokF1_neg_std'])} | "
            f"{fmt(r['BERT_all_mean'], r['BERT_all_std'])} | {fmt(r['BERT_pos_mean'], r['BERT_pos_std'])} | {fmt(r['BERT_neg_mean'], r['BERT_neg_std'])} | "
            f"{fmt(r['PI_BERT_mean'], r['PI_BERT_std'])} |"
        )


if __name__ == "__main__":
    main()
