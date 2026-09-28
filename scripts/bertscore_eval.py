"""
Compute BERTScore, EM, and token-F1 metrics for the extracted evaluation pairs.

Metrics per (model, dataset) combination:
  All           : mean score over all records
  pos (positive): mean score for the role that maps to 'positive' (q1 for nevir; q2 for negations_free/controlled)
  neg (negated) : mean score for the role that maps to 'negated'
  Per-instance  : fraction of pairs where BOTH pos and neg meet the threshold

Usage:
  python scripts/bertscore_eval.py [--threshold 0.5] [--input run/extracted_pairs.csv]
                                   [--filter-mixed]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from bert_score import score as bert_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from verl.utils.reward_score.qa_em import em_check, token_f1_score

from eval_utils import accumulate_pair_scores, pair_coverage, per_instance_score

DEFAULT_CSV = BASE_DIR / "run" / "extracted_pairs.csv"

SWAP_ROLES = {
    "negations_free_with_answer": True,
    "negations_free_sentence_with_answer": True,
    "negation_controlled_with_answers": True,
    "nevir_with_answers": False,
}

TRIVIAL = {"yes", "no", "none", "not mentioned", "not specified"}


def is_trivial(answer: str) -> bool:
    return answer.lower().strip() in TRIVIAL


def mixed_pair_ids(df: pd.DataFrame) -> set[str]:
    """Return pair_ids where exactly one of q1/q2 golden answer is trivial."""
    pair_answers: dict[str, dict[str, str]] = defaultdict(dict)
    for _, r in df[df["query_role"].isin(["q1", "q2"])].iterrows():
        golden = json.loads(r["golden_answers_json"])[0]
        pair_answers[r["pair_id"]][r["query_role"]] = golden
    return {
        pid for pid, roles in pair_answers.items()
        if "q1" in roles and "q2" in roles
        and is_trivial(roles["q1"]) != is_trivial(roles["q2"])
    }


def compute_bertscore_best_of(
    hypotheses: list[str],
    references_per_item: list[list[str]],
    batch_size: int = 64,
    device: str | None = None,
) -> list[float]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    flat_hyps, flat_refs, item_slices = [], [], []
    for hyp, refs in zip(hypotheses, references_per_item):
        start = len(flat_hyps)
        for ref in refs:
            flat_hyps.append(hyp)
            flat_refs.append(ref)
        item_slices.append((start, len(flat_hyps)))

    _, _, F1 = bert_score(
        flat_hyps, flat_refs,
        lang="en", rescale_with_baseline=True,
        device=device, batch_size=batch_size, verbose=False,
    )
    F1 = F1.tolist()
    return [max(F1[s:e]) for s, e in item_slices]


def mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_CSV))
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold for per-instance 'both correct' (default: 0.5)")
    parser.add_argument("--filter-mixed", action="store_true",
                        help="Exclude pairs where one golden answer is trivial (yes/no/none) "
                             "and the other is substantive.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Per-instance threshold : {args.threshold}")
    print(f"Filter mixed pairs     : {args.filter_mixed}")
    print()

    rows = []
    for (model, dataset), gdf in df.groupby(["model", "dataset"]):
        print(f"{'='*60}")
        print(f"model={model}  dataset={dataset}  n={len(gdf)}")

        gdf = gdf.copy().reset_index(drop=True)

        if args.filter_mixed:
            excluded = mixed_pair_ids(gdf)
            n_excl = gdf["pair_id"].isin(excluded).sum()
            print(f"  --filter-mixed: excluding {len(excluded)} mixed pairs ({n_excl} records)")
            gdf = gdf[~gdf["pair_id"].isin(excluded)].reset_index(drop=True)
            print(f"  Records after filter: {len(gdf)}")

        hyps = gdf["extracted_answer"].fillna("").tolist()
        refs_per_item = [json.loads(r) for r in gdf["golden_answers_json"]]

        # --- BERTScore ---
        bs_f1 = compute_bertscore_best_of(hyps, refs_per_item)
        gdf["bs_f1"] = bs_f1

        # --- EM and token-F1 (verl qa_em functions) ---
        gdf["em"] = [em_check(h, r) for h, r in zip(hyps, refs_per_item)]
        gdf["tok_f1"] = [token_f1_score(h, r) for h, r in zip(hyps, refs_per_item)]

        def split_mean(col, role):
            vals = gdf.loc[gdf["query_role"] == role, col].tolist()
            return mean(vals), len(vals)

        swap = SWAP_ROLES.get(dataset, False)
        pos_role, neg_role = ("q2", "q1") if swap else ("q1", "q2")

        bs_all = mean(bs_f1);          bs_pos, n_pos = split_mean("bs_f1", pos_role); bs_neg, n_neg = split_mean("bs_f1", neg_role)
        em_all = mean(gdf["em"].tolist()); em_pos, _  = split_mean("em", pos_role);   em_neg, _     = split_mean("em", neg_role)
        f1_all = mean(gdf["tok_f1"].tolist()); f1_pos, _ = split_mean("tok_f1", pos_role); f1_neg, _ = split_mean("tok_f1", neg_role)

        bs_pairs  = accumulate_pair_scores(gdf, "bs_f1")
        em_pairs  = accumulate_pair_scores(gdf, "em")
        total_ids, complete, pair_frac = pair_coverage(bs_pairs)
        bs_pi, bs_pi_n = per_instance_score(bs_pairs, args.threshold)
        em_pi, em_pi_n = per_instance_score(em_pairs, threshold=1.0)

        print(f"  EM   : all={em_all:.4f}  pos={em_pos:.4f} (n={n_pos})  neg={em_neg:.4f} (n={n_neg})  per-inst={em_pi:.4f} (n={em_pi_n})")
        print(f"  tokF1: all={f1_all:.4f}  pos={f1_pos:.4f}  neg={f1_neg:.4f}")
        print(f"  BERT : all={bs_all:.4f}  pos={bs_pos:.4f}  neg={bs_neg:.4f}  per-inst={bs_pi:.4f} (n={bs_pi_n})")
        print(f"  Pairs: {complete}/{total_ids} ({pair_frac:.1%})")

        rows.append({
            "model": model, "dataset": dataset,
            "filter_mixed": args.filter_mixed,
            "n_records": len(gdf),
            "EM_all": round(em_all, 4), "EM_pos": round(em_pos, 4), "EM_neg": round(em_neg, 4),
            "EM_per_instance": em_pi, "EM_pi_n": em_pi_n,
            "tokF1_all": round(f1_all, 4), "tokF1_pos": round(f1_pos, 4), "tokF1_neg": round(f1_neg, 4),
            "BERT_all": round(bs_all, 4), "BERT_pos": round(bs_pos, 4), "BERT_neg": round(bs_neg, 4),
            "BERT_per_instance": bs_pi, "BERT_pi_n": bs_pi_n,
            "complete_pairs": complete, "total_pairs": total_ids, "pair_fraction": round(pair_frac, 4),
        })

    results_df = pd.DataFrame(rows)

    # --- Summary table ---
    tag = "filter-mixed" if args.filter_mixed else "unfiltered"
    print(f"\n{'='*90}")
    print(f"Summary ({tag}, per-instance thr={args.threshold})")
    print(f"{'='*90}")
    hdr = f"{'Model':<12} {'Dataset':<34} {'EM':>6} {'pos':>6} {'neg':>6} {'tF1':>6} {'pos':>6} {'neg':>6} {'BERT':>6} {'pos':>6} {'neg':>6} {'PI-EM':>7} {'PI-BS':>7} {'Pairs':>12}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in results_df.iterrows():
        pair_str = f"{int(r['complete_pairs'])}/{int(r['total_pairs'])} ({r['pair_fraction']:.0%})"
        print(
            f"{r['model']:<12} {r['dataset']:<34} "
            f"{r['EM_all']:>6.4f} {r['EM_pos']:>6.4f} {r['EM_neg']:>6.4f} "
            f"{r['tokF1_all']:>6.4f} {r['tokF1_pos']:>6.4f} {r['tokF1_neg']:>6.4f} "
            f"{r['BERT_all']:>6.4f} {r['BERT_pos']:>6.4f} {r['BERT_neg']:>6.4f} "
            f"{r['EM_per_instance']:>7.4f} {r['BERT_per_instance']:>7.4f} {pair_str:>12}"
        )

    suffix = "_filtered" if args.filter_mixed else ""
    out = BASE_DIR / "run" / f"eval_results{suffix}.csv"
    results_df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
