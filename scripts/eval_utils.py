"""
Shared evaluation metric utilities.

Used by:
  scripts/bertscore_eval.py
  scripts/llm_judge_eval.py
"""

from collections import defaultdict


def per_instance_score(
    pair_scores: dict[str, dict[str, float]],
    threshold: float = 0.5,
) -> tuple[float, int]:
    """Fraction of complete pairs where BOTH q1 AND q2 meet the threshold.

    Works for both continuous scores (BERTScore F1) and binary judgments
    (True == 1.0, False == 0.0) because both compare against the threshold.
    """
    complete = [
        (v["q1"] >= threshold and v["q2"] >= threshold)
        for v in pair_scores.values()
        if "q1" in v and "q2" in v
    ]
    if not complete:
        return float("nan"), 0
    return round(sum(complete) / len(complete), 4), len(complete)


def pair_coverage(pair_scores: dict[str, dict]) -> tuple[int, int, float]:
    """Return (total_pair_ids, complete_pairs, pair_fraction) from a pair_scores dict."""
    total = len(pair_scores)
    complete = sum(1 for v in pair_scores.values() if len(v) == 2)
    frac = complete / total if total else float("nan")
    return total, complete, round(frac, 4)


def accumulate_pair_scores(
    df,
    score_col: str,
) -> dict[str, dict[str, float]]:
    """Group per-row scores by (pair_id, query_role), average over multiple runs.

    Returns pair_scores: pair_id -> {"q1": mean_score, "q2": mean_score}
    Only considers rows where query_role is 'q1' or 'q2'.
    """
    known = df[df["query_role"].isin(["q1", "q2"])]
    pair_role: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for _, row in known.iterrows():
        pair_role[row["pair_id"]][row["query_role"]].append(row[score_col])
    return {
        pid: {role: sum(vals) / len(vals) for role, vals in roles.items()}
        for pid, roles in pair_role.items()
    }
