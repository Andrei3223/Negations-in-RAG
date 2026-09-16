"""Validation metrics, including the paired metrics negation datasets are built for.

Ported from the `main` branch's RayPPOTrainer._validate (commit 9dd7534, "negation
metrics"), which the `subsearch` branch never had: every SubSearch run until now
logged only per-source EM (val/test_score/*), not whether a model answers BOTH
queries of a minimal pair correctly -- the quantity NeVIR-style data measures.

Kept as a pure function of plain lists so it can be tested without a trainer.

Two deliberate changes from main:

1. Which query of a pair is negated is decided PER DATA SOURCE, not by main's
   single global `negation.swap_q_roles`. The two conventions coexist in the same
   validation set (combined_*_decomposed/val.parquet holds both), so one flag
   necessarily mislabels one of them. Checked on that val file:
     nevir-with-answers                   q2 carries more negation cues in 44 pairs, q1 in 17
     negations-free-sentence-with-answer  q1 carries more in 24 pairs ("undiscovered",
                                          "unconventional"), q2 in 1
   NeVIR pairs are contrastive rather than strictly positive/negated ("hardly ever"
   vs "always"), so positive_/negated_ splits are approximate for it; the paired
   accuracy does not depend on that labelling.

2. A query counts as correct when its score reaches `correct_threshold` (the
   reward's full score), not when it is merely > 0. Validation uses plain EM today
   so both agree, but with val_add_format on a wrong answer earns 0.2-0.3 format
   credit and `> 0` would count it as correct.
"""

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def compute_val_metrics(
    scores: Sequence[float],
    data_sources: Sequence[str],
    extra_infos: Sequence[Optional[dict]],
    *,
    pair_sources: Iterable[str],
    q1_negated_sources: Iterable[str],
    correct_threshold: float = 1.0,
    f1_scores: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Aggregate per-sample validation scores into logged metrics.

    Returns, per data source:
      val/test_score/{ds}           mean score (unchanged from before)
    and, for data sources in `pair_sources` whose rows carry
    extra_info.pair_id and extra_info.query_role in {q1, q2}:
      val/per_instance_acc/{ds}     fraction of complete pairs with BOTH queries correct
      val/n_pairs/{ds}              complete pairs found -- a sanity check, since
                                    subsampling (val_data_num) or a missing role
                                    silently drops pairs
      val/positive_score/{ds}       mean score on the non-negated query
      val/negated_score/{ds}        mean score on the negated query
    and when `f1_scores` is given:
      val/test_f1/{ds}, val/positive_f1/{ds}, val/negated_f1/{ds}
    """
    n = len(scores)
    if len(data_sources) != n or len(extra_infos) != n:
        raise ValueError(f"length mismatch: scores={n}, data_sources={len(data_sources)}, "
                         f"extra_infos={len(extra_infos)}")
    if f1_scores is not None and len(f1_scores) != n:
        raise ValueError(f"length mismatch: scores={n}, f1_scores={len(f1_scores)}")

    pair_sources = set(pair_sources)
    q1_negated_sources = set(q1_negated_sources)

    by_source: Dict[str, List[float]] = {}
    for s, ds in zip(scores, data_sources):
        by_source.setdefault(ds, []).append(float(s))
    metrics = {f'val/test_score/{ds}': float(np.mean(v)) for ds, v in by_source.items()}

    pos: Dict[str, List[float]] = {}
    neg: Dict[str, List[float]] = {}
    pos_f1: Dict[str, List[float]] = {}
    neg_f1: Dict[str, List[float]] = {}
    pairs: Dict[tuple, Dict[str, float]] = {}

    for i in range(n):
        ds = data_sources[i]
        if ds not in pair_sources:
            continue
        info = extra_infos[i] if isinstance(extra_infos[i], dict) else {}
        role = info.get('query_role')
        pair_id = info.get('pair_id')
        if role not in ('q1', 'q2'):
            continue

        negated_role = 'q1' if ds in q1_negated_sources else 'q2'
        is_negated = role == negated_role
        (neg if is_negated else pos).setdefault(ds, []).append(float(scores[i]))
        if f1_scores is not None:
            (neg_f1 if is_negated else pos_f1).setdefault(ds, []).append(float(f1_scores[i]))

        if pair_id is not None:
            # A duplicate role within a pair would silently overwrite; keep the first.
            pairs.setdefault((ds, pair_id), {}).setdefault(role, float(scores[i]))

    per_instance: Dict[str, List[float]] = {}
    for (ds, _), roles in pairs.items():
        if 'q1' in roles and 'q2' in roles:
            both = roles['q1'] >= correct_threshold and roles['q2'] >= correct_threshold
            per_instance.setdefault(ds, []).append(1.0 if both else 0.0)

    for ds, v in per_instance.items():
        metrics[f'val/per_instance_acc/{ds}'] = float(np.mean(v))
        metrics[f'val/n_pairs/{ds}'] = float(len(v))
    for ds, v in pos.items():
        metrics[f'val/positive_score/{ds}'] = float(np.mean(v))
    for ds, v in neg.items():
        metrics[f'val/negated_score/{ds}'] = float(np.mean(v))

    if f1_scores is not None:
        f1_by_source: Dict[str, List[float]] = {}
        for s, ds in zip(f1_scores, data_sources):
            f1_by_source.setdefault(ds, []).append(float(s))
        for ds, v in f1_by_source.items():
            metrics[f'val/test_f1/{ds}'] = float(np.mean(v))
        for ds, v in pos_f1.items():
            metrics[f'val/positive_f1/{ds}'] = float(np.mean(v))
        for ds, v in neg_f1.items():
            metrics[f'val/negated_f1/{ds}'] = float(np.mean(v))

    return metrics
