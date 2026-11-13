#!/usr/bin/env python3
"""Re-score finished eval runs on a curated subset of their dataset.

The negations-free test set was curated after some baselines ran
(scripts/data_process/drop_conflicting_pairs.py dropped 10 of 294 pairs; the
originals are in <dataset>/pre_conflict_filter/). A run that logged every row
(reward_model.log_sample_rate=1) can be re-scored on the surviving pairs without
rerunning: each logged answer is aligned to its row by position, checked against
the row's question and gold answer, and the in-run metrics are recomputed.

Before trusting the subset numbers, the script recomputes the metrics on ALL rows
and requires them to equal the run's own logged metrics.

Usage:
  python3 scripts/rescore_on_curated.py eval_logs/nfwa-searchr1-golden-r*-350725
"""
import argparse
import os
import re
import sys
from pathlib import Path
from statistics import mean

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts'))
from verl.utils.reward_score.qa_em import em_check, token_f1_score  # noqa: E402
from build_eval_index import in_run_metrics  # noqa: E402

ANSI = re.compile(r'\x1b\[[0-9;]*m')
PFX = re.compile(r'^\([a-z_]+ pid=\d+\)\s*')
SAMPLE = re.compile(r'^\[SAMPLE_R\d+_I\d+\]\s*')
GOLD_ITEM = re.compile(r"'([^']*)'|\"([^\"]*)\"")
DATASET = BASE / 'data/negations_with_answer_rag/negations-free-with-answer'


def traces(log_path):
    """(question, extracted answer or None, gold answers) for every logged row, in order."""
    cur, extracted = None, None
    for raw in open(log_path, errors='ignore'):
        line = SAMPLE.sub('', PFX.sub('', ANSI.sub('', raw.rstrip('\n'))))
        if line.startswith('Reasoning Trace: '):
            cur, extracted = [line[len('Reasoning Trace: '):]], None
        elif cur is not None and line.startswith('Extracted answer: '):
            extracted = line[len('Extracted answer: '):].strip()
        elif cur is not None and extracted is not None and line.startswith('Golden answers: '):
            text = '\n'.join(cur)
            q = text.split('Question:', 1)[1].split('<|im_end|>', 1)[0].strip()
            # numpy repr: ['a' 'b'] (no commas), quotes switch to "..." around apostrophes
            gold = [a or b for a, b in GOLD_ITEM.findall(line[len('Golden answers: '):])]
            yield q, (None if extracted == 'None' else extracted), gold
            cur = extracted = None
        elif cur is not None and extracted is None:
            cur.append(line)


def metrics(df, q1_negated=True):
    """Same definitions as verl/trainer/ppo/negation_metrics.compute_val_metrics."""
    neg, pos = ('q1', 'q2') if q1_negated else ('q2', 'q1')
    pairs = df.pivot_table(index='pair_id', columns='role', values='em', aggfunc='first')
    pairs = pairs.dropna()
    return {
        'rows': len(df), 'pairs': len(pairs),
        'test_score': df.em.mean(),
        'per_instance_acc': ((pairs.q1 >= 1) & (pairs.q2 >= 1)).mean(),
        'positive_score': df[df.role == pos].em.mean(),
        'negated_score': df[df.role == neg].em.mean(),
        'test_f1': df.f1.mean(),
        'positive_f1': df[df.role == pos].f1.mean(),
        'negated_f1': df[df.role == neg].f1.mean(),
        'no_answer_rate': df['empty'].mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', help='eval_logs run folders')
    ap.add_argument('--full', default=str(DATASET / 'pre_conflict_filter/test_clean2_docs.parquet'))
    ap.add_argument('--curated', default=str(DATASET / 'test_clean2_docs.parquet'))
    ap.add_argument('--out', default=str(BASE / 'run/rescored_curated.csv'))
    args = ap.parse_args()

    full = pd.read_parquet(args.full)
    curated = pd.read_parquet(args.curated)
    keep = {str(e['pair_id']) for e in curated.extra_info}
    rows = pd.DataFrame({
        'pair_id': [str(e['pair_id']) for e in full.extra_info],
        'role': [e['query_role'] for e in full.extra_info],
        'question': [p[0]['content'].split('Question:', 1)[1].strip() for p in full.prompt],
        'target': [[str(t) for t in r['ground_truth']['target']] for r in full.reward_model],
    })

    out = []
    for run in args.runs:
        run = run.rstrip('/')
        log = os.path.join(run, 'eval.log')
        logged = list(traces(log))
        if len(logged) != len(rows):
            sys.exit(f'{run}: {len(logged)} logged rows, dataset has {len(rows)} -- not a full log')
        df = rows.copy()
        for i, (q, ans, gold) in enumerate(logged):
            if q != df.question[i] or gold != df.target[i]:
                sys.exit(f'{run}: row {i} does not align (question or gold answer differs)')
        answers = [a for _, a, _ in logged]
        df['em'] = [float(em_check(a, t)) if a is not None else 0.0 for a, t in zip(answers, df.target)]
        df['f1'] = [token_f1_score(a, t) if a is not None else 0.0 for a, t in zip(answers, df.target)]
        df['empty'] = [float(a is None or not a.strip()) for a in answers]

        # Guard: on all rows the recomputation must reproduce the run's own metrics.
        logged_m = in_run_metrics(open(log, errors='ignore').read())
        allm = metrics(df)
        for k in ('test_score', 'per_instance_acc', 'positive_score', 'negated_score', 'test_f1'):
            if abs(allm[k] - logged_m[k]) > 1e-6:
                sys.exit(f'{run}: recomputed {k}={allm[k]:.6f} != logged {logged_m[k]:.6f}')

        cur = metrics(df[df.pair_id.isin(keep)])
        name = os.path.basename(run)
        out.append({'run': name, 'subset': 'all (294 pairs)', **allm})
        out.append({'run': name, 'subset': f'curated ({cur["pairs"]} pairs)', **cur})
        print(f'{name}: reproduces logged metrics on all rows; curated EM {cur["test_score"]:.4f} '
              f'(was {allm["test_score"]:.4f}), pair acc {cur["per_instance_acc"]:.4f} '
              f'(was {allm["per_instance_acc"]:.4f})')

    pd.DataFrame(out).to_csv(args.out, index=False)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
