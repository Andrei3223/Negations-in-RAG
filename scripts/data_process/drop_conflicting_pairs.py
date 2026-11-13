"""
Drop negation pairs with unreliable labels from the clean base files, per split.

A pair is dropped (both halves, so no pair is left incomplete) when any of its rows is:

  conflict        the same question with the same document carries a DIFFERENT gold
                  answer elsewhere in the split. Upstream negations-free reuses one
                  half-pair (question + document) in two pairs of different negation
                  types; generate_answers.py answers q2 once per pair, primed with that
                  pair's q1 and answer1, so the same q2 got 'yes' in the type-4 pair and
                  'no' in the type-2 pair (the document supports 'yes'). Every pair that
                  contains the question is dropped: which label is right cannot be told
                  without re-answering, and one side is often a mislabel.
  missing_answer  the gold answer is empty / 'nan' / 'none' (a None answer that
                  create_clean2.py wrote as the string 'nan').
  val_dup_train   a validation row repeats a train question (same document and answer).
  val_dup_test    a validation row repeats a test question. Validation picks
                  checkpoints, so it must not contain test or train questions.

Conflicts are detected by question text + document, so the _docs files (which
carry the documents) are read; the base files are then filtered by pair_id, and
the _docs files must be regenerated from them:

  python scripts/data_process/drop_conflicting_pairs.py            # all datasets
  python scripts/data_process/negations_golden_docs.py --dataset all --split all
  python scripts/data_process/build_golden_training.py --setup ...  (each setup)

Before the first change, every file it rewrites is copied to
<dataset dir>/pre_conflict_filter/ (never overwritten), so evals run on the old
files stay reproducible. Every dropped pair is logged to
<dataset dir>/dropped_pairs.csv. Idempotent: a second run finds nothing to drop.
"""

import argparse
import collections
import os
import re
import shutil
import string
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import negations_golden_docs as ngd  # noqa: E402  -- same registry of base files and splits

LOG_COLS = ['split', 'pair_id', 'reason', 'question', 'answer', 'detail']


def norm(s):
    s = str(s).lower()
    s = ''.join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def question(prompt):
    return prompt[0]['content'].rsplit('Question:', 1)[-1].strip()


def target(rm):
    return str(rm['ground_truth']['target'][0]).strip()


def docs_path(base):
    return base.replace('.parquet', '_docs.parquet')


def find_drops(name):
    """(split, pair_id) -> list of log rows, for one dataset."""
    cfg = ngd.DATASETS[name]
    frames = {}
    for split in ngd.SPLITS:
        path = docs_path(cfg['base'][split])
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path}: run negations_golden_docs.py --split all first')
        frames[split] = pd.read_parquet(path)

    drops = collections.defaultdict(list)

    def log(split, pid, reason, q, a, detail=''):
        drops[(split, str(pid))].append(dict(split=split, pair_id=str(pid), reason=reason,
                                             question=q, answer=a, detail=detail))

    for split, df in frames.items():
        # Missing answers from the BASE file: negations_golden_docs.py already drops
        # those pairs from _docs, so the _docs file alone would never report them.
        base = pd.read_parquet(cfg['base'][split])
        for ei, p, rm in zip(base.extra_info, base.prompt, base.reward_model):
            if norm(target(rm)) in ('', 'nan', 'none'):
                log(split, ei['pair_id'], 'missing_answer', question(p), target(rm))
        by_key = collections.defaultdict(list)
        for ei, p, rm, doc in zip(df.extra_info, df.prompt, df.reward_model, df.gold_doc):
            q, a = question(p), target(rm)
            by_key[(q.lower(), doc)].append((ei['pair_id'], a))
        for (q, _), rows in by_key.items():
            answers = sorted({norm(a) for _, a in rows})
            if len(answers) > 1:
                for pid, a in rows:
                    others = [f'pair {p}: {x}' for p, x in rows if p != pid]
                    log(split, pid, 'conflict', q, a, '; '.join(others))

    # Validation must not repeat train or test questions. Checked on what remains
    # after the in-split drops above.
    def remaining_questions(split):
        df = frames[split]
        keep = [(split, str(ei['pair_id'])) not in drops for ei in df.extra_info]
        return {question(p).lower() for p, k in zip(df.prompt, keep) if k}
    for other in ('train', 'test'):
        seen = remaining_questions(other)
        for ei, p, rm in zip(frames['validation'].extra_info, frames['validation'].prompt,
                             frames['validation'].reward_model):
            if question(p).lower() in seen:
                log('validation', ei['pair_id'], f'val_dup_{other}', question(p), target(rm))
    return drops


def backup(path):
    d = os.path.join(os.path.dirname(path), 'pre_conflict_filter')
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, os.path.basename(path))
    if not os.path.exists(dst):   # keep the ORIGINAL; never overwrite with a filtered copy
        shutil.copy2(path, dst)


def apply(name, drops, dry_run):
    cfg = ngd.DATASETS[name]
    log_rows = [r for rows in drops.values() for r in rows]
    for split in ngd.SPLITS:
        ids = {pid for (s, pid) in drops if s == split}
        for path in (cfg['base'][split], docs_path(cfg['base'][split])):
            df = pd.read_parquet(path)
            keep = [str(ei['pair_id']) not in ids for ei in df.extra_info]
            n_drop = len(df) - sum(keep)
            print(f'   {split:<10} {os.path.basename(path):<28} {len(df):>5} rows -> {sum(keep):>5}'
                  f'  (-{n_drop} rows, -{len(ids)} pairs)')
            if n_drop and not dry_run:
                backup(path)
                tmp = f'{path}.tmp'
                df[keep].reset_index(drop=True).to_parquet(tmp, index=False)
                os.replace(tmp, path)
    if log_rows and not dry_run:
        log_path = os.path.join(os.path.dirname(cfg['base']['train']), 'dropped_pairs.csv')
        new = pd.DataFrame(log_rows, columns=LOG_COLS)
        if os.path.exists(log_path):   # append, so earlier runs stay recorded
            new = pd.concat([pd.read_csv(log_path, dtype=str), new], ignore_index=True)
        new.to_csv(log_path, index=False)
        print(f'   log: {log_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=sorted(ngd.DATASETS) + ['all'])
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()
    names = sorted(ngd.DATASETS) if args.dataset == 'all' else [args.dataset]
    for name in names:
        drops = find_drops(name)
        reasons = collections.Counter((r['split'], r['reason'])
                                      for rows in drops.values() for r in rows)
        pairs = collections.Counter(s for s, _ in drops)
        print(f'== {name}: {sum(pairs.values())} pairs to drop {dict(pairs)}')
        for (split, reason), n in sorted(reasons.items()):
            print(f'   {split:<10} {reason:<15} {n} rows')
        apply(name, drops, args.dry_run)


if __name__ == '__main__':
    main()
