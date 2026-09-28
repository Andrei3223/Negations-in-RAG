#!/usr/bin/env python3
"""Write eval_logs/EVALUATIONS.md: every evaluation run with its parameters,
metrics and a link to its log.

Sources, per run folder eval_logs/<name>/:
  - .hydra/config.yaml   parameters (runs launched after the Hydra eval setup);
                         older runs fall back to what parse_eval_log.py can
                         read from the log text
  - eval.log             in-run metrics ("Initial validation metrics"), rows
                         evaluated and rows that never produced an <answer>
                         (from the ACTIVE_TRAJ_NUM lines)
  - run/eval_results_3x_runs.csv   BERTScore and offline EM, written by
                         scripts/eval_3x.py -- only for runs whose log holds an
                         answer for every row (see the note in the output)

Usage:
  python3 scripts/build_eval_index.py
"""
import glob
import os
import re
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from omegaconf import OmegaConf

BASE = Path(__file__).resolve().parent.parent
LOGS = BASE / 'eval_logs'
sys.path.insert(0, str(BASE / 'scripts'))
from parse_eval_log import parse_file  # noqa: E402

RUN_RE = re.compile(r'^(nevir|nfs|nfwa)-(searchr1|subsearch)-([a-z0-9_]+)-r(\d+)-(\d+)$')
ANSI = re.compile(r'\x1b\[[0-9;]*m')
PID = re.compile(r'\([a-z_]+ pid=\d+\)')

DS_NAME = {'nevir': 'nevir-with-answers', 'nfs': 'negations-free-sentence-with-answer',
           'nfwa': 'negations-free-with-answer'}
RETR_DESC = {
    'wiki': 'live e5 retrieval over wiki-18, top-3',
    'wiki_5': 'live e5 retrieval over wiki-18, top-5',
    'custom': 'live e5 retrieval over the custom doc1/doc2 corpus, top-1 (single passage; see caveat)',
    'custom_2': 'live e5 retrieval over the custom corpus, top-2 (second hit often the pair\'s other half)',
    'custom_x2': 'live e5 retrieval over the custom corpus, top-1 shown twice',
    'running': 'live wiki-18 retrieval via an already-running server',
    'golden': 'no retrieval: gold document shown `golden_copies` times',
    'golden1': 'no retrieval: gold document shown once (old runs; see caveat)',
    'golden_hn': 'no retrieval: gold + hard negative (other half of the pair)',
    'golden_rn': 'no retrieval: gold + random document from another topic/pair',
}


def in_run_metrics(text):
    """Metrics dict from the 'Initial validation metrics' block (pprint wraps lines)."""
    i = text.find('Initial validation metrics')
    if i < 0:
        return {}
    blk = text[i:i + 4000]
    blk = blk[:blk.find('}') + 1]
    blk = PID.sub('', ANSI.sub('', blk)).replace('\n', ' ').replace('"', '').replace("'", '')
    return {k: float(v) for k, v in re.findall(r'val/([a-z_0-9]+)/[a-z0-9-]+:\s*([0-9.]+)', blk)}


def rows_and_unanswered(text):
    trajs = [list(map(int, re.findall(r'\d+', t))) for t in re.findall(r'ACTIVE_TRAJ_NUM: \[([0-9, ]+)\]', text)]
    if not trajs:
        return None, None
    return sum(t[0] for t in trajs), sum(t[-1] for t in trajs)


def params(run_dir, text):
    cfg_path = run_dir / '.hydra' / 'config.yaml'
    if cfg_path.exists():
        # Hydra saves the config unresolved (e.g. val_batch_size is an oc.select).
        try:
            c = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        except Exception:
            c = yaml.safe_load(cfg_path.read_text())
        r = c.get('retriever', {})
        ro = c['actor_rollout_ref']['rollout']
        mode = r.get('mode', 'live')
        return {
            'model': c.get('model_id', '?'),
            'val_file': os.path.relpath(c['data']['val_files'], 'data/negations_with_answer_rag'),
            'mode': mode,
            'docs': (r.get('golden_copies', 1) if mode == 'golden' else 2) if mode != 'live'
                    else f"top-{r.get('topk')}" + (f" ×{r['repeat_docs']}" if r.get('repeat_docs', 1) > 1 else ''),
            'seed': ro.get('seed', 0), 'temp': ro.get('temperature'),
            'max_turns': c.get('max_turns'), 'batch': c['data'].get('val_batch_size'),
            'drop_last': c['data'].get('val_drop_last', True),
            'source': 'hydra',
        }
    p = parse_file(str(run_dir / 'eval.log'))['config']
    ds_model = re.search(r'Dataset:\s+(\S+)\s+Model:\s+(\S+)', ANSI.sub('', text))
    return {
        'model': ds_model.group(2) if ds_model else '?',
        'val_file': 'test_clean (pre-Hydra run)',
        'mode': 'live', 'docs': f"top-{p.get('retriever_topk', '?')}",
        'seed': 0, 'temp': p.get('temperature'), 'max_turns': p.get('max_turns'),
        'batch': p.get('val_batch_size'), 'drop_last': True, 'source': 'log',
    }


def empty_stats(r):
    """Rows with no usable answer, from the best source available for the run.

    Returns (count, rate, pos_rate, neg_rate, source) or None. Sources, best first:
      run  -- in-run metric val/n_no_answer (runs after it was added): exact
      log  -- eval_3x re-parse of every logged answer (N_EMPTY): exact, but a literal
              answer "None" is indistinguishable from no answer in those logs
      traj -- rows still active after the last turn (ACTIVE_TRAJ_NUM): every row,
              counts only "never emitted <answer>", no positive/negated split
    """
    m, b = r['metrics'], r.get('bert')
    if 'n_no_answer' in m:
        return (int(m['n_no_answer']), m.get('no_answer_rate'), m.get('positive_no_answer_rate'),
                m.get('negated_no_answer_rate'), 'run')
    if b is not None and 'EMPTY_all' in b and b['EMPTY_all'] == b['EMPTY_all']:
        # N_EMPTY counts log records (pre-Hydra logs hold two per row), so derive
        # the row count from the rate instead.
        n = round(b['EMPTY_all'] * r['rows']) if r['rows'] else int(b['N_EMPTY'])
        return n, b['EMPTY_all'], b['EMPTY_pos'], b['EMPTY_neg'], 'log'
    if r['rows']:
        return r['unans'], r['unans'] / r['rows'], None, None, 'traj'
    return None


def pct(v):
    return '—' if v is None or v != v else f'{v:.0%}'


def f(v, nd=3):
    return '—' if v is None or (isinstance(v, float) and v != v) else f'{v:.{nd}f}'


def ms(vals, nd=3):
    vals = [v for v in vals if v is not None and v == v]
    if not vals:
        return '—'
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f'{st.mean(vals):.{nd}f} ± {sd:.{nd}f}'


def main():
    bert = {}
    runs_csv = BASE / 'run' / 'eval_results_3x_runs.csv'
    if runs_csv.exists():
        for r in pd.read_csv(runs_csv).to_dict('records'):
            bert[r['label']] = r

    runs, others = [], []
    for log in sorted(LOGS.glob('*/eval.log')):
        d = log.parent
        text = log.read_text(errors='ignore')
        m = RUN_RE.match(d.name)
        met = in_run_metrics(text)
        rows, unans = rows_and_unanswered(text)
        rec = {'name': d.name, 'link': f'{d.name}/eval.log', 'metrics': met, 'rows': rows, 'unans': unans,
               'finished': datetime.fromtimestamp(log.stat().st_mtime).strftime('%Y-%m-%d')}
        if not m:
            ds = re.search(r'Dataset:\s+(\S+)\s+Model:\s+(\S+)', ANSI.sub('', text))
            rec.update(dataset=ds.group(1) if ds else '?', model=ds.group(2) if ds else '?')
            others.append(rec)
            continue
        ds_tag, model_tag, retr, rep, job = m.groups()
        rec.update(ds=ds_tag, model_tag=model_tag, retr=retr, rep=int(rep), job=job,
                   label=f'{ds_tag}-{model_tag}-{retr}-r{rep}', **params(d, text))
        rec['bert'] = bert.get(rec['label'])
        runs.append(rec)

    out = []
    w = out.append
    w('# Evaluation runs\n')
    w(f'Generated by `scripts/build_eval_index.py` on {datetime.now():%Y-%m-%d %H:%M}. '
      'Regenerate after new runs finish; do not edit by hand.\n')
    w('## How to read this\n')
    w('- **Run name** is `<dataset>-<model>-<retriever>-r<repeat>-<slurm job>`; each links to its full log.')
    w('- **Datasets:** `nfwa` = negations-free-with-answer (all 10 negation types), '
      '`nfs` = negations-free-sentence-with-answer (sentence-level types 0/1/5/6 only), '
      '`nevir` = nevir-with-answers.')
    w('- **Retriever modes:**')
    for k, v in RETR_DESC.items():
        w(f'  - `{k}`: {v}')
    w('- **Metrics** come from the run itself: EM (exact match, all rows), EM pos / EM neg (the non-negated and '
      'negated question of each pair), **pair acc** (both halves of a pair correct), token F1.')
    w('- **Empty answers** = rows with no usable answer: no `<answer>` tag (the row used up `max_turns`) or an empty '
      'one. They score 0. Shown as count (rate), then the rate on the positive / negated question. '
      'The source is marked: **run** = in-run metric `val/n_no_answer` (exact); **log** = every logged answer '
      're-parsed by `eval_3x.py` (logs before 2026-09-16 record a missing answer as "Beijing", the prompt\'s '
      'example, and are counted as empty); **traj** = rows still unfinished after the last turn (all rows, '
      'no pos/neg split). **traj** is an upper bound, since a row may write `<answer>` in the same turn as a '
      'search and still be scored: measured on runs that have both, SearchR1 agrees (nevir custom: 840 vs 841 '
      'rows) but SubSearch nearly doubles (nevir custom: 19 vs 35).')
    w('- **BERTScore** (roberta-large F1, rescaled; best over gold answers) and **pair BERT** (both halves ≥ 0.5) '
      'come from `scripts/eval_3x.py`, which re-scores every row\'s extracted answer from the log.\n')

    w('## Caveats\n')
    w('- **Repeats before the seed fix are identical.** Runs with `seed 0` all sampled with vLLM\'s default seed, '
      'so their repeats reproduce each other exactly (sd = 0). Only runs with seeds 1–3 are real repeats.')
    w('- **`golden1` is not a valid measurement.** With a single passage the model writes its own fake '
      '"Doc 2"/"Doc 3" and loops until it runs out of turns (see the No answer column). '
      'The `custom` retriever (top-1) has the same problem. `golden` is being rerun with two copies.')
    w('- **Row coverage differs.** Pre-Hydra runs dropped the last partial batch (nevir 1300 of 1324 rows, '
      'nfs 200 of 244). Runs with `drop_last False` score every row, so the two are not exactly comparable.')
    w('- **Pre-2026-09-16 logs record a missing answer as "Beijing".** The old extractor read the prompt\'s example '
      '`<answer> Beijing </answer>`. EM is unaffected (no gold answer contains Beijing), but BERTScore for those '
      'rows compares "Beijing" to the gold answer; for `searchr1` + `custom` that is 44–65% of rows.')
    w('- **BERTScore is missing for the golden runs.** Validation prints a row\'s answer only with probability '
      '1/`log_sample_rate` (64), so those logs hold ~1.5% of answers and cannot be re-scored. '
      'Only pre-Hydra runs (which printed every row) have BERTScore.\n')

    # --- summary per experiment ---
    groups = defaultdict(list)
    for r in runs:
        groups[(r['ds'], r['model_tag'], r['retr'])].append(r)
    w('## Summary by experiment (mean ± sd over repeats)\n')
    w('| dataset | model | retriever | repeats (complete/total) | seeds | rows | empty answers | EM | pair acc | F1 | BERTScore | pair BERT |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|')
    order = {'nfwa': 0, 'nfs': 1, 'nevir': 2}
    for (ds, mt, retr), rs in sorted(groups.items(), key=lambda kv: (order[kv[0][0]], kv[0][1], kv[0][2])):
        g = lambda k: [x['metrics'].get(k) for x in rs]
        b = lambda k: [x['bert'][k] for x in rs if x['bert'] is not None]
        seeds = sorted({x['seed'] for x in rs})
        rows = rs[0]['rows']
        unans = [e[1] for e in (empty_stats(x) for x in rs) if e and e[1] is not None]
        done = sum(1 for x in rs if x['metrics'])
        w(f"| {ds} | {mt} | `{retr}` | {done}/{len(rs)} | {','.join(map(str, seeds))} | {rows or '—'} | "
          f"{ms([u * 100 for u in unans], 1) + '%' if unans else '—'} | {ms(g('test_score'))} | "
          f"{ms(g('per_instance_acc'))} | {ms(g('test_f1'))} | {ms(b('BERT_all'))} | {ms(b('PI_BERT'))} |")
    w('')

    # --- every run ---
    w('## All runs\n')
    for ds in ('nfwa', 'nfs', 'nevir'):
        rs = sorted([r for r in runs if r['ds'] == ds], key=lambda r: (r['model_tag'], r['retr'], r['job'], r['rep']))
        if not rs:
            continue
        w(f'### {ds}: {DS_NAME[ds]}\n')
        w('| run | finished | model | retriever | docs | eval file | seed | temp | max turns | batch / drop_last | rows | empty answers | empty pos / neg | EM | EM pos | EM neg | pair acc | F1 | BERTScore | pair BERT |')
        w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for r in rs:
            m, b = r['metrics'], r['bert']
            e = empty_stats(r)
            ua = f"{e[0]} ({pct(e[1])}) {e[4]}" if e else '—'
            pn = f"{pct(e[2])} / {pct(e[3])}" if e and e[2] is not None else '—'
            w(f"| [{r['name']}]({r['link']}) | {r['finished']} | {r['model']} | `{r['retr']}` | "
              f"{r['docs']} | {r['val_file']} | {r['seed']} | {r['temp']} | {r['max_turns']} | "
              f"{r['batch']} / {r['drop_last']} | {r['rows'] or '—'} | {ua} | {pn} | {f(m.get('test_score'))} | "
              f"{f(m.get('positive_score'))} | {f(m.get('negated_score'))} | {f(m.get('per_instance_acc'))} | "
              f"{f(m.get('test_f1'))} | {f(b['BERT_all']) if b else '—'} | {f(b['PI_BERT']) if b else '—'} |")
        w('')

    if others:
        w('## Other runs (earlier ad-hoc evaluations, not named by the current convention)\n')
        w('| run | finished | dataset | model | rows | EM | pair acc | F1 |')
        w('|---|---|---|---|---|---|---|---|')
        for r in others:
            m = r['metrics']
            w(f"| [{r['name']}]({r['link']}) | {r['finished']} | {r['dataset']} | {r['model']} | {r['rows'] or '—'} | "
              f"{f(m.get('test_score'))} | {f(m.get('per_instance_acc'))} | {f(m.get('test_f1'))} |")
        w('')

    (LOGS / 'EVALUATIONS.md').write_text('\n'.join(out))
    print(f"wrote {LOGS / 'EVALUATIONS.md'}: {len(runs)} runs, {len(others)} other runs, "
          f"{sum(1 for r in runs if r['bert'])} with BERTScore")


if __name__ == '__main__':
    main()
