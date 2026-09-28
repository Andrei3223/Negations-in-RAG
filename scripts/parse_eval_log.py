#!/usr/bin/env python3
"""Parse negation-eval log files and print a markdown summary.

Usage:
    python scripts/parse_eval_log.py eval_logs/*/eval.log
    python run/parse_eval_log.py eval_logs/slurm/subsearch-eval-neg-top4-5agents-318653.out
"""

import re
import sys
import os

ANSI = re.compile(r'\x1b\[[0-9;]*m')
RAY_PREFIX = re.compile(r'^\([a-z_]+ pid=\d+\)\s+', re.MULTILINE)

DATASET_LABELS = {
    'negations-free-with-answer':          'negations-free',
    'negation-controlled-with-answers':    'negation-controlled',
    'nevir-with-answers':                  'nevir',
    'negations-free-sentence-with-answer': 'negations-free-sentence',
}

METRIC_COLS = [
    ('test_score',       'EM All (q1+q2)'),
    ('positive_score',   'EM positive'),
    ('negated_score',    'EM negated'),
    ('per_instance_acc', 'EM Per-instance'),
    ('test_f1',          'F1 All'),
    ('positive_f1',      'F1 positive'),
    ('negated_f1',       'F1 negated'),
]

CONFIG_FIELDS = [
    ('model',                r"^Mode:\s+(.+)$",                         None),
    ('dataset',              r"^Dataset:\s+(.+)$",                      None),
    ('experiment_name',      r"'experiment_name':\s+'([^']+)'",         None),
    ('max_turns',            r"'max_turns':\s+(\d+)",                   int),
    ('n_agent',              r"'n_agent':\s+(\d+)",                     int),
    ('val_batch_size',       r"'val_batch_size':\s+(\d+)",              int),
    ('max_response_length',  r"'max_response_length':\s+(\d+)",         int),
    ('max_start_length',     r"'max_start_length':\s+(\d+)",            int),
    ('max_obs_length',       r"'max_obs_length':\s+(\d+)",              int),
    ('temperature',          r"'temperature':\s+([\d.]+)",              float),
    ('gpu_memory_util',      r"'gpu_memory_utilization':\s+([\d.]+)",   float),
    ('retriever_topk',       r"'topk':\s+(\d+)",                       int),
    ('retrieval_wait_s',     r"Retrieval server is ready \(waited ~(\d+)s\)", int),
    ('swap_q_roles',         r"'swap_q_roles':\s+(True|False)",              None),
]


def clean(text: str) -> str:
    text = ANSI.sub('', text)
    text = RAY_PREFIX.sub('', text)
    return text


def parse_file(path: str) -> dict:
    raw = open(path).read()
    text = clean(raw)
    # join lines for multi-line value extraction
    joined = ' '.join(l.strip() for l in text.splitlines())

    result = {'run_name': os.path.splitext(os.path.basename(path))[0]}

    # --- config ---
    config = {}
    for key, pattern, cast in CONFIG_FIELDS:
        m = re.search(pattern, text, re.MULTILINE)
        if not m:
            m = re.search(pattern, joined)
        if m:
            val = m.group(1).strip()
            config[key] = cast(val) if cast else val
    result['config'] = config

    # --- metrics ---
    idx = joined.find('Initial validation metrics')
    if idx == -1:
        result['metrics'] = None
        return result

    block = joined[idx:idx + 1000]
    metrics = {}
    # dataset suffix pattern
    ds_suffix = r'/(negations?-free-with-answer|negation-controlled-with-answers|nevir-with-answers|negations-free-sentence-with-answer)'
    for key, _ in METRIC_COLS:
        pat = rf"'val/{re.escape(key)}{ds_suffix}':\s*[^0-9-]*?([\d.]+(?:e[+-]?\d+)?)"
        m = re.search(pat, block)
        if m:
            metrics[key] = round(float(m.group(2)), 4)
    result['metrics'] = metrics or None
    return result


def fmt(v) -> str:
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)


def render_markdown(parsed: dict) -> str:
    lines = []
    run = parsed['run_name']
    cfg = parsed['config']
    metrics = parsed['metrics']

    lines.append(f'## {run}')
    lines.append('')

    # --- parameters ---
    lines.append('### Parameters')
    lines.append('')
    lines.append('| Parameter | Value |')
    lines.append('|---|---|')

    param_rows = [
        ('Model',              cfg.get('model', '—')),
        ('Dataset',            DATASET_LABELS.get(cfg.get('dataset', ''), cfg.get('dataset', '—'))),
        ('Experiment name',    cfg.get('experiment_name', '—')),
        ('Max turns',          fmt(cfg.get('max_turns'))),
        ('N agents',           fmt(cfg.get('n_agent'))),
        ('Val batch size',     fmt(cfg.get('val_batch_size'))),
        ('Max response len',   fmt(cfg.get('max_response_length'))),
        ('Max start len',      fmt(cfg.get('max_start_length'))),
        ('Max obs len',        fmt(cfg.get('max_obs_length'))),
        ('Temperature',        fmt(cfg.get('temperature'))),
        ('GPU mem util',       fmt(cfg.get('gpu_memory_util'))),
        ('Retriever top-k',    fmt(cfg.get('retriever_topk'))),
        ('Retrieval wait (s)', fmt(cfg.get('retrieval_wait_s'))),
        ('Swap q roles',       cfg.get('swap_q_roles', '—')),
    ]
    for name, val in param_rows:
        lines.append(f'| {name} | {val} |')

    lines.append('')

    # --- metrics ---
    lines.append('### Metrics')
    lines.append('')
    if metrics is None:
        lines.append('*Run did not complete — no metrics found.*')
    else:
        header = ' | '.join(col for _, col in METRIC_COLS)
        sep    = ' | '.join('---' for _ in METRIC_COLS)
        vals   = ' | '.join(fmt(metrics.get(key)) for key, _ in METRIC_COLS)
        lines.append(f'| {header} |')
        lines.append(f'| {sep} |')
        lines.append(f'| {vals} |')

    lines.append('')
    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = sys.argv[1:]
    blocks = []
    for path in paths:
        if not os.path.exists(path):
            print(f'# ERROR: file not found: {path}', file=sys.stderr)
            continue
        parsed = parse_file(path)
        blocks.append(render_markdown(parsed))

    print('\n'.join(blocks))


if __name__ == '__main__':
    main()
