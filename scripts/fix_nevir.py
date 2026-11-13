"""
Fix annotation mistakes in nevir_with_answers by comparing with nevir_with_answer.

Strategy:
  - Regression: new file has none-like answer but old file had a specific answer -> restore old
  - Both-none: both files agree on none-like -> keep, flag as still problematic
"""
import csv, re, os
from pathlib import Path

RAW_DATA_DIR = os.environ.get('RAW_DATA_DIR', '/home/16207041/data')  # raw negation datasets

NONE_PAT = re.compile(
    r'not mention|not specif|not stated|not given|\bnone\b|\bn/a\b',
    re.IGNORECASE
)

OLD_DIR = Path(RAW_DATA_DIR, 'nevir_with_answer')
NEW_DIR = Path(RAW_DATA_DIR, 'nevir_with_answers')
OUT_DIR = Path(RAW_DATA_DIR, 'nevir_with_answers_fixed')
OUT_DIR.mkdir(exist_ok=True)

log_rows = []   # (split, id, col, old_ans, new_ans, action)
drop_stats = {}

for split in ['train', 'validation', 'test']:
    old_path = OLD_DIR / f'{split}.csv'
    new_path = NEW_DIR / f'{split}.csv'
    out_path = OUT_DIR / f'{split}.csv'

    with open(old_path, newline='', encoding='utf-8') as f:
        old_rows = {r['id']: r for r in csv.DictReader(f)}

    with open(new_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        new_rows = list(reader)

    fixed_rows = []
    for r in new_rows:
        row = dict(r)
        rid = row['id']
        for col in ['answer1', 'answer2']:
            v = row[col].strip()
            if NONE_PAT.search(v):
                old_ans = old_rows.get(rid, {}).get(col, '').strip()
                if old_ans and not NONE_PAT.search(old_ans):
                    # Regression: restore old answer
                    log_rows.append((split, rid, col, v, old_ans, 'FIXED: restored old answer'))
                    row[col] = old_ans
                else:
                    # Both-none: flag as still problematic
                    log_rows.append((split, rid, col, v, old_ans or '(missing)', 'STILL_PROBLEMATIC'))
        fixed_rows.append(row)

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fixed_rows)

    # Dropped version: remove none-like answers AND same-answer rows
    dropped_path = OUT_DIR / f'{split}_dropped.csv'
    dropped_rows = [
        r for r in fixed_rows
        if not any(NONE_PAT.search(r[col].strip()) for col in ['answer1', 'answer2'])
        and r['answer1'].strip().lower() != r['answer2'].strip().lower()
    ]
    n_dropped = len(fixed_rows) - len(dropped_rows)
    with open(dropped_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dropped_rows)

    n_none_only = sum(
        1 for r in fixed_rows
        if any(NONE_PAT.search(r[col].strip()) for col in ['answer1','answer2'])
        and r['answer1'].strip().lower() != r['answer2'].strip().lower()
    )
    n_same_only = sum(
        1 for r in fixed_rows
        if not any(NONE_PAT.search(r[col].strip()) for col in ['answer1','answer2'])
        and r['answer1'].strip().lower() == r['answer2'].strip().lower()
    )
    n_both = sum(
        1 for r in fixed_rows
        if any(NONE_PAT.search(r[col].strip()) for col in ['answer1','answer2'])
        and r['answer1'].strip().lower() == r['answer2'].strip().lower()
    )
    drop_stats[split] = {
        'total': len(fixed_rows), 'kept': len(dropped_rows), 'dropped': n_dropped,
        'none_only': n_none_only, 'same_only': n_same_only, 'both': n_both,
    }
    print(f'{split}: wrote {len(fixed_rows)} rows -> {out_path}')
    print(f'{split}: wrote {len(dropped_rows)} rows (dropped {n_dropped}) -> {dropped_path}')

# Write log
log_path = OUT_DIR / 'fix_log.txt'
fixed  = [r for r in log_rows if r[5].startswith('FIXED')]
remain = [r for r in log_rows if r[5].startswith('STILL')]

counts = {}
for split in ['train', 'validation', 'test']:
    counts[split] = {
        'fixed':  sum(1 for r in fixed  if r[0] == split),
        'remain': sum(1 for r in remain if r[0] == split),
    }

with open(log_path, 'w', encoding='utf-8') as f:
    f.write('=' * 70 + '\n')
    f.write('NEVIR_WITH_ANSWERS FIX LOG\n')
    f.write('=' * 70 + '\n\n')

    f.write('SUMMARY\n')
    f.write('-' * 40 + '\n')
    total_fixed  = len(fixed)
    total_remain = len(remain)
    f.write(f'Total rows fixed:            {total_fixed}\n')
    f.write(f'Total rows still problematic:{total_remain}\n\n')
    for split in ['train', 'validation', 'test']:
        f.write(f'  {split:12s}: fixed={counts[split]["fixed"]:4d}  still_problematic={counts[split]["remain"]:4d}\n')

    f.write('\n\nFIXED ROWS (regression: old file had real answer, new file had none-like)\n')
    f.write('-' * 70 + '\n')
    for split in ['train', 'validation', 'test']:
        split_fixed = [r for r in fixed if r[0] == split]
        if split_fixed:
            f.write(f'\n[{split.upper()}] ({len(split_fixed)} rows)\n')
            for (sp, rid, col, bad_ans, good_ans, action) in split_fixed:
                f.write(f'  id={rid:10s} {col}: "{bad_ans}" -> "{good_ans}"\n')

    f.write('\n\nSTILL PROBLEMATIC (both old and new have none-like answer)\n')
    f.write('-' * 70 + '\n')
    for split in ['train', 'validation', 'test']:
        split_remain = [r for r in remain if r[0] == split]
        if split_remain:
            f.write(f'\n[{split.upper()}] ({len(split_remain)} rows)\n')
            for (sp, rid, col, cur_ans, old_ans, action) in split_remain:
                f.write(f'  id={rid:10s} {col}: "{cur_ans}"\n')

    f.write('\n\nFINAL DROPPED TABLES (none-like OR answer1==answer2 removed)\n')
    f.write('-' * 70 + '\n')
    hdr = f'  {"split":12s}  {"original":>10}  {"kept":>8}  {"dropped":>8}  {"drop%":>6}  {"none_only":>10}  {"same_only":>10}  {"both":>6}\n'
    f.write(hdr)
    for split in ['train', 'validation', 'test']:
        s = drop_stats[split]
        pct = s['dropped'] / s['total'] * 100
        f.write(f'  {split:12s}  {s["total"]:>10}  {s["kept"]:>8}  {s["dropped"]:>8}  {pct:>5.1f}%'
                f'  {s["none_only"]:>10}  {s["same_only"]:>10}  {s["both"]:>6}\n')
    to = sum(d['total']      for d in drop_stats.values())
    tk = sum(d['kept']       for d in drop_stats.values())
    td = sum(d['dropped']    for d in drop_stats.values())
    tn = sum(d['none_only']  for d in drop_stats.values())
    ts = sum(d['same_only']  for d in drop_stats.values())
    tb = sum(d['both']       for d in drop_stats.values())
    f.write(f'  {"TOTAL":12s}  {to:>10}  {tk:>8}  {td:>8}  {td/to*100:>5.1f}%'
            f'  {tn:>10}  {ts:>10}  {tb:>6}\n')

print(f'\nLog written to {log_path}')
print(f'Regressions fixed: {total_fixed}  |  Still problematic (none-like): {total_remain}')
print()
print(f'{"split":12s}  {"original":>10}  {"kept":>8}  {"dropped":>8}  {"drop%":>6}  {"none_only":>10}  {"same_only":>10}  {"both":>6}')
for split in ['train', 'validation', 'test']:
    s = drop_stats[split]
    pct = s['dropped'] / s['total'] * 100
    print(f'{split:12s}  {s["total"]:>10}  {s["kept"]:>8}  {s["dropped"]:>8}  {pct:>5.1f}%'
          f'  {s["none_only"]:>10}  {s["same_only"]:>10}  {s["both"]:>6}')
to = sum(d['total']     for d in drop_stats.values())
tk = sum(d['kept']      for d in drop_stats.values())
td = sum(d['dropped']   for d in drop_stats.values())
tn = sum(d['none_only'] for d in drop_stats.values())
ts = sum(d['same_only'] for d in drop_stats.values())
tb = sum(d['both']      for d in drop_stats.values())
print(f'{"TOTAL":12s}  {to:>10}  {tk:>8}  {td:>8}  {td/to*100:>5.1f}%'
      f'  {tn:>10}  {ts:>10}  {tb:>6}')
