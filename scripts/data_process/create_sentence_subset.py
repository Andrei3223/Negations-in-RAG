"""
Create negations-free-sentence-with-answer raw source data by filtering
negations_free_with_answer _clean2 splits to sentence-level negation types (0,1,5,6).

  0 - Sentential (no/not/none)
  1 - Exceptors (others/besides/but/except)
  5 - Affix (un-, in-, non-)
  6 - Implicit

Output: $RAW_DATA_DIR/negations_free_sentence_with_answer/{split}.{csv,parquet}

After running this, generate chat-format parquets with:
  python3 scripts/data_process/negations_with_answer_rag.py --dataset negations-free-sentence-with-answer
"""

import os
import pandas as pd

RAW_DATA_DIR = os.environ.get('RAW_DATA_DIR', '/home/16207041/data')  # raw negation datasets

SRC_DIR = os.path.join(RAW_DATA_DIR, 'negations_free_with_answer')
OUT_DIR = os.path.join(RAW_DATA_DIR, 'negations_free_sentence_with_answer')
SENTENCE_TYPES = {0, 1, 5, 6}

os.makedirs(OUT_DIR, exist_ok=True)

for split in ['train', 'validation', 'test']:
    df = pd.read_csv(f'{SRC_DIR}/{split}_clean2.csv')
    filtered = df[df['negation'].isin(SENTENCE_TYPES)].copy()
    filtered.to_csv(f'{OUT_DIR}/{split}.csv', index=False)
    filtered.to_parquet(f'{OUT_DIR}/{split}.parquet', index=False)
    counts = filtered['negation'].value_counts().sort_index()
    print(f'{split}: {len(df)} → {len(filtered)} pairs  '
          + '  '.join(f'type{t}={c}' for t, c in counts.items()))
