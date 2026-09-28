"""
Create further-cleaned dataset versions by removing any pair where either
answer1 or answer2 is a none-like string (e.g. "not mentioned", "none", "n/a").

For negations-free and negation-controlled:
  Starts from the existing _clean parquets (which already removed answer1==answer2
  and yes/no boolean answers) and produces _clean2 parquets.

For nevir:
  Source data is already dropped/deduped; produces first-time _clean parquets.

Outputs (chat-format parquets, 2 records per pair):
  data/negations_with_answer_rag/negations-free-with-answer/{split}_clean2.parquet
  data/negations_with_answer_rag/negation-controlled-with-answers/{split}_clean2.parquet
  data/negations_with_answer_rag/nevir-with-answers/{split}_clean.parquet

Usage:
  cd /home/16207041/Negations-in-RAG
  python3 scripts/data_process/create_clean2.py
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAG_DIR = os.path.join(BASE_DIR, 'data', 'negations_with_answer_rag')

NONE_LIKE = {
    'not mentioned', 'none', 'n/a', 'not specified', 'none mentioned',
    'none specified', 'none given', 'not specified in document',
    'information not provided', 'not applicable',
}

DATASETS = {
    'negations-free-with-answer': {
        'source_dir': '/home/16207041/data/negations_free_with_answer',
        'source_ext': 'csv',
        'pair_id_col': None,
        'negation_col': 'negation',
        'has_clean': True,
        'out_suffix': '_clean2',
    },
    'negation-controlled-with-answers': {
        'source_dir': '/home/16207041/data/negation_controlled_with_answers',
        'source_ext': 'csv',
        'pair_id_col': None,
        'negation_col': 'negation',
        'has_clean': True,
        'out_suffix': '_clean2',
    },
    'nevir-with-answers': {
        'source_dir': '/home/16207041/data/nevir_with_answers',
        'source_ext': 'parquet',
        'pair_id_col': 'id',
        'negation_col': None,
        'has_clean': False,
        'out_suffix': '_clean',
    },
}

SPLITS = ['train', 'validation', 'test']


def has_none_like(s) -> bool:
    return str(s).strip().lower() in NONE_LIKE


def make_prompt(question: str) -> str:
    question = question.strip()
    if not question.endswith('?'):
        question += '?'
    return (
        "Answer the given question. "
        "You must conduct reasoning inside <think> and </think> first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between "
        "<information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside "
        "<answer> and </answer>, without detailed illustrations. "
        f"For example, <answer> Beijing </answer>. Question: {question}\n"
    )


def to_records(df: pd.DataFrame, split: str, data_source: str, pair_id_col, negation_col) -> list:
    records = []
    for idx, row in df.iterrows():
        pair_id = str(row[pair_id_col]) if pair_id_col else str(idx)
        negation_type = (
            int(row[negation_col])
            if negation_col and pd.notna(row.get(negation_col))
            else None
        )
        for role, q_col, a_col in [('q1', 'q1', 'answer1'), ('q2', 'q2', 'answer2')]:
            records.append({
                'data_source': data_source,
                'prompt': [{'role': 'user', 'content': make_prompt(str(row[q_col]))}],
                'ability': 'fact-reasoning',
                'reward_model': {
                    'style': 'rule',
                    'ground_truth': {'target': [str(row[a_col]).strip()]},
                },
                'extra_info': {
                    'split': split,
                    'pair_id': pair_id,
                    'query_role': role,
                    'negation_type': negation_type,
                },
            })
    return records


def get_clean_pair_ids(rag_ds_dir: str, split: str) -> set:
    """Extract pair_ids from an existing _clean processed parquet."""
    path = os.path.join(rag_ds_dir, f'{split}_clean.parquet')
    df = pd.read_parquet(path)
    return {str(r['pair_id']) for r in df['extra_info']}


def print_stats_header():
    print(f"\n{'='*90}")
    print(f"{'Dataset':<30} {'Split':<12} {'Raw':>6} {'_clean':>8} {'_clean2':>8} "
          f"{'raw→clean':>12} {'clean→clean2':>14}")
    print(f"{'-'*90}")


def main():
    print_stats_header()

    for ds_name, cfg in DATASETS.items():
        rag_ds_dir = os.path.join(RAG_DIR, ds_name)
        os.makedirs(rag_ds_dir, exist_ok=True)

        for split in SPLITS:
            # --- Load raw source ---
            src_path = os.path.join(cfg['source_dir'], f'{split}.{cfg["source_ext"]}')
            if not os.path.exists(src_path):
                print(f"  SKIP {ds_name} {split}: source not found at {src_path}")
                continue

            raw = (pd.read_csv(src_path) if cfg['source_ext'] == 'csv'
                   else pd.read_parquet(src_path))
            n_raw = len(raw)

            # --- Determine clean_ids (pairs approved by existing _clean) ---
            if cfg['has_clean']:
                clean_ids = get_clean_pair_ids(rag_ds_dir, split)
                id_col = cfg['pair_id_col']
                raw_ids = {
                    str(row[id_col]) if id_col else str(idx)
                    for idx, row in raw.iterrows()
                }
                n_clean = len(clean_ids)
            else:
                # nevir: no existing _clean, start from full source
                clean_ids = None
                n_clean = None

            # --- Apply none-like filter ---
            removed_clean2 = []
            kept_rows = []
            for idx, row in raw.iterrows():
                pair_id = (str(row[cfg['pair_id_col']]) if cfg['pair_id_col']
                           else str(idx))
                # Skip pairs not in the existing _clean set
                if clean_ids is not None and pair_id not in clean_ids:
                    continue
                # Check none-like
                a1_bad = has_none_like(row['answer1'])
                a2_bad = has_none_like(row['answer2'])
                if a1_bad or a2_bad:
                    reason = []
                    if a1_bad:
                        reason.append(f'a1={repr(str(row["answer1"]))}')
                    if a2_bad:
                        reason.append(f'a2={repr(str(row["answer2"]))}')
                    removed_clean2.append((pair_id, ', '.join(reason)))
                else:
                    kept_rows.append(idx)

            clean2_df = raw.loc[kept_rows]
            n_clean2 = len(clean2_df)

            # --- Print statistics row ---
            raw_to_clean = (f'{n_raw - n_clean} ({(n_raw - n_clean) / n_raw:.1%})'
                            if n_clean is not None else '—')
            clean_to_clean2 = (
                f'{n_clean - n_clean2} ({(n_clean - n_clean2) / n_clean:.1%})'
                if n_clean is not None
                else f'{n_raw - n_clean2} ({(n_raw - n_clean2) / n_raw:.1%})'
            )
            clean_str = str(n_clean) if n_clean is not None else '—'
            print(f"{ds_name:<30} {split:<12} {n_raw:>6} {clean_str:>8} {n_clean2:>8} "
                  f"{raw_to_clean:>12} {clean_to_clean2:>14}")

            # --- Print removed pairs detail ---
            if removed_clean2:
                label = 'clean→clean2' if cfg['has_clean'] else 'raw→clean'
                print(f"  Removed {label} ({split}):")
                for pid, reason in removed_clean2:
                    print(f"    [{pid}] {reason}")

            # --- Write output parquet ---
            records = to_records(
                clean2_df, split, ds_name,
                cfg['pair_id_col'], cfg['negation_col'],
            )
            out_name = f'{split}{cfg["out_suffix"]}.parquet'
            out_path = os.path.join(rag_ds_dir, out_name)
            pd.DataFrame(records).to_parquet(out_path, index=False)

        print(f"{'-'*90}")

    print()


if __name__ == '__main__':
    main()
