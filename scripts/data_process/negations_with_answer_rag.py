"""
Preprocess negation QA datasets to parquet format for live-retrieval eval.

Supported datasets (--dataset):
  negations-free-with-answer        /home/16207041/data/negations_free_with_answer
  negation-controlled-with-answers  /home/16207041/data/negation_controlled_with_answers
  nevir-with-answers                /home/16207041/data/nevir_with_answers

Schema (all three): q1, q2, answer1, answer2, [negation (int 0-9)], [id]

Each row expands into TWO records. Documents are NOT included in the prompt —
the model retrieves from wiki-18 via live search (same as NQ pipeline).

  q1 record: question=q1, ground_truth=[answer1], query_role='q1'
  q2 record: question=q2, ground_truth=[answer2], query_role='q2'
"""

import os
import argparse
import pandas as pd


DATASETS = {
    'negations-free-with-answer': {
        'source_dir': '/home/16207041/data/negations_free_with_answer',
        'pair_id_col': None,   # use row index
        'negation_col': 'negation',
    },
    'negation-controlled-with-answers': {
        'source_dir': '/home/16207041/data/negation_controlled_with_answers',
        'pair_id_col': None,
        'negation_col': 'negation',
    },
    'nevir-with-answers': {
        'source_dir': '/home/16207041/data/nevir_with_answers',
        'pair_id_col': 'id',   # dataset has its own id field
        'negation_col': None,
    },
    'negations-free-sentence-with-answer': {
        'source_dir': '/home/16207041/data/negations_free_sentence_with_answer',
        'pair_id_col': None,
        'negation_col': 'negation',
    },
}

SPLITS = ['train', 'validation', 'test']


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


def process_split(df: pd.DataFrame, split_name: str, data_source: str, cfg: dict) -> list:
    records = []
    for idx, row in df.iterrows():
        pair_id = str(row[cfg['pair_id_col']]) if cfg['pair_id_col'] else str(idx)
        negation_type = int(row[cfg['negation_col']]) if cfg['negation_col'] and pd.notna(row[cfg['negation_col']]) else None

        for query_role, question_col, answer_col in [
            ('q1', 'q1', 'answer1'),
            ('q2', 'q2', 'answer2'),
        ]:
            record = {
                'data_source': data_source,
                'prompt': [{'role': 'user', 'content': make_prompt(str(row[question_col]))}],
                'ability': 'fact-reasoning',
                'reward_model': {
                    'style': 'rule',
                    'ground_truth': {'target': [str(row[answer_col]).strip()]},
                },
                'extra_info': {
                    'split': split_name,
                    'pair_id': pair_id,
                    'query_role': query_role,
                    'negation_type': negation_type,
                },
            }
            records.append(record)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=list(DATASETS.keys()) + ['all'],
        default='all',
        help='Which dataset to process (default: all)',
    )
    parser.add_argument('--local_dir', default='./data/negations_with_answer_rag')
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]

    for data_source in targets:
        cfg = DATASETS[data_source]
        out_dir = os.path.join(args.local_dir, data_source)
        os.makedirs(out_dir, exist_ok=True)
        print(f'\n=== {data_source} ===')

        for split in SPLITS:
            src = os.path.join(cfg['source_dir'], f'{split}.parquet')
            if not os.path.exists(src):
                print(f'  Skipping {split} (not found)')
                continue

            df = pd.read_parquet(src)
            records = process_split(df, split, data_source, cfg)
            out_path = os.path.join(out_dir, f'{split}.parquet')
            pd.DataFrame(records).to_parquet(out_path, index=False)
            print(f'  {split}: {len(df)} rows → {len(records)} records → {out_path}')
