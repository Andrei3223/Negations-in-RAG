"""
Preprocess GPT4o-Negation-Controlled and GPT4o-Negation-Free datasets to parquet format.

Dataset schema: split, topic, q1, doc1, q2, doc2, negation (int 0-9)
  - q1 is relevant only to doc1
  - q2 is relevant only to doc2

Each row is expanded into TWO parquet entries (one per query). Both documents are
always present in the context so the model must discriminate between them.

Task is framed as document selection:
  - q1 entry: ground truth = ["document 1"]
  - q2 entry: ground truth = ["document 2"]

pair_id and query_role in extra_info enable Per-Instance Accuracy:
a pair scores 1 only when both q1→"document 1" AND q2→"document 2".
"""

import os
import argparse
import datasets
import pandas as pd

from verl.utils.hdfs_io import copy, makedirs


HF_REPO_MAP = {
    'gpt4o-negation-controlled': 'RoxanaMaria/gpt4o-negation-controlled',
    'gpt4o-negation-free':       'RoxanaMaria/gpt4o-negation-free',
}

INSTRUCTION = (
    "You are given a question and two documents. "
    "Identify which document answers the question. "
    "Show your reasoning in <think> </think> tags. "
    "Then output the relevant document label in <answer> </answer> tags, "
    "for example <answer> document 1 </answer> or <answer> document 2 </answer>."
)

# Ground truth labels aligned with the instruction wording
ANSWER_FOR_Q1 = ["document 1"]
ANSWER_FOR_Q2 = ["document 2"]


def make_context(doc1_text: str, doc2_text: str) -> str:
    return (
        f"Document 1: {doc1_text.strip()}\n\n"
        f"Document 2: {doc2_text.strip()}\n"
    )


def make_prompt(question: str, context: str) -> str:
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\n{context}"


def process_split(hf_split, split_name: str, data_source: str) -> list:
    """
    Expand each row into two records: one for (q1, doc1, doc2) and one for (q2, doc1, doc2).
    """
    records = []
    for idx, example in enumerate(hf_split):
        pair_id = str(idx)
        doc1 = example['doc1']
        doc2 = example['doc2']
        context = make_context(doc1, doc2)
        negation_type = example.get('negation', None)

        for query_role, question_key, targets in [
            ('q1', 'q1', ANSWER_FOR_Q1),
            ('q2', 'q2', ANSWER_FOR_Q2),
        ]:
            question = example[question_key].strip()
            prompt_text = make_prompt(question, context)

            record = {
                'data_source': data_source,
                'prompt': [{'role': 'user', 'content': prompt_text}],
                'ability': 'fact-reasoning',
                'reward_model': {
                    'style': 'rule',
                    'ground_truth': {'target': targets},
                },
                'extra_info': {
                    'split': split_name,
                    'index': idx,
                    'pair_id': pair_id,
                    'query_role': query_role,
                    'negation_type': negation_type,
                },
            }
            records.append(record)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/negation_rag')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument(
        '--dataset',
        choices=list(HF_REPO_MAP.keys()) + ['both'],
        default='both',
        help='Which negation dataset to process',
    )
    args = parser.parse_args()

    sources_to_process = (
        list(HF_REPO_MAP.keys()) if args.dataset == 'both' else [args.dataset]
    )

    for data_source in sources_to_process:
        hf_repo = HF_REPO_MAP[data_source]
        print(f"Loading {hf_repo} ...")
        raw = datasets.load_dataset(hf_repo)

        available_splits = list(raw.keys())
        print(f"  Available splits: {available_splits}")
        print(f"  Columns: {raw[available_splits[0]].column_names}")

        out_dir = os.path.join(args.local_dir, data_source)
        os.makedirs(out_dir, exist_ok=True)

        for split_name in available_splits:
            records = process_split(raw[split_name], split_name, data_source)
            out_path = os.path.join(out_dir, f'{split_name}.parquet')
            pd.DataFrame(records).to_parquet(out_path, index=False)
            print(f"  Wrote {len(records)} entries ({len(records) // 2} pairs) → {out_path}")

        if args.hdfs_dir is not None:
            makedirs(args.hdfs_dir)
            copy(src=out_dir, dst=os.path.join(args.hdfs_dir, data_source))
