"""
Preprocess the ExcluIR dataset to parquet format.

Dataset: thijmennijdam/ExcluIR
Schema: split, q1, q2, doc1, doc2
  - q1 asks about information in doc1, explicitly excluding doc2
  - q2 asks about information in doc2, explicitly excluding doc1

Same document-selection framing as negation_rag.py:
  - q1 entry: ground truth = ["document 1"]
  - q2 entry: ground truth = ["document 2"]

pair_id (row index) and query_role in extra_info enable Per-Instance Accuracy.
"""

import os
import argparse
import datasets
import pandas as pd

from verl.utils.hdfs_io import copy, makedirs


HF_REPO = 'thijmennijdam/ExcluIR'
DATA_SOURCE = 'exclusir'

INSTRUCTION = (
    "You are given a question and two documents. "
    "Identify which document answers the question. "
    "Show your reasoning in <think> </think> tags. "
    "Then output the relevant document label in <answer> </answer> tags, "
    "for example <answer> document 1 </answer> or <answer> document 2 </answer>."
)

ANSWER_FOR_Q1 = ["document 1"]
ANSWER_FOR_Q2 = ["document 2"]


def make_context(doc1_text: str, doc2_text: str) -> str:
    return (
        f"Document 1: {doc1_text.strip()}\n\n"
        f"Document 2: {doc2_text.strip()}\n"
    )


def make_prompt(question: str, context: str) -> str:
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\n{context}"


def process_split(hf_split, split_name: str) -> list:
    records = []
    for idx, example in enumerate(hf_split):
        # ExcluIR has no id field — use row index as pair_id
        pair_id = str(idx)
        doc1 = example['doc1']
        doc2 = example['doc2']
        context = make_context(doc1, doc2)

        for query_role, question_key, targets in [
            ('q1', 'q1', ANSWER_FOR_Q1),
            ('q2', 'q2', ANSWER_FOR_Q2),
        ]:
            prompt_text = make_prompt(example[question_key].strip(), context)
            record = {
                'data_source': DATA_SOURCE,
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
                },
            }
            records.append(record)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/exclusir_rag')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    print(f"Loading {HF_REPO} ...")
    raw = datasets.load_dataset(HF_REPO)
    available_splits = list(raw.keys())
    print(f"  Available splits: {available_splits}")
    print(f"  Columns: {raw[available_splits[0]].column_names}")

    os.makedirs(args.local_dir, exist_ok=True)

    for split_name in available_splits:
        records = process_split(raw[split_name], split_name)
        out_path = os.path.join(args.local_dir, f'{split_name}.parquet')
        pd.DataFrame(records).to_parquet(out_path, index=False)
        print(f"  Wrote {len(records)} entries ({len(records) // 2} pairs) → {out_path}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
