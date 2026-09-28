"""
Preprocess the BoolQ dataset to parquet format.

Dataset: google/boolq
Schema: question (str), answer (bool), passage (str)

The passage is embedded in the prompt as context. The model reads the passage
and answers yes/no. Ground truth: ["yes"] if answer==True, ["no"] if answer==False.

Splits available: train (9,427), validation (3,270) — no labeled test split.
"""

import os
import argparse
import datasets
import pandas as pd

from verl.utils.hdfs_io import copy, makedirs


HF_REPO = 'google/boolq'
DATA_SOURCE = 'boolq'

INSTRUCTION = (
    "Answer the yes/no question based on the provided passage. "
    "Show your reasoning in <think> </think> tags and return the final answer "
    "in <answer> </answer> tags, for example <answer> yes </answer> or <answer> no </answer>."
)


def make_prompt(question: str, passage: str) -> str:
    question = question.strip()
    if not question.endswith('?'):
        question += '?'
    return (
        f"{INSTRUCTION}\n\n"
        f"Passage: {passage.strip()}\n\n"
        f"Question: {question}\n"
    )


def process_split(hf_split, split_name: str) -> list:
    records = []
    for idx, example in enumerate(hf_split):
        target = ["yes"] if example['answer'] else ["no"]
        prompt_text = make_prompt(example['question'], example['passage'])
        record = {
            'data_source': DATA_SOURCE,
            'prompt': [{'role': 'user', 'content': prompt_text}],
            'ability': 'fact-reasoning',
            'reward_model': {
                'style': 'rule',
                'ground_truth': {'target': target},
            },
            'extra_info': {
                'split': split_name,
                'index': idx,
            },
        }
        records.append(record)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/boolq_rag')
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
        # Save validation split as both 'validation.parquet' and 'test.parquet'
        # so it can be used as a drop-in test set (BoolQ has no labeled test split)
        out_path = os.path.join(args.local_dir, f'{split_name}.parquet')
        pd.DataFrame(records).to_parquet(out_path, index=False)
        print(f"  Wrote {len(records)} entries → {out_path}")
        if split_name == 'validation':
            test_path = os.path.join(args.local_dir, 'test.parquet')
            pd.DataFrame(records).to_parquet(test_path, index=False)
            print(f"  Copied validation → {test_path} (no labeled test split exists)")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
