"""
Generate short NQ-style answers for each doc-query pair in the negation-controlled dataset.

Two-phase per batch:
  Phase 1 — generate answer1 for all rows (q1 + doc1)
  Phase 2 — generate answer2 for all rows (q2 + doc2 + answer1 as reference)

Passing answer1 into phase 2 prevents the model from blindly copying it for
antonym/contrary/polar questions where the correct answer is the logical opposite.

Optional fix pass (--no_fix to disable):
  After generation, rows where answer1 == answer2 are re-run with a stricter
  reformulation prompt that explicitly forbids copying the related answer.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_client import call_with_retry, make_client  # noqa: E402

load_dotenv(Path(__file__).parent / '.env')

NEGATION_TYPES = {
    0: 'sentential',
    1: 'exceptor',
    2: 'contradiction',
    3: 'contrary',
    4: 'subcontradiction',
    5: 'affixal',
    6: 'implicit',
    7: 'immediate antonyms',
    8: 'mid antonyms',
    9: 'polar antonyms',
}

# Types whose q2 questions are existential yes/no by structure
BINARY_TYPES = {'contradiction', 'contrary', 'subcontradiction'}

NEGATION_HINTS = {
    'sentential':         "The entire proposition is negated. Answer what is actually true under the negation.",
    'exceptor':           "The original question used 'except X'; this one removes that exclusion. Include X in the answer.",
    'contradiction':      "This is a yes/no existential question. Answer ONLY 'yes' or 'no' — nothing else.",
    'contrary':           "This is a yes/no existential question. Answer ONLY 'yes' or 'no' — nothing else.",
    'subcontradiction':   "This is a yes/no existential question. Answer ONLY 'yes' or 'no' — nothing else.",
    'affixal':            "A negating prefix/suffix (un-, non-, in-, -less …) flips the meaning. Give the factual answer in 1-4 words.",
    'implicit':           "The negation is implicit — a key word was replaced by its opposite concept. Identify the shift and answer in 1-4 words.",
    'immediate antonyms': "A direct antonym replaces the key word. Derive the factual opposite answer (e.g. if lost→80%, then found→20%). 1-4 words.",
    'mid antonyms':       "A mid-scale antonym replaces the key word. Derive the factual opposite answer in 1-4 words.",
    'polar antonyms':     "A polar antonym replaces the key word. Derive the logically opposite factual answer in 1-4 words.",
}

SYSTEM_Q1 = (
    "Answer questions as briefly as possible. "
    "Output only the answer, 1-5 words maximum, no punctuation, no explanation. "
    "For yes/no questions output only 'yes' or 'no'."
)

# NevIR: doc2 was rewritten specifically to answer q2 — extract directly, don't copy answer1
SYSTEM_Q2_NEVIR = (
    "Answer questions as briefly as possible. 1-5 words max, no punctuation.\n\n"
    "The document was specifically written to answer this question. Extract the answer directly.\n\n"
    "Rules:\n"
    "- Do NOT copy the related answer.\n"
    "- Give a factual answer in 1-5 words taken from the document."
)

SYSTEM_FIX = """\
Answer questions as briefly as possible. No punctuation.

You are given a document, a related question with its answer, and a NEW question.
The new question is the negation/opposite of the related question.

STRICT RULES:
- Your answer MUST be different from the related answer. If you cannot find a different answer, output "not mentioned".
- Do NOT repeat, paraphrase, or negate the related answer (e.g. do NOT output "not X" where X is the related answer).
- Search the document specifically for information about the concept in the NEW question.
- If the document contains no information about the new question's concept, output "not mentioned".
- For yes/no questions (contradiction/contrary/subcontradiction): answer ONLY 'yes' or 'no'.
- For antonym questions: derive the logical opposite from facts in the document (e.g. lost→80% means found→20%). If no such fact exists, output "not mentioned".
- For affixal questions (un-, non-, in-…): look for the positive/base form in the document."""


def system_q2(negation_type: str) -> str:
    hint = NEGATION_HINTS.get(negation_type, "The question negates a related question.")
    binary = negation_type in BINARY_TYPES
    answer_rule = (
        "- Answer ONLY 'yes' or 'no' — no extra words."
        if binary else
        "- Give a factual answer in 1-4 words. Do NOT copy the related answer unless the document explicitly confirms it.\n"
        "- For antonym/polar types: if related answer is A, derive the logical opposite of A."
    )
    return (
        f"Answer questions as briefly as possible. No punctuation.\n\n"
        f"Negation type: {negation_type}. {hint}\n\n"
        f"Rules:\n"
        f"- You are given the related question and its answer as reference.\n"
        f"{answer_rule}"
    )


def call_api(client, model: str, system: str, user: str, retries: int = 3) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return call_with_retry(client, model, messages, max_tokens=30, retries=retries) or "N/A"


def run_batch(client: AzureOpenAI, model: str, system: str, user_msgs: list[str]) -> list[str]:
    with ThreadPoolExecutor(max_workers=len(user_msgs)) as pool:
        futures = {
            pool.submit(call_api, client, model, system, msg): i
            for i, msg in enumerate(user_msgs)
        }
        results = [None] * len(user_msgs)
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results


def process_batch(client: AzureOpenAI, model: str, rows: list[dict], dataset: str) -> list[tuple[str, str]]:
    # ── Phase 1: answer1 for every row ───────────────────────────────────────
    answers1 = run_batch(
        client, model, SYSTEM_Q1,
        [f"Document: {row['doc1']}\nQuestion: {row['q1']}" for row in rows],
    )

    # ── Phase 2: answer2 using answer1 as reference ───────────────────────────
    if dataset == 'nevir':
        # doc2 was rewritten to answer q2 — single shared prompt, batch call
        answers2 = run_batch(
            client, model, SYSTEM_Q2_NEVIR,
            [
                f"Document: {row['doc2']}\n"
                f"Related question: {row['q1']}\n"
                f"Related answer: {answers1[i]}\n"
                f"Question to answer: {row['q2']}"
                for i, row in enumerate(rows)
            ],
        )
    else:
        # negation_controlled: per-row prompt based on negation type
        with ThreadPoolExecutor(max_workers=len(rows)) as pool:
            futures = {
                pool.submit(
                    call_api, client, model,
                    system_q2(NEGATION_TYPES.get(row.get('negation'), 'negation')),
                    (
                        f"Document: {row['doc2']}\n"
                        f"Related question: {row['q1']}\n"
                        f"Related answer: {answers1[i]}\n"
                        f"Question to answer: {row['q2']}"
                    ),
                ): i
                for i, row in enumerate(rows)
            }
            answers2 = [None] * len(rows)
            for fut in as_completed(futures):
                answers2[futures[fut]] = fut.result()

    return list(zip(answers1, answers2))


def fix_same_answers(client: AzureOpenAI, model: str, df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    same_idx = df[
        df['answer1'].str.lower().str.strip() == df['answer2'].str.lower().str.strip()
    ].index.tolist()

    if not same_idx:
        print("    No same-answer rows found.")
        return df

    print(f"    {len(same_idx)} rows with answer1 == answer2 → re-generating with fix prompt")
    rows_todo = df.loc[same_idx].to_dict('records')
    fixed = []

    for start in range(0, len(rows_todo), batch_size):
        chunk = rows_todo[start:start + batch_size]
        with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            futures = {
                pool.submit(
                    call_api, client, model, SYSTEM_FIX,
                    (
                        f"Document: {row['doc2']}\n"
                        f"Related question: {row['q1']}\n"
                        f"Related answer (DO NOT USE THIS): {row['answer1']}\n"
                        f"Negation type: {NEGATION_TYPES.get(row.get('negation'), 'negation')}\n"
                        f"New question: {row['q2']}"
                    ),
                ): i
                for i, row in enumerate(chunk)
            }
            chunk_results = [None] * len(chunk)
            for fut in as_completed(futures):
                chunk_results[futures[fut]] = fut.result()
        fixed.extend(chunk_results)
        print(f"    {min(start + batch_size, len(rows_todo))}/{len(rows_todo)} re-generated")

    still_same = sum(
        1 for i, idx in enumerate(same_idx)
        if df.loc[idx, 'answer1'].lower().strip() == fixed[i].lower().strip()
    )
    for i, idx in enumerate(same_idx):
        df.loc[idx, 'answer2'] = fixed[i]

    print(f"    Changed: {len(same_idx) - still_same} / {len(same_idx)} | still same: {still_same}")
    return df


# Single-token function words that are never valid standalone answers
_BAD_TOKENS = {
    'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its',
    'their', 'we', 'us', 'i', 'me', 'my', 'you', 'the', 'a', 'an',
}

def _is_bad_answer(answer: str) -> bool:
    return answer.strip().lower() in _BAD_TOKENS


def fix_pronoun_answers(client: AzureOpenAI, model: str, df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Re-run rows where either answer is a bare pronoun or article."""
    bad1 = df['answer1'].apply(_is_bad_answer)
    bad2 = df['answer2'].apply(_is_bad_answer)
    bad_idx = df[bad1 | bad2].index.tolist()

    if not bad_idx:
        return df

    print(f"    {len(bad_idx)} rows with pronoun/article answers → re-generating")

    def fix_one(row: dict, col: str) -> str:
        doc_key = 'doc1' if col == 'answer1' else 'doc2'
        q_key = 'q1' if col == 'answer1' else 'q2'
        return call_api(
            client, model, SYSTEM_Q1,
            f"Document: {row[doc_key]}\nQuestion: {row[q_key]}\n"
            f"Important: answer with a proper noun or phrase, never a pronoun.",
        )

    for start in range(0, len(bad_idx), batch_size):
        chunk_idx = bad_idx[start:start + batch_size]
        with ThreadPoolExecutor(max_workers=len(chunk_idx)) as pool:
            futures = {}
            for idx in chunk_idx:
                row = df.loc[idx].to_dict()
                if _is_bad_answer(row['answer1']):
                    futures[pool.submit(fix_one, row, 'answer1')] = (idx, 'answer1')
                if _is_bad_answer(row['answer2']):
                    futures[pool.submit(fix_one, row, 'answer2')] = (idx, 'answer2')
            for fut in as_completed(futures):
                idx, col = futures[fut]
                df.loc[idx, col] = fut.result()

    return df


def process_split(df: pd.DataFrame, model: str, batch_size: int, fix: bool, dataset: str) -> pd.DataFrame:
    client = make_client()
    rows = df.to_dict('records')
    answers1, answers2 = [], []

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        results = process_batch(client, model, chunk, dataset)
        for a1, a2 in results:
            answers1.append(a1)
            answers2.append(a2)
        print(f"    {min(start + batch_size, len(rows))}/{len(rows)} rows done")

    df = df.copy()
    df['answer1'] = answers1
    df['answer2'] = answers2

    if fix:
        print("  Running fix pass for same-answer rows ...")
        df = fix_same_answers(client, model, df, batch_size)

    print("  Running pronoun/article check ...")
    df = fix_pronoun_answers(client, model, df, batch_size)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['negation_controlled', 'nevir'],
                        default='negation_controlled')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--model', default='gpt-4.1-mini')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'])
    parser.add_argument('--no_fix', action='store_true', help='Skip the same-answer fix pass')
    parser.add_argument('--force', action='store_true', help='Re-generate even if output already exists')
    args = parser.parse_args()

    # Apply dataset-specific defaults when not explicitly set
    DATASET_DEFAULTS = {
        'negation_controlled': ('./data/negation_controlled', './data/negation_controlled_with_answers'),
        'nevir':               ('./data/nevir',               './data/nevir_with_answers'),
    }
    default_data, default_out = DATASET_DEFAULTS[args.dataset]
    if args.data_dir is None:
        args.data_dir = default_data
    if args.output_dir is None:
        args.output_dir = default_out

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name in args.splits:
        in_path = os.path.join(args.data_dir, f'{split_name}.parquet')
        out_path = os.path.join(args.output_dir, f'{split_name}.parquet')

        if not os.path.exists(in_path):
            print(f"  [{split_name}] input not found: {in_path}, skipping")
            continue

        df = pd.read_parquet(in_path)

        # Resume: skip rows that already have valid answers
        if not args.force and os.path.exists(out_path):
            done_df = pd.read_parquet(out_path)
            if 'answer1' in done_df.columns and 'answer2' in done_df.columns:
                n_done = done_df['answer1'].notna().sum()
                print(f"  [{split_name}] resuming — {n_done}/{len(df)} rows already done")
                if n_done == len(df):
                    print(f"  [{split_name}] already complete, skipping")
                    continue
                missing_idx = done_df[done_df['answer1'].isna()].index
                df_todo = df.loc[missing_idx].copy()
                df_done = done_df.drop(index=missing_idx)
            else:
                df_todo = df
                df_done = None
        else:
            df_todo = df
            df_done = None

        print(f"  [{split_name}] processing {len(df_todo)} rows with {args.model} (batch_size={args.batch_size}, dataset={args.dataset}) ...")
        df_todo = process_split(df_todo, args.model, args.batch_size, fix=not args.no_fix, dataset=args.dataset)

        if df_done is not None:
            result_df = pd.concat([df_done, df_todo]).sort_index()
        else:
            result_df = df_todo

        result_df.to_parquet(out_path, index=False)
        result_df.to_csv(out_path.replace('.parquet', '.csv'), index=False)
        print(f"  [{split_name}] {len(result_df)} rows → {out_path}")
        print(f"    Sample answer1: {result_df['answer1'].iloc[0]!r}")
        print(f"    Sample answer2: {result_df['answer2'].iloc[0]!r}")
