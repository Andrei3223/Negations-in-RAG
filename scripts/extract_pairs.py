"""
Parse evaluation log files and produce a CSV with one row per evaluation record.
Each row contains: question (from parquet), golden_answers, extracted_answer,
query_role (q1/q2), pair_id, model, dataset.

Source parquet in $RAW_DATA_DIR stores one row per PAIR with explicit
q1/q2/answer1/answer2 columns, so pair_id = row index and role is determined
by matching the golden answer against answer1 (→ q1) or answer2 (→ q2).

The log trajectories interleave question text and golden-answer blocks for
different records, so we match records to the parquet via their ground-truth
answer (not by the nearby question text in the log).
"""

import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", "/home/16207041/data"))

LOGS = [
    # SearchR1 runs
    {
        "log": "eval_logs/slurm/searchr1-eval-325918.out",
        "dataset": "negation_controlled_with_answers",
        "model": "searchr1",
        "eval_parquet": "data/negations_with_answer_rag/negation-controlled-with-answers/test_clean2.parquet",
        "src_split": "test_clean2",
    },
    {
        "log": "eval_logs/slurm/searchr1-eval-325919.out",
        "dataset": "negations_free_with_answer",
        "model": "searchr1",
        "eval_parquet": "data/negations_with_answer_rag/negations-free-with-answer/test_clean2.parquet",
        "src_split": "test_clean2",
    },
    {
        "log": "eval_logs/slurm/searchr1-eval-325920.out",
        "dataset": "nevir_with_answers",
        "model": "searchr1",
        "eval_parquet": "data/negations_with_answer_rag/nevir-with-answers/test.parquet",
    },
    # em-grpo runs
    {
        "log": "eval_logs/searchr1-eval-328648/eval.log",
        "dataset": "negations_free_sentence_with_answer",
        "model": "em-grpo",
        "eval_parquet": "data/negations_with_answer_rag/negations-free-sentence-with-answer/test_clean.parquet",
        "src_split": "test",
    },
    # SubSearch runs
    {
        "log": "eval_logs/searchr1-eval-328647/eval.log",
        "dataset": "negations_free_sentence_with_answer",
        "model": "subsearch",
        "eval_parquet": "data/negations_with_answer_rag/negations-free-sentence-with-answer/test_clean.parquet",
        "src_split": "test",
    },
    {
        "log": "eval_logs/slurm/searchr1-eval-326647.out",
        "dataset": "negations_free_with_answer",
        "model": "subsearch",
        "eval_parquet": "data/negations_with_answer_rag/negations-free-with-answer/test_clean2.parquet",
        "src_split": "test_clean2",
    },
    {
        "log": "eval_logs/slurm/searchr1-eval-326648.out",
        "dataset": "negation_controlled_with_answers",
        "model": "subsearch",
        "eval_parquet": "data/negations_with_answer_rag/negation-controlled-with-answers/test_clean2.parquet",
        "src_split": "test_clean2",
    },
    {
        "log": "eval_logs/slurm/searchr1-eval-326630.out",
        "dataset": "nevir_with_answers",
        "model": "subsearch",
        "eval_parquet": "data/negations_with_answer_rag/nevir-with-answers/test.parquet",
    },
    # custom-index runs — negations_free_sentence (topk=1, data/custom_index/)
    {
        "log": "eval_logs/searchr1-eval-328731/eval.log",
        "dataset": "negations_free_sentence_with_answer",
        "model": "em-grpo-custom",
        "eval_parquet": "data/negations_with_answer_rag/negations-free-sentence-with-answer/test_clean.parquet",
        "src_split": "test",
    },
    {
        "log": "eval_logs/searchr1-eval-328732/eval.log",
        "dataset": "negations_free_sentence_with_answer",
        "model": "subsearch-custom",
        "eval_parquet": "data/negations_with_answer_rag/negations-free-sentence-with-answer/test_clean.parquet",
        "src_split": "test",
    },
    # custom-index runs — nevir (topk=1, data/custom_index/)
    {
        "log": "eval_logs/searchr1-eval-328649/eval.log",
        "dataset": "nevir_with_answers",
        "model": "em-grpo-custom",
        "eval_parquet": "data/negations_with_answer_rag/nevir-with-answers/test_clean.parquet",
        "src_split": "test_clean",
    },
    {
        "log": "eval_logs/searchr1-eval-328651/eval.log",
        "dataset": "nevir_with_answers",
        "model": "em-grpo-custom-2",
        "eval_parquet": "data/negations_with_answer_rag/nevir-with-answers/test_clean.parquet",
        "src_split": "test_clean",
    },
    {
        "log": "eval_logs/searchr1-eval-329206/eval.log",
        "dataset": "nevir_with_answers",
        "model": "subsearch-custom",
        "eval_parquet": "data/negations_with_answer_rag/nevir-with-answers/test_clean.parquet",
        "src_split": "test_clean",
    },
]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_RE = re.compile(r"^\([^)]+\)\s*")


def clean(line: str) -> str:
    return RAY_RE.sub("", ANSI_RE.sub("", line)).strip()


def valid_pair_ids_from_eval_parquet(eval_parquet: str) -> set[str]:
    """
    Read the processed eval parquet and return the set of pair_ids it contains.
    Each row in the processed parquet has extra_info['pair_id'].
    """
    df = pd.read_parquet(BASE_DIR / eval_parquet)
    return {str(row["extra_info"]["pair_id"]) for _, row in df.iterrows()}


def build_lookup(
    dataset: str,
    valid_pair_ids: set[str] | None = None,
    src_split: str = "test",
) -> dict[tuple, list[dict]]:
    """
    Build answer → [{pair_id, query_role, question}] from the source parquet,
    restricted to pairs in valid_pair_ids (if provided).

    Each row in the source parquet is one pair:
      columns: q1, q2, answer1, answer2 (+ optional id, topic, doc1, doc2)
      pair_id = id column if present, else row index (str)

    src_split selects which parquet file to read (e.g. "test", "test_clean2").
    Restricting to valid_pair_ids prevents wrong matches from source pairs that
    were not included in the evaluation split.
    """
    path = DATA_DIR / dataset / f"{src_split}.parquet"
    df = pd.read_parquet(path)

    lookup: dict[tuple, list[dict]] = defaultdict(list)
    id_col = "id" if "id" in df.columns else None
    for i, row in df.iterrows():
        pair_id = str(row[id_col]) if id_col else str(i)
        if valid_pair_ids is not None and pair_id not in valid_pair_ids:
            continue
        lookup[tuple([row["answer1"]])].append(
            {"pair_id": pair_id, "query_role": "q1", "question": row["q1"]}
        )
        lookup[tuple([row["answer2"]])].append(
            {"pair_id": pair_id, "query_role": "q2", "question": row["q2"]}
        )

    return dict(lookup)


SAMPLE_RE = re.compile(r"^\[SAMPLE_R\d+_I\d+\]\s*")
SEPARATOR = "--------------------------------"
QUOTED_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def parse_gold(text: str) -> list[str]:
    """Gold answers from a logged list, in either repr.

    Python lists print as ['a', 'b']; the numpy arrays newer logs print have no
    commas, ['a' 'b'], which ast.literal_eval silently joins into 'ab'. So read
    each quoted item on its own (numpy switches to "..." around apostrophes).
    """
    return [ast.literal_eval(tok) for tok in QUOTED_RE.findall(text)]


def parse_log(log_path: str) -> list[dict]:
    """Return ordered list of {golden_answers, extracted_answer} dicts.

    Two log formats:
      old: a bare separator line, then "Golden answers:" and "Extracted answer:"
           within the next few lines (one block per record);
      new: every line prefixed "[SAMPLE_R<r>_I<i>] ", and a multi-line
           "Reasoning Trace:" between the separator and "Extracted answer:",
           which now comes BEFORE "Golden answers:".
    A record is emitted once both fields have been seen since the last
    separator, in whichever order they appear. A long gold list wraps onto
    continuation lines (numpy line width), so it is read until its closing "]".
    """
    records = []
    golden = extracted = None
    gold_buf = None            # an unfinished, wrapped "Golden answers:" list
    in_block = False
    with open(log_path) as f:
        for raw in f:
            line = SAMPLE_RE.sub("", clean(raw))
            if gold_buf is not None:
                gold_buf += " " + line
                if line.endswith("]") or gold_buf.count("\n") > 20:
                    golden, gold_buf = parse_gold(gold_buf), None
            elif line.startswith(SEPARATOR):
                # startswith, not ==: Ray streams worker output asynchronously, so
                # another process's print can land on the same line (seen once:
                # "...----Stopping retrieval server (PID ...)").
                in_block, golden, extracted = True, None, None
                continue
            elif not in_block:
                continue
            elif line.startswith("Golden answers:") and golden is None:
                text = line[len("Golden answers:"):].strip()
                if text.endswith("]"):
                    golden = parse_gold(text)
                else:
                    gold_buf = text
            elif line.startswith("Extracted answer:") and extracted is None:
                extracted = line[len("Extracted answer:"):].strip()
            if golden is not None and extracted is not None:
                records.append({"golden_answers": golden, "extracted_answer": extracted})
                in_block, golden, extracted = False, None, None

    return records


def match_records(log_records: list[dict], lookup: dict) -> list[dict]:
    """
    Match each log record to a source-parquet entry via the golden-answer tuple.

    Primary match: consume entries in dataset order (handles different pairs
    that share the same answer string).
    Fallback: when the queue is exhausted, re-use the last matched entry for
    that answer key — covers n_val_response > 1 (multiple evaluation runs of
    the same item producing identical golden answers).
    """
    queues: dict[tuple, list[dict]] = {k: list(v) for k, v in lookup.items()}
    last_matched: dict[tuple, dict] = {}
    result = []

    for rec in log_records:
        gt_key = tuple(rec["golden_answers"])
        queue = queues.get(gt_key, [])

        if queue:
            info = queue.pop(0)
            last_matched[gt_key] = info
            result.append(
                {**rec, "question": info["question"], "query_role": info["query_role"],
                 "pair_id": info["pair_id"], "matched": True}
            )
        elif gt_key in last_matched:
            info = last_matched[gt_key]
            result.append(
                {**rec, "question": info["question"], "query_role": info["query_role"],
                 "pair_id": info["pair_id"], "matched": True}
            )
        else:
            result.append(
                {**rec, "question": None, "query_role": "unknown",
                 "pair_id": "unknown", "matched": False}
            )

    return result


def main():
    all_rows = []

    for cfg in LOGS:
        log_path = BASE_DIR / cfg["log"]
        print(f"\n{'='*60}")
        print(f"Log   : {cfg['log']}")
        print(f"Model : {cfg['model']}   Dataset: {cfg['dataset']}")

        raw = parse_log(str(log_path))
        print(f"Records found in log: {len(raw)}")

        valid_ids = None
        if "eval_parquet" in cfg:
            valid_ids = valid_pair_ids_from_eval_parquet(cfg["eval_parquet"])
            print(f"Restricting lookup to {len(valid_ids)} pair_ids from {cfg['eval_parquet']}")
        lookup = build_lookup(cfg["dataset"], valid_ids, cfg.get("src_split", "test"))
        matched_recs = match_records(raw, lookup)

        n_matched = sum(r["matched"] for r in matched_recs)
        n_unknown = len(matched_recs) - n_matched
        print(f"Matched: {n_matched}   Unmatched: {n_unknown}")
        if n_unknown:
            for r in matched_recs:
                if not r["matched"]:
                    print(f"  WARNING no match for golden={r['golden_answers']}")

        q1 = sum(1 for r in matched_recs if r["query_role"] == "q1")
        q2 = sum(1 for r in matched_recs if r["query_role"] == "q2")
        # Count shared pairs (both q1 and q2 present for same pair_id)
        from collections import defaultdict as _dd
        roles_per_pair: dict = _dd(set)
        for r in matched_recs:
            if r["query_role"] in ("q1", "q2"):
                roles_per_pair[r["pair_id"]].add(r["query_role"])
        shared = sum(1 for v in roles_per_pair.values() if len(v) == 2)
        print(f"q1={q1}  q2={q2}  unknown={n_unknown}  shared_pairs={shared}")

        for r in matched_recs:
            all_rows.append(
                {
                    "question": r["question"],
                    "golden_answers_json": json.dumps(r["golden_answers"]),
                    "extracted_answer": r["extracted_answer"],
                    "query_role": r["query_role"],
                    "pair_id": r["pair_id"],
                    "model": cfg["model"],
                    "dataset": cfg["dataset"],
                    "log_file": cfg["log"],
                }
            )

    out = BASE_DIR / "run" / "extracted_pairs.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nSaved {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()
