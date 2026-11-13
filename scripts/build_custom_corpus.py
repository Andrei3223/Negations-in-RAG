import json
import os
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "/home/16207041/data")  # raw negation datasets

DATASETS = [
    {
        "name": "negations_free_sentence_with_answer",
        "dir": Path(RAW_DATA_DIR, "negations_free_sentence_with_answer"),
        "splits": ["train", "validation", "test"],
        "title_col": "topic",
    },
    {
        "name": "nevir_with_answers",
        "dir": Path(RAW_DATA_DIR, "nevir_with_answers"),
        "splits": ["train", "validation", "test"],
        "title_col": "id",
        "title_prefix": "nevir_",
    },
]

OUTPUT_PATH = Path(os.environ.get("REPO_ROOT", "/home/16207041/Negations-in-RAG"), "data/custom_index/corpus.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

seen: set[int] = set()
docs: list[dict] = []
doc_id = 0

for ds in DATASETS:
    prefix = ds.get("title_prefix", "")
    for split in ds["splits"]:
        csv_path = ds["dir"] / f"{split}.csv"
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            title = f"{prefix}{row[ds['title_col']]}"
            for doc_col in ("doc1", "doc2"):
                text = row.get(doc_col)
                if not isinstance(text, str) or not text.strip():
                    continue
                text = text.strip()
                key = hash(text)
                if key in seen:
                    continue
                seen.add(key)
                contents = f'"{title}"\n{text}'
                docs.append({"id": str(doc_id), "contents": contents})
                doc_id += 1

with open(OUTPUT_PATH, "w") as f:
    for doc in docs:
        f.write(json.dumps(doc) + "\n")

print(f"Written {len(docs)} unique documents to {OUTPUT_PATH}")
