"""Freeze the capability-probe data (run once; outputs are committed).

    uv run --with datasets python scripts/freeze_capability_data.py

* data/capability/fineweb_edu_val.jsonl: first 256 FineWeb-Edu (sample-10BT)
  documents with >= 600 tokens, in stream order, truncated to 3000 chars
  (the probe scores the first 512 tokens). Held-out from nothing in
  particular; it is a fixed proxy for general LM quality, not a real val split.
* data/capability/mmlu_lite.jsonl: 500 MMLU test questions sampled with a fixed
  seed across all subjects, plus 5 fixed few-shot exemplars from the dev split.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data" / "capability"
N_DOCS, MIN_TOKENS, MAX_CHARS = 256, 600, 3000
N_QUESTIONS, N_SHOTS, SEED = 500, 5, 20240601


def freeze_fineweb() -> None:
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    docs = []
    for row in stream:
        if row["token_count"] >= MIN_TOKENS:
            docs.append({"id": row["id"], "url": row["url"], "text": row["text"][:MAX_CHARS]})
            if len(docs) == N_DOCS:
                break
    with (OUT / "fineweb_edu_val.jsonl").open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")


def freeze_mmlu() -> None:
    test = list(load_dataset("cais/mmlu", "all", split="test"))
    dev = list(load_dataset("cais/mmlu", "all", split="dev"))
    rng = random.Random(SEED)
    rng.shuffle(test)
    rng.shuffle(dev)
    rows = [{"kind": "shot", **_strip(q)} for q in dev[:N_SHOTS]]
    rows += [{"kind": "test", **_strip(q)} for q in test[:N_QUESTIONS]]
    with (OUT / "mmlu_lite.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _strip(q: dict) -> dict:
    return {"subject": q["subject"], "question": q["question"], "choices": q["choices"], "answer": int(q["answer"])}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    freeze_fineweb()
    freeze_mmlu()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
