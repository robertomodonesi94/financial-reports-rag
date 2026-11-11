"""Gold dataset: one JSON object per line.

{"id": "q001", "question": "...", "answer": "gold answer", "doc_ids": ["3M_2018_10K"], "pages": [["3M_2018_10K", 59]],
 "type": "factual|numeric|reasoning|unanswerable", "source": "financebench|mine", "reviewed": true}

pages are zero-based PDF pages (optional). Unanswerable questions have an empty answer: the system must abstain.
They may still name doc_ids, to test "the answer is not in this report".
"""
from __future__ import annotations

import json
import random
from pathlib import Path

TYPES = {"factual", "numeric", "reasoning", "unanswerable"}


def validate(row: dict) -> None:
    for key in ("id", "question", "type"):
        if not row.get(key):
            raise ValueError(f"missing '{key}' in {row}")
    if row["type"] not in TYPES:
        raise ValueError(f"unknown type '{row['type']}' in {row['id']}")
    if row["type"] == "unanswerable":
        if row.get("answer"):
            raise ValueError(f"unanswerable question must have an empty answer: {row['id']}")
    elif not (row.get("answer") and row.get("doc_ids")):
        raise ValueError(f"answerable question needs 'answer' and 'doc_ids': {row['id']}")


def load_dataset(path: str | Path, only_reviewed: bool = True) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if only_reviewed:
        rows = [r for r in rows if r.get("reviewed", True)]
    for r in rows:
        r.setdefault("doc_ids", [])
        r.setdefault("pages", [])
        r.setdefault("answer", "")
        validate(r)
    return rows


def sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n answerable questions at random (fixed seed) plus every unanswerable one. n=0 keeps everything."""
    if not n:
        return rows
    una = [r for r in rows if r["type"] == "unanswerable"]
    ans = [r for r in rows if r["type"] != "unanswerable"]
    random.Random(seed).shuffle(ans)
    return ans[:n] + una


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
