"""Propose candidate questions from random chunks. They are DRAFTS: a human must review and fix every one."""
from __future__ import annotations

import random

from ..agent import extract_json
from ..chunking import Chunk
from ..llm import LLM

PROMPT = (
    "Read the passage from a company financial report ({title}, page {page}). Write ONE specific question that can be "
    "answered only from this passage (a reported figure with its period, a segment, a risk, an accounting choice), naming "
    "the company and the fiscal year, and its short answer (at most 15 words). "
    'Reply with JSON only: {{"question": "...", "answer": "..."}}\n\nPassage:\n{passage}'
)


def propose_questions(chunks: list[Chunk], llm: LLM, n: int = 40, seed: int = 0, min_words: int = 120) -> list[dict]:
    rng = random.Random(seed)
    pool = [c for c in chunks if len(c.text.split()) >= min_words]
    rng.shuffle(pool)
    rows: list[dict] = []
    for c in pool:
        if len(rows) >= n:
            break
        data = extract_json(llm.chat([{"role": "user", "content": PROMPT.format(passage=c.text, title=c.title, page=c.page + 1)}], max_new_tokens=120))
        if not data or not data.get("question") or not data.get("answer"):
            continue
        rows.append({
            "id": f"q{len(rows) + 1:03d}", "question": str(data["question"]).strip(), "answer": str(data["answer"]).strip(),
            "doc_ids": [c.doc_id], "pages": [[c.doc_id, c.page]], "type": "factual", "source": "mine", "reviewed": False,
            "source_chunk": c.id,
        })
    return rows
