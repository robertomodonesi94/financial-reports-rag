"""Retrieval-augmented answering with numbered citations (document + page) and explicit abstention."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .llm import ABSTAIN_TEXT, LLM
from .retrieve import Hit, Retriever

SYSTEM_PROMPT = (
    "You answer questions about company financial reports (10-K, 10-Q, 8-K, earnings releases) using ONLY the numbered "
    "sources provided. Cite every claim with its source number in square brackets, like [1] or [2][3]. "
    "Report figures with their unit and period (e.g. 'USD 1,577 million in FY2018'). Show any calculation you make. "
    f"If the sources do not contain the answer, reply exactly: {ABSTAIN_TEXT} "
    "Be concise and never invent numbers."
)
_CITE = re.compile(r"\[(\d+)\]")
_ABSTAIN = re.compile(r"(don'?t|do not|cannot|can'?t) (know|answer)|not (stated|mentioned|provided|contained)|no information", re.I)


def extract_citations(text: str) -> list[int]:
    seen: list[int] = []
    for m in _CITE.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN.search(text))


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] {h.chunk.title}, page {h.chunk.page + 1} ({h.chunk.doc_id})\n{h.chunk.text}" for i, h in enumerate(hits, 1))


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)

    def cited_doc_ids(self) -> list[str]:
        ids: list[str] = []
        for n in self.citations:
            if 1 <= n <= len(self.sources) and self.sources[n - 1]["doc_id"] not in ids:
                ids.append(self.sources[n - 1]["doc_id"])
        return ids

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLM, top_k: int = 5, max_new_tokens: int = 400, temperature: float = 0.0):
        self.retriever, self.llm = retriever, llm
        self.top_k, self.max_new_tokens, self.temperature = top_k, max_new_tokens, temperature

    def ask(self, question: str, doc_ids: list[str] | None = None) -> Answer:
        """doc_ids limits the search to the given reports (the 'single document' setting of FinanceBench)."""
        hits = self.retriever.search(question, self.top_k, doc_ids=doc_ids)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{format_context(hits)}\n\nQuestion: {question}"},
        ]
        text = self.llm.chat(messages, max_new_tokens=self.max_new_tokens, temperature=self.temperature).strip()
        cites = extract_citations(text)
        valid = [n for n in cites if 1 <= n <= len(hits)]
        return Answer(
            question=question,
            text=text,
            abstained=is_abstention(text),
            citations=valid,
            invalid_citations=[n for n in cites if n not in valid],
            sources=[{"n": i, "chunk_id": h.chunk.id, "doc_id": h.chunk.doc_id, "page": h.chunk.page, "title": h.chunk.title,
                      "score": round(h.score, 4)} for i, h in enumerate(hits, 1)],
        )
