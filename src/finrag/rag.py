"""Retrieval-augmented answering with numbered citations (document + page) and explicit abstention."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace

from .llm import ABSTAIN_TEXT, LLM
from .query import expand_glossary, guess_docs, rewrite_with_llm
from .retrieve import Hit, Retriever

SYSTEM_PROMPT = (
    "You answer questions about company financial reports (10-K, 10-Q, 8-K, earnings releases) using ONLY the numbered "
    "sources provided. Cite every claim with its source number in square brackets, like [1] or [2][3]. "
    "Report figures with their unit and period (e.g. 'USD 1,577 million in FY2018'); if the question asks for a unit "
    "(e.g. USD billions), convert to that unit. Show any calculation you make. "
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


def fit_context(hits: list[Hit], max_chars: int, min_chars: int = 300) -> list[Hit]:
    """Keep the best-ranked chunks within a character budget; the last one is cut, the rest dropped.
    Long prompts run out of GPU memory on a T4 (tables are expensive: every digit is a token)."""
    if max_chars <= 0:
        return hits
    out, used = [], 0
    for h in hits:
        room = max_chars - used
        if room < min_chars:
            break
        if len(h.chunk.text) > room:
            h = Hit(replace(h.chunk, text=h.chunk.text[:room]), h.score, h.dense_rank, h.bm25_rank)
        out.append(h)
        used += len(h.chunk.text)
    return out


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    doc_filter: list[str] | None = None
    filter_source: str = "none"      # none | given | guessed

    def cited_doc_ids(self) -> list[str]:
        ids: list[str] = []
        for n in self.citations:
            if 1 <= n <= len(self.sources) and self.sources[n - 1]["doc_id"] not in ids:
                ids.append(self.sources[n - 1]["doc_id"])
        return ids

    def to_dict(self) -> dict:
        return asdict(self)


REWRITE_MODES = {"none", "glossary", "llm"}


class RAGPipeline:
    """query_rewrite: none | glossary | llm (extra search phrasings, fused with the original question).
    auto_filter: when no report is given, search only the reports of the company and years named in the question."""

    def __init__(self, retriever: Retriever, llm: LLM, top_k: int = 5, max_new_tokens: int = 400, temperature: float = 0.0,
                 query_rewrite: str = "none", auto_filter: bool = False, docs: dict[str, dict] | None = None,
                 max_context_chars: int = 10000):
        if query_rewrite not in REWRITE_MODES:
            raise ValueError(f"query_rewrite must be one of {sorted(REWRITE_MODES)}, got '{query_rewrite}'")
        self.retriever, self.llm = retriever, llm
        self.top_k, self.max_new_tokens, self.temperature = top_k, max_new_tokens, temperature
        self.query_rewrite, self.auto_filter, self.docs = query_rewrite, auto_filter, docs or {}
        self.max_context_chars = max_context_chars

    def retrieve(self, question: str, doc_ids: list[str] | None = None) -> tuple[list[Hit], dict]:
        """The chunks the answer will be based on, plus how they were searched (queries, report filter)."""
        source = "given" if doc_ids else "none"
        if not doc_ids and self.auto_filter:
            doc_ids = guess_docs(question, self.docs)
            source = "guessed" if doc_ids else "none"
        queries = [question]
        if self.query_rewrite == "glossary":
            extra = expand_glossary(question)
            if extra:
                queries.append(question + " " + extra)
        elif self.query_rewrite == "llm":
            queries.append(rewrite_with_llm(question, self.llm))
        hits = self.retriever.search_multi(queries, self.top_k, doc_ids=doc_ids)
        if not hits and source == "guessed":  # a wrong guess must not hide everything
            hits, doc_ids, source = self.retriever.search_multi(queries, self.top_k), None, "none"
        return hits, {"search_queries": [q for q in queries if q], "doc_filter": doc_ids, "filter_source": source}

    def ask(self, question: str, doc_ids: list[str] | None = None) -> Answer:
        """doc_ids limits the search to the given reports (the 'single document' setting of FinanceBench)."""
        hits, info = self.retrieve(question, doc_ids)
        hits = fit_context(hits, self.max_context_chars)
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
            **info,
        )
