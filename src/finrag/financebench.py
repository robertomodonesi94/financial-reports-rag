"""FinanceBench open sample (Islam et al., 2023, arXiv:2311.11944): questions, document list and PDFs.

Source: https://github.com/patronus-ai/financebench. Nothing is redistributed by this repository:
files are downloaded at run time into data/ (git-ignored). Check the dataset licence before any non-personal use.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import requests

RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
FILES = {"questions": "data/financebench_open_source.jsonl", "documents": "data/financebench_document_information.jsonl"}
HEADERS = {"User-Agent": "finrag/0.1 (portfolio research project)"}


def _get(url: str, timeout: int = 120) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.content


def download_benchmark(dest: str | Path) -> dict[str, Path]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, rel in FILES.items():
        path = dest / Path(rel).name
        if not path.exists():
            path.write_bytes(_get(f"{RAW}/{rel}"))
        out[key] = path
    return out


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def select_docs(questions: list[dict], max_docs: int = 0) -> list[str]:
    """Documents referenced by the questions, the most-asked first. max_docs=0 keeps all of them."""
    counts = Counter(q["doc_name"] for q in questions)
    for q in questions:  # evidence can sit in a second document
        for e in q.get("evidence") or []:
            name = e.get("evidence_doc_name") or e.get("doc_name")
            if name and name not in counts:
                counts[name] = 0
    ranked = [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ranked[:max_docs] if max_docs else ranked


def download_pdf(doc_name: str, dest: str | Path, pause: float = 0.5) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{doc_name}.pdf"
    if path.exists() and path.stat().st_size > 0:
        return path
    data = _get(f"{RAW}/pdfs/{doc_name}.pdf", timeout=300)
    if not data.startswith(b"%PDF"):
        raise ValueError(f"{doc_name}: the download is not a PDF")
    path.write_bytes(data)
    time.sleep(pause)
    return path


def pdf_pages(path: str | Path) -> list[str]:
    """Plain text of every page (index = zero-based page number)."""
    try:
        import pymupdf
    except ImportError:  # older PyMuPDF releases
        import fitz as pymupdf
    with pymupdf.open(str(path)) as doc:
        return [page.get_text() for page in doc]


def doc_title(meta: dict) -> str:
    parts = [meta.get("company"), str(meta.get("doc_period") or ""), (meta.get("doc_type") or "").upper()]
    return " ".join(p for p in parts if p) or meta.get("doc_name", "")


def question_type(q: dict) -> str:
    reasoning = (q.get("question_reasoning") or "").lower()
    if "numerical" in reasoning:
        return "numeric"
    if reasoning.startswith("information extraction"):
        return "factual"
    return "reasoning"


def to_gold(questions: list[dict], keep_docs: set[str] | None = None) -> list[dict]:
    """Convert FinanceBench rows to this project's gold format. Questions whose documents are not indexed are dropped."""
    rows = []
    for q in questions:
        pages, docs = [], []
        for e in q.get("evidence") or []:
            name = e.get("evidence_doc_name") or e.get("doc_name") or q["doc_name"]
            if name not in docs:
                docs.append(name)
            try:
                pages.append([name, int(e.get("evidence_page_num"))])
            except (TypeError, ValueError):
                pass
        docs = docs or [q["doc_name"]]
        if keep_docs is not None and not set(docs) <= keep_docs:
            continue
        rows.append({
            "id": q["financebench_id"], "question": q["question"], "answer": q["answer"], "doc_ids": docs, "pages": pages,
            "type": question_type(q), "source": "financebench", "question_type": q.get("question_type"),
            "company": q.get("company"), "reviewed": True,
        })
    return rows
