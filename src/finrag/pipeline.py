"""Glue code: data download, ingestion (PDF pages -> chunks -> index) and construction of the question-answering objects."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from . import financebench as fb
from .agent import Agent
from .chunking import Chunk, chunk_pages
from .config import Config
from .index import Embedder, VectorIndex, get_embedder
from .llm import LLM, get_llm
from .rag import RAGPipeline
from .retrieve import Retriever
from .tools import ToolBox


def fetch(cfg: Config, log: Callable[[str], None] = print) -> list[str]:
    """Download the FinanceBench questions, the document list and the PDFs the questions need."""
    paths = fb.download_benchmark(cfg.benchmark_dir)
    questions = fb.read_jsonl(paths["questions"])
    info = {d["doc_name"]: d for d in fb.read_jsonl(paths["documents"])}
    names = fb.select_docs(questions, cfg.max_docs)
    log(f"{len(questions)} questions; downloading {len(names)} reports into {cfg.pdf_dir}")
    ok, meta = [], {}
    for name in names:
        try:
            fb.download_pdf(name, cfg.pdf_dir)
        except Exception as e:  # keep going: the report list is long
            log(f"  {name}: download failed ({type(e).__name__}: {e})")
            continue
        d = info.get(name, {"doc_name": name})
        meta[name] = {"doc_id": name, "title": fb.doc_title(d), "company": d.get("company"), "doc_type": d.get("doc_type"),
                      "period": d.get("doc_period"), "sector": d.get("gics_sector") or d.get("comany_sector_gics"),
                      "source_url": d.get("doc_link")}
        ok.append(name)
    cfg.meta_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{len(ok)}/{len(names)} reports ready")
    return ok


def ingest(cfg: Config, log: Callable[[str], None] = print,
           read_pages: Callable[[Path], list[str]] = fb.pdf_pages) -> VectorIndex:
    """Index every PDF in data/pdf/. Metadata comes from data/documents.json when present (your own PDFs work too)."""
    pdfs = sorted(cfg.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF in {cfg.pdf_dir}: run `python -m finrag fetch` or copy your PDFs there")
    meta = json.loads(cfg.meta_path.read_text(encoding="utf-8")) if cfg.meta_path.exists() else {}
    chunks: list[Chunk] = []
    for path in pdfs:
        doc_id = path.stem
        m = meta.setdefault(doc_id, {"doc_id": doc_id, "title": doc_id})
        try:
            pages = read_pages(path)
        except Exception as e:  # damaged PDF: skip it, say so
            log(f"  {doc_id}: unreadable ({type(e).__name__}); skipped")
            continue
        dc = chunk_pages(doc_id, m.get("title") or doc_id, pages, cfg.chunk_words, cfg.overlap_words)
        m["pages"] = len(pages)
        empty = sum(1 for p in pages if not p.strip())
        log(f"  {doc_id}: {len(pages):3d} pages, {len(dc):4d} chunks" + (f", {empty} pages without text (scanned?)" if empty else ""))
        chunks.extend(dc)
    embedder = get_embedder(cfg.embedder)
    index = VectorIndex.build(chunks, embedder)
    index.save(cfg.index_dir)
    cfg.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"Index saved: {len(chunks)} chunks from {len(pdfs)} documents -> {cfg.index_dir}")
    return index


@dataclass
class Pipeline:
    cfg: Config
    index: VectorIndex
    embedder: Embedder
    llm: LLM
    docs: dict[str, dict]

    def retriever(self, **overrides) -> Retriever:
        cfg = replace(self.cfg, **overrides) if overrides else self.cfg
        return Retriever(self.index, self.embedder, hybrid=cfg.hybrid)

    def rag(self, **overrides) -> RAGPipeline:
        cfg = replace(self.cfg, **overrides) if overrides else self.cfg
        return RAGPipeline(self.retriever(**overrides), self.llm, cfg.top_k, cfg.max_new_tokens, cfg.temperature)

    def agent(self, **overrides) -> Agent:
        cfg = replace(self.cfg, **overrides) if overrides else self.cfg
        return Agent(self.llm, ToolBox(self.retriever(**overrides), self.docs), cfg.agent_max_steps, cfg.max_new_tokens, cfg.temperature)


def load_pipeline(cfg: Config, llm: LLM | None = None) -> Pipeline:
    index = VectorIndex.load(cfg.index_dir)
    embedder = get_embedder(index.embedder_name)
    docs = json.loads(cfg.meta_path.read_text(encoding="utf-8")) if cfg.meta_path.exists() else {}
    return Pipeline(cfg, index, embedder, llm or get_llm(cfg), docs)
