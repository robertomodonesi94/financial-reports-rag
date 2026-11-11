import json

import pytest

from finrag import financebench as fb
from finrag import pipeline
from finrag.config import Config
from finrag.llm import ABSTAIN_TEXT, FakeLLM

QUESTIONS = [
    {"financebench_id": "fb1", "company": "ACME", "doc_name": "ACME_2022_10K", "question": "What is ACME's FY2022 capex?",
     "answer": "$1577.00", "question_type": "metrics-generated", "question_reasoning": "Information extraction",
     "evidence": [{"evidence_text": "...", "doc_name": "ACME_2022_10K", "evidence_page_num": "1"}]},
    {"financebench_id": "fb2", "company": "ACME", "doc_name": "ACME_2022_10K", "question": "Capex growth 2021-2022?",
     "answer": "14.9%", "question_type": "novel-generated", "question_reasoning": "Numerical reasoning",
     "evidence": [{"evidence_text": "...", "doc_name": "ACME_2022_10K", "evidence_page_num": 1}]},
    {"financebench_id": "fb3", "company": "BETA", "doc_name": "BETA_2023_10K", "question": "Is Beta's margin improving?",
     "answer": "No, it declined.", "question_type": "domain-relevant", "question_reasoning": None,
     "evidence": [{"evidence_text": "...", "doc_name": "BETA_2023_10K", "evidence_page_num": 1},
                  {"evidence_text": "...", "doc_name": "BETA_2022_10K", "evidence_page_num": 40}]},
]
DOCS = [{"doc_name": "ACME_2022_10K", "company": "ACME", "gics_sector": "Industrials", "doc_type": "10k", "doc_period": 2022,
         "doc_link": "https://example.com/acme.pdf"}]


def test_question_type_mapping():
    assert [fb.question_type(q) for q in QUESTIONS] == ["factual", "numeric", "reasoning"]


def test_select_docs_most_asked_first_and_includes_evidence_docs():
    assert fb.select_docs(QUESTIONS) == ["ACME_2022_10K", "BETA_2023_10K", "BETA_2022_10K"]
    assert fb.select_docs(QUESTIONS, max_docs=1) == ["ACME_2022_10K"]


def test_to_gold_keeps_pages_and_filters_missing_docs():
    rows = fb.to_gold(QUESTIONS)
    assert rows[0]["pages"] == [["ACME_2022_10K", 1]] and rows[0]["doc_ids"] == ["ACME_2022_10K"]
    assert rows[2]["doc_ids"] == ["BETA_2023_10K", "BETA_2022_10K"] and rows[0]["reviewed"] is True
    kept = fb.to_gold(QUESTIONS, keep_docs={"ACME_2022_10K", "BETA_2023_10K"})
    assert [r["id"] for r in kept] == ["fb1", "fb2"]  # fb3 needs a report that is not indexed


def test_doc_title():
    assert fb.doc_title(DOCS[0]) == "ACME 2022 10K"


def test_pdf_pages_reads_a_real_pdf(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in ("Cash flow statement capex 1,577", "Risk factors"):
        doc.new_page().insert_text((72, 72), text)
    path = tmp_path / "X_2022_10K.pdf"
    doc.save(str(path))
    pages = fb.pdf_pages(path)
    assert len(pages) == 2 and "1,577" in pages[0] and "Risk" in pages[1]


def test_fetch_downloads_benchmark_and_pdfs(tmp_path, monkeypatch):
    def fake_get(url, timeout=120):
        if url.endswith("financebench_open_source.jsonl"):
            return "\n".join(json.dumps(q) for q in QUESTIONS).encode()
        if url.endswith("financebench_document_information.jsonl"):
            return json.dumps(DOCS[0]).encode()
        if "BETA_2022" in url:
            raise ConnectionError("offline")
        return b"%PDF-1.4 fake"

    monkeypatch.setattr(fb, "_get", fake_get)
    monkeypatch.setattr(fb.time, "sleep", lambda s: None)
    cfg = Config(data_dir=str(tmp_path))
    logs: list[str] = []
    ok = pipeline.fetch(cfg, log=logs.append)
    assert ok == ["ACME_2022_10K", "BETA_2023_10K"]
    assert any("BETA_2022_10K: download failed" in line for line in logs)
    meta = json.loads(cfg.meta_path.read_text())
    assert meta["ACME_2022_10K"]["sector"] == "Industrials" and meta["ACME_2022_10K"]["title"] == "ACME 2022 10K"


def test_ingest_builds_index_and_skips_broken_pdfs(tmp_path):
    cfg = Config(data_dir=str(tmp_path), embedder="hashing", llm_backend="fake", chunk_words=30, overlap_words=5)
    cfg.pdf_dir.mkdir(parents=True)
    for name in ("ACME_2022_10K", "BROKEN_2022_10K", "MINE_2024_REPORT"):
        (cfg.pdf_dir / f"{name}.pdf").write_bytes(b"%PDF")
    cfg.meta_path.write_text(json.dumps({"ACME_2022_10K": {"doc_id": "ACME_2022_10K", "title": "ACME 2022 10K"}}))

    def fake_pages(path):
        if "BROKEN" in path.name:
            raise RuntimeError("bad xref")
        return ["capital expenditure was $1,577 million " * 5, ""]

    logs: list[str] = []
    index = pipeline.ingest(cfg, log=logs.append, read_pages=fake_pages)
    assert {c.doc_id for c in index.chunks} == {"ACME_2022_10K", "MINE_2024_REPORT"}  # your own PDFs work without metadata
    assert any("BROKEN_2022_10K: unreadable" in line for line in logs)
    assert any("1 pages without text" in line for line in logs)
    meta = json.loads(cfg.meta_path.read_text())
    assert meta["ACME_2022_10K"]["pages"] == 2 and meta["MINE_2024_REPORT"]["title"] == "MINE_2024_REPORT"

    pipe = pipeline.load_pipeline(cfg, llm=FakeLLM(responses=[ABSTAIN_TEXT]))
    assert len(pipe.docs) == 3 and pipe.rag().ask("What was ACME capex?").abstained


def test_ingest_requires_pdfs(tmp_path):
    with pytest.raises(ValueError):
        pipeline.ingest(Config(data_dir=str(tmp_path)))
