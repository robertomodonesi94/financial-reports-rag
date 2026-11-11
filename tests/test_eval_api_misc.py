import json

import pytest
from fastapi.testclient import TestClient

from finrag import llm as llm_module
from finrag.api import create_app
from finrag.config import Config
from finrag.eval.dataset import load_dataset, sample, save_jsonl, validate
from finrag.eval.generate import propose_questions
from finrag.eval.metrics import citation_precision, doc_hit, normalize, number_match, numbers, page_hit, summarize, token_f1
from finrag.eval.run import compare, evaluate, to_markdown
from finrag.llm import ABSTAIN_TEXT, FakeLLM, OpenAICompatLLM, fold_system

GOLD = [
    {"id": "q1", "question": "What was ACME's capital expenditure in FY2022?", "answer": "$1577.00", "doc_ids": ["ACME_2022_10K"],
     "pages": [["ACME_2022_10K", 1]], "type": "factual"},
    {"id": "q2", "question": "What was Beta's gross margin in fiscal 2023?", "answer": "22.4%", "doc_ids": ["BETA_2023_10K"],
     "pages": [["BETA_2023_10K", 1]], "type": "numeric"},
    {"id": "q3", "question": "Who won the 2018 football World Cup?", "answer": "", "doc_ids": [], "type": "unanswerable"},
]


def test_numbers_and_number_match():
    assert numbers("Capex was $1,577.00 million in FY2022, (12.5) and 8.7%") == [1577.0, -12.5, 8.7]
    assert numbers("In 2021 and 2022") == [] and numbers("$2,022 million") == [2022.0]
    assert number_match("It was USD 1,577 million [1].", "$1577.00") is True
    assert number_match("about 1.58 billion", "$1577.00") is True         # rounding within 1%, million vs billion
    assert number_match("about 1.7 billion", "$1577.00") is False         # outside the 1% tolerance
    assert number_match("a ratio of 0.224", "22.4%") is True              # ratio vs percent
    assert number_match("22%", "22.4%") is False
    assert number_match("anything", "Yes, it improved.") is None


def test_metrics():
    assert normalize("The ACME  report!") == "acme report"
    assert token_f1("the gross margin", "margin") == pytest.approx(2 / 3)
    assert token_f1("", "") == 1.0 and token_f1("x", "y") == 0.0
    assert doc_hit(["a", "b"], ["b"]) and not doc_hit(["a"], ["b"])
    assert page_hit([("a", 3), ("b", 1)], [["b", 1]]) is True and page_hit([("b", 2)], [["b", 1]]) is False
    assert page_hit([("b", 2)], []) is None
    assert citation_precision(["a", "b"], ["a"]) == 0.5 and citation_precision([], ["a"]) is None


def test_summarize_separates_answerable_and_unanswerable():
    rows = [
        {"type": "factual", "abstained": False, "f1": 1.0, "number_match": True, "doc_hit": 1.0, "page_hit": 1.0,
         "citation_precision": 1.0, "invalid_citations": []},
        {"type": "reasoning", "abstained": True, "f1": 0.0, "number_match": None, "doc_hit": 0.0, "page_hit": None,
         "citation_precision": None, "invalid_citations": [4]},
        {"type": "unanswerable", "abstained": True, "f1": 0.0, "number_match": None},
    ]
    s = summarize(rows)
    assert s["n_answerable"] == 2 and s["n_unanswerable"] == 1
    assert s["number_match"] == 1.0 and s["page_hit"] == 1.0 and s["doc_hit"] == 0.5
    assert s["false_abstention"] == 0.5 and s["abstention_accuracy"] == 1.0
    assert s["citation_precision"] == 1.0 and s["invalid_citation_rate"] == 0.5


def test_dataset_validation_and_roundtrip(tmp_path):
    with pytest.raises(ValueError):
        validate({"id": "x", "question": "?", "type": "factual"})  # answerable without answer
    with pytest.raises(ValueError):
        validate({"id": "x", "question": "?", "type": "unanswerable", "answer": "42"})
    with pytest.raises(ValueError):
        validate({"id": "x", "question": "?", "type": "weird"})
    path = tmp_path / "gold.jsonl"
    save_jsonl(GOLD + [{"id": "q4", "question": "draft?", "type": "factual", "reviewed": False}], path)
    rows = load_dataset(path)
    assert [r["id"] for r in rows] == ["q1", "q2", "q3"]  # drafts are skipped
    assert [r["id"] for r in sample(rows, 1)] in (["q1", "q3"], ["q2", "q3"])  # unanswerable always kept
    assert sample(rows, 0) == rows


def answer_from_context(messages):
    text = messages[-1]["content"]
    question = text.rsplit("Question:", 1)[-1]  # decide from the question, not from the retrieved context
    if "football" in question.lower():
        return ABSTAIN_TEXT
    if "capital expenditure" in question and "1,577" in text:
        return "ACME's capital expenditure was $1,577 million in FY2022 [1]."
    return "Beta's gross margin was 22.4% in fiscal 2023 [1]."


def test_evaluate_and_compare_end_to_end(pipeline):
    pipeline.llm = FakeLLM(fn=answer_from_context)
    res = evaluate(GOLD, pipeline, "rag")
    s = res["summary"]
    assert s["n_answerable"] == 2 and s["n_unanswerable"] == 1
    assert s["abstention_accuracy"] == 1.0 and s["number_match"] == 1.0 and s["doc_hit"] == 1.0 and s["page_hit"] >= 0.5
    variants = {"dense": {"hybrid": False}, "hybrid": {"hybrid": True}, "per_doc": {"hybrid": True, "scope": "per_doc"}}
    results = compare(GOLD, pipeline, variants)
    assert results["per_doc"]["summary"]["doc_hit"] == 1.0
    assert all(r["sources"] for r in results["per_doc"]["rows"][:2])
    table = to_markdown(results)
    assert "| dense |" in table and "| per_doc |" in table and "2 answerable and 1 unanswerable" in table


def test_evaluate_agent_mode_counts_invalid_steps(pipeline):
    pipeline.llm = FakeLLM(fn=lambda m: "not json" if len(m) < 4 else '{"final": "$1,577 million", "sources": ["ACME_2022_10K"]}')
    res = evaluate(GOLD[:1], pipeline, "agent", scope="per_doc")
    row = res["rows"][0]
    assert row["invalid_steps"] == 1 and row["number_match"] is True and row["citation_precision"] == 1.0


def test_propose_questions_marks_drafts_unreviewed(chunks):
    llm = FakeLLM(fn=lambda m: json.dumps({"question": "What was ACME's FY2022 capex?", "answer": "$1,577 million"}))
    rows = propose_questions(chunks, llm, n=2, min_words=20)
    assert len(rows) == 2 and all(r["reviewed"] is False and r["doc_ids"] and r["pages"] for r in rows)
    bad = FakeLLM(fn=lambda m: "not json")
    assert propose_questions(chunks, bad, n=2, min_words=20) == []


def test_api(pipeline):
    pipeline.llm = FakeLLM(fn=answer_from_context)
    client = TestClient(create_app(pipeline))
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["documents"] == 3
    r = client.post("/ask", json={"question": "What was ACME's capital expenditure in FY2022?"})
    assert r.status_code == 200 and r.json()["mode"] == "rag" and "1,577" in r.json()["text"] and r.json()["sources"]
    r = client.post("/ask", json={"question": "What was the gross margin?", "doc_ids": ["BETA_2023_10K"]})
    assert {s["doc_id"] for s in r.json()["sources"]} == {"BETA_2023_10K"}
    assert client.post("/ask", json={"question": "ab"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "mode": "other"}).status_code == 422
    pipeline.llm = FakeLLM(fn=lambda m: (_ for _ in ()).throw(RuntimeError("backend down")))
    assert client.post("/ask", json={"question": "valid question"}).status_code == 502
    pipeline.llm = FakeLLM(responses=['{"final": "no idea", "sources": []}'])
    a = client.post("/ask", json={"question": "valid question", "mode": "agent"}).json()
    assert a["mode"] == "agent" and a["finished"] is True


def test_openai_compat_llm_builds_request(monkeypatch):
    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": " hello "}}]}

    def fake_post(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return Resp()

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    out = OpenAICompatLLM("http://srv:8000/v1/", "m", api_key="k").chat([{"role": "user", "content": "hi"}], max_new_tokens=5)
    assert out == "hello" and seen["url"] == "http://srv:8000/v1/chat/completions"
    assert seen["json"]["max_tokens"] == 5 and seen["headers"]["Authorization"] == "Bearer k"


def test_fold_system_and_config(tmp_path):
    msgs = [{"role": "system", "content": "RULES"}, {"role": "user", "content": "question"}]
    folded = fold_system(msgs)
    assert len(folded) == 1 and folded[0]["content"].startswith("RULES") and folded[0]["content"].endswith("question")
    y = tmp_path / "c.yaml"
    y.write_text("top_k: 7\nhybrid: false\n", encoding="utf-8")
    cfg = Config.load(y, top_k=9, llm_model=None)
    assert cfg.top_k == 9 and cfg.hybrid is False and cfg.llm_model.startswith("mistralai")
    y.write_text("nope: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Config.load(y)
