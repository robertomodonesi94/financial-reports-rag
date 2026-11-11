import json

import pytest

from finrag.agent import Agent, extract_json
from finrag.llm import ABSTAIN_TEXT, FakeLLM
from finrag.rag import RAGPipeline, extract_citations, is_abstention
from finrag.tools import ToolBox, ToolError, safe_eval


def test_extract_citations_and_abstention():
    assert extract_citations("Capex was $1,577 million [1][3] and also [1] [12]") == [1, 3, 12]
    assert is_abstention(ABSTAIN_TEXT) and is_abstention("I cannot answer that.")
    assert not is_abstention("Revenue was $43,452 million [1].")


def test_rag_answer_with_valid_and_invalid_citations(retriever):
    llm = FakeLLM(responses=["Capex was $1,577 million in FY2022 [1], up from $1,373 million [2][9]."])
    ans = RAGPipeline(retriever, llm, top_k=3).ask("What was ACME's capital expenditure in 2022?")
    assert ans.citations == [1, 2] and ans.invalid_citations == [9]
    assert not ans.abstained and len(ans.sources) == 3
    assert ans.cited_doc_ids() and all("page" in s for s in ans.sources)
    prompt = llm.calls[0][1]["content"]
    assert "Sources:" in prompt and "page 2 (ACME_2022_10K)" in prompt and "Question: What was ACME" in prompt


def test_rag_restricted_to_one_report(retriever):
    ans = RAGPipeline(retriever, FakeLLM(responses=["x [1]"]), top_k=3).ask("operating income", doc_ids=["BETA_2023_10K"])
    assert {s["doc_id"] for s in ans.sources} == {"BETA_2023_10K"}


def test_rag_abstains(retriever):
    ans = RAGPipeline(retriever, FakeLLM(responses=[ABSTAIN_TEXT])).ask("Who won the 2018 World Cup?")
    assert ans.abstained and ans.citations == []


def test_extract_json_handles_noise_and_nesting():
    assert extract_json('Sure! {"tool": "calculate", "args": {"expression": "1+1"}} thanks') == {"tool": "calculate", "args": {"expression": "1+1"}}
    assert extract_json('{"final": "a {brace} inside"}') == {"final": "a {brace} inside"}
    assert extract_json("no json here") is None
    assert extract_json('{"broken": ') is None


def test_safe_eval_allows_arithmetic_only():
    assert safe_eval("2 * (3 + 4) / 7") == 2
    assert safe_eval("(1577 - 1373) / 1373 * 100") == pytest.approx(14.858, rel=1e-3)
    for bad in ("__import__('os').system('ls')", "abs(-1)", "x + 1", "2 ** 99", "1 / 0", "1 +", "True + 1"):
        with pytest.raises(ToolError):
            safe_eval(bad)


def make_agent(retriever, docs, responses, max_steps=5):
    return Agent(FakeLLM(responses=responses), ToolBox(retriever, docs), max_steps=max_steps)


def test_agent_searches_calculates_then_answers(retriever, docs):
    agent = make_agent(retriever, docs, [
        json.dumps({"tool": "search_corpus", "args": {"query": "capital expenditure", "doc_id": "ACME_2022_10K"}}),
        json.dumps({"tool": "calculate", "args": {"expression": "(1577 - 1373) / 1373 * 100"}}),
        json.dumps({"final": "Capex grew by about 14.9% in FY2022 (ACME_2022_10K).", "sources": ["ACME_2022_10K"]}),
    ])
    res = agent.run("By how much did ACME's capex grow in 2022?")
    assert res.finished and "14.9" in res.answer
    assert res.cited_ids == ["ACME_2022_10K"] and res.unsupported_ids == []
    assert [s["type"] for s in res.steps] == ["tool", "tool", "final"]
    assert "14.85" in res.steps[1]["result_preview"]


def test_agent_flags_citations_it_never_saw(retriever, docs):
    agent = make_agent(retriever, docs, [json.dumps({"final": "It is 42 (see BETA_2023_10K).", "sources": ["NOPE_2020_10K"]})])
    res = agent.run("anything")
    assert res.finished and set(res.unsupported_ids) == {"NOPE_2020_10K", "BETA_2023_10K"}


def test_agent_recovers_from_invalid_output_and_tool_errors(retriever, docs):
    agent = make_agent(retriever, docs, [
        "I think the answer is blue",
        json.dumps({"tool": "get_document_info", "args": {"doc_id": "NOPE_2020_10K"}}),
        json.dumps({"tool": "calculate", "args": {"expression": "1795 / 43452 * 100"}}),
        json.dumps({"final": "Operating margin is about 4.1%", "sources": []}),
    ])
    res = agent.run("Beta operating margin?")
    assert res.finished
    assert [s["type"] for s in res.steps] == ["invalid", "tool", "tool", "final"]
    assert "unknown doc_id" in res.steps[1]["result_preview"]
    assert "4.13" in res.steps[2]["result_preview"]


def test_agent_gives_up_after_max_steps(retriever, docs):
    agent = make_agent(retriever, docs, ['{"tool": "list_documents", "args": {}}'] * 3, max_steps=3)
    res = agent.run("loop forever")
    assert not res.finished and res.answer == ABSTAIN_TEXT


def test_toolbox_validation(retriever, docs):
    tb = ToolBox(retriever, docs)
    with pytest.raises(ToolError):
        tb.run("search_corpus", {"query": "  "})
    with pytest.raises(ToolError):
        tb.run("search_corpus", {"query": "capex", "doc_id": "NOPE_2020_10K"})
    with pytest.raises(ToolError):
        tb.run("nope", {})
    assert len(tb.run("search_corpus", {"query": "revenue", "k": 100})) <= 8
    hits = tb.run("search_corpus", {"query": "revenue", "doc_id": "BETA_2023_10K"})
    assert {h["doc_id"] for h in hits} == {"BETA_2023_10K"} and all(h["page"] >= 1 for h in hits)
    assert tb.run("get_document_info", {"doc_id": "GAMMA_2023Q2_10Q"})["period"] == 2023
    assert len(tb.run("list_documents", {})) == 3
    assert [d["doc_id"] for d in tb.run("list_documents", {"company": "beta"})] == ["BETA_2023_10K"]
