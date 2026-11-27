from finrag.eval.run import compare, to_markdown
from finrag.llm import FakeLLM
from finrag.query import expand_glossary, guess_docs, rewrite_with_llm, years_in
from finrag.rag import RAGPipeline

DOCS = {
    "3M_2018_10K": {"company": "3M", "period": 2018},
    "3M_2022_10K": {"company": "3M", "period": 2022},
    "AMERICANEXPRESS_2022_10K": {"company": "American Express", "period": 2022},
    "JPMORGAN_2021Q1_10Q": {"company": "JPMorgan", "period": 2021},
    "JPMORGAN_2022_10K": {"company": "JPMorgan", "period": 2022},
    "COCACOLA_2021_10K": {"company": "Coca-Cola", "period": 2021},
    "BESTBUY_2024Q2_10Q": {"company": "Best Buy", "period": 2024},
    "MINE_2024_REPORT": {"title": "my own PDF"},
}


def test_expand_glossary_uses_filing_vocabulary():
    assert "purchases of property, plant and equipment" in expand_glossary("What is the FY2018 capital expenditure for 3M?")
    assert "cost of sales" in expand_glossary("What is Coca Cola's FY2021 COGS % margin?")
    assert expand_glossary("Who is the CEO?") == ""


def test_years_in_question():
    assert years_in("FY2018 capex") == {2018}
    assert years_in("From FY21 to FY22") == {2021, 2022}
    assert years_in("between FY 2023 and Q2 of FY2024") == {2023, 2024}
    assert years_in("as of Q2'2023") == {2023}
    assert years_in("no year here, 150 stores") == set()


def test_guess_docs_company_years_and_aliases():
    assert guess_docs("What is the FY2018 capital expenditure amount for 3M?", DOCS) == ["3M_2018_10K"]
    assert guess_docs("What is 3M's revenue?", DOCS) == ["3M_2018_10K", "3M_2022_10K"]            # no year: all 3M reports
    assert guess_docs("Does AMEX have an improving operating margin as of 2022?", DOCS) == ["AMERICANEXPRESS_2022_10K"]
    assert guess_docs("If JPM went bankrupt by the end of 2021 Q1...", DOCS) == ["JPMORGAN_2021Q1_10Q"]
    assert guess_docs("What is Coca Cola's FY2021 COGS % margin?", DOCS) == ["COCACOLA_2021_10K"]
    assert guess_docs("What is 3M's FY2030 revenue?", DOCS) == ["3M_2018_10K", "3M_2022_10K"]     # unknown year: keep the company
    assert guess_docs("What is the average gross margin of retailers?", DOCS) is None


def test_rewrite_with_llm_cleans_the_reply_and_survives_errors():
    llm = FakeLLM(responses=['Query: "3M purchases of property, plant and equipment 2018"\nextra text'])
    assert rewrite_with_llm("capex 3M 2018?", llm) == "3M purchases of property, plant and equipment 2018"
    broken = FakeLLM(fn=lambda m: (_ for _ in ()).throw(RuntimeError("down")))
    assert rewrite_with_llm("anything", broken) == ""


def test_rag_auto_filter_and_glossary(retriever, docs):
    rag = RAGPipeline(retriever, FakeLLM(fn=lambda m: "x [1]"), top_k=3, query_rewrite="glossary", auto_filter=True, docs=docs)
    hits, info = rag.retrieve("What was ACME's capital expenditure in FY2022?")
    assert info["filter_source"] == "guessed" and info["doc_filter"] == ["ACME_2022_10K"]
    assert {h.chunk.doc_id for h in hits} == {"ACME_2022_10K"} and (hits[0].chunk.page == 1)
    assert len(info["search_queries"]) == 2 and "purchases of property" in info["search_queries"][1]
    ans = rag.ask("What was ACME's capital expenditure in FY2022?")
    assert ans.doc_filter == ["ACME_2022_10K"] and ans.search_queries
    # no company in the question: no filter, search everything
    _, info = rag.retrieve("Which company paid a dividend?")
    assert info["filter_source"] == "none" and info["doc_filter"] is None


def test_rag_given_report_wins_over_the_guess(retriever, docs):
    rag = RAGPipeline(retriever, FakeLLM(fn=lambda m: "x"), auto_filter=True, docs=docs)
    _, info = rag.retrieve("ACME capital expenditure FY2022", doc_ids=["BETA_2023_10K"])
    assert info["filter_source"] == "given" and info["doc_filter"] == ["BETA_2023_10K"]


def test_rag_rejects_unknown_rewrite_mode(retriever):
    import pytest
    with pytest.raises(ValueError):
        RAGPipeline(retriever, FakeLLM(responses=[]), query_rewrite="magic")


def test_search_multi_fuses_and_respects_filter(retriever):
    hits = retriever.search_multi(["capex", "purchases of property, plant and equipment"], k=3)
    assert hits and len({h.chunk.id for h in hits}) == len(hits)
    assert ("ACME_2022_10K", 1) in {(h.chunk.doc_id, h.chunk.page) for h in hits}   # the cash-flow page is found
    only = retriever.search_multi(["revenue", "net sales"], k=5, doc_ids=["BETA_2023_10K"])
    assert {h.chunk.doc_id for h in only} == {"BETA_2023_10K"}
    assert retriever.search_multi(["", "  "], k=2) == retriever.search("", 2)


def test_retrieval_only_evaluation(pipeline):
    gold = [
        {"id": "q1", "question": "What was ACME's capital expenditure in FY2022?", "answer": "$1577.00",
         "doc_ids": ["ACME_2022_10K"], "pages": [["ACME_2022_10K", 1]], "type": "factual"},
        {"id": "q2", "question": "What was Beta's gross margin in fiscal 2023?", "answer": "22.4%",
         "doc_ids": ["BETA_2023_10K"], "pages": [["BETA_2023_10K", 1]], "type": "numeric"},
        {"id": "q3", "question": "Who won the World Cup?", "answer": "", "doc_ids": [], "pages": [], "type": "unanswerable"},
    ]
    pipeline.llm = FakeLLM(fn=lambda m: (_ for _ in ()).throw(AssertionError("retrieval mode must not generate")))
    variants = {"base": {}, "auto": {"auto_filter": True}, "all": {"auto_filter": True, "query_rewrite": "glossary", "top_k": 5}}
    res = compare(gold, pipeline, variants, mode="retrieval")
    assert res["base"]["summary"]["n_answerable"] == 2            # unanswerable questions are skipped
    assert res["auto"]["summary"]["filter_used"] == 1.0 and res["auto"]["summary"]["filter_correct"] == 1.0
    assert res["all"]["summary"]["page_hit"] == 1.0
    table = to_markdown(res)
    assert "filter_correct" in table and "| auto |" in table and "number_match" not in table


def test_glossary_covers_ratio_questions_from_the_error_analysis():
    assert "property, plant and equipment, net" in expand_glossary("What is the FY2018 fixed asset turnover ratio for CVS Health?")
    assert "accounts receivable" in expand_glossary("What is Amcor's year end FY2020 net AR?")
    assert "inventories" in expand_glossary("What is Kraft Heinz's FY2019 inventory turnover ratio?")
    assert "long-term debt" in expand_glossary("Has Microsoft increased its debt on balance sheet?")


def test_split_dev_test_is_disjoint_and_keeps_block3_questions_in_dev():
    from finrag.eval.dataset import sample, split_dev_test
    rows = [{"id": f"q{i}", "type": "factual"} for i in range(150)] + [{"id": "u1", "type": "unanswerable"}]
    dev, test = split_dev_test(rows, n_dev=40)
    assert len(dev) == 40 and len(test) == 111 and not {r["id"] for r in dev} & {r["id"] for r in test}
    assert {r["id"] for r in sample(rows, 10, seed=0) if r["type"] != "unanswerable"} <= {r["id"] for r in dev}
    assert any(r["type"] == "unanswerable" for r in test)
