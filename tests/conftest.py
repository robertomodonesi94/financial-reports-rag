import pytest

from finrag.chunking import chunk_pages
from finrag.config import Config
from finrag.index import HashingEmbedder, VectorIndex
from finrag.llm import FakeLLM
from finrag.pipeline import Pipeline
from finrag.retrieve import Retriever

# Three tiny made-up reports, one string per PDF page.
CORPUS = {
    "ACME_2022_10K": ("ACME 2022 10K", [
        "ACME Corporation annual report for fiscal year 2022. Business overview: ACME designs and sells warehouse robots "
        "in Europe and North America through direct sales and distributors.",
        "Consolidated statement of cash flows. Purchases of property, plant and equipment (capital expenditure) were "
        "$1,577 million in 2022 and $1,373 million in 2021. Net cash provided by operating activities was $6,439 million.",
        "Risk factors. Supply chain disruptions and higher interest rates could reduce demand for our products and our margins.",
    ]),
    "BETA_2023_10K": ("BETA 2023 10K", [
        "Beta Retail annual report for fiscal year 2023. Beta operates about one thousand stores selling consumer electronics.",
        "Consolidated income statement. Revenue was $43,452 million and gross margin was 22.4% in fiscal 2023. "
        "Operating income was $1,795 million, down from $2,021 million.",
    ]),
    "GAMMA_2023Q2_10Q": ("GAMMA 2023 10Q", [
        "Gamma Pharma quarterly report for the second quarter of 2023. Research and development expenses were $2,100 million, "
        "mainly for oncology trials. The board declared a dividend of $0.75 per share.",
    ]),
}


@pytest.fixture(scope="session")
def docs():
    return {d: {"doc_id": d, "title": t, "company": d.split("_")[0], "doc_type": d.split("_")[-1].lower(),
                "period": int(d.split("_")[1][:4]), "sector": "Test", "pages": len(p)} for d, (t, p) in CORPUS.items()}


@pytest.fixture(scope="session")
def chunks():
    out = []
    for d, (title, pages) in CORPUS.items():
        out.extend(chunk_pages(d, title, pages, size=30, overlap=5))
    return out


@pytest.fixture(scope="session")
def embedder():
    return HashingEmbedder()


@pytest.fixture(scope="session")
def index(chunks, embedder):
    return VectorIndex.build(chunks, embedder)


@pytest.fixture
def retriever(index, embedder):
    return Retriever(index, embedder, hybrid=True)


@pytest.fixture
def pipeline(index, embedder, docs):
    cfg = Config(embedder="hashing", llm_backend="fake", top_k=3)
    return Pipeline(cfg, index, embedder, FakeLLM(responses=[]), docs)
