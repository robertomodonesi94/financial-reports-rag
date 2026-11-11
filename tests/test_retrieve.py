from finrag.retrieve import BM25, Retriever, tokenize


def test_bm25_prefers_documents_with_the_rare_term():
    bm = BM25(["revenue and margin", "dividend per share", "revenue revenue revenue"])
    scores = bm.scores("dividend")
    assert scores[1] > scores[0] and scores[1] > scores[2]
    assert bm.scores("unseen") == [0.0, 0.0, 0.0]


def test_tokenize_lowercases():
    assert tokenize("Capex, $1,577.00") == ["capex", "1", "577", "00"]


def test_hybrid_returns_k_hits_with_rank_information(retriever):
    hits = retriever.search("what was the gross margin and revenue of Beta", k=3)
    assert len(hits) == 3
    assert (hits[0].chunk.doc_id, hits[0].chunk.page) == ("BETA_2023_10K", 1)
    assert all(h.dense_rank or h.bm25_rank for h in hits)
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_dense_only_mode(index, embedder):
    r = Retriever(index, embedder, hybrid=False)
    hits = r.search("research and development expenses oncology trials", k=2)
    assert len(hits) == 2 and hits[0].bm25_rank is None
    assert hits[0].chunk.doc_id == "GAMMA_2023Q2_10Q"


def test_hybrid_rescues_exact_keyword_match(index, embedder):
    # "0.75" only matters lexically: BM25 must bring the dividend chunk in
    hits = Retriever(index, embedder, hybrid=True).search("0.75", k=3)
    assert any(h.chunk.doc_id == "GAMMA_2023Q2_10Q" for h in hits)


def test_search_restricted_to_documents(retriever):
    for hybrid in (True, False):
        retriever.hybrid = hybrid
        hits = retriever.search("operating income", k=5, doc_ids=["ACME_2022_10K"])
        assert hits and {h.chunk.doc_id for h in hits} == {"ACME_2022_10K"}
    assert retriever.search("operating income", k=5, doc_ids=["NOPE_2020_10K"]) == []
