import numpy as np
import pytest

from finrag.chunking import chunk_pages, clean_text, make_chunks
from finrag.index import HashingEmbedder, VectorIndex
from finrag.retrieve import Retriever


def test_clean_text_undoes_hyphenation_and_whitespace():
    assert clean_text("operat-\ning  income\n rose") == "operating income rose"


def test_make_chunks_window_and_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = make_chunks("ACME_2022_10K", "T", text, size=40, overlap=10, page=7)
    assert [len(c.text.split()) for c in chunks] == [40, 40, 40]
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    assert chunks[1].id == "ACME_2022_10K#p7-1" and chunks[1].page == 7
    assert chunks[-1].text.split()[-1] == "w99"


def test_make_chunks_edge_cases():
    assert make_chunks("D", "T", "   ", 10, 2) == []
    assert len(make_chunks("D", "T", "one two three", 10, 2)) == 1
    with pytest.raises(ValueError):
        make_chunks("D", "T", "a b c", 5, 5)


def test_chunk_pages_keeps_page_numbers_and_skips_empty_pages():
    chunks = chunk_pages("D", "T", ["first page text", "  ", "third page " * 30], size=20, overlap=5)
    assert chunks[0].page == 0 and {c.page for c in chunks[1:]} == {2}
    assert all(c.doc_id == "D" for c in chunks)


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.encode(["capital expenditure"]), e.encode(["capital expenditure"])
    assert np.allclose(a, b)
    assert pytest.approx(float(np.linalg.norm(a[0])), rel=1e-5) == 1.0
    assert np.allclose(e.encode([""]), 0)


def test_index_search_ranks_relevant_chunk_first(index, embedder):
    q = embedder.encode(["purchases of property plant and equipment capital expenditure"])[0]
    top = index.search(q, 3)
    best = index.chunks[top[0][0]]
    assert (best.doc_id, best.page) == ("ACME_2022_10K", 1)
    assert top[0][1] >= top[1][1] >= top[2][1]


def test_index_search_with_mask(index, embedder):
    q = embedder.encode(["capital expenditure"])[0]
    mask = np.array([c.doc_id == "BETA_2023_10K" for c in index.chunks])
    hits = index.search(q, 10, mask=mask)
    assert hits and all(index.chunks[i].doc_id == "BETA_2023_10K" for i, _ in hits)
    assert index.search(q, 3, mask=np.zeros(len(index.chunks), dtype=bool)) == []


def test_index_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    assert [c.id for c in loaded.chunks] == [c.id for c in index.chunks]
    assert [c.page for c in loaded.chunks] == [c.page for c in index.chunks]
    assert np.allclose(loaded.matrix, index.matrix)
    assert loaded.embedder_name == "hashing"


def test_retriever_rejects_mismatched_embedder(index):
    class Other:
        name, dim = "other", 8

        def encode(self, texts):
            return np.zeros((len(texts), 8), dtype=np.float32)

    with pytest.raises(ValueError):
        Retriever(index, Other())
