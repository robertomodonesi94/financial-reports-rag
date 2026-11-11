"""Retrieval: dense search, BM25, and reciprocal rank fusion (hybrid)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .index import Embedder, VectorIndex

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for d, length in zip(self.docs, self.lengths):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1.0)))
            out.append(s)
        return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


class Retriever:
    def __init__(self, index: VectorIndex, embedder: Embedder, hybrid: bool = True, pool: int = 20, rrf_k: int = 60):
        if embedder.name != index.embedder_name:
            raise ValueError(f"Index built with '{index.embedder_name}' but embedder is '{embedder.name}'")
        self.index, self.embedder, self.hybrid, self.pool, self.rrf_k = index, embedder, hybrid, pool, rrf_k
        self._bm25 = BM25([c.text for c in index.chunks]) if hybrid else None

    def search(self, query: str, k: int = 5, doc_ids: list[str] | None = None) -> list[Hit]:
        """Top-k chunks. doc_ids restricts the search to some documents (e.g. the report the user picked)."""
        allowed = None
        if doc_ids:
            wanted = set(doc_ids)
            allowed = np.array([c.doc_id in wanted for c in self.index.chunks], dtype=bool)
            if not allowed.any():
                return []
        qvec = self.embedder.encode([query])[0]
        dense = self.index.search(qvec, max(k, self.pool) if self.hybrid else k, mask=allowed)
        if not self.hybrid:
            return [Hit(self.index.chunks[i], s, dense_rank=r + 1) for r, (i, s) in enumerate(dense[:k])]
        bm = self._bm25.scores(query)
        candidates = range(len(bm)) if allowed is None else np.flatnonzero(allowed)
        bm_order = [int(i) for i in sorted(candidates, key=lambda i: -bm[i]) if bm[i] > 0][: self.pool]
        fused: dict[int, float] = {}
        ranks: dict[int, dict[str, int]] = {}
        for r, (i, _) in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
            ranks.setdefault(i, {})["dense"] = r + 1
        for r, i in enumerate(bm_order):
            fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + r + 1)
            ranks.setdefault(i, {})["bm25"] = r + 1
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.index.chunks[i], fused[i], ranks[i].get("dense"), ranks[i].get("bm25")) for i in best]
