"""Embeddings and a small exact vector index (cosine similarity with NumPy)."""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from .chunking import Chunk


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic lexical embedder (hashing trick). No download, no GPU: used for tests and as a baseline."""

    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in re.findall(r"\w+", text.lower()):
                out[i, zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        out = np.log1p(out)  # sublinear term frequency
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional heavy dependency

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


def get_embedder(name: str) -> Embedder:
    return HashingEmbedder() if name == "hashing" else SentenceTransformerEmbedder(name)


class VectorIndex:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, embedder_name: str):
        if len(chunks) != len(matrix):
            raise ValueError("chunks and matrix must have the same length")
        self.chunks = chunks
        self.matrix = matrix
        self.embedder_name = embedder_name

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, batch: int = 128) -> "VectorIndex":
        parts = [embedder.encode([c.text for c in chunks[i:i + batch]]) for i in range(0, len(chunks), batch)]
        matrix = np.vstack(parts) if parts else np.zeros((0, embedder.dim), dtype=np.float32)
        return cls(chunks, matrix, embedder.name)

    def search(self, query_vec: np.ndarray, k: int = 5, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Exact cosine search. mask (bool per chunk) keeps only some chunks."""
        if not len(self.chunks):
            return []
        scores = self.matrix @ query_vec.astype(np.float32)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
            k = min(k, int(mask.sum()))
            if k == 0:
                return []
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "matrix.npy", self.matrix)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (d / "embedder.json").write_text(json.dumps({"name": self.embedder_name}), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        d = Path(directory)
        chunks = [Chunk(**json.loads(line)) for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        meta = json.loads((d / "embedder.json").read_text(encoding="utf-8"))
        return cls(chunks, np.load(d / "matrix.npy"), meta["name"])
