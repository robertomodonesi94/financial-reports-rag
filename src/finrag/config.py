"""Configuration: one dataclass, loadable from YAML, overridable from the command line."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # data
    data_dir: str = "data"
    max_docs: int = 0            # 0 = every report referenced by the FinanceBench questions
    # chunking (inside each PDF page, so every chunk keeps its page number)
    chunk_words: int = 220
    overlap_words: int = 40
    # retrieval
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2"  # "hashing" = no-download lexical baseline
    top_k: int = 5
    hybrid: bool = True
    # generation
    llm_backend: str = "hf"  # hf | openai_compat | fake
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"  # or Qwen/Qwen2.5-7B-Instruct (check the licence)
    llm_base_url: str = "http://localhost:8000/v1"
    max_new_tokens: int = 400
    temperature: float = 0.0
    # agent
    agent_max_steps: int = 5

    @property
    def pdf_dir(self) -> Path:
        return Path(self.data_dir) / "pdf"

    @property
    def benchmark_dir(self) -> Path:
        return Path(self.data_dir) / "financebench"

    @property
    def meta_path(self) -> Path:
        return Path(self.data_dir) / "documents.json"

    @property
    def index_dir(self) -> Path:
        return Path(self.data_dir) / "index"

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "Config":
        values: dict = {}
        if path and Path(path).exists():
            values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        values.update({k: v for k, v in overrides.items() if v is not None and k in known})
        return cls(**values)
