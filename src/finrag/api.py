"""FastAPI service. Start with:  uvicorn finrag.api:app_factory --factory --host 0.0.0.0 --port 8080

Set FINRAG_CONFIG to a YAML file. With llm_backend=openai_compat the image needs no GPU: it calls an external LLM server.
"""
from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Config
from .pipeline import Pipeline, load_pipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["rag", "agent"] = "rag"
    doc_ids: list[str] | None = Field(default=None, max_length=10, description="search only these reports")


def create_app(pipeline: Pipeline) -> FastAPI:
    app = FastAPI(title="finrag", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(pipeline.index.chunks), "documents": len(pipeline.docs)}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        try:
            if req.mode == "rag":
                return {"mode": "rag", **pipeline.rag().ask(req.question, doc_ids=req.doc_ids).to_dict()}
            q = req.question + (f"\n(The relevant report: {', '.join(req.doc_ids)}.)" if req.doc_ids else "")
            return {"mode": "agent", **pipeline.agent().run(q).to_dict()}
        except Exception as e:  # surface backend failures as a clean 502
            raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e

    return app


def app_factory() -> FastAPI:
    cfg = Config.load(os.environ.get("FINRAG_CONFIG", "configs/default.yaml"))
    return create_app(load_pipeline(cfg))
