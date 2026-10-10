"""FastAPI service exposing the hybrid retriever.

    uvicorn app.main:app --port 8000      # Swagger UI at http://localhost:8000/docs
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from app.searcher import Searcher, build_default_searcher


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000, examples=["Vitamin D deficiency increases risk of fractures"])
    k: int = Field(10, ge=1, le=50, description="number of results")
    rerank: bool = Field(True, description="apply the cross-encoder reranker")


class SearchHit(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: float


class SearchResponse(BaseModel):
    query: str
    reranked: bool
    latency_ms: float
    hits: list[SearchHit]


def create_app(searcher_factory: Callable[[], Searcher]) -> FastAPI:
    """Build the app; the searcher is created once at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.searcher = searcher_factory()
        yield

    app = FastAPI(
        title="RAG Retrieval Lab — hybrid search",
        description="BM25 + dense weighted fusion with optional cross-encoder reranking.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, request: Request) -> SearchResponse:
        t0 = time.perf_counter()
        hits = request.app.state.searcher.search(req.query, k=req.k, rerank=req.rerank)
        return SearchResponse(
            query=req.query,
            reranked=req.rerank,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            hits=[SearchHit(**h.__dict__) for h in hits],
        )

    return app


app = create_app(build_default_searcher)
