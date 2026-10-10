"""Hybrid search engine behind the API: BM25 + dense weighted fusion, optional cross-encoder rerank."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from raglab.fusion import weighted_fusion
from raglab.rerank import RERANKERS, Reranker
from raglab.retrievers import EMBEDDERS, BM25Retriever, Embedder

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Hit:
    doc_id: str
    title: str
    snippet: str
    score: float


class Searcher(Protocol):
    def search(self, query: str, k: int, rerank: bool) -> list[Hit]: ...


class HybridSearcher:
    """In-memory hybrid retriever: weighted BM25 + dense fusion, then an optional
    cross-encoder rerank of the first `rerank_depth` fused candidates."""

    def __init__(
        self,
        doc_ids: list[str],
        titles: list[str],
        texts: list[str],
        embedder_key: str = "bge-base",
        alpha: float = 0.8,
        reranker_key: str | None = "gte-modernbert",
        candidates: int = 100,
        rerank_depth: int = 10,
        cache_key: str | None = None,
    ) -> None:
        self.doc_ids, self.titles, self.texts = doc_ids, titles, texts
        self.docs = [f"{t} {x}".strip() for t, x in zip(titles, texts)]
        self.position = {d: i for i, d in enumerate(doc_ids)}
        self.alpha = alpha
        self.candidates = candidates
        self.rerank_depth = rerank_depth
        self.bm25 = BM25Retriever(self.docs, doc_ids)
        self.embedder = Embedder(EMBEDDERS[embedder_key], ROOT / "data" / "cache")
        self.doc_emb = self.embedder.encode_docs(self.docs, cache_key=cache_key)
        self.reranker = Reranker(RERANKERS[reranker_key]) if reranker_key else None

    def search(self, query: str, k: int = 10, rerank: bool = True) -> list[Hit]:
        sparse = self.bm25.scores(query)
        dense = self.doc_emb @ self.embedder.encode_queries([query])[0]
        n = self.candidates
        cand = np.union1d(np.argsort(-sparse)[:n], np.argsort(-dense)[:n])
        ids = [self.doc_ids[i] for i in cand]
        fused = weighted_fusion(
            {"q": dict(zip(ids, dense[cand].tolist()))},
            {"q": dict(zip(ids, sparse[cand].tolist()))},
            self.alpha,
            depth=n,
        )["q"]
        ranked = [(self.position[d], s) for d, s in fused]
        if rerank and self.reranker is not None:
            head, tail = ranked[: self.rerank_depth], ranked[self.rerank_depth :]
            ce = self.reranker.score(query, [self.docs[i] for i, _ in head])
            ranked = [(head[j][0], float(ce[j])) for j in np.argsort(-ce, kind="stable")] + tail
        return [
            Hit(self.doc_ids[i], self.titles[i], self.texts[i][:300], float(s))
            for i, s in ranked[:k]
        ]


def build_default_searcher() -> HybridSearcher:
    """Index a BEIR corpus chosen by env vars.

    Defaults are the benchmark's best quality/latency trade-off on SciFact: BGE-base,
    alpha 0.8, GTE-ModernBERT reranking the top 10 (nDCG@10 0.771, ~160 ms on a laptop GPU).
    """
    from raglab.data import load_beir

    name = os.environ.get("RAGLAB_DATASET", "scifact")
    ds = load_beir(name)
    return HybridSearcher(
        ds.doc_ids,
        ds.titles,
        ds.texts,
        embedder_key=os.environ.get("RAGLAB_EMBEDDER", "bge-base"),
        alpha=float(os.environ.get("RAGLAB_ALPHA", "0.8")),
        reranker_key=os.environ.get("RAGLAB_RERANKER", "gte-modernbert") or None,
        rerank_depth=int(os.environ.get("RAGLAB_RERANK_DEPTH", "10")),
        cache_key=name,
    )
