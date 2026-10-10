"""Second-stage reranking with cross-encoders."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from raglab.metrics import Run


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    label: str
    model_id: str


RERANKERS: dict[str, RerankerSpec] = {
    s.key: s
    for s in [
        RerankerSpec("ms-minilm", "MiniLM-L6 cross-encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        RerankerSpec("bge-reranker", "BGE-reranker-base", "BAAI/bge-reranker-base"),
        RerankerSpec("gte-modernbert", "GTE-reranker-ModernBERT", "Alibaba-NLP/gte-reranker-modernbert-base"),
    ]
}


class Reranker:
    """Scores (query, document) pairs jointly; one forward batch per query."""

    def __init__(self, spec: RerankerSpec, max_length: int = 512) -> None:
        self.spec = spec
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"dtype": torch.float16} if device == "cuda" else {}
        self.model = CrossEncoder(spec.model_id, max_length=max_length, device=device, model_kwargs=kwargs)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """Raw relevance logits. Ranking only needs a monotone score, and skipping the
        sigmoid avoids fp16 ties among very negative logits."""
        if not docs:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(
            self.model.predict(
                [(query, d) for d in docs],
                batch_size=len(docs),
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
            ),
            dtype=np.float32,
        )

    def score_run(
        self, queries: dict[str, str], run: Run, doc_text: dict[str, str], depth: int = 100
    ) -> tuple[dict[str, np.ndarray], list[float]]:
        """Cross-encoder scores for the top-`depth` candidates of each query, plus per-query latency (ms)."""
        scores: dict[str, np.ndarray] = {}
        latency_ms: list[float] = []
        for qid, text in queries.items():
            docs = [doc_text[d] for d, _ in run.get(qid, [])[:depth]]
            t0 = time.perf_counter()
            scores[qid] = self.score(text, docs)
            latency_ms.append((time.perf_counter() - t0) * 1000)
        return scores, latency_ms


def apply_rerank(run: Run, ce_scores: dict[str, np.ndarray], depth: int) -> Run:
    """Reorder the first `depth` candidates by cross-encoder score; keep the tail in first-stage order."""
    out: Run = {}
    for qid, ranked in run.items():
        head, tail = ranked[:depth], ranked[depth:]
        s = ce_scores[qid][: len(head)]
        order = np.argsort(-s, kind="stable")
        out[qid] = [(head[i][0], float(s[i])) for i in order] + tail
    return out
