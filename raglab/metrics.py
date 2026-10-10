"""Ranking metrics matching trec_eval conventions (linear gain, log2 discount)."""

from __future__ import annotations

import math

import pandas as pd

from raglab.data import Qrels

Run = dict[str, list[tuple[str, float]]]  # query id -> [(doc id, score)] sorted by score desc


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    """nDCG@k with graded gains; ideal ranking built from all judged-relevant docs."""
    ideal = sorted((g for g in rels.values() if g > 0), reverse=True)[:k]
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg([max(rels.get(d, 0), 0) for d in ranked[:k]]) / idcg


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int = 100) -> float:
    """Share of relevant documents that appear in the top k."""
    relevant = {d for d, g in rels.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def mrr_at_k(ranked: list[str], rels: dict[str, int], k: int = 10) -> float:
    """Reciprocal rank of the first relevant document within the top k."""
    for i, d in enumerate(ranked[:k]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(run: Run, qrels: Qrels, query_ids: list[str]) -> pd.DataFrame:
    """Per-query nDCG@10, Recall@100 and MRR@10. Missing queries score zero."""
    rows = []
    for qid in query_ids:
        ranked = [d for d, _ in run.get(qid, [])]
        rels = qrels[qid]
        rows.append(
            {
                "qid": qid,
                "ndcg@10": ndcg_at_k(ranked, rels, 10),
                "recall@100": recall_at_k(ranked, rels, 100),
                "mrr@10": mrr_at_k(ranked, rels, 10),
            }
        )
    return pd.DataFrame(rows)


def summarize(per_query: pd.DataFrame) -> dict[str, float]:
    """Mean of each metric column."""
    return {c: float(per_query[c].mean()) for c in ("ndcg@10", "recall@100", "mrr@10")}
