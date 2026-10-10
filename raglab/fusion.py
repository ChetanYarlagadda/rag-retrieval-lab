"""Hybrid retrieval: reciprocal rank fusion and weighted score fusion."""

from __future__ import annotations

from raglab.metrics import Run

CandidateScores = dict[str, dict[str, float]]  # query id -> doc id -> raw score


def rrf(runs: list[Run], k: int = 60, depth: int = 100) -> Run:
    """Reciprocal rank fusion (Cormack et al., 2009): score(d) = sum_r 1 / (k + rank_r(d))."""
    fused: Run = {}
    for qid in set().union(*runs):
        acc: dict[str, float] = {}
        for run in runs:
            for rank, (doc, _) in enumerate(run.get(qid, [])[:depth], start=1):
                acc[doc] = acc.get(doc, 0.0) + 1.0 / (k + rank)
        fused[qid] = sorted(acc.items(), key=lambda x: (-x[1], x[0]))[:depth]
    return fused


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {d: 0.0 for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def weighted_fusion(dense: CandidateScores, sparse: CandidateScores, alpha: float, depth: int = 100) -> Run:
    """alpha * minmax(dense) + (1 - alpha) * minmax(sparse) over a shared candidate set.

    Both inputs must hold exact scores for the same candidates of each query (the
    union of the two first-stage top-k lists), so no document is penalised just
    for missing from one list.
    """
    fused: Run = {}
    for qid, d_scores in dense.items():
        s_scores = sparse[qid]
        if set(d_scores) != set(s_scores):
            raise ValueError(f"candidate sets differ for query {qid}")
        nd, ns = _minmax(d_scores), _minmax(s_scores)
        combined = {d: alpha * nd[d] + (1 - alpha) * ns[d] for d in d_scores}
        fused[qid] = sorted(combined.items(), key=lambda x: (-x[1], x[0]))[:depth]
    return fused
