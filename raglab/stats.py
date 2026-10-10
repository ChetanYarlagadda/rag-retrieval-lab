"""Paired significance testing over per-query metric values."""

from __future__ import annotations

import numpy as np
import pandas as pd


def paired_test(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Compare system B against system A on the same queries.

    Returns the mean difference (B - A), a 95% percentile-bootstrap CI over queries,
    and a two-sided p-value from a paired sign-flip randomisation test.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    rng = np.random.default_rng(seed)
    diff = b - a
    observed = diff.mean()

    boots = rng.choice(diff, size=(n_resamples, diff.size), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    signs = rng.choice([-1.0, 1.0], size=(n_resamples, diff.size))
    null = (signs * diff).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_resamples + 1)

    return {"mean_diff": float(observed), "ci_low": float(ci_low), "ci_high": float(ci_high), "p_value": float(p)}


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def comparison_pairs(best_dense: str, best_reranker: str) -> list[tuple[str, str]]:
    """The (baseline, system) comparisons reported for every dataset."""
    hybrid, hybrid_rr = "Hybrid (weighted)", f"Hybrid + {best_reranker}"
    return [
        ("BM25", best_dense),
        (best_dense, hybrid),
        (hybrid, "Hybrid (RRF)"),
        (hybrid, "Hybrid + MiniLM-L6 cross-encoder"),
        (hybrid, "Hybrid + BGE-reranker-base"),
        (hybrid, hybrid_rr),
        (best_dense, hybrid_rr),
        ("Hybrid + MiniLM-L6 cross-encoder", hybrid_rr),
        ("Hybrid + BGE-reranker-base", hybrid_rr),
        (hybrid_rr, f"BM25 + {best_reranker}"),
    ]


def significance_table(per_query: pd.DataFrame, best_dense: dict[str, str], best_reranker: str) -> pd.DataFrame:
    """Paired tests on per-query nDCG@10 for every dataset, with Holm correction within each dataset."""
    rows = []
    for name, dense_label in best_dense.items():
        pq = per_query[per_query.dataset == name].pivot(index="qid", columns="method", values="ndcg@10")
        block = [
            {"dataset": name, "baseline": a, "system": b, **paired_test(pq[a].to_numpy(), pq[b].to_numpy())}
            for a, b in comparison_pairs(dense_label, best_reranker)
        ]
        for row, adj in zip(block, holm([r["p_value"] for r in block])):
            row["p_holm"] = adj
        rows.extend(block)
    return pd.DataFrame(rows)
