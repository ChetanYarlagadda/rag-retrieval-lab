import numpy as np
import pytest

from raglab.retrievers import BM25Retriever
from raglab.stats import holm, paired_test


def test_paired_test_identical_systems():
    a = np.random.default_rng(1).random(200)
    res = paired_test(a, a.copy())
    assert res["mean_diff"] == 0.0
    assert res["p_value"] > 0.9


def test_paired_test_detects_consistent_gain():
    rng = np.random.default_rng(2)
    a = rng.random(300)
    res = paired_test(a, a + 0.05 + rng.normal(0, 0.01, 300))
    assert res["mean_diff"] == pytest.approx(0.05, abs=0.01)
    assert res["ci_low"] > 0
    assert res["p_value"] < 0.001


def test_bm25_ranks_matching_document_first():
    docs = [
        "Stocks and bonds are common investment vehicles.",
        "Statins lower cholesterol and reduce cardiovascular risk.",
        "The quarterly dividend was paid to shareholders.",
    ]
    bm25 = BM25Retriever(docs, ["fin", "med", "div"])
    run = bm25.search({"q": "does cholesterol medication reduce heart risk"}, k=3)
    assert run["q"][0][0] == "med"


def test_bm25_handles_stopword_only_query():
    bm25 = BM25Retriever(["alpha beta", "gamma delta"], ["a", "b"])
    assert bm25.search({"q": "the and of"}, k=2)["q"] == []


def test_holm_adjustment_matches_hand_computation():
    # sorted p: 0.01 (x3 = 0.03), 0.02 (x2 = 0.04), 0.04 (x1 = 0.04); monotone, capped at 1
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])


def test_significance_table_runs_every_comparison_with_holm():
    import pandas as pd

    from raglab.stats import comparison_pairs, significance_table

    rng = np.random.default_rng(3)
    methods = {m for pair in comparison_pairs("Dense", "RR") for m in pair}
    rows = [
        {"dataset": "toy", "method": m, "qid": str(q), "ndcg@10": rng.random()}
        for m in methods
        for q in range(50)
    ]
    table = significance_table(pd.DataFrame(rows), {"toy": "Dense"}, "RR")
    assert len(table) == len(comparison_pairs("Dense", "RR"))
    assert (table.p_holm >= table.p_value).all() and (table.p_holm <= 1).all()
