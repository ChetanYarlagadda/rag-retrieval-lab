import math

import pytest

from raglab.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_perfect_ranking_is_one():
    rels = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rels, 10) == pytest.approx(1.0)


def test_ndcg_matches_hand_computation():
    # relevant docs a (grade 2) and b (grade 1); ranking puts b first, a third.
    rels = {"a": 2, "b": 1}
    dcg = 1 / math.log2(2) + 2 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "x", "a"], rels, 10) == pytest.approx(dcg / idcg)


def test_ndcg_respects_cutoff():
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, k=2) == 0.0


def test_recall_and_mrr():
    rels = {"a": 1, "b": 1, "c": 0}
    assert recall_at_k(["x", "a", "c"], rels, 100) == pytest.approx(0.5)
    assert mrr_at_k(["x", "c", "b"], rels, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y"], rels, 10) == 0.0


def test_evaluate_scores_missing_queries_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": [("a", 1.0)]}
    df = evaluate(run, qrels, ["q1", "q2"]).set_index("qid")
    assert df.loc["q1", "ndcg@10"] == pytest.approx(1.0)
    assert df.loc["q2", "ndcg@10"] == 0.0
