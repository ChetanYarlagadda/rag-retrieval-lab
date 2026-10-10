import pytest

from raglab.fusion import rrf, weighted_fusion


def test_rrf_rewards_agreement():
    run_a = {"q": [("x", 9.0), ("y", 8.0), ("z", 7.0)]}
    run_b = {"q": [("y", 0.9), ("w", 0.8), ("x", 0.7)]}
    fused = [d for d, _ in rrf([run_a, run_b], k=60)["q"]]
    # y is ranked 2nd and 1st -> best combined; w and z appear in one list only.
    assert fused[0] == "y"
    assert fused.index("x") < fused.index("w")
    assert set(fused) == {"x", "y", "z", "w"}


def test_weighted_fusion_extremes_follow_single_system():
    dense = {"q": {"a": 0.9, "b": 0.5, "c": 0.1}}
    sparse = {"q": {"a": 1.0, "b": 3.0, "c": 7.0}}
    assert [d for d, _ in weighted_fusion(dense, sparse, alpha=1.0)["q"]] == ["a", "b", "c"]
    assert [d for d, _ in weighted_fusion(dense, sparse, alpha=0.0)["q"]] == ["c", "b", "a"]


def test_weighted_fusion_requires_shared_candidates():
    with pytest.raises(ValueError):
        weighted_fusion({"q": {"a": 1.0}}, {"q": {"b": 1.0}}, alpha=0.5)
