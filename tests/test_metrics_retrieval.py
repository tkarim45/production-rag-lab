"""Retrieval metrics vs hand-computed cases (docs/03-metrics-catalog.md)."""

import math

from harness.metrics import retrieval as R


def test_recall_precision_hitrate():
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}  # z never retrieved
    # top-3 = a,b,c → 1 hit (b) of 3 relevant
    assert R.recall_at_k(ranked, relevant, 3) == 1 / 3
    assert R.precision_at_k(ranked, relevant, 3) == 1 / 3
    assert R.hit_rate_at_k(ranked, relevant, 3) == 1.0
    # top-1 = a → no hit
    assert R.hit_rate_at_k(ranked, relevant, 1) == 0.0
    assert R.precision_at_k(ranked, relevant, 1) == 0.0


def test_reciprocal_rank():
    assert R.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5      # first relevant at rank 2
    assert R.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert R.reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_average_precision():
    # relevant at ranks 1 and 3 → (1/1 + 2/3) / 2
    ap = R.average_precision(["a", "x", "b", "y"], {"a", "b"})
    assert math.isclose(ap, (1.0 + 2 / 3) / 2)


def test_ndcg_binary_perfect_and_partial():
    # perfect ranking → 1.0
    assert math.isclose(R.ndcg_at_k(["a", "b"], {"a", "b"}, 2), 1.0)
    # single relevant at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2)=1
    got = R.ndcg_at_k(["x", "a"], {"a"}, 2)
    assert math.isclose(got, (1 / math.log2(3)) / 1.0)


def test_ndcg_graded_gains():
    ranked = ["a", "b", "c"]
    gains = {"a": 3.0, "b": 2.0, "c": 0.0}
    dcg = 3 / math.log2(2) + 2 / math.log2(3)
    idcg = 3 / math.log2(2) + 2 / math.log2(3)  # already ideal
    assert math.isclose(R.ndcg_at_k(ranked, {"a", "b"}, 3, gains=gains), dcg / idcg)


def test_empty_relevant_is_zero_not_crash():
    assert R.recall_at_k(["a"], set(), 1) == 0.0
    assert R.average_precision(["a"], set()) == 0.0
