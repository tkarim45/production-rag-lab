"""Retrieval metrics — need relevance judgments (qrels).

Inputs are always:
  ranked: list[str]        chunk ids in retrieved order (rank 1 first)
  relevant: set[str]       the gold-relevant chunk ids for the query
  k: int                   cutoff

Graded-relevance metrics (NDCG) accept an optional `gains: dict[str, float]`; when absent,
binary relevance (gain 1.0 for relevant) is assumed.

All functions are pure and match the standard IR definitions; verified in
tests/test_metrics_retrieval.py against hand-computed cases.
"""

from __future__ import annotations

import math
from typing import Iterable


def _topk(ranked: list[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    return ranked[:k]


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for c in _topk(ranked, k) if c in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    top = _topk(ranked, k)
    if not top:
        return 0.0
    hits = sum(1 for c in top if c in relevant)
    return hits / len(top)


def hit_rate_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant doc is in the top-k, else 0.0."""
    return 1.0 if any(c in relevant for c in _topk(ranked, k)) else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant doc (0.0 if none). MRR = mean over queries."""
    for i, c in enumerate(ranked, start=1):
        if c in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked: list[str], gains: dict[str, float], k: int) -> float:
    total = 0.0
    for i, c in enumerate(_topk(ranked, k), start=1):
        g = gains.get(c, 0.0)
        if g:
            total += g / math.log2(i + 1)
    return total


def ndcg_at_k(
    ranked: list[str], relevant: set[str], k: int, gains: dict[str, float] | None = None
) -> float:
    if gains is None:
        gains = {c: 1.0 for c in relevant}
    dcg = dcg_at_k(ranked, gains, k)
    # ideal ranking: gains sorted descending
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(ranked: list[str], relevant: set[str]) -> float:
    """AP for one query; MAP = mean of AP across queries."""
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, c in enumerate(ranked, start=1):
        if c in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(relevant)


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
