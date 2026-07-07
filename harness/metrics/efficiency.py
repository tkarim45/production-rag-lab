"""Efficiency metrics — first-class, not afterthoughts.

Latency percentiles (nearest-rank), cost aggregation, and the headline production number:
cost per correct answer = total cost / (#queries * mean correctness). A +2pt correctness
that triples cost is a different decision than a free one — the leaderboard shows both.
"""

from __future__ import annotations


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. p in [0, 100]."""
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be in [0, 100]")
    s = sorted(values)
    if p == 0:
        return s[0]
    import math

    rank = math.ceil(p / 100 * len(s))
    return s[min(rank, len(s)) - 1]


def latency_summary(totals_ms: list[float]) -> dict[str, float]:
    return {
        "p50_ms": percentile(totals_ms, 50),
        "p95_ms": percentile(totals_ms, 95),
        "p99_ms": percentile(totals_ms, 99),
        "mean_ms": sum(totals_ms) / len(totals_ms) if totals_ms else 0.0,
    }


def cost_per_correct_answer(total_cost_usd: float, n_queries: int, mean_correctness: float) -> float:
    """total cost / number of correct answers. inf if nothing is correct (honest)."""
    correct = n_queries * mean_correctness
    if correct <= 0:
        return float("inf")
    return total_cost_usd / correct
