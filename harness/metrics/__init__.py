"""Metrics — every number in docs/03-metrics-catalog.md, defined once, unit-tested.

Split by family:
- retrieval.py  — need relevance judgments (qrels): recall@k, precision@k, hit_rate@k,
                  mrr, ndcg@k, average_precision/MAP.
- answer.py     — need gold answers: exact_match, token_f1.
- efficiency.py — latency percentiles, cost aggregation.

All functions are pure and operate on plain ids/strings/lists so they can be tested
against hand-computed cases with no pipeline involved.
"""

from harness.metrics import answer, efficiency, retrieval  # noqa: F401
