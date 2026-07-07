"""Aggregate a batch of PipelineResults into the leaderboard metrics dict.

Computes every catalog metric that is key-free (retrieval + EM/F1 + latency/cost) over all
queries, and splits the end-to-end metrics by hop type (single vs multi) because aggregates
hide the multi-hop failure of query-transform methods. LLM-judged metrics (groundedness,
answer-correctness) are added in Phase 11; here `answer_correctness` falls back to token-F1
so cost-per-correct-answer has a real denominator in Phase 0.
"""

from __future__ import annotations

from typing import Any

from harness.contract import PipelineResult
from harness.metrics import answer as A
from harness.metrics import efficiency as E
from harness.metrics import retrieval as R

_KS = (1, 3, 5, 10)


def _retrieval_block(results: list[PipelineResult]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in _KS:
        out[f"recall@{k}"] = R.mean(
            R.recall_at_k(r.retrieved_chunk_ids, r.query.relevant_chunk_ids, k) for r in results
        )
        out[f"precision@{k}"] = R.mean(
            R.precision_at_k(r.retrieved_chunk_ids, r.query.relevant_chunk_ids, k) for r in results
        )
        out[f"hit_rate@{k}"] = R.mean(
            R.hit_rate_at_k(r.retrieved_chunk_ids, r.query.relevant_chunk_ids, k) for r in results
        )
        out[f"ndcg@{k}"] = R.mean(
            R.ndcg_at_k(r.retrieved_chunk_ids, r.query.relevant_chunk_ids, k) for r in results
        )
    out["mrr"] = R.mean(
        R.reciprocal_rank(r.retrieved_chunk_ids, r.query.relevant_chunk_ids) for r in results
    )
    out["map"] = R.mean(
        R.average_precision(r.retrieved_chunk_ids, r.query.relevant_chunk_ids) for r in results
    )
    return out


def _answer_block(results: list[PipelineResult], suffix: str = "") -> dict[str, float]:
    graded = [r for r in results if r.query.gold_answer is not None]
    if not graded:
        return {}
    em = R.mean(A.exact_match(r.answer, r.query.gold_answer) for r in graded)
    f1 = R.mean(A.token_f1(r.answer, r.query.gold_answer) for r in graded)
    return {f"em{suffix}": em, f"token_f1{suffix}": f1}


def score(results: list[PipelineResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("no results to score")

    metrics: dict[str, Any] = {}
    metrics.update(_retrieval_block(results))
    metrics.update(_answer_block(results))

    # split e2e by hop type
    for hop in ("single", "multi"):
        subset = [r for r in results if r.query.hop_type == hop]
        if subset:
            metrics.update(_answer_block(subset, suffix=f"_{hop}"))

    # efficiency
    totals = [r.stage_latency_ms.get("total_ms", 0.0) for r in results]
    metrics.update({f"latency_{k}": v for k, v in E.latency_summary(totals).items()})
    total_cost = sum(r.cost_usd for r in results)
    metrics["total_cost_usd"] = total_cost
    metrics["cost_per_query_usd"] = total_cost / len(results)
    # correctness proxy for Phase 0 = token_f1 (Phase 11 swaps in a calibrated judge)
    correctness = metrics.get("token_f1", 0.0)
    metrics["cost_per_correct_usd"] = E.cost_per_correct_answer(total_cost, len(results), correctness)

    metrics["n_queries"] = len(results)
    return metrics
