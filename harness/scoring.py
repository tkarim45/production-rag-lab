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


def _ranked_and_relevant(r: PipelineResult) -> tuple[list[str], set[str]]:
    """Pick the granularity: doc-level if the query carries doc qrels, else chunk-level.

    Doc-level ranking dedups chunks to their doc so chunking strategies are comparable on
    the same relevance judgments (a finer chunker that ranks the right passage higher wins).
    """
    if r.query.relevant_doc_ids:
        return r.retrieved_doc_ids, r.query.relevant_doc_ids
    return r.retrieved_chunk_ids, r.query.relevant_chunk_ids


def _retrieval_block(results: list[PipelineResult]) -> dict[str, float]:
    out: dict[str, float] = {}
    ranked_rel = [_ranked_and_relevant(r) for r in results]
    for k in _KS:
        out[f"recall@{k}"] = R.mean(R.recall_at_k(ra, rel, k) for ra, rel in ranked_rel)
        out[f"precision@{k}"] = R.mean(R.precision_at_k(ra, rel, k) for ra, rel in ranked_rel)
        out[f"hit_rate@{k}"] = R.mean(R.hit_rate_at_k(ra, rel, k) for ra, rel in ranked_rel)
        out[f"ndcg@{k}"] = R.mean(R.ndcg_at_k(ra, rel, k) for ra, rel in ranked_rel)
    out["mrr"] = R.mean(R.reciprocal_rank(ra, rel) for ra, rel in ranked_rel)
    out["map"] = R.mean(R.average_precision(ra, rel) for ra, rel in ranked_rel)
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

    # Phase 10 generator signals, when the generator reports them
    if any(r.extra for r in results):
        cited = [r for r in results if "has_citation" in r.extra]
        if cited:
            metrics["citation_rate"] = R.mean(1.0 if r.extra["has_citation"] else 0.0 for r in cited)
        abst = [r for r in results if "abstained" in r.extra]
        if abst:
            metrics["abstain_rate"] = R.mean(1.0 if r.extra["abstained"] else 0.0 for r in abst)

    metrics["n_queries"] = len(results)
    return metrics
