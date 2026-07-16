"""Shared eval-run plumbing for the ops module.

Both A/B testing and the CI gate need the same two primitives:

1. "run this config on the eval set" — without polluting the leaderboard (a gate that
   appends a row every CI build would rewrite history on every red build).
2. "give me the **per-query** values of a metric" — an aggregate mean cannot be
   bootstrapped or PSI'd; you need the length-n vector behind it.

Defined once here so `ab.py`, `drift.py` and `gate.py` share one definition of a metric
instead of quietly drifting apart. The stage option labels (`tfidf`, `hashing`, `bm25`, …)
are the *same* labels `harness.sweep` uses — reused, not re-declared.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from harness import scoring
from harness.contract import PipelineResult
from harness.data import load_dataset
from harness.metrics import answer as A
from harness.metrics import retrieval as R
from harness.sweep import _BASE, _STAGE_DEFAULTS


def base_config(dataset: str = "builtin_docs") -> dict[str, Any]:
    """A fresh copy of the sweep's base config (Phase 3 winner: tfidf + flat + dense)."""
    cfg = copy.deepcopy(_BASE)
    cfg["dataset"] = dataset
    return cfg


def _set_stage(cfg: dict[str, Any], stage: str, option: str) -> None:
    if stage not in _STAGE_DEFAULTS:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(_STAGE_DEFAULTS)}")
    component = _STAGE_DEFAULTS[stage].get(option) or {"name": option}
    cfg[stage] = None if component.get("name") is None else dict(component)


def variant(stage: str, option: str, dataset: str = "builtin_docs",
            pin: dict[str, str] | None = None) -> dict[str, Any]:
    """Base config with one stage swapped to a sweep option label (e.g. embedder=hashing).

    `pin` fixes other stages to non-default options (e.g. `{"embedder": "hashing"}` to A/B a
    chunker on a deliberately *weak* first stage, the way Phase 9 measured contextual
    retrieval). Without it, an A/B can only ever reproduce the base condition.
    """
    cfg = base_config(dataset)
    for pinned_stage, pinned_option in (pin or {}).items():
        _set_stage(cfg, pinned_stage, pinned_option)
    _set_stage(cfg, stage, option)
    return cfg


def run_results(cfg: dict[str, Any], pipeline=None) -> list[PipelineResult]:
    """Build the pipeline from a config dict and run every eval query. No side effects."""
    from harness import config as cfgmod  # local: importing modules/ at call time, not import time

    if pipeline is None:
        pipeline = cfgmod.build_pipeline(cfg)
    docs, queries = load_dataset(cfg["dataset"])
    pipeline.build(docs)
    return [pipeline.run_query(q) for q in queries]


def run_metrics(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[PipelineResult]]:
    """Run a config and return (aggregate metrics, per-query results)."""
    results = run_results(cfg)
    return scoring.score(results), results


# ── per-query metric extraction ───────────────────────────────────────────────
# The granularity rule (doc-level qrels when present, else chunk-level) lives in
# harness.scoring and MUST match, or the bootstrap CI would be computed on a different
# metric than the leaderboard reports. Imported, never re-implemented.

_ranked_and_relevant = scoring._ranked_and_relevant


def _retrieval_metric(base: str, k: int) -> Callable[[PipelineResult], float]:
    fns = {
        "recall": R.recall_at_k,
        "precision": R.precision_at_k,
        "hit_rate": R.hit_rate_at_k,
        "ndcg": R.ndcg_at_k,
    }
    if base not in fns:
        raise ValueError(f"unknown metric {base}@{k}; expected one of {sorted(fns)}")
    fn = fns[base]
    return lambda r: fn(*_ranked_and_relevant(r), k)


def per_query(results: list[PipelineResult], metric: str) -> list[float]:
    """The length-n vector of per-query values behind an aggregate metric.

    Supports `recall@k`, `precision@k`, `hit_rate@k`, `ndcg@k`, `mrr`, `map`, `token_f1`, `em`.
    Answer metrics silently drop ungraded queries (no gold answer) — both arms of an A/B run
    the same eval set in the same order, so the pairing survives the filter.
    """
    if "@" in metric:
        base, k = metric.split("@", 1)
        fn = _retrieval_metric(base, int(k))
        return [fn(r) for r in results]
    if metric == "mrr":
        return [R.reciprocal_rank(*_ranked_and_relevant(r)) for r in results]
    if metric == "map":
        return [R.average_precision(*_ranked_and_relevant(r)) for r in results]
    if metric in ("token_f1", "em"):
        fn = A.token_f1 if metric == "token_f1" else A.exact_match
        return [fn(r.answer, r.query.gold_answer) for r in results if r.query.gold_answer is not None]
    if metric == "latency_ms":
        return [r.stage_latency_ms.get("total_ms", 0.0) for r in results]
    raise ValueError(f"unsupported per-query metric {metric!r}")
