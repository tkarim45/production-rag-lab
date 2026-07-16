"""CLI: the Phase 12 cache benchmark — what do the two caches actually buy?

    python -m harness.cache_report                       # offline, key-free
    python -m harness.cache_report --generator claude    # real Bedrock Haiku

Two measurements, both on real runs of the real pipeline:

A. **Embedding cache** — cold build (empty cache) vs warm build (same corpus, same embedder,
   fresh process-equivalent instance) vs no cache at all. The number is the rebuild-time
   saving, and the interesting part is how it depends entirely on how expensive the embedder
   is: a cache in front of something cheap is overhead with extra steps.

B. **Semantic cache** — replay a query workload (every eval query, `--passes` times) through a
   `semantic_cached` generator at several thresholds. Pass 1 is 20 *distinct* questions, so
   **any hit in pass 1 is by construction a false hit** — a different question served someone
   else's answer. Pass 2 is exact repeats, so its hits are the honest upside. Ground truth for
   the audit is the gold answer already on each Query: a hit whose source query has a
   different gold answer is a false hit. The cache never sees the gold labels — only the audit
   does.

Writes results/serving_cache_report.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from typing import Any

import modules.serving  # noqa: F401  registers `cached` + `semantic_cached`
from harness import config as cfgmod
from harness.contract import Chunk, PipelineResult, Query
from harness.data import load_dataset
from harness.latency_report import stage_breakdown
from harness.registry import available
from harness.registry import build as build_component
from harness.runner import RESULTS

_GENERATORS = {
    "extractive_mock": {"name": "extractive_mock"},
    "claude": {"name": "claude_prompted", "style": "grounded"},
}

# same base pipeline as harness.sweep so Phase 12 numbers are comparable to every other phase
_BASE: dict[str, Any] = {
    "dataset": "builtin_docs",
    "retrieve_k": 10,
    "final_k": 5,
    "chunker": {"name": "fixed", "size": 60, "overlap": 10},
    "embedder": {"name": "tfidf"},
    "index": {"name": "flat"},
    "retriever": {"name": "dense"},
    "reranker": None,
    "query_transformer": None,
    "assembler": {"name": "concat"},
    "generator": {"name": "extractive_mock"},
}


def _pipeline_with(cfg_patch: dict[str, Any], dataset: str):
    cfg = copy.deepcopy(_BASE)
    cfg["dataset"] = dataset
    cfg.update(cfg_patch)
    return cfgmod.build_pipeline(cfg)


# ── A. embedding cache: cold vs warm ──────────────────────────────────────────


def embedding_cache_bench(dataset: str, bases: list[str]) -> list[dict[str, Any]]:
    docs, _ = load_dataset(dataset)
    rows: list[dict[str, Any]] = []

    for base in bases:
        # steady-state only: a neural embedder's FIRST forward pass in a process pays torch's
        # one-time warm-up, which would otherwise be charged to whichever run went first and
        # flatter every later one. Burn it here so cold-vs-warm measures the cache, not torch.
        warm = build_component("embedder", base)
        warm.encode_chunks([Chunk("warmup::0", "warmup", "warm up the embedder")])

        # no cache — the control
        p = _pipeline_with({"embedder": {"name": base}}, dataset)
        p.build(copy.deepcopy(docs))
        uncached_ms = p.build_stats["embed_ms"]

        with tempfile.TemporaryDirectory(prefix="rag_embed_cache_") as tmp:
            patch = {"embedder": {"name": "cached", "base": base, "cache_dir": tmp}}

            cold = _pipeline_with(patch, dataset)
            cold.build(copy.deepcopy(docs))
            cold_ms = cold.build_stats["embed_ms"]
            cold_stats = cold.embedder.stats

            # a *fresh* wrapper on the same directory = what the next deploy/rebuild sees
            warm = _pipeline_with(patch, dataset)
            warm.build(copy.deepcopy(docs))
            warm_ms = warm.build_stats["embed_ms"]
            warm_stats = warm.embedder.stats

        rows.append({
            "base_embedder": base,
            "n_chunks": cold.build_stats["n_chunks"],
            "dim": cold.build_stats["embed_dim"],
            "uncached_embed_ms": uncached_ms,
            "cold_embed_ms": cold_ms,
            "warm_embed_ms": warm_ms,
            "saving_vs_cold_pct": (1 - warm_ms / cold_ms) * 100 if cold_ms > 0 else 0.0,
            "speedup_vs_cold": (cold_ms / warm_ms) if warm_ms > 0 else float("inf"),
            "cold_hit_rate": cold_stats["hit_rate"],
            "warm_hit_rate": warm_stats["hit_rate"],
            "cold_writes": cold_stats["writes"],
            "namespace": cold_stats["namespace"],
        })
    return rows


# ── B. semantic cache: hit rate vs false hits ─────────────────────────────────


def _audit(results: list[PipelineResult], gold: dict[str, str | None]) -> dict[str, Any]:
    """Classify every served answer. A hit is FALSE when the cached entry it came from
    answers a question with a different gold answer."""
    hits = [r for r in results if r.extra.get("cache_hit")]
    false_hits = [
        r for r in hits
        if gold.get(r.extra.get("cache_source_query_id")) != gold.get(r.query.query_id)
    ]
    n = len(results)
    return {
        "n_requests": n,
        "hits": len(hits),
        "hit_rate": len(hits) / n if n else 0.0,
        "true_hits": len(hits) - len(false_hits),
        "false_hits": len(false_hits),
        "false_hit_rate_of_requests": len(false_hits) / n if n else 0.0,
        "false_hit_rate_of_hits": (len(false_hits) / len(hits)) if hits else 0.0,
        "examples": [
            {
                "asked": r.query.text,
                "served_answer_for": r.extra.get("cache_source_text"),
                "cosine": round(float(r.extra.get("cache_sim", 0.0)), 4),
            }
            for r in false_hits[:3]
        ],
    }


def semantic_cache_bench(
    dataset: str, thresholds: list[float], gen_key: str, cache_embedder: str, passes: int
) -> list[dict[str, Any]]:
    docs, queries = load_dataset(dataset)
    gold = {q.query_id: q.gold_answer for q in queries}
    rows: list[dict[str, Any]] = []

    for t in thresholds:
        patch = {"generator": {
            "name": "semantic_cached",
            "base": _GENERATORS[gen_key]["name"],
            "threshold": t,
            "embedder": cache_embedder,
            **{k: v for k, v in _GENERATORS[gen_key].items() if k != "name"},
        }}
        p = _pipeline_with(patch, dataset)
        p.build(copy.deepcopy(docs))

        results: list[PipelineResult] = []
        for _ in range(passes):
            for q in queries:
                # a fresh Query object per request — the workload is a replayed stream,
                # not the same object handed back (that would leak state into the cache)
                results.append(p.run_query(Query(
                    query_id=q.query_id, text=q.text, gold_answer=q.gold_answer,
                    relevant_chunk_ids=set(q.relevant_chunk_ids),
                    relevant_doc_ids=set(q.relevant_doc_ids), hop_type=q.hop_type,
                )))

        row = {"threshold": t, "cache_embedder": cache_embedder, "generator": gen_key,
               "passes": passes}
        row.update(_audit(results, gold))
        per_pass = len(results) // passes
        row["pass1_hits"] = sum(1 for r in results[:per_pass] if r.extra.get("cache_hit"))
        row["pass2_hits"] = sum(1 for r in results[per_pass:] if r.extra.get("cache_hit"))
        row["total_cost_usd"] = sum(r.cost_usd for r in results)
        gen_ms = [r.stage_latency_ms.get("generate_ms", 0.0) for r in results]
        hit_ms = [r.stage_latency_ms.get("generate_ms", 0.0)
                  for r in results if r.extra.get("cache_hit")]
        miss_ms = [r.stage_latency_ms.get("generate_ms", 0.0)
                   for r in results if not r.extra.get("cache_hit")]
        row["mean_generate_ms"] = sum(gen_ms) / len(gen_ms) if gen_ms else 0.0
        row["mean_generate_ms_on_hit"] = sum(hit_ms) / len(hit_ms) if hit_ms else 0.0
        row["mean_generate_ms_on_miss"] = sum(miss_ms) / len(miss_ms) if miss_ms else 0.0
        row["latency"] = stage_breakdown(results)["stages"]
        rows.append(row)
    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────


def _print_embed(rows: list[dict[str, Any]]) -> None:
    print("\n=== A. embedding cache — rebuild cost (cold vs warm) ===")
    print(f"{'base embedder':<22}{'no cache':>11}{'cold':>11}{'warm':>11}"
          f"{'saving':>10}{'speedup':>10}")
    for r in rows:
        print(f"{r['base_embedder']:<22}{r['uncached_embed_ms']:>10.1f}ms"
              f"{r['cold_embed_ms']:>10.1f}ms{r['warm_embed_ms']:>10.1f}ms"
              f"{r['saving_vs_cold_pct']:>9.1f}%{r['speedup_vs_cold']:>9.1f}x")


def _print_semantic(rows: list[dict[str, Any]]) -> None:
    print("\n=== B. semantic cache — hit rate vs false hits ===")
    print(f"{'threshold':<11}{'hit rate':>10}{'hits':>7}{'true':>7}{'FALSE':>7}"
          f"{'false/hits':>12}{'p1 hits':>9}{'p2 hits':>9}{'cost $':>11}")
    for r in rows:
        print(f"{r['threshold']:<11.2f}{r['hit_rate']:>9.1%}{r['hits']:>7}{r['true_hits']:>7}"
              f"{r['false_hits']:>7}{r['false_hit_rate_of_hits']:>11.1%}"
              f"{r['pass1_hits']:>9}{r['pass2_hits']:>9}{r['total_cost_usd']:>11.4f}")
    for r in rows:
        for ex in r["examples"]:
            print(f"  ! t={r['threshold']:.2f} cos={ex['cosine']:.3f} asked "
                  f"{ex['asked']!r}\n      → served the answer to {ex['served_answer_for']!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-lab-cache", description="Phase 12 cache benchmark.")
    ap.add_argument("--dataset", default="builtin_docs")
    ap.add_argument("--generator", default="extractive_mock", choices=sorted(_GENERATORS))
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.99, 0.90, 0.80, 0.70, 0.60])
    ap.add_argument("--passes", type=int, default=2, help="times the query stream is replayed")
    ap.add_argument("--cache-embedder", default="auto",
                    help="embedder keying the semantic cache (auto = best available)")
    ap.add_argument("--embed-bases", nargs="+", default=None,
                    help="base embedders for the embedding-cache bench (default: auto)")
    args = ap.parse_args(argv)

    have_st = "sentence_transformer" in available("embedder")
    cache_embedder = args.cache_embedder
    if cache_embedder == "auto":
        cache_embedder = "sentence_transformer" if have_st else "hashing"
    bases = args.embed_bases or (["hashing", "sentence_transformer"] if have_st else ["hashing"])

    embed_rows = embedding_cache_bench(args.dataset, bases)
    _print_embed(embed_rows)

    sem_rows = semantic_cache_bench(
        args.dataset, args.thresholds, args.generator, cache_embedder, args.passes
    )
    _print_semantic(sem_rows)

    report = {
        "dataset": args.dataset,
        "generator": args.generator,
        "semantic_cache_embedder": cache_embedder,
        "embedding_cache": embed_rows,
        "semantic_cache": sem_rows,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "serving_cache_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
