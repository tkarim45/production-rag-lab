"""Sweep runner — benchmark a grid of component choices on a base config.

    python -m harness.sweep --dataset builtin_docs --vary chunker \
        --options fixed recursive sentence paragraph structural semantic parent_child

Runs one config per option (all other stages fixed to the base config), appends each to the
leaderboard, and prints a compact comparison table. This is how every phase produces its
"variant leaderboard" — the repo's core deliverable.
"""

from __future__ import annotations

import argparse
import copy
from typing import Any

from harness import config as cfgmod
from harness import leaderboard, scoring
from harness.data import load_dataset
from harness.runner import RESULTS

# option-label → full component config (name + params). Lets a sweep option like
# "quantized_binary" expand to {name: quantized, mode: binary}.
_STAGE_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "chunker": {
        "fixed": {"name": "fixed", "size": 60, "overlap": 10},
        "recursive": {"name": "recursive", "size": 60, "overlap": 10},
        "sentence": {"name": "sentence", "per_chunk": 2, "overlap": 1},
        "paragraph": {"name": "paragraph"},
        "structural": {"name": "structural"},
        "semantic": {"name": "semantic", "threshold": 0.25},
        "parent_child": {"name": "parent_child"},
        # Phase 9 contextual retrieval — same base params as `fixed` so the A/B is fair
        "contextual": {"name": "contextual", "base": "fixed", "size": 60, "overlap": 10},
        "contextual_llm": {"name": "contextual_llm", "base": "fixed", "size": 60, "overlap": 10},
    },
    "embedder": {
        "hashing": {"name": "hashing", "dim": 512},
        "tfidf": {"name": "tfidf"},
        "quantized_int8": {"name": "quantized", "base": "tfidf", "mode": "int8"},
        "quantized_binary": {"name": "quantized", "base": "tfidf", "mode": "binary"},
        "matryoshka_64": {"name": "matryoshka", "base": "tfidf", "dim": 64},
        "matryoshka_128": {"name": "matryoshka", "base": "tfidf", "dim": 128},
        "matryoshka_256": {"name": "matryoshka", "base": "tfidf", "dim": 256},
    },
    "index": {
        "flat": {"name": "flat"},
        "ivf": {"name": "ivf", "nlist": 8, "nprobe": 2},
        "ivf_nprobe4": {"name": "ivf", "nlist": 8, "nprobe": 4},
        "ivf_aggressive": {"name": "ivf", "nlist": 12, "nprobe": 1},
        "bm25": {"name": "bm25"},
        "hnsw": {"name": "hnsw"},
    },
    "retriever": {
        "dense": {"name": "dense"},
        "sparse": {"name": "sparse"},
        "hybrid": {"name": "hybrid", "fusion": "rrf"},
        "hybrid_weighted": {"name": "hybrid", "fusion": "weighted", "alpha": 0.5},
        "mmr": {"name": "mmr", "lam": 0.6},
    },
    "query_transformer": {
        "none": {"name": None},
        "prf": {"name": "prf"},
        "multiquery_prf": {"name": "multiquery_prf"},
    },
    "reranker": {
        "none": {"name": None},
        "lexical": {"name": "lexical"},
        "cross_encoder": {"name": "cross_encoder"},
        "llm": {"name": "llm"},
    },
    "assembler": {
        "concat": {"name": "concat"},
        "reorder": {"name": "reorder"},
        "dedup": {"name": "dedup"},
        "budget": {"name": "budget", "max_words": 120},
        "parent": {"name": "parent"},
    },
    "generator": {
        "extractive_mock": {"name": "extractive_mock"},
        "bare": {"name": "claude_prompted", "style": "bare"},
        "grounded": {"name": "claude_prompted", "style": "grounded"},
        "cite_forced": {"name": "claude_prompted", "style": "cite_forced"},
        "abstain": {"name": "claude_prompted", "style": "abstain"},
        "grounded_t07": {"name": "claude_prompted", "style": "grounded", "temperature": 0.7},
    },
}

_BASE = {
    "dataset": "builtin_docs",
    "retrieve_k": 10,
    "final_k": 5,
    "chunker": {"name": "fixed", "size": 60, "overlap": 10},
    "embedder": {"name": "tfidf"},          # Phase 3 winner — meaningful dense baseline
    "index": {"name": "flat"},
    "retriever": {"name": "dense"},
    "reranker": None,
    "query_transformer": None,
    "assembler": {"name": "concat"},
    "generator": {"name": "extractive_mock"},
}


def _run_one(cfg: dict[str, Any], label: str) -> dict[str, Any]:
    pipeline = cfgmod.build_pipeline(cfg)
    docs, queries = load_dataset(cfg["dataset"])
    pipeline.build(docs)
    results = [pipeline.run_query(q) for q in queries]
    metrics = scoring.score(results)
    row = {"config": label, "dataset": cfg["dataset"],
           "stages": {"chunker": cfg["chunker"]["name"], "embedder": cfg["embedder"]["name"],
                      "index": cfg["index"]["name"], "retriever": cfg["retriever"]["name"],
                      "reranker": (cfg["reranker"] or {}).get("name") if cfg["reranker"] else None,
                      "generator": cfg["generator"]["name"]},
           "build_stats": pipeline.build_stats, **metrics}
    leaderboard.append_row(row)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-lab-sweep")
    ap.add_argument("--dataset", default="builtin_docs")
    ap.add_argument("--vary", default="chunker", choices=list(_STAGE_DEFAULTS))
    ap.add_argument("--options", nargs="+", required=True)
    ap.add_argument("--phase", default="sweep", help="label prefix for leaderboard rows")
    # pin other stages, e.g. rerank on a deliberately WEAK first stage: --embedder hashing
    ap.add_argument("--embedder", help="override the base embedder (e.g. hashing, tfidf)")
    ap.add_argument("--chunker", help="override the base chunker (e.g. parent_child)")
    args = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    rows = []
    for opt in args.options:
        cfg = copy.deepcopy(_BASE)
        cfg["dataset"] = args.dataset
        if args.embedder:
            cfg["embedder"] = dict(_STAGE_DEFAULTS["embedder"].get(args.embedder, {"name": args.embedder}))
        if args.chunker:
            cfg["chunker"] = dict(_STAGE_DEFAULTS["chunker"].get(args.chunker, {"name": args.chunker}))
        component = _STAGE_DEFAULTS[args.vary].get(opt) or {"name": opt}
        cfg[args.vary] = None if component.get("name") is None else dict(component)
        row = _run_one(cfg, f"{args.phase}_{args.vary}={opt}")
        row["_option"] = opt
        rows.append(row)

    leaderboard.render()
    cols = ["recall@1", "recall@5", "mrr", "ndcg@10", "map", "token_f1",
            "latency_p50_ms"]
    print(f"\n=== sweep: {args.vary} on {args.dataset} ===")
    print(f"{'option':<16} " + " ".join(f"{c:>13}" for c in cols) + "   n_chunks")
    rows.sort(key=lambda r: -r["recall@5"])
    for r in rows:
        opt = r.get("_option", r["config"].split("=")[-1])
        vals = " ".join(f"{r.get(c, 0):>13.4f}" for c in cols)
        print(f"{opt:<16} {vals}   {r['build_stats']['n_chunks']}")
    print(f"\nLeaderboard → {RESULTS / 'leaderboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
