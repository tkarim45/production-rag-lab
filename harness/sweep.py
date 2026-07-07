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

# sensible defaults so a bare option name (e.g. "sentence") gets reasonable params
_STAGE_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "chunker": {
        "fixed": {"size": 60, "overlap": 10},
        "recursive": {"size": 60, "overlap": 10},
        "sentence": {"per_chunk": 2, "overlap": 1},
        "paragraph": {},
        "structural": {},
        "semantic": {"threshold": 0.25},
        "parent_child": {},
    },
}

_BASE = {
    "dataset": "builtin_docs",
    "retrieve_k": 10,
    "final_k": 5,
    "chunker": {"name": "fixed", "size": 60, "overlap": 10},
    "embedder": {"name": "hashing", "dim": 512},
    "index": {"name": "flat"},
    "retriever": {"name": "dense"},
    "reranker": None,
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
    args = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    rows = []
    for opt in args.options:
        cfg = copy.deepcopy(_BASE)
        cfg["dataset"] = args.dataset
        params = _STAGE_DEFAULTS[args.vary].get(opt, {})
        cfg[args.vary] = {"name": opt, **params}
        row = _run_one(cfg, f"{args.phase}_{args.vary}={opt}")
        rows.append(row)

    leaderboard.render()
    cols = ["recall@1", "recall@5", "mrr", "ndcg@10", "map", "token_f1",
            "latency_p50_ms"]
    print(f"\n=== sweep: {args.vary} on {args.dataset} ===")
    print(f"{'option':<16} " + " ".join(f"{c:>13}" for c in cols) + "   n_chunks")
    rows.sort(key=lambda r: -r["recall@5"])
    for r in rows:
        opt = r["config"].split("=")[-1]
        vals = " ".join(f"{r.get(c, 0):>13.4f}" for c in cols)
        print(f"{opt:<16} {vals}   {r['build_stats']['n_chunks']}")
    print(f"\nLeaderboard → {RESULTS / 'leaderboard.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
