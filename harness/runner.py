"""Runner — the one command that ties Phase 0 together.

    python -m harness.runner configs/naive.yaml

Loads a config → builds the pipeline → loads the dataset → builds the index once → runs
every query → scores → writes results/<config>.json → appends the leaderboard row → renders
the leaderboard. This is the seam every later phase reuses: a phase is "a grid of configs
run through this."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from harness import config as cfgmod
from harness import leaderboard, scoring
from harness.data import load_dataset

RESULTS = Path(__file__).resolve().parent.parent / "results"


def run_config(path: str | Path) -> dict[str, Any]:
    cfg, pipeline = cfgmod.load(path)
    docs, queries = load_dataset(cfg["dataset"])

    pipeline.build(docs)
    results = [pipeline.run_query(q) for q in queries]
    metrics = scoring.score(results)

    config_name = Path(path).stem
    row: dict[str, Any] = {
        "config": config_name,
        "dataset": cfg["dataset"],
        "stages": {
            k: (cfg.get(k) or {}).get("name") if isinstance(cfg.get(k), dict) else None
            for k in ("chunker", "embedder", "index", "retriever", "reranker", "assembler", "generator")
        },
        "build_stats": pipeline.build_stats,
        **metrics,
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{config_name}.json").write_text(json.dumps(row, indent=2))
    leaderboard.append_row(row)
    leaderboard.render()
    return row


def _print_summary(row: dict[str, Any]) -> None:
    print(f"\n=== {row['config']}  (dataset: {row['dataset']}, n={row['n_queries']}) ===")
    for key in ("recall@1", "recall@5", "mrr", "ndcg@10", "map", "em", "token_f1",
                "token_f1_single", "token_f1_multi", "latency_p50_ms", "cost_per_query_usd"):
        if key in row:
            v = row[key]
            print(f"  {key:<22} {v:.4f}" if isinstance(v, float) else f"  {key:<22} {v}")
    print(f"\nLeaderboard → {RESULTS / 'leaderboard.md'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-lab", description="Run a RAG pipeline config through the harness.")
    ap.add_argument("config", nargs="?", default="configs/naive.yaml", help="path to a pipeline YAML")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not Path(args.config).exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    row = run_config(args.config)
    if not args.quiet:
        _print_summary(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
