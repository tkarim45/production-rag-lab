"""CLI: where does the latency budget actually go?

    python -m harness.latency_report configs/naive.yaml

Runs a config over its dataset, reads `PipelineResult.stage_latency_ms` off every query, and
prints the per-stage p50/p95 plus each stage's share of the mean end-to-end budget. Writes
results/latency_<config>.json.

The point of the report is triage. "Optimize the pipeline" is not actionable; "one stage eats
N% of p50" is. A stage worth 0.3% of the budget cannot be optimized into a win no matter how
clever the fix — the breakdown is what stops you from tuning the wrong thing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import modules.serving  # noqa: F401  registers the Phase 12 cache wrappers
from harness import config as cfgmod
from harness.contract import PipelineResult
from harness.data import load_dataset
from harness.metrics import efficiency as E
from harness.runner import RESULTS

_TOTAL = "total_ms"


def stage_breakdown(results: list[PipelineResult]) -> dict[str, Any]:
    """Per-stage p50/p95/mean + share of the mean end-to-end budget.

    Stages appear in pipeline order (the order the Pipeline recorded them). `pct_of_budget`
    is mean(stage) / mean(total) — means, not percentiles, because percentiles don't add up:
    p95(retrieve) + p95(generate) != p95(total), and a "budget share" must sum to ~100%.
    """
    if not results:
        raise ValueError("no results to report")

    order: list[str] = []
    for r in results:
        for k in r.stage_latency_ms:
            if k != _TOTAL and k not in order:
                order.append(k)

    totals = [r.stage_latency_ms.get(_TOTAL, 0.0) for r in results]
    mean_total = sum(totals) / len(totals) if totals else 0.0

    stages: dict[str, dict[str, float]] = {}
    for name in order:
        vals = [r.stage_latency_ms[name] for r in results if name in r.stage_latency_ms]
        mean = sum(vals) / len(vals) if vals else 0.0
        stages[name] = {
            "p50_ms": E.percentile(vals, 50),
            "p95_ms": E.percentile(vals, 95),
            "mean_ms": mean,
            "pct_of_budget": (mean / mean_total * 100) if mean_total > 0 else 0.0,
            "n": len(vals),
        }

    return {
        "n_queries": len(results),
        "stages": stages,
        "total": {
            "p50_ms": E.percentile(totals, 50),
            "p95_ms": E.percentile(totals, 95),
            "mean_ms": mean_total,
        },
    }


def _print_report(label: str, report: dict[str, Any]) -> None:
    print(f"\n=== latency budget: {label}  (n={report['n_queries']}) ===")
    print(f"{'stage':<18}{'p50 ms':>12}{'p95 ms':>12}{'mean ms':>12}{'% of budget':>14}")
    for name, s in report["stages"].items():
        print(f"{name:<18}{s['p50_ms']:>12.2f}{s['p95_ms']:>12.2f}"
              f"{s['mean_ms']:>12.2f}{s['pct_of_budget']:>13.1f}%")
    t = report["total"]
    print(f"{'-' * 68}")
    print(f"{'total':<18}{t['p50_ms']:>12.2f}{t['p95_ms']:>12.2f}{t['mean_ms']:>12.2f}"
          f"{100.0:>13.1f}%")


def report_config(path: str | Path) -> dict[str, Any]:
    cfg, pipeline = cfgmod.load(path)
    docs, queries = load_dataset(cfg["dataset"])
    pipeline.build(docs)
    results = [pipeline.run_query(q) for q in queries]

    report = stage_breakdown(results)
    report["config"] = Path(path).stem
    report["dataset"] = cfg["dataset"]
    report["build_stats"] = pipeline.build_stats
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rag-lab-latency", description="Per-stage latency budget for a pipeline config."
    )
    ap.add_argument("config", nargs="?", default="configs/naive.yaml")
    args = ap.parse_args(argv)

    if not Path(args.config).exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 2

    report = report_config(args.config)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"latency_{report['config']}.json"
    out.write_text(json.dumps(report, indent=2))
    _print_report(report["config"], report)
    print(f"\nReport → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
