"""CI regression gate CLI — the merge blocker.

    python -m harness.gate configs/naive.yaml --golden configs/golden_gate.yaml
    python -m harness.gate configs/naive.yaml --min recall@5=0.9 --max latency_p50_ms=50

Exit codes are the contract (this is a CI job, not a report):
    0  green — every threshold met
    1  red   — at least one regression; block the merge
    2  usage — bad config path / bad threshold syntax / no thresholds

Lives in `harness/` because it's an entry point like `harness.runner` and `harness.sweep`;
the logic it calls lives in `modules/ops/gate.py` with the rest of Phase 13.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from modules.ops.gate import EXIT_USAGE, load_golden, run_gate


def _parse_thresholds(pairs: list[str] | None) -> dict[str, float]:
    """`recall@5=0.9` → {"recall@5": 0.9}. Bad syntax is a usage error, not a silent skip."""
    out: dict[str, float] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"expected metric=value, got {pair!r}")
        metric, _, raw = pair.partition("=")
        try:
            out[metric.strip()] = float(raw)
        except ValueError:
            raise ValueError(f"threshold for {metric!r} is not a number: {raw!r}") from None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rag-lab-gate",
        description="Run a config against golden thresholds; exit nonzero on regression.",
    )
    ap.add_argument("config", nargs="?", default="configs/naive.yaml", help="pipeline YAML")
    ap.add_argument("--golden", help="YAML of golden thresholds ({min: {...}, max: {...}})")
    ap.add_argument("--min", dest="mins", nargs="+", metavar="METRIC=VALUE",
                    help="quality floors, e.g. recall@5=0.9 mrr=0.8")
    ap.add_argument("--max", dest="maxes", nargs="+", metavar="METRIC=VALUE",
                    help="cost/latency ceilings, e.g. latency_p50_ms=50")
    ap.add_argument("--dataset", help="override the config's dataset")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    config = args.config
    mins: dict[str, float] = {}
    maxes: dict[str, float] = {}
    dataset = args.dataset

    try:
        if args.golden:
            if not Path(args.golden).exists():
                print(f"golden file not found: {args.golden}", file=sys.stderr)
                return EXIT_USAGE
            golden = load_golden(args.golden)
            config = golden.get("config", config) if args.config == "configs/naive.yaml" else config
            dataset = dataset or golden.get("dataset")
            mins.update(golden["min"])
            maxes.update(golden["max"])
        # explicit flags win over the golden file (that's how you locally tighten a bar)
        mins.update(_parse_thresholds(args.mins))
        maxes.update(_parse_thresholds(args.maxes))
    except ValueError as e:
        print(f"usage error: {e}", file=sys.stderr)
        return EXIT_USAGE

    if not Path(config).exists():
        print(f"config not found: {config}", file=sys.stderr)
        return EXIT_USAGE
    if not mins and not maxes:
        print("usage error: no thresholds given (--golden or --min/--max). "
              "A gate that checks nothing always passes.", file=sys.stderr)
        return EXIT_USAGE

    try:
        report = run_gate(config, mins=mins, maxes=maxes, dataset=dataset)
    except (KeyError, ValueError) as e:
        print(f"usage error: {e}", file=sys.stderr)
        return EXIT_USAGE

    if not args.quiet:
        print(report.render())
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
