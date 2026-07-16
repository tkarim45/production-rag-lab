"""CI regression gate (Phase 13) — golden thresholds that fail the build.

Every other artifact in this repo *informs* a human. This one **blocks a merge**. A RAG
pipeline has no compiler and no type error for "recall quietly dropped 12 points because
someone retuned the chunker": the only thing standing between that commit and production is
a job that runs the eval set and exits nonzero.

Design decisions worth defending:
- **Two directions.** `min` gates quality (recall, MRR, F1 — regressions fall). `max` gates
  cost and latency (regressions rise). A gate with only `min` merges a change that keeps
  recall and triples the bill.
- **No leaderboard write.** The gate must be runnable on every PR, including red ones.
  Appending a leaderboard row per CI build would rewrite the phase history with noise.
- **Thresholds are the measured baseline minus a margin**, checked into
  `configs/golden_gate.yaml`. A threshold set at the exact measured value fails on the first
  reordering of a tie; a threshold set at 0 never fails at all.

The honest caveat this phase must state: on a 20-query eval set, one query flipping moves
recall@5 by 0.05, which is why the margin exists — and why a gate this small catches
*breakage*, not *regression*. Distinguishing a real 2-point regression from noise needs the
CI in `ab.py`, and a CI that tight needs hundreds of queries.

CLI (see `harness/gate.py`):
    python -m harness.gate configs/naive.yaml --golden configs/golden_gate.yaml
    python -m harness.gate configs/naive.yaml --min recall@5=0.9 --max latency_p50_ms=50
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from modules.ops import runs

# exit codes — the gate's actual API surface in CI
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


@dataclass(frozen=True)
class Check:
    metric: str
    direction: str        # "min" (value must be >=) | "max" (value must be <=)
    threshold: float
    value: float
    passed: bool

    def render(self) -> str:
        op = ">=" if self.direction == "min" else "<="
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.metric:<22} {self.value:>10.4f} {op} {self.threshold:<10.4f}"


@dataclass
class GateReport:
    config: str
    dataset: str
    checks: list[Check]
    metrics: dict[str, Any]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def exit_code(self) -> int:
        return EXIT_PASS if self.passed else EXIT_FAIL

    def render(self) -> str:
        head = f"\n=== CI gate: {self.config} ({self.dataset}, n={self.metrics['n_queries']}) ==="
        body = [c.render() for c in self.checks]
        failed = [c for c in self.checks if not c.passed]
        if self.passed:
            tail = f"\nGREEN — {len(self.checks)}/{len(self.checks)} checks passed. exit 0"
        else:
            names = ", ".join(c.metric for c in failed)
            tail = (f"\nRED — {len(failed)}/{len(self.checks)} check(s) failed ({names}). "
                    f"Regression gate blocks the merge. exit 1")
        return "\n".join([head, *body, tail])


def load_golden(path: str | Path) -> dict[str, Any]:
    """Read a golden-threshold YAML: {config?, dataset?, min: {...}, max: {...}}."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"golden file {path} must be a mapping")
    data.setdefault("min", {})
    data.setdefault("max", {})
    return data


def check_metrics(metrics: dict[str, Any], mins: dict[str, float],
                  maxes: dict[str, float]) -> list[Check]:
    """Compare an already-computed metrics dict against thresholds. Pure — trivially testable."""
    checks: list[Check] = []
    for metric, threshold in sorted(mins.items()):
        if metric not in metrics:
            raise KeyError(f"gate references unknown metric {metric!r}; "
                           f"available: {sorted(k for k in metrics if not k.startswith('_'))}")
        value = float(metrics[metric])
        checks.append(Check(metric, "min", float(threshold), value, value >= threshold))
    for metric, threshold in sorted(maxes.items()):
        if metric not in metrics:
            raise KeyError(f"gate references unknown metric {metric!r}")
        value = float(metrics[metric])
        checks.append(Check(metric, "max", float(threshold), value, value <= threshold))
    if not checks:
        raise ValueError("gate has no thresholds — a gate that checks nothing always passes")
    return checks


def run_gate(config_path: str | Path, mins: dict[str, float] | None = None,
             maxes: dict[str, float] | None = None, dataset: str | None = None) -> GateReport:
    """Run a pipeline config on its eval set and check the metrics against the golden bar."""
    from harness import config as cfgmod

    cfg = cfgmod.load_config(config_path)
    if dataset:
        cfg["dataset"] = dataset
    metrics, _ = runs.run_metrics(cfg)
    return GateReport(
        config=str(config_path),
        dataset=cfg["dataset"],
        checks=check_metrics(metrics, mins or {}, maxes or {}),
        metrics=metrics,
    )
