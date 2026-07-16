"""A/B testing two RAG configs (Phase 13) — the delta, and whether you may believe it.

Every earlier phase in this repo produced a leaderboard: config A scored 0.95, config B
scored 0.90, A wins. **That inference is not licensed by the data.** A single run on 20
queries gives one number per arm with no uncertainty attached; the delta could be the
technique or it could be four queries that happened to break the right way.

This module is the correction. It runs both configs on the **same eval set**, extracts the
per-query metric vectors, and computes a **paired bootstrap 95% CI** on the delta:

    for b in 1..B:  resample query indices with replacement (the SAME indices for both arms)
                    delta_b = mean(B[idx]) − mean(A[idx])
    CI = the 2.5th and 97.5th percentiles of {delta_b}

Pairing is the whole trick: both arms answer the identical query, so resampling *queries*
(not arms independently) cancels the "is this query easy?" variance and leaves only the
"is this config better?" variance. It's from scratch — numpy's `default_rng` and a
percentile, ~10 lines — because the interesting part is what the interval *means*.

**If the CI contains zero, you have not measured a winner.** That is the honest core of this
phase, and on a 20-query set it is the usual outcome.

CLI:
    python -m modules.ops.ab --vary embedder --a tfidf --b hashing \\
        --metrics recall@1 recall@5 mrr ndcg@10 token_f1
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np

from modules.ops import runs

DEFAULT_METRICS = ("recall@1", "recall@5", "mrr", "ndcg@10", "token_f1")


def bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    paired: bool = True,
) -> dict[str, Any]:
    """Percentile bootstrap CI on mean(B) − mean(A).

    `paired=True` (default, correct for a shared eval set) resamples query indices once per
    replicate and applies them to both arms. `paired=False` resamples each arm independently
    — the right choice only when the two arms saw different queries, and always wider.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("both arms need at least one value")
    if paired and a.size != b.size:
        raise ValueError(f"paired bootstrap needs equal-length arms: {a.size} vs {b.size}")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    rng = np.random.default_rng(seed)
    if paired:
        idx = rng.integers(0, a.size, size=(n_boot, a.size))
        deltas = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    else:
        ia = rng.integers(0, a.size, size=(n_boot, a.size))
        ib = rng.integers(0, b.size, size=(n_boot, b.size))
        deltas = b[ib].mean(axis=1) - a[ia].mean(axis=1)

    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    delta = float(b.mean() - a.mean())
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": delta,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n": int(a.size),
        "n_boot": int(n_boot),
        "paired": bool(paired),
        "confidence": 1 - alpha,
    }


def compare_results(results_a, results_b, metrics: Sequence[str] = DEFAULT_METRICS,
                    n_boot: int = 10_000, seed: int = 0) -> dict[str, dict[str, Any]]:
    """Bootstrap every metric from two already-run result sets (no re-running)."""
    out: dict[str, dict[str, Any]] = {}
    for m in metrics:
        va, vb = runs.per_query(results_a, m), runs.per_query(results_b, m)
        out[m] = bootstrap_ci(va, vb, n_boot=n_boot, seed=seed)
    return out


def ab_test(
    stage: str,
    option_a: str,
    option_b: str,
    dataset: str = "builtin_docs",
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_boot: int = 10_000,
    seed: int = 0,
    pin: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run two configs (one stage swapped) on the same eval set and bootstrap every metric.

    `pin` fixes the other stages (e.g. `{"embedder": "hashing"}`), so an A/B can reproduce
    the exact condition an earlier phase reported a win under — and re-check it.
    """
    results_a = runs.run_results(runs.variant(stage, option_a, dataset, pin=pin))
    results_b = runs.run_results(runs.variant(stage, option_b, dataset, pin=pin))
    return {
        "stage": stage,
        "a": option_a,
        "b": option_b,
        "dataset": dataset,
        "pin": pin or {},
        "n_queries": len(results_a),
        "metrics": compare_results(results_a, results_b, metrics, n_boot=n_boot, seed=seed),
    }


def render(report: dict[str, Any]) -> str:
    r = report
    pin = "".join(f", {k}={v}" for k, v in (r.get("pin") or {}).items())
    lines = [
        f"\n=== A/B: {r['stage']} — A={r['a']} vs B={r['b']} "
        f"({r['dataset']}, n={r['n_queries']} queries{pin}) ===",
        f"{'metric':<12} {'A':>8} {'B':>8} {'delta':>9} {'95% CI':>20}  verdict",
    ]
    for name, s in r["metrics"].items():
        ci = f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"
        call = "B wins" if s["excludes_zero"] and s["delta"] > 0 else (
            "A wins" if s["excludes_zero"] else "inconclusive")
        lines.append(f"{name:<12} {s['mean_a']:>8.3f} {s['mean_b']:>8.3f} "
                     f"{s['delta']:>+9.3f} {ci:>20}  {call}")
    n_boot = next(iter(r["metrics"].values()))["n_boot"]
    lines.append(f"\npaired percentile bootstrap, B={n_boot:,} resamples. "
                 f"'inconclusive' = the 95% CI contains 0.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-lab-ab", description="A/B two RAG configs with a bootstrap CI.")
    ap.add_argument("--vary", default="embedder", help="stage to swap (embedder, chunker, retriever, …)")
    ap.add_argument("--a", required=True, help="control option label (e.g. tfidf)")
    ap.add_argument("--b", required=True, help="treatment option label (e.g. hashing)")
    ap.add_argument("--dataset", default="builtin_docs")
    ap.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    ap.add_argument("--pin", nargs="+", metavar="STAGE=OPTION", default=[],
                    help="fix other stages, e.g. --pin embedder=hashing")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    pin: dict[str, str] = {}
    for item in args.pin:
        if "=" not in item:
            ap.error(f"--pin expects STAGE=OPTION, got {item!r}")
        stage, _, option = item.partition("=")
        pin[stage.strip()] = option.strip()

    report = ab_test(args.vary, args.a, args.b, dataset=args.dataset, metrics=args.metrics,
                     n_boot=args.n_boot, seed=args.seed, pin=pin)
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
