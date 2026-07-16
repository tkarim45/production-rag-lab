"""Significance testing (Phase 11) — from-scratch bootstrap CI.

The reporting rule from docs/03-metrics-catalog.md: **don't crown a winner on a 1-run delta.**
With 20 queries, a +0.05 mean difference is very likely noise. The bootstrap answers "if I
resampled my eval set, would config A still beat config B?" without assuming normality.

Paired bootstrap: resample the same query indices for both configs (they're evaluated on the
same questions), so the CI reflects per-query differences, not two independent samples.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap CI for a list of per-item scores."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = arr[rng.randint(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def paired_delta_ci(
    a: list[float], b: list[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Paired bootstrap CI for (mean(a) − mean(b)) over the SAME items.

    Returns mean_delta, lo, hi, and `significant` = 1.0 when the CI excludes 0.
    """
    if len(a) != len(b):
        raise ValueError("paired bootstrap needs equal-length, aligned score lists")
    if not a:
        return {"mean_delta": 0.0, "lo": 0.0, "hi": 0.0, "significant": 0.0}
    rng = np.random.RandomState(seed)
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    means = diff[rng.randint(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_delta": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "significant": 1.0 if (lo > 0 or hi < 0) else 0.0,
    }
