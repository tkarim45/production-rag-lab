"""Observability & ops (Phase 13).

Phases 0–10 answered "which config scores best?". This phase answers the two questions that
come *after* you ship it: **is it still working, and how would I know?**

Nothing here registers a pipeline component — ops is not a stage, it wraps the whole thing.
That's why this package is imported explicitly by its tests and CLIs rather than through the
registry.

Exports:
- `tracing` — per-request traces/spans/retrieval-sets to SQLite (`TracedPipeline`, `Tracer`).
- `drift` — from-scratch PSI (0.1 / 0.25 bands) over query embeddings and retrieval quality.
- `ab` — paired bootstrap 95% CI on a metric delta between two configs. The honest core:
  a one-run leaderboard delta is not a winner until the interval clears zero.
- `gate` — golden-threshold CI regression gate; `python -m harness.gate` exits nonzero.
- `runs` — shared "run a config / extract the per-query metric vector" plumbing.
"""

from modules.ops import ab, drift, gate, runs, tracing  # noqa: F401
