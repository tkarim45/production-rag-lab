"""Serving & performance (Phase 12).

The layer between "the pipeline answers correctly" and "the pipeline answers correctly under
load, at a price you can pay". Registers two cache wrappers over the existing contract —

- `cached` (embedder): a disk cache in front of any embedder, keyed by sha256(text) +
  embedder identity (name + params + dim). Kills the rebuild cost of re-embedding an
  unchanged corpus. Refuses corpus-fit embedders (tfidf) because their vectors are NOT a
  pure function of the text — the honest limit of the technique.
- `semantic_cached` (generator): serve a stored answer when a new query's cosine to a cached
  one is ≥ threshold. Cheap and fast — and wrong whenever the threshold lets a *different*
  question through (a false hit).

Plus `harness.latency_report` (per-stage p50/p95 + share of the latency budget),
`harness.cache_report` (the Phase 12 benchmark), and an optional FastAPI app in `api.py`
(degrades cleanly when fastapi is absent — core stays numpy + pyyaml).
"""

from modules.serving import embedders  # noqa: F401
from modules.serving import generators  # noqa: F401
