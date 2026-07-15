"""Advanced retrieval flows (Phase 9).

Ships **Contextual Retrieval** (Anthropic, Sept 2024): prepend situating context to each
chunk *before* embedding/indexing, so an isolated chunk stops being ambiguous. Two variants:
- `contextual` — deps-free: prefix the doc title/section (the structural context you already
  have from ingestion metadata).
- `contextual_llm` — the real thing: Claude writes a one-line situating blurb per chunk
  (optional, needs creds; the paper's method).

Implemented as *chunker wrappers* — the context is baked into chunk.text before the embedder
sees it, which is exactly where the paper puts it. Iterative/agentic flows (Self-RAG, CRAG,
Adaptive) are deferred: they need an LLM control loop + a harder corpus to show a difference.
"""

from modules.flows import contextual  # noqa: F401
