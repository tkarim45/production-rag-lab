"""Query understanding (Phase 6).

Registers query transformers that run before retrieval:
- `prf` — pseudo-relevance feedback: retrieve once, harvest top terms from the top docs,
  append them to the query, retrieve again. Deterministic, key-free, genuinely helps recall
  on vocabulary-mismatch queries.
- `multiquery_prf` — return [original, PRF-expanded] so the pipeline RRF-fuses both.
- `hyde` / `multiquery_llm` — real LLM transforms via Claude/Bedrock (optional). Generate a
  hypothetical answer (HyDE) or N paraphrases, retrieve for each, fuse.

The honest finding this phase exposes: query-transform helps single-hop recall but does
little for multi-hop (you need a bridge doc, not a rephrase).
"""

from modules.query import transformers  # noqa: F401
