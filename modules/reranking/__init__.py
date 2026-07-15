"""Reranking (Phase 7).

Second-stage precision: re-score the first stage's candidates with a stronger (slower)
model. Registers `lexical` (deps-free BM25-style pointwise scorer), `cross_encoder`
(sentence-transformers, optional), and `llm` (Claude/Bedrock pointwise, optional).

The lab's Phase 7 benchmark deliberately uses a WEAK first stage (the `hashing` embedder,
recall@1 0.55) — a reranker can only prove its worth when first-stage recall is high but
precision is low. On the saturated tfidf stage there is nothing left to fix.
"""

from modules.reranking import rerankers  # noqa: F401
