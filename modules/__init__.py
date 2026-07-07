"""Modules — one package per RAG layer; each registers swappable stage implementations.

Phase 0 ships `baseline` (the naive end-to-end pipeline). Later phases add sibling
packages (chunking, embeddings, indexing, retrieval, query, reranking, context, flows, …),
each registering more implementations of the same `harness.contract` interfaces.

Importing this package imports every module so its `@register(...)` decorators run and the
component registry is populated before a config is loaded.
"""

from modules import baseline  # noqa: F401
