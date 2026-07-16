"""Modules — one package per RAG layer; each registers swappable stage implementations.

Phase 0 ships `baseline` (the naive end-to-end pipeline). Later phases add sibling
packages (chunking, embeddings, indexing, retrieval, query, reranking, context, flows, …),
each registering more implementations of the same `harness.contract` interfaces.

Importing this package imports every module so its `@register(...)` decorators run and the
component registry is populated before a config is loaded.
"""

from modules import baseline  # noqa: F401
from modules import chunking  # noqa: F401
from modules import context  # noqa: F401
from modules import embeddings  # noqa: F401
from modules import evaluation  # noqa: F401
from modules import flows  # noqa: F401
from modules import generation  # noqa: F401
from modules import indexing  # noqa: F401
from modules import query  # noqa: F401
from modules import reranking  # noqa: F401
from modules import retrieval  # noqa: F401
