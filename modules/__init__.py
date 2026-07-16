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
# Phase 12/14 register stages too (embedder=cached, generator=semantic_cached,
# retriever=acl, assembler=screened|spotlight). Importing only makes them *nameable* in a
# config — nothing is applied unless a config asks for it, so the opt-in property holds.
# `modules.ops` (Phase 13) is deliberately absent: it registers no stage, it wraps the
# pipeline (see harness/gate.py, modules/ops/).
from modules import security  # noqa: F401
from modules import serving  # noqa: F401
