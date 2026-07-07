"""Baseline module — the naive end-to-end RAG the whole leaderboard is measured against.

Registers deps-free defaults so Phase 0 runs on numpy alone:
  chunker=fixed · embedder=hashing · index=flat · retriever=dense · assembler=concat ·
  generator=extractive_mock (+ optional generator=claude via Bedrock).

Later phases replace any one of these with a better implementation and read the delta off
the shared leaderboard.
"""

from modules.baseline import (  # noqa: F401
    assemblers,
    chunkers,
    embedders,
    generators,
    indexes,
    retrievers,
)
