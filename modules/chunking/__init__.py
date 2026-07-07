"""Chunking (Phase 2) — the chunker family, benchmarked on one corpus.

Registers: recursive, sentence, paragraph, semantic, structural, parent-child (small-to-big).
The Phase 0 `fixed` chunker lives in `modules/baseline`. All share the `Chunker` interface,
so the harness swaps them via config and the sweep reads the retrieval/storage tradeoff off
the shared leaderboard.
"""

from modules.chunking import chunkers  # noqa: F401
