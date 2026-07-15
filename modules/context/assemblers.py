"""Context assembler implementations (Phase 8). `assemble(query, scored) -> str`.

All emit numbered, citable blocks `[n] (chunk_id) text` so citation metrics stay possible —
an assembler that drops provenance breaks Phase 11's citation accuracy by construction.
"""

from __future__ import annotations

import re

import numpy as np

from harness.contract import Query, Scored
from harness.registry import register

_WORD = re.compile(r"[a-z0-9]+")


def _fmt(scored: list[Scored]) -> str:
    return "\n\n".join(f"[{i}] ({s.chunk.chunk_id}) {s.chunk.text}" for i, s in enumerate(scored, 1))


@register("assembler", "reorder")
class LostInTheMiddleAssembler:
    """Mitigate the U-shaped attention curve (Liu et al. 2023): models attend to the START
    and END of context and miss the MIDDLE. So place the most relevant chunks at the edges
    and bury the weakest in the middle. Input is assumed sorted best-first."""

    name = "reorder"

    def assemble(self, query: Query, scored: list[Scored]) -> str:
        # best → front, 2nd best → back, 3rd → front+1, 4th → back-1, ...
        head: list[Scored] = []
        tail: list[Scored] = []
        for i, s in enumerate(scored):
            (head if i % 2 == 0 else tail).append(s)
        return _fmt(head + list(reversed(tail)))


@register("assembler", "dedup")
class DedupAssembler:
    """Drop near-duplicate retrieved chunks (cosine ≥ threshold to an already-kept chunk).
    Near-dups crowd the window and waste tokens without adding information."""

    name = "dedup"

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def assemble(self, query: Query, scored: list[Scored]) -> str:
        kept: list[Scored] = []
        for s in scored:
            if s.chunk.embedding is None:
                kept.append(s)
                continue
            dup = any(
                t.chunk.embedding is not None
                and float(np.dot(s.chunk.embedding, t.chunk.embedding)) >= self.threshold
                for t in kept
            )
            if not dup:
                kept.append(s)
        return _fmt(kept)


@register("assembler", "budget")
class TokenBudgetAssembler:
    """Pack chunks best-first until a word budget is hit — cost-aware assembly. Prevents the
    silent truncation you get when you blindly stuff k chunks into a fixed window."""

    name = "budget"

    def __init__(self, max_words: int = 120):
        self.max_words = max_words

    def assemble(self, query: Query, scored: list[Scored]) -> str:
        kept, used = [], 0
        for s in scored:
            w = len(s.chunk.text.split())
            if used + w > self.max_words and kept:
                break
            kept.append(s)
            used += w
        return _fmt(kept)


@register("assembler", "parent")
class ParentExpansionAssembler:
    """Small-to-big: retrieve the precise small chunk, but feed the LLM its parent paragraph
    (from `metadata['parent_text']`, set by the parent_child chunker). Precision in retrieval,
    context in generation. Falls back to the chunk text when there's no parent."""

    name = "parent"

    def assemble(self, query: Query, scored: list[Scored]) -> str:
        blocks, seen = [], set()
        for i, s in enumerate(scored, 1):
            text = s.chunk.metadata.get("parent_text", s.chunk.text)
            if text in seen:      # two children of the same parent → emit the parent once
                continue
            seen.add(text)
            blocks.append(f"[{i}] ({s.chunk.chunk_id}) {text}")
        return "\n\n".join(blocks)
