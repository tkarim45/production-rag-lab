"""Chunkers. Phase 0 ships the fixed-size splitter; Phase 2 adds the other 7 + the sweep."""

from __future__ import annotations

from harness.contract import Chunk, Document
from harness.registry import register


@register("chunker", "fixed")
class FixedSizeChunker:
    """Fixed-size word-window chunker with overlap.

    Defaults (`size` large enough to keep each built-in doc as one chunk) give chunk ids
    of the form `{doc_id}::{i}`, which is exactly what the built-in qrels reference. Phase 2
    will sweep size × overlap and show the retrieval/storage tradeoff.
    """

    name = "fixed"

    def __init__(self, size: int = 256, overlap: int = 0):
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= overlap < size:
            raise ValueError("overlap must be in [0, size)")
        self.size = size
        self.overlap = overlap

    def run(self, docs: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.size - self.overlap
        for doc in docs:
            words = doc.text.split()
            if not words:
                continue
            i = 0
            idx = 0
            while i < len(words):
                window = words[i : i + self.size]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::{idx}",
                        doc_id=doc.doc_id,
                        text=" ".join(window),
                        metadata=dict(doc.metadata),
                    )
                )
                idx += 1
                if i + self.size >= len(words):
                    break
                i += step
        return chunks
