"""Chunker implementations (Phase 2). All deps-free; semantic uses the injected embedder.

Each yields `Chunk`s with ids `{doc_id}::{i}`. Retrieval is scored at doc level (the qrels
in builtin_docs are doc-level), so a chunker wins by ranking a passage from the right doc
higher / with less dilution — not by matching a chunk id.
"""

from __future__ import annotations

import re

import numpy as np

from harness.contract import Chunk, Document
from harness.registry import register

_SENT = re.compile(r"(?<=[.!?])\s+")


def _mk(doc: Document, idx: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"{doc.doc_id}::{idx}", doc_id=doc.doc_id, text=text, metadata=dict(doc.metadata))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


@register("chunker", "recursive")
class RecursiveChunker:
    """Split on the coarsest separator that keeps chunks under `size` words, recursing
    paragraph → sentence → word. Approximates LangChain's RecursiveCharacterTextSplitter."""

    name = "recursive"

    def __init__(self, size: int = 60, overlap: int = 10):
        self.size, self.overlap = size, overlap

    def _pack(self, units: list[str]) -> list[str]:
        out, cur, n = [], [], 0
        for u in units:
            w = len(u.split())
            if n + w > self.size and cur:
                out.append(" ".join(cur))
                # word overlap
                back = " ".join(out[-1].split()[-self.overlap :]) if self.overlap else ""
                cur, n = ([back] if back else []), (len(back.split()) if back else 0)
            cur.append(u)
            n += w
        if cur:
            out.append(" ".join(cur))
        return out

    def run(self, docs: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in docs:
            paras = [p for p in re.split(r"\n\s*\n", doc.text) if p.strip()] or [doc.text]
            units: list[str] = []
            for p in paras:
                units.extend(_sentences(p) or [p])
            for i, t in enumerate(self._pack(units)):
                chunks.append(_mk(doc, i, t))
        return chunks


@register("chunker", "sentence")
class SentenceChunker:
    """One chunk per N sentences with sentence overlap."""

    name = "sentence"

    def __init__(self, per_chunk: int = 2, overlap: int = 1):
        if overlap >= per_chunk:
            raise ValueError("overlap must be < per_chunk")
        self.per_chunk, self.overlap = per_chunk, overlap

    def run(self, docs: list[Document]) -> list[Chunk]:
        chunks = []
        step = self.per_chunk - self.overlap
        for doc in docs:
            sents = _sentences(doc.text)
            i = idx = 0
            while i < len(sents):
                chunks.append(_mk(doc, idx, " ".join(sents[i : i + self.per_chunk])))
                idx += 1
                if i + self.per_chunk >= len(sents):
                    break
                i += step
        return chunks


@register("chunker", "paragraph")
class ParagraphChunker:
    """One chunk per paragraph (blank-line separated); falls back to whole doc."""

    name = "paragraph"

    def run(self, docs: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in docs:
            paras = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()] or [doc.text]
            for i, p in enumerate(paras):
                chunks.append(_mk(doc, i, p))
        return chunks


@register("chunker", "structural")
class StructuralChunker:
    """Layout-aware: split on markdown-style headings if present, else paragraphs.
    Real corpora carry section structure; here it degrades to paragraph on plain text."""

    name = "structural"

    def run(self, docs: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in docs:
            parts = re.split(r"\n(?=#{1,6}\s)", doc.text)
            if len(parts) == 1:
                parts = [p for p in re.split(r"\n\s*\n", doc.text) if p.strip()] or [doc.text]
            for i, p in enumerate(parts):
                chunks.append(_mk(doc, i, p.strip()))
        return chunks


@register("chunker", "semantic")
class SemanticChunker:
    """Embedding-similarity boundaries: start a new chunk when consecutive sentences'
    embeddings drop below a similarity threshold. Uses the pipeline's embedder."""

    name = "semantic"

    def __init__(self, embedder_name: str = "hashing", threshold: float = 0.3, max_sents: int = 6):
        from harness.registry import build

        self._emb = build("embedder", embedder_name)
        self.threshold, self.max_sents = threshold, max_sents

    def run(self, docs: list[Document]) -> list[Chunk]:
        from harness.contract import Query

        chunks = []
        for doc in docs:
            sents = _sentences(doc.text)
            if not sents:
                continue
            # embed each sentence (reuse the embedder's query path for a single string)
            vecs = [self._emb.encode_query(Query(query_id="_", text=s)).embedding for s in sents]
            groups, cur = [], [0]
            for i in range(1, len(sents)):
                sim = float(np.dot(vecs[i], vecs[i - 1]))
                if sim < self.threshold or len(cur) >= self.max_sents:
                    groups.append(cur)
                    cur = [i]
                else:
                    cur.append(i)
            groups.append(cur)
            for idx, g in enumerate(groups):
                chunks.append(_mk(doc, idx, " ".join(sents[j] for j in g)))
        return chunks


@register("chunker", "parent_child")
class ParentChildChunker:
    """Small-to-big: index small (sentence) chunks but carry the parent paragraph in
    metadata so assembly can feed the bigger context. Here we emit the small chunks for
    retrieval; `metadata['parent_text']` holds the paragraph for Phase 8 to expand."""

    name = "parent_child"

    def __init__(self, child_sents: int = 1):
        self.child_sents = child_sents

    def run(self, docs: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in docs:
            paras = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()] or [doc.text]
            idx = 0
            for p in paras:
                sents = _sentences(p) or [p]
                for i in range(0, len(sents), self.child_sents):
                    child = " ".join(sents[i : i + self.child_sents])
                    c = _mk(doc, idx, child)
                    c.metadata["parent_text"] = p
                    chunks.append(c)
                    idx += 1
        return chunks
