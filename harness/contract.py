"""The pipeline contract — the fixed interfaces every stage implements.

A production RAG pipeline is a chain of swappable stages:

    ingest → chunk → embed → index → retrieve → rerank → assemble → generate

Each stage is a small class with a `run(...)` method and a fixed I/O type, so any
implementation of a stage (8 chunkers, 5 indexes, …) is interchangeable and benchmarkable
on the same harness. Later phases add new *implementations* of these interfaces — never new
interfaces (unless a whole new layer is genuinely needed, added here first).

Design choices:
- Plain dataclasses, not a framework — cheap to read, cheap to serialize.
- Embeddings are carried on the objects (numpy arrays) so stages stay stateless-ish and
  the runner can cache them.
- Everything is typed by intent, not by a heavy schema lib, to keep Phase 0 dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

# ── Data objects that flow through the pipeline ───────────────────────────────


@dataclass
class Document:
    """A source document after ingestion/parsing."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit produced by a chunker.

    `chunk_id` is globally unique and is what qrels/relevance judgments reference — the
    chunk id IS the citation key downstream.
    """

    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass
class Query:
    """An evaluation query with optional gold labels for scoring."""

    query_id: str
    text: str
    # gold labels (present in eval sets, absent at serving time)
    gold_answer: str | None = None
    relevant_chunk_ids: set[str] = field(default_factory=set)
    # doc-level relevance (used when the corpus has multi-chunk docs and qrels are at doc
    # granularity — decouples relevance judgments from the chunker under test).
    relevant_doc_ids: set[str] = field(default_factory=set)
    hop_type: str = "single"  # "single" | "multi" — split metrics by this
    embedding: np.ndarray | None = None


@dataclass
class Scored:
    """A chunk with a retrieval/rerank score."""

    chunk: Chunk
    score: float


@dataclass
class PipelineResult:
    """Everything one query produced — fed to the metrics layer."""

    query: Query
    retrieved: list[Scored]          # after retrieve (+ rerank if present)
    context: str                     # assembled context string
    answer: str                      # generated answer
    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)   # e.g. {"in": .., "out": ..}
    cost_usd: float = 0.0

    @property
    def retrieved_chunk_ids(self) -> list[str]:
        return [s.chunk.chunk_id for s in self.retrieved]

    @property
    def retrieved_doc_ids(self) -> list[str]:
        """Doc ids in retrieved order, deduped to first occurrence (doc-level ranking)."""
        seen: set[str] = set()
        out: list[str] = []
        for s in self.retrieved:
            if s.chunk.doc_id not in seen:
                seen.add(s.chunk.doc_id)
                out.append(s.chunk.doc_id)
        return out


# ── Stage interfaces (Protocols) ──────────────────────────────────────────────
# Each stage declares `name` (for the registry/leaderboard) and a `run` method.
# Implementations live in `modules/`. A pipeline may omit optional stages (rerank).


@runtime_checkable
class Chunker(Protocol):
    name: str

    def run(self, docs: list[Document]) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Populate `.embedding` on each chunk (in place, returns them)."""
        ...

    def encode_query(self, query: Query) -> Query:
        """Populate `.embedding` on the query."""
        ...


@runtime_checkable
class Index(Protocol):
    name: str

    def build(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: Query, k: int) -> list[Scored]:
        """Return top-k chunks for the query, highest score first."""
        ...


@runtime_checkable
class Retriever(Protocol):
    name: str

    def retrieve(self, query: Query, k: int) -> list[Scored]: ...

    # optional: hybrid/sparse retrievers implement this to build their own sub-indexes
    # (e.g. a BM25 side) from the corpus after the dense index is built.
    # def bind_corpus(self, chunks: list[Chunk], index: Index, embedder: Embedder) -> None: ...


@runtime_checkable
class QueryTransformer(Protocol):
    """Query understanding stage (Phase 6): rewrite/expand/multi-query/HyDE.

    `expand` returns one or more Queries to retrieve for. It may call `retrieve_fn`
    internally (e.g. pseudo-relevance feedback retrieves once, then expands). When it
    returns >1 query, the pipeline retrieves each and RRF-fuses the results.
    """

    name: str

    def expand(self, query: Query, retrieve_fn) -> list["Query"]: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: Query, scored: list[Scored], k: int) -> list[Scored]: ...


@runtime_checkable
class Assembler(Protocol):
    name: str

    def assemble(self, query: Query, scored: list[Scored]) -> str:
        """Build the context string handed to the generator."""
        ...


@runtime_checkable
class Generator(Protocol):
    name: str

    def generate(self, query: Query, context: str) -> dict[str, Any]:
        """Return {"answer": str, "tokens": {"in": int, "out": int}, "cost_usd": float}."""
        ...
