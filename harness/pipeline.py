"""The Pipeline — runs the stage chain over a corpus + queries and times every stage.

Build order per run:
  1. ingest (given docs) → chunk → embed chunks → build index          [once]
  2. per query: embed query → retrieve → [rerank] → assemble → generate [per query]

The index/embed/chunk work is done once and reused across all queries (the expensive part
is cached). Latency is captured per stage so the efficiency metrics are real, not modeled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from harness.contract import (
    Assembler,
    Chunker,
    Document,
    Embedder,
    Generator,
    Index,
    PipelineResult,
    Query,
    QueryTransformer,
    Reranker,
    Retriever,
    Scored,
)
from harness.fusion import rrf_fuse


@dataclass
class Pipeline:
    chunker: Chunker
    embedder: Embedder
    index: Index
    assembler: Assembler
    generator: Generator
    reranker: Reranker | None = None
    retriever: Retriever | None = None            # Phase 5: dense/sparse/hybrid
    query_transformer: QueryTransformer | None = None  # Phase 6: rewrite/expand/multi-query
    retrieve_k: int = 10          # candidates pulled from the index
    final_k: int = 5              # kept after rerank (or truncation) for assembly
    # populated by build()
    _chunks: list = None          # type: ignore[assignment]
    build_stats: dict[str, Any] = None  # type: ignore[assignment]

    def build(self, docs: list[Document]) -> None:
        """One-time: chunk → embed → index (+ bind corpus to a hybrid retriever)."""
        stats: dict[str, Any] = {}

        t = time.perf_counter()
        chunks = self.chunker.run(docs)
        stats["chunk_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        chunks = self.embedder.encode_chunks(chunks)
        stats["embed_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        self.index.build(chunks)
        stats["index_build_ms"] = (time.perf_counter() - t) * 1000

        # let a retriever build its own sub-indexes (e.g. BM25 side of a hybrid)
        if self.retriever is not None:
            if hasattr(self.retriever, "bind_corpus"):
                self.retriever.bind_corpus(chunks, self.index, self.embedder)
            elif hasattr(self.retriever, "bind"):
                self.retriever.bind(self.index)

        stats["n_docs"] = len(docs)
        stats["n_chunks"] = len(chunks)
        stats["embed_dim"] = self.embedder.dim
        self._chunks = chunks
        self.build_stats = stats

    def _retrieve(self, query: Query, k: int) -> list[Scored]:
        """Route through the retriever if present, else straight to the index."""
        if self.retriever is not None:
            return self.retriever.retrieve(query, k)
        return self.index.search(query, k)

    def run_query(self, query: Query) -> PipelineResult:
        if self.build_stats is None:
            raise RuntimeError("call build(docs) before run_query()")

        lat: dict[str, float] = {}

        t = time.perf_counter()
        query = self.embedder.encode_query(query)
        lat["embed_query_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        if self.query_transformer is not None:
            # transformer may fan out to multiple queries (multi-query/expansion); RRF-fuse.
            # each returned query is re-embedded (PRF/HyDE change the text).
            queries = self.query_transformer.expand(query, self._retrieve) or [query]
            embedded = [self.embedder.encode_query(q) for q in queries]
            if len(embedded) == 1:
                scored = self._retrieve(embedded[0], self.retrieve_k)
            else:
                lists = [self._retrieve(q, self.retrieve_k) for q in embedded]
                scored = rrf_fuse(lists, top=self.retrieve_k)
        else:
            scored = self._retrieve(query, self.retrieve_k)
        lat["retrieve_ms"] = (time.perf_counter() - t) * 1000

        if self.reranker is not None:
            t = time.perf_counter()
            scored = self.reranker.rerank(query, scored, self.final_k)
            lat["rerank_ms"] = (time.perf_counter() - t) * 1000
        else:
            scored = scored[: self.final_k]

        t = time.perf_counter()
        context = self.assembler.assemble(query, scored)
        lat["assemble_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        gen = self.generator.generate(query, context)
        lat["generate_ms"] = (time.perf_counter() - t) * 1000

        lat["total_ms"] = sum(
            v for k, v in lat.items() if k in
            ("embed_query_ms", "retrieve_ms", "rerank_ms", "assemble_ms", "generate_ms")
        )

        return PipelineResult(
            query=query,
            retrieved=scored,
            context=context,
            answer=gen["answer"],
            stage_latency_ms=lat,
            tokens=gen.get("tokens", {}),
            cost_usd=float(gen.get("cost_usd", 0.0)),
            extra={k: v for k, v in gen.items()
                   if k not in ("answer", "tokens", "cost_usd")},
        )
