"""Retrieval strategies (Phase 5).

Registers retrievers that route the pipeline's query path: `sparse` (BM25), `hybrid`
(dense + BM25 fused by RRF or weighted score), and `mmr` (diversity/redundancy control over
a base retriever). The Phase 0 `dense` retriever lives in modules/baseline. Each `hybrid`/
`sparse` builds its own BM25 side via `bind_corpus` at pipeline build time.
"""

from modules.retrieval import retrievers  # noqa: F401
