"""Right-to-be-forgotten (Phase 14) — delete a document from the index *and every cache*.

"Delete the document" sounds like a one-liner. It is a distributed problem, because by the
time a doc has been through this lab's pipeline its text exists in at least four places:

1. the chunk list,
2. the dense index (a float matrix — the vectors are derived from the text),
3. any retriever's own sub-index (the BM25 side of `hybrid`/`sparse` has its own posting
   lists and never hears about your delete),
4. the answer cache (a cached answer is a *copy of the source text*, laundered through a
   model).

A delete that reaches 3 of the 4 is not a delete. `SecureStore` exists to own all four
behind one call, because the only reliable way to make a deletion complete is to give it a
single seam to go through.

**The tombstone trap.** The cheap implementation marks the chunk deleted and moves on —
no rebuild, O(1), reversible, and it is what most systems ship first. A tombstone is a
*read-path convention*: it protects only the read paths that remember to check the flag.
The index does not check it. BM25 does not check it. The cache does not check it. So the
`tombstone` mode here honestly does not filter reads, and the bench measures the text coming
straight back out. Hard delete costs an index rebuild (flat/IVF have no delete primitive);
that rebuild cost is reported rather than hidden, because it is the real reason teams reach
for the tombstone.
"""

from __future__ import annotations

import time
from typing import Any

from harness.contract import Chunk, Query, Scored
from modules.security.audit import AuditLog


class AnswerCache:
    """Query → answer cache that knows which docs each entry was derived from.

    That provenance is the entire design. A cache keyed only by query text cannot be purged
    by doc — you'd have to flush the whole cache on every deletion request (correct but
    brutal) or leave the copy sitting there (fast and unlawful). Recording `doc_ids` at set
    time is what makes a targeted purge possible.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, frozenset[str]]] = {}
        self.hits = 0
        self.misses = 0

    def set(self, key: str, answer: str, doc_ids: list[str] | tuple[str, ...]) -> None:
        self._entries[key] = (answer, frozenset(doc_ids))

    def get(self, key: str) -> str | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry[0]

    def purge_doc(self, doc_id: str) -> int:
        """Evict every entry derived from `doc_id`. Returns entries evicted."""
        doomed = [k for k, (_, docs) in self._entries.items() if doc_id in docs]
        for k in doomed:
            del self._entries[k]
        return len(doomed)

    def __len__(self) -> int:
        return len(self._entries)


class SecureStore:
    """Owns chunks + index + retriever + cache, so a delete has one place to reach.

    Not a new pipeline stage — it's the governance seam *around* the existing stages. The
    index and retriever are the same objects the Pipeline uses; this store just holds the
    references a deletion needs and knows the rebuild order.
    """

    def __init__(
        self,
        index: Any,
        embedder: Any,
        retriever: Any | None = None,
        cache: AnswerCache | None = None,
        audit: AuditLog | None = None,
    ):
        self.index = index
        self.embedder = embedder
        self.retriever = retriever
        self.cache = cache if cache is not None else AnswerCache()
        self.audit = audit
        self._chunks: list[Chunk] = []
        self.last_rebuild_ms: float = 0.0

    # ── build / read ──────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        t = time.perf_counter()
        self.index.build(self._chunks)
        if self.retriever is not None:
            # The retriever's own sub-indexes (BM25) must be rebuilt too — this is the line
            # a naive delete forgets, and it's why BM25 keeps answering for deleted docs.
            if hasattr(self.retriever, "bind_corpus"):
                self.retriever.bind_corpus(self._chunks, self.index, self.embedder)
            elif hasattr(self.retriever, "bind"):
                self.retriever.bind(self.index)
        self.last_rebuild_ms = (time.perf_counter() - t) * 1000

    def search(self, query: Query, k: int) -> list[Scored]:
        """The read path, deliberately naive: it does NOT check tombstones.

        That is the honest part. A store whose search filtered tombstones would be asserting
        that every read path in the system remembers to — and the whole finding is that they
        don't. Retrieval goes through the retriever when present (so the BM25 path is
        exercised), else straight to the index.
        """
        if self.retriever is not None:
            return self.retriever.retrieve(query, k)
        return self.index.search(query, k)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    @property
    def doc_ids(self) -> set[str]:
        return {c.doc_id for c in self._chunks}

    # ── delete ────────────────────────────────────────────────────────────────

    def delete(self, doc_id: str, mode: str = "hard") -> dict[str, Any]:
        """Delete `doc_id`. `mode="hard"` purges + rebuilds; `mode="tombstone"` only marks."""
        if mode not in ("hard", "tombstone"):
            raise ValueError("mode must be 'hard' or 'tombstone'")
        if doc_id not in self.doc_ids:
            raise KeyError(f"doc {doc_id!r} not in store")

        n_before = len(self._chunks)

        if mode == "tombstone":
            # O(1), no rebuild — and no effect on any read path.
            marked = 0
            for c in self._chunks:
                if c.doc_id == doc_id:
                    c.metadata["deleted"] = True
                    marked += 1
            report = {
                "mode": "tombstone",
                "doc_id": doc_id,
                "chunks_marked": marked,
                "chunks_removed": 0,
                "chunks_after": n_before,
                "cache_entries_purged": 0,
                "rebuild_ms": 0.0,
            }
        else:
            self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
            purged = self.cache.purge_doc(doc_id)     # (4) the copy people forget
            self._rebuild()                           # (2) index + (3) BM25 sub-index
            report = {
                "mode": "hard",
                "doc_id": doc_id,
                "chunks_marked": 0,
                "chunks_removed": n_before - len(self._chunks),
                "chunks_after": len(self._chunks),
                "cache_entries_purged": purged,
                "rebuild_ms": self.last_rebuild_ms,
            }

        if self.audit is not None:
            # doc_id is logged (it's an identifier, and deletions must be provable);
            # the text is not (that's the thing being deleted).
            self.audit.record("delete", doc_ids=(doc_id,), **{k: v for k, v in report.items()
                                                              if k != "doc_id"})
        return report

    def is_retrievable(self, doc_id: str, queries: list[Query], k: int = 10) -> bool:
        """Ground truth for "is it gone": can ANY query still surface this doc's text?

        The test that matters. Not "is the flag set" — whether the bytes come back.
        """
        for q in queries:
            q = self.embedder.encode_query(q)
            if any(s.chunk.doc_id == doc_id for s in self.search(q, k)):
                return True
        return False
