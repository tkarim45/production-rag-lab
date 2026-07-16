"""Embedding cache (Phase 12) — a disk cache wrapper over any embedder.

Re-embedding an unchanged corpus on every rebuild is pure waste: the embedding of a text is a
deterministic function of (text, model). Cache it on disk and a rebuild only pays for the
chunks that actually changed.

The cache key is `sha256(text)`; the *namespace* (the directory) is the embedder's identity —
base name + declared params + dim + model id. The namespace is what makes the cache safe: two
embedders that disagree about what a vector means must never read each other's files. Change
the model, change the dim, change any param → new namespace → cold cache. That is the correct
behavior, not a miss to optimize away.

**The honest limit.** This is only sound when the embedding is a pure function of the text.
`tfidf` is corpus-fit — its vectors depend on the IDF of whatever corpus it last saw, so the
same text embeds differently in a different corpus. Caching it would serve vectors from
another corpus's statistics and silently corrupt retrieval, so `cached` refuses a corpus-fit
base rather than pretend. Hashing / neural encoders are pure and cache correctly.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from harness.contract import Chunk, Query
from harness.registry import build, register

# Embedders whose vectors depend on the corpus they were fit on, not just the text.
# Caching these across builds is unsound — see the module docstring.
_CORPUS_FIT = frozenset({"tfidf"})

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "embeddings"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _identity_chain(embedder) -> list[str]:
    """Names of an embedder and every base it wraps (`quantized`→`tfidf`→…)."""
    names: list[str] = []
    cur = embedder
    for _ in range(8):  # wrappers are shallow; the bound just stops a cycle
        if cur is None:
            break
        names.append(str(getattr(cur, "name", type(cur).__name__)))
        cur = getattr(cur, "base", None)
    return names


@register("embedder", "cached")
class CachedEmbedder:
    """Disk cache in front of any (pure) embedder. Reports hits/misses/writes."""

    name = "cached"

    def __init__(
        self,
        base: str = "hashing",
        cache_dir: str | Path | None = None,
        model_id: str | None = None,
        **base_params,
    ):
        self.base = build("embedder", base, **base_params)
        chain = _identity_chain(self.base)
        corpus_fit = sorted(set(chain) & _CORPUS_FIT)
        if corpus_fit:
            raise ValueError(
                f"cannot cache a corpus-fit embedder ({', '.join(corpus_fit)}): its vectors "
                "depend on the corpus it was fit on, not only on the text, so a cached vector "
                "is not reusable across builds. Cache a pure embedder (hashing, "
                "sentence_transformer) instead."
            )
        self._chain = chain
        # model_id makes the namespace explicit for embedders whose identity is a model name
        self.model_id = str(model_id or base_params.get("model") or base)
        self._dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._mem: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0

    # ── identity / key ────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        return self.base.dim

    @property
    def namespace(self) -> str:
        """Directory name encoding *which* embedder these vectors came from."""
        ident = "-".join(self._chain)
        return _SAFE.sub("_", f"{ident}__{self.model_id}__d{self.dim}")

    def key(self, text: str) -> str:
        """Full cache key: embedder identity + content hash."""
        return f"{self.namespace}/{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    def _path(self, text: str) -> Path:
        ns, digest = self.key(text).split("/")
        return self._dir / ns / f"{digest}.npy"

    @property
    def stats(self) -> dict[str, object]:
        looked_up = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": (self.hits / looked_up) if looked_up else 0.0,
            "namespace": self.namespace,
            "cache_dir": str(self._dir),
        }

    # ── store ─────────────────────────────────────────────────────────────────

    def _get(self, text: str) -> np.ndarray | None:
        k = self.key(text)
        vec = self._mem.get(k)
        if vec is not None:
            self.hits += 1
            return vec
        p = self._path(text)
        if p.exists():
            vec = np.load(p)
            self._mem[k] = vec
            self.hits += 1
            return vec
        self.misses += 1
        return None

    def _put(self, text: str, vec: np.ndarray) -> None:
        if vec is None:
            return
        k = self.key(text)
        self._mem[k] = vec
        p = self._path(text)
        p.parent.mkdir(parents=True, exist_ok=True)
        # write-then-rename: a crash mid-write must not leave a truncated vector behind
        # (np.save appends `.npy` to a path, so hand it an open handle instead)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, vec)
        tmp.replace(p)
        self.writes += 1

    # ── Embedder interface ────────────────────────────────────────────────────

    def encode_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        misses: list[Chunk] = []
        for c in chunks:
            vec = self._get(c.text)
            if vec is None:
                misses.append(c)
            else:
                c.embedding = vec
        if misses:
            self.base.encode_chunks(misses)
            for c in misses:
                self._put(c.text, c.embedding)
        return chunks

    def encode_query(self, query: Query) -> Query:
        vec = self._get(query.text)
        if vec is None:
            query = self.base.encode_query(query)
            self._put(query.text, query.embedding)
            return query
        query.embedding = vec
        return query
