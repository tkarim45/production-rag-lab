"""Phase 5 retrievers + fusion, Phase 6 query transformers."""

import numpy as np

import modules  # noqa: F401
from harness.contract import Chunk, Query, Scored
from harness.fusion import rrf_fuse, weighted_fuse
from harness.registry import build


def _corpus():
    vecs = np.eye(4, dtype=np.float32)
    texts = ["alpha alpha topic", "beta beta topic", "gamma gamma topic", "delta delta topic"]
    return [Chunk(f"d{i}::0", f"d{i}", texts[i], embedding=vecs[i]) for i in range(4)]


# ── fusion ────────────────────────────────────────────────────────────────────

def test_rrf_fuse_rewards_consensus():
    a = [Scored(Chunk("x::0", "x", ""), 0.9), Scored(Chunk("y::0", "y", ""), 0.8)]
    b = [Scored(Chunk("y::0", "y", ""), 0.7), Scored(Chunk("z::0", "z", ""), 0.6)]
    fused = rrf_fuse([a, b])
    # y appears in both lists → should rank first
    assert fused[0].chunk.chunk_id == "y::0"


def test_weighted_fuse_normalizes():
    a = [Scored(Chunk("x::0", "x", ""), 100.0), Scored(Chunk("y::0", "y", ""), 0.0)]
    b = [Scored(Chunk("y::0", "y", ""), 1.0)]
    fused = weighted_fuse([a, b], weights=[0.5, 0.5])
    assert {s.chunk.chunk_id for s in fused} == {"x::0", "y::0"}


# ── retrievers ─────────────────────────────────────────────────────────────────

def test_sparse_retriever_is_bm25():
    r = build("retriever", "sparse")
    r.bind_corpus(_corpus(), None, None)
    q = Query("q", "gamma")
    assert r.retrieve(q, 1)[0].chunk.doc_id == "d2"


def test_hybrid_retriever_fuses_dense_and_sparse():
    chunks = _corpus()
    dense = build("index", "flat"); dense.build([Chunk(c.chunk_id, c.doc_id, c.text, embedding=c.embedding.copy()) for c in chunks])
    r = build("retriever", "hybrid", fusion="rrf")
    r.bind_corpus(chunks, dense, None)
    q = Query("q", "beta"); q.embedding = np.array([0, 1, 0, 0], dtype=np.float32)
    out = r.retrieve(q, 2)
    assert out and out[0].chunk.doc_id == "d1"   # both dense + bm25 agree on beta


def test_mmr_returns_k_diverse():
    chunks = _corpus()
    dense = build("index", "flat"); dense.build([Chunk(c.chunk_id, c.doc_id, c.text, embedding=c.embedding.copy()) for c in chunks])
    r = build("retriever", "mmr", lam=0.5)
    r.bind_corpus(chunks, dense, None)
    q = Query("q", "alpha"); q.embedding = np.array([1, 0, 0, 0], dtype=np.float32)
    out = r.retrieve(q, 3)
    assert len(out) == 3
    assert len({s.chunk.chunk_id for s in out}) == 3   # no duplicates


# ── query transformers ─────────────────────────────────────────────────────────

def test_prf_expands_query_text():
    chunks = _corpus()
    dense = build("index", "flat"); dense.build([Chunk(c.chunk_id, c.doc_id, c.text, embedding=c.embedding.copy()) for c in chunks])
    r = build("retriever", "sparse"); r.bind_corpus(chunks, None, None)
    prf = build("query_transformer", "prf", fb_docs=1, add_terms=3)
    q = Query("q", "alpha")
    out = prf.expand(q, r.retrieve)
    assert len(out) == 1
    assert "alpha" in out[0].text  # original terms preserved


def test_multiquery_prf_returns_two():
    chunks = _corpus()
    r = build("retriever", "sparse"); r.bind_corpus(chunks, None, None)
    mq = build("query_transformer", "multiquery_prf")
    out = mq.expand(Query("q", "alpha"), r.retrieve)
    assert len(out) == 2
