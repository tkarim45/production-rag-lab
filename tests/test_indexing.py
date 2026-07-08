"""Phase 4 indexes: BM25 lexical scoring, IVF approximate recall vs Flat."""

import numpy as np

import modules  # noqa: F401
from harness.contract import Chunk, Query
from harness.registry import build


def _embedded_chunks():
    # 3 orthogonal-ish unit vectors
    vecs = np.eye(3, dtype=np.float32)
    return [Chunk(f"d{i}::0", f"d{i}", f"doc about topic {w}", embedding=vecs[i])
            for i, w in enumerate(["alpha", "beta", "gamma"])]


def test_bm25_ranks_lexical_match_first():
    idx = build("index", "bm25")
    idx.build([Chunk("a::0", "a", "the cat sat on the mat"),
               Chunk("b::0", "b", "quantum physics and particles"),
               Chunk("c::0", "c", "a dog and a cat play")])
    out = idx.search(Query("q", "cat"), k=3)
    assert out[0].chunk.doc_id in ("a", "c")   # both mention cat
    assert out[0].score > 0


def test_ivf_matches_flat_on_easy_case():
    chunks = _embedded_chunks()
    flat = build("index", "flat"); flat.build([Chunk(c.chunk_id, c.doc_id, c.text, embedding=c.embedding.copy()) for c in chunks])
    ivf = build("index", "ivf", nlist=3, nprobe=3)  # probe all cells → exact
    ivf.build(chunks)
    q = Query("q", "topic alpha"); q.embedding = np.array([1, 0, 0], dtype=np.float32)
    assert ivf.search(q, 1)[0].chunk.doc_id == "d0"
    assert flat.search(q, 1)[0].chunk.doc_id == "d0"


def test_ivf_nprobe1_can_miss():
    # with nprobe=1 and many cells, a query near a cell boundary may miss — approximate
    chunks = _embedded_chunks()
    ivf = build("index", "ivf", nlist=3, nprobe=1)
    ivf.build(chunks)
    q = Query("q", "x"); q.embedding = np.array([1, 0, 0], dtype=np.float32)
    res = ivf.search(q, 3)
    assert len(res) >= 1   # returns from the single probed cell


def test_bm25_score_formula_positive_idf():
    idx = build("index", "bm25", k1=1.5, b=0.75)
    idx.build([Chunk("a::0", "a", "rare unique term here"), Chunk("b::0", "b", "common common common")])
    # 'unique' appears in 1 of 2 docs → positive idf, ranks doc a
    out = idx.search(Query("q", "unique"), 2)
    assert out and out[0].chunk.doc_id == "a"
