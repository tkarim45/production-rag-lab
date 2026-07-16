"""Phase 12 serving: embedding cache, semantic cache (incl. false hits), latency budget."""

import numpy as np
import pytest

import modules  # noqa: F401 populate registry
import modules.serving  # noqa: F401 Phase 12 components (not yet wired into modules/__init__)
from harness.contract import Chunk, PipelineResult, Query
from harness.latency_report import stage_breakdown
from harness.registry import build


def _chunks():
    return [
        Chunk("d::0", "d", "the cat sat on the mat"),
        Chunk("d::1", "d", "a dog ran in the park"),
        Chunk("d::2", "d", "quantum physics studies particles"),
    ]


# ── embedding cache ───────────────────────────────────────────────────────────


def test_embedding_cache_cold_misses_then_warm_hits(tmp_path):
    cold = build("embedder", "cached", base="hashing", cache_dir=str(tmp_path), dim=64)
    cold.encode_chunks(_chunks())
    assert (cold.hits, cold.misses, cold.writes) == (0, 3, 3)

    # a fresh wrapper on the same dir = the next rebuild: every chunk served from disk
    warm = build("embedder", "cached", base="hashing", cache_dir=str(tmp_path), dim=64)
    warm_chunks = warm.encode_chunks(_chunks())
    assert (warm.hits, warm.misses, warm.writes) == (3, 0, 0)
    assert warm.stats["hit_rate"] == 1.0

    # and the cached vectors are the ones the base would have produced
    truth = build("embedder", "hashing", dim=64).encode_chunks(_chunks())
    for got, want in zip(warm_chunks, truth):
        assert np.allclose(got.embedding, want.embedding)


def test_embedding_cache_key_includes_model_id_and_dim(tmp_path):
    a = build("embedder", "cached", base="hashing", cache_dir=str(tmp_path), dim=64)
    b = build("embedder", "cached", base="hashing", cache_dir=str(tmp_path), dim=128)
    c = build("embedder", "cached", base="hashing", cache_dir=str(tmp_path), dim=64,
              model_id="hashing-v2")

    text = "the cat sat on the mat"
    # same text, different embedder identity → different key AND different namespace
    assert a.key(text) != b.key(text)          # dim is part of the identity
    assert a.key(text) != c.key(text)          # so is the model id
    assert a.namespace != b.namespace != c.namespace
    # same text, same identity → same key (that is what makes a hit possible)
    a2 = build("embedder", "cached", base="hashing", cache_dir=str(tmp_path), dim=64)
    assert a.key(text) == a2.key(text)


def test_embedding_cache_namespaces_do_not_bleed(tmp_path):
    """A dim-64 cache must never serve a dim-128 embedder — that would corrupt retrieval."""
    a = build("embedder", "cached", base="hashing", cache_dir=str(tmp_path), dim=64)
    a.encode_chunks(_chunks())
    b = build("embedder", "cached", base="hashing", cache_dir=str(tmp_path), dim=128)
    chunks = b.encode_chunks(_chunks())
    assert b.hits == 0 and b.misses == 3        # cold despite the shared directory
    assert all(len(c.embedding) == 128 for c in chunks)


def test_embedding_cache_rejects_corpus_fit_embedder(tmp_path):
    """tfidf vectors depend on the corpus it was fit on, so they are not cacheable by text."""
    with pytest.raises(ValueError, match="corpus-fit"):
        build("embedder", "cached", base="tfidf", cache_dir=str(tmp_path))
    # …including when a wrapper hides it (`quantized` defaults to a tfidf base)
    with pytest.raises(ValueError, match="corpus-fit"):
        build("embedder", "cached", base="quantized", cache_dir=str(tmp_path))


def test_embedding_cache_query_reuses_chunk_vector(tmp_path):
    e = build("embedder", "cached", base="hashing", cache_dir=str(tmp_path), dim=64)
    e.encode_chunks(_chunks())
    hits_before = e.hits
    e.encode_query(Query("q", "the cat sat on the mat"))  # identical text
    assert e.hits == hits_before + 1


# ── semantic cache ────────────────────────────────────────────────────────────


class _CountingGenerator:
    """A base generator that reports how many times it was actually called + a real cost."""

    name = "counting"

    def __init__(self):
        self.calls = 0

    def generate(self, query, context):
        self.calls += 1
        return {"answer": f"answer for {query.text}", "tokens": {"in": 100, "out": 10},
                "cost_usd": 0.001}


def _semantic(threshold=0.95):
    g = build("generator", "semantic_cached", base="extractive_mock", threshold=threshold,
              embedder="hashing")
    g.base = _CountingGenerator()
    return g


def test_semantic_cache_serves_exact_repeat_for_free():
    g = _semantic(threshold=0.95)
    first = g.generate(Query("q1", "who invented the world wide web"), "ctx")
    assert first["cache_hit"] is False and g.base.calls == 1

    second = g.generate(Query("q1-repeat", "who invented the world wide web"), "ctx")
    assert second["cache_hit"] is True
    assert second["answer"] == first["answer"]
    assert second["cost_usd"] == 0.0 and second["tokens"] == {"in": 0, "out": 0}
    assert g.base.calls == 1                      # the base was never called again
    assert second["cache_sim"] == pytest.approx(1.0, abs=1e-5)
    assert g.stats["hit_rate"] == 0.5


def test_semantic_cache_misses_a_dissimilar_query():
    g = _semantic(threshold=0.95)
    g.generate(Query("q1", "who invented the world wide web"), "ctx")
    out = g.generate(Query("q2", "how many chambers does the human heart have"), "ctx")
    assert out["cache_hit"] is False
    assert g.base.calls == 2
    assert g.stats["entries"] == 2


def test_semantic_cache_false_hit_at_a_low_threshold():
    """The honest headline: lower the threshold and a *different* question gets served
    someone else's answer — confidently, at zero cost."""
    strict, loose = _semantic(threshold=0.99), _semantic(threshold=0.30)
    a = Query("q1", "who discovered penicillin and in what year")
    b = Query("q2", "who discovered penicillin resistance in bacteria")

    strict.generate(a, "ctx")
    strict_out = strict.generate(b, "ctx")
    assert strict_out["cache_hit"] is False       # strict threshold answers b properly

    loose.generate(a, "ctx")
    loose_out = loose.generate(b, "ctx")
    assert loose_out["cache_hit"] is True         # loose threshold serves a's answer to b
    assert loose_out["cache_source_query_id"] == "q1"
    assert loose_out["answer"] == f"answer for {a.text}"   # …the wrong answer
    assert loose.base.calls == 1


def test_semantic_cache_uses_pipeline_query_embedding_when_no_embedder():
    g = build("generator", "semantic_cached", base="extractive_mock", threshold=0.95)
    g.base = _CountingGenerator()
    q1, q2 = Query("q1", "anything"), Query("q2", "totally different words")
    q1.embedding = np.array([1.0, 0.0], dtype=np.float32)
    q2.embedding = np.array([1.0, 0.0], dtype=np.float32)
    g.generate(q1, "ctx")
    out = g.generate(q2, "ctx")
    assert out["cache_hit"] is True               # keyed on the vector, not the string
    assert g.base.calls == 1


def test_semantic_cache_threshold_validated():
    with pytest.raises(ValueError, match="cosine"):
        build("generator", "semantic_cached", threshold=1.5)


# ── latency budget ────────────────────────────────────────────────────────────


def _result(lat):
    lat = dict(lat)
    lat["total_ms"] = sum(lat.values())
    return PipelineResult(query=Query("q", "t"), retrieved=[], context="", answer="a",
                          stage_latency_ms=lat)


def test_stage_breakdown_budget_math():
    # generate dominates: 1 + 9 + 90 = 100ms total
    results = [_result({"retrieve_ms": 1.0, "assemble_ms": 9.0, "generate_ms": 90.0})] * 4
    rep = stage_breakdown(results)
    assert rep["n_queries"] == 4
    assert rep["total"]["mean_ms"] == pytest.approx(100.0)
    assert rep["stages"]["generate_ms"]["pct_of_budget"] == pytest.approx(90.0)
    assert rep["stages"]["retrieve_ms"]["pct_of_budget"] == pytest.approx(1.0)
    # shares of the budget must sum to 100% (why the math uses means, not percentiles)
    assert sum(s["pct_of_budget"] for s in rep["stages"].values()) == pytest.approx(100.0)


def test_stage_breakdown_percentiles_are_per_stage():
    results = [_result({"retrieve_ms": float(i), "generate_ms": 10.0}) for i in (1, 2, 3, 4)]
    rep = stage_breakdown(results)
    assert rep["stages"]["retrieve_ms"]["p50_ms"] == 2.0      # nearest-rank
    assert rep["stages"]["retrieve_ms"]["p95_ms"] == 4.0
    assert rep["stages"]["retrieve_ms"]["mean_ms"] == pytest.approx(2.5)
    assert rep["stages"]["generate_ms"]["n"] == 4
    assert rep["stages"]["generate_ms"]["pct_of_budget"] == pytest.approx(10 / 12.5 * 100)


def test_stage_breakdown_keeps_pipeline_order_and_needs_results():
    rep = stage_breakdown([_result({"embed_query_ms": 1.0, "retrieve_ms": 2.0,
                                    "assemble_ms": 1.0, "generate_ms": 6.0})])
    assert list(rep["stages"]) == ["embed_query_ms", "retrieve_ms", "assemble_ms", "generate_ms"]
    with pytest.raises(ValueError):
        stage_breakdown([])


# ── optional API ──────────────────────────────────────────────────────────────


def test_api_health_and_ask():
    from modules.serving import api

    if not api.HAVE_FASTAPI:
        pytest.skip("fastapi not installed (optional dep)")
    from fastapi.testclient import TestClient

    client = TestClient(api.create_app("configs/naive.yaml"))

    health = client.get("/health").json()
    assert health["status"] == "ok" and health["n_chunks"] > 0

    r = client.post("/ask", json={"query": "what is the capital of France?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] and body["citations"]
    assert "generate_ms" in body["latency_ms"] and "total_ms" in body["latency_ms"]
    assert client.post("/ask", json={"query": ""}).status_code == 422
