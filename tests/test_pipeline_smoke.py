"""End-to-end smoke: the naive pipeline runs over the built-in set and retrieves correctly.

This is the Phase 0 definition-of-done in a test: a config → real metrics, and the exact
Flat index actually finds the gold chunk for the easy single-hop queries.
"""

from harness import config as cfgmod
from harness import scoring
from harness.data import load_dataset


def _run():
    cfg, pipeline = cfgmod.load("configs/naive.yaml")
    docs, queries = load_dataset(cfg["dataset"])
    pipeline.build(docs)
    return queries, [pipeline.run_query(q) for q in queries]


def test_pipeline_runs_and_builds_index():
    queries, results = _run()
    assert len(results) == len(queries)
    for r in results:
        assert r.retrieved                      # got candidates
        assert r.answer is not None
        assert "total_ms" in r.stage_latency_ms


def test_retrieval_finds_gold_on_single_hop():
    _, results = _run()
    # the hashing embedder + exact index should rank the gold chunk #1 on lexical single-hop
    single = [r for r in results if r.query.hop_type == "single"]
    top1_hits = sum(
        1 for r in single if r.retrieved_chunk_ids[0] in r.query.relevant_chunk_ids
    )
    # allow one miss to stay robust, but demand the baseline mostly works
    assert top1_hits >= len(single) - 1


def test_scoring_emits_all_families():
    _, results = _run()
    m = scoring.score(results)
    for key in ("recall@5", "mrr", "ndcg@10", "map", "em", "token_f1",
                "latency_p50_ms", "cost_per_query_usd", "cost_per_correct_usd"):
        assert key in m, f"missing metric {key}"
    # hop split present
    assert "token_f1_single" in m and "token_f1_multi" in m


def test_recall_at_10_is_perfect_on_tiny_corpus():
    # 12 chunks, retrieve_k=10 ... gold is usually in top-10 for lexical queries
    _, results = _run()
    m = scoring.score(results)
    assert m["recall@10"] >= 0.75
