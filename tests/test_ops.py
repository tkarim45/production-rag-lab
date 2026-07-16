"""Phase 13 ops: PSI drift, bootstrap CI, CI regression gate, per-request tracing."""

import math

import numpy as np
import pytest

import modules  # noqa: F401  populates registry
from harness import gate as gate_cli
from harness.contract import Chunk, PipelineResult, Query, Scored
from modules.ops import ab, drift, runs
from modules.ops import gate as gate_mod
from modules.ops.tracing import Tracer, TracedPipeline

# ── Drift: PSI ────────────────────────────────────────────────────────────────
#
# The reference case, computed by hand:
#   expected proportions [0.5, 0.5], actual [0.25, 0.75]
#   PSI = (0.25-0.50)*ln(0.25/0.50) + (0.75-0.50)*ln(0.75/0.50)
#       = (-0.25)*(-0.693147)      + (0.25)*(0.405465)
#       =  0.173287                +  0.101366
#       =  0.274653
HAND_PSI = 0.2746530721670273


def test_psi_from_proportions_matches_hand_computed_case():
    assert drift.psi_from_proportions([0.5, 0.5], [0.25, 0.75]) == pytest.approx(HAND_PSI, abs=1e-9)


def test_psi_hand_computed_case_through_the_binning_path():
    """Same hand-computed answer, but reached by binning raw samples with explicit edges."""
    edges = np.array([-np.inf, 0.5, np.inf])
    expected = [0.1, 0.2, 0.8, 0.9]          # 2 below 0.5, 2 above  → [0.50, 0.50]
    actual = [0.1, 0.6, 0.7, 0.8]            # 1 below 0.5, 3 above  → [0.25, 0.75]
    assert drift.psi(expected, actual, edges=edges) == pytest.approx(HAND_PSI, abs=1e-9)


def test_psi_hand_computed_case_is_above_the_significant_band():
    # sanity on the constant itself: this hand case is a genuine "significant" shift
    assert HAND_PSI > drift.SIGNIFICANT_SHIFT
    assert drift.verdict(HAND_PSI) == "significant"


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    assert drift.psi(x, x) == pytest.approx(0.0, abs=1e-12)


def test_psi_is_nonnegative_and_detects_a_real_shift_at_adequate_n():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.0, 1.0, 10_000)
    same = rng.normal(0.0, 1.0, 10_000)
    shifted = rng.normal(1.0, 1.0, 10_000)
    assert drift.psi(ref, same) >= 0.0
    assert drift.psi(ref, same) < drift.NO_SHIFT          # no drift → "none" band
    assert drift.psi(ref, shifted) > drift.SIGNIFICANT_SHIFT   # 1-sigma shift → "significant"


def test_verdict_bands_are_the_conventional_thresholds():
    assert drift.verdict(0.05) == "none"
    assert drift.verdict(0.10) == "moderate"      # boundary belongs to the upper band
    assert drift.verdict(0.20) == "moderate"
    assert drift.verdict(0.25) == "significant"
    assert drift.verdict(9.9) == "significant"


def test_epsilon_floor_keeps_an_empty_bin_finite():
    """Without the floor an empty bin makes PSI +inf, which on small samples is constant."""
    value = drift.psi_from_proportions([0.5, 0.5], [1.0, 0.0])
    assert math.isfinite(value)
    assert value > drift.SIGNIFICANT_SHIFT


def test_psi_grows_as_epsilon_shrinks_so_it_is_an_implementation_choice():
    # the honest finding, pinned as a test: on an empty bin, PSI's magnitude is set by a
    # constant we chose, not by the data.
    coarse = drift.psi_from_proportions([0.5, 0.5], [1.0, 0.0], epsilon=1e-2)
    fine = drift.psi_from_proportions([0.5, 0.5], [1.0, 0.0], epsilon=1e-6)
    assert fine > coarse


def test_psi_categorical_handles_binary_outcomes():
    # 50/50 vs 25/75 over labels {0, 1} — the same hand-computed number
    expected = [0, 0, 1, 1]
    actual = [0, 1, 1, 1]
    assert drift.psi_categorical(expected, actual) == pytest.approx(HAND_PSI, abs=1e-9)


def test_psi_auto_selects_categorical_for_low_cardinality_reference():
    # quantile bins on {0,1} would collapse to one bin and report a meaningless 0.0
    assert drift.psi([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(HAND_PSI, abs=1e-9)


def test_quantile_edges_are_open_ended_so_out_of_range_values_are_counted():
    edges = drift.quantile_edges([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], bins=5)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    # a sample far outside the reference range still lands in a bin (drift must be visible)
    assert drift.psi([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [99, 99, 99, 99, 99]) > 0.0


def test_psi_rejects_empty_samples():
    with pytest.raises(ValueError):
        drift.psi([], [1.0])
    with pytest.raises(ValueError):
        drift.psi([1.0], [])


def test_psi_from_proportions_rejects_misaligned_vectors():
    with pytest.raises(ValueError):
        drift.psi_from_proportions([0.5, 0.5], [1.0])


# ── Drift: the three surfaces ─────────────────────────────────────────────────


def test_embedding_drift_is_zero_against_itself():
    rng = np.random.default_rng(0)
    vecs = rng.normal(0, 1, (200, 8))
    d = drift.embedding_drift(vecs, vecs)
    assert d["psi"] == pytest.approx(0.0, abs=1e-12)
    assert d["verdict"] == "none"
    assert d["dim"] == 8 and d["n_reference"] == 200


def test_embedding_drift_fires_on_a_translated_cloud():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (2000, 4))
    shifted = rng.normal(3, 1, (2000, 4))
    d = drift.embedding_drift(ref, shifted)
    assert d["psi"] > drift.SIGNIFICANT_SHIFT
    assert d["verdict"] == "significant"


def test_embedding_drift_rejects_mismatched_dims():
    with pytest.raises(ValueError):
        drift.embedding_drift(np.zeros((4, 8)), np.zeros((4, 6)))


def test_query_drift_embeds_unembedded_queries():
    embedder = __import__("harness.registry", fromlist=["build"]).build("embedder", "hashing", dim=32)
    ref = [Query(f"r{i}", f"what is retrieval augmented generation {i}") for i in range(30)]
    act = [Query(f"a{i}", f"how do I bake sourdough bread number {i}") for i in range(30)]
    d = drift.query_drift(ref, act, embedder=embedder)
    assert d["psi"] > 0.0
    assert set(d) >= {"psi_centroid_cosine", "psi_per_dim_mean", "verdict"}


def test_query_drift_without_an_embedder_on_raw_queries_raises():
    with pytest.raises(ValueError, match="unembedded"):
        drift.query_drift([Query("a", "x")], [Query("b", "y")])


def test_rolling_mean_windows_the_series():
    assert drift.rolling_mean([1, 1, 0, 0], window=2) == [1.0, 0.5, 0.0]
    assert drift.rolling_mean([1, 2], window=5) == []      # not enough points yet
    with pytest.raises(ValueError):
        drift.rolling_mean([1, 2, 3], window=0)


def test_retrieval_quality_drift_reports_mean_delta_alongside_psi():
    ref = [1.0] * 18 + [0.0] * 2      # 90% hit rate
    act = [1.0] * 10 + [0.0] * 10     # 50% hit rate
    d = drift.retrieval_quality_drift(ref, act, window=5)
    assert d["reference_mean"] == pytest.approx(0.9)
    assert d["actual_mean"] == pytest.approx(0.5)
    assert d["delta"] == pytest.approx(-0.4)
    assert d["psi"] > drift.SIGNIFICANT_SHIFT
    assert d["distinct_reference_values"] == 2
    assert len(d["rolling_actual"]) == 16


# ── A/B: bootstrap CI ─────────────────────────────────────────────────────────


def test_bootstrap_ci_on_identical_arms_is_zero_and_contains_zero():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    s = ab.bootstrap_ci(vals, vals, n_boot=2000, seed=0)
    assert s["delta"] == pytest.approx(0.0)
    # paired resampling: identical arms cancel exactly on every replicate
    assert s["ci_lo"] == pytest.approx(0.0) and s["ci_hi"] == pytest.approx(0.0)
    assert not s["excludes_zero"]


def test_bootstrap_ci_brackets_the_true_delta_and_excludes_zero_when_separation_is_large():
    a = [0.0] * 50
    b = [1.0] * 50
    s = ab.bootstrap_ci(a, b, n_boot=2000, seed=0)
    assert s["delta"] == pytest.approx(1.0)
    assert s["ci_lo"] <= s["delta"] <= s["ci_hi"]
    assert s["excludes_zero"]


def test_bootstrap_ci_calls_a_small_noisy_delta_inconclusive():
    """The honest core: a 1-in-20 difference is not a winner."""
    a = [1.0] * 19 + [0.0]
    b = [1.0] * 20
    s = ab.bootstrap_ci(a, b, n_boot=5000, seed=0)
    assert s["delta"] == pytest.approx(0.05)
    assert not s["excludes_zero"]     # one discordant query can never clear the bar


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 40), rng.normal(0.3, 1, 40)
    first = ab.bootstrap_ci(a, b, n_boot=1000, seed=7)
    second = ab.bootstrap_ci(a, b, n_boot=1000, seed=7)
    assert first == second
    assert ab.bootstrap_ci(a, b, n_boot=1000, seed=8)["ci_lo"] != first["ci_lo"]


def test_bootstrap_ci_narrows_as_n_grows():
    rng = np.random.default_rng(0)
    small_a, small_b = rng.normal(0, 1, 20), rng.normal(0.5, 1, 20)
    big_a, big_b = rng.normal(0, 1, 2000), rng.normal(0.5, 1, 2000)
    small = ab.bootstrap_ci(small_a, small_b, n_boot=2000, seed=0)
    big = ab.bootstrap_ci(big_a, big_b, n_boot=2000, seed=0)
    assert (big["ci_hi"] - big["ci_lo"]) < (small["ci_hi"] - small["ci_lo"])


def test_bootstrap_ci_unpaired_is_wider_than_paired_on_correlated_arms():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 200)
    a = base
    b = base + 0.2                     # perfectly correlated arms, constant +0.2 effect
    paired = ab.bootstrap_ci(a, b, n_boot=2000, seed=0, paired=True)
    unpaired = ab.bootstrap_ci(a, b, n_boot=2000, seed=0, paired=False)
    assert (paired["ci_hi"] - paired["ci_lo"]) < (unpaired["ci_hi"] - unpaired["ci_lo"])


def test_bootstrap_ci_validates_inputs():
    with pytest.raises(ValueError):
        ab.bootstrap_ci([], [1.0])
    with pytest.raises(ValueError, match="equal-length"):
        ab.bootstrap_ci([1.0, 2.0], [1.0], paired=True)
    with pytest.raises(ValueError, match="alpha"):
        ab.bootstrap_ci([1.0], [1.0], alpha=1.5)


def test_agrees_with_the_phase_11_bootstrap_written_independently():
    """Cross-validation, not duplication.

    Phase 11's `modules/evaluation/significance.paired_delta_ci` bootstraps the *difference
    vector*; this module resamples shared query indices across both arms. Those are the same
    estimator by algebra (mean(b[i]) - mean(a[i]) == mean((b-a)[i])) but different code and a
    different RNG stream. Two independent implementations landing on the same interval is
    evidence the statistic is right; a divergence here means one of them is wrong.
    """
    significance = pytest.importorskip("modules.evaluation.significance")
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, 200).tolist()
    b = rng.normal(0.4, 1.0, 200).tolist()

    mine = ab.bootstrap_ci(a, b, n_boot=20_000, seed=0)
    # theirs computes mean(a) - mean(b); ours is mean(b) - mean(a) → flip and swap bounds
    theirs = significance.paired_delta_ci(b, a, n_boot=20_000, seed=0)

    assert mine["delta"] == pytest.approx(theirs["mean_delta"], abs=1e-12)  # exact: no sampling
    assert mine["ci_lo"] == pytest.approx(theirs["lo"], abs=0.02)           # Monte-Carlo noise
    assert mine["ci_hi"] == pytest.approx(theirs["hi"], abs=0.02)
    assert mine["excludes_zero"] == bool(theirs["significant"])


def test_confidence_level_is_reported_and_wider_alpha_narrows_the_interval():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 100), rng.normal(0.5, 1, 100)
    ci95 = ab.bootstrap_ci(a, b, n_boot=3000, seed=0, alpha=0.05)
    ci80 = ab.bootstrap_ci(a, b, n_boot=3000, seed=0, alpha=0.20)
    assert ci95["confidence"] == pytest.approx(0.95)
    assert (ci80["ci_hi"] - ci80["ci_lo"]) < (ci95["ci_hi"] - ci95["ci_lo"])


# ── A/B: per-query metric extraction ──────────────────────────────────────────


def _result(query_id="q1", ranked_docs=("d1", "d2"), relevant=("d1",), answer="paris", gold="paris"):
    q = Query(query_id, "text", gold_answer=gold, relevant_doc_ids=set(relevant))
    retrieved = [Scored(Chunk(f"{d}::0", d, f"text of {d}"), 1.0 - i * 0.1)
                 for i, d in enumerate(ranked_docs)]
    return PipelineResult(query=q, retrieved=retrieved, context="ctx", answer=answer,
                          stage_latency_ms={"total_ms": 1.5})


def test_per_query_returns_a_vector_not_a_mean():
    results = [_result("q1", ("d1", "d2"), ("d1",)),      # hit at rank 1
               _result("q2", ("d9", "d1"), ("d1",))]      # hit at rank 2
    assert runs.per_query(results, "recall@1") == [1.0, 0.0]
    assert runs.per_query(results, "mrr") == [1.0, 0.5]
    assert len(runs.per_query(results, "token_f1")) == 2


def test_per_query_drops_ungraded_queries_for_answer_metrics():
    results = [_result("q1", gold="paris"), _result("q2", gold=None)]
    assert len(runs.per_query(results, "em")) == 1        # only the graded one
    assert len(runs.per_query(results, "recall@1")) == 2  # retrieval needs no gold answer


def test_per_query_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="unsupported"):
        runs.per_query([_result()], "bleu")


def test_variant_pins_other_stages():
    cfg = runs.variant("chunker", "contextual", pin={"embedder": "hashing"})
    assert cfg["chunker"]["name"] == "contextual"
    assert cfg["embedder"]["name"] == "hashing"
    with pytest.raises(ValueError, match="unknown stage"):
        runs.variant("nonsense", "x")


# ── Gate ──────────────────────────────────────────────────────────────────────


def test_check_metrics_min_direction_passes_and_fails():
    metrics = {"recall@5": 0.95}
    assert gate_mod.check_metrics(metrics, {"recall@5": 0.90}, {})[0].passed
    assert not gate_mod.check_metrics(metrics, {"recall@5": 0.99}, {})[0].passed


def test_check_metrics_max_direction_catches_a_cost_regression():
    metrics = {"cost_per_query_usd": 0.02}
    assert not gate_mod.check_metrics(metrics, {}, {"cost_per_query_usd": 0.001})[0].passed
    assert gate_mod.check_metrics(metrics, {}, {"cost_per_query_usd": 0.05})[0].passed


def test_check_metrics_threshold_is_inclusive_on_both_directions():
    assert gate_mod.check_metrics({"m": 0.5}, {"m": 0.5}, {})[0].passed
    assert gate_mod.check_metrics({"m": 0.5}, {}, {"m": 0.5})[0].passed


def test_gate_rejects_an_unknown_metric_rather_than_silently_passing():
    with pytest.raises(KeyError, match="unknown metric"):
        gate_mod.check_metrics({"recall@5": 1.0}, {"recall@99": 0.5}, {})


def test_a_gate_with_no_thresholds_is_an_error_not_a_pass():
    with pytest.raises(ValueError, match="checks nothing"):
        gate_mod.check_metrics({"recall@5": 1.0}, {}, {})


def test_gate_report_exit_code_and_rendering():
    green = gate_mod.GateReport("c", "d", gate_mod.check_metrics({"m": 1.0}, {"m": 0.5}, {}),
                                {"n_queries": 10})
    red = gate_mod.GateReport("c", "d", gate_mod.check_metrics({"m": 0.1}, {"m": 0.5}, {}),
                              {"n_queries": 10})
    assert green.passed and green.exit_code == gate_mod.EXIT_PASS == 0
    assert not red.passed and red.exit_code == gate_mod.EXIT_FAIL == 1
    assert "GREEN" in green.render() and "PASS" in green.render()
    assert "RED" in red.render() and "blocks the merge" in red.render()


def test_load_golden_defaults_missing_sections():
    golden = gate_mod.load_golden("configs/golden_gate.yaml")
    assert golden["config"] == "configs/naive.yaml"
    assert golden["min"]["recall@5"] == 0.90
    assert golden["max"]["cost_per_query_usd"] == 0.001


# ── Gate CLI: the exit codes are the contract ─────────────────────────────────


def test_cli_exits_zero_when_the_golden_bar_is_met(capsys):
    code = gate_cli.main(["--golden", "configs/golden_gate.yaml"])
    assert code == 0
    assert "GREEN" in capsys.readouterr().out


def test_cli_exits_one_on_a_regression(capsys):
    # same config, an unmeetable bar → the gate must block, not warn
    code = gate_cli.main(["configs/naive.yaml", "--min", "recall@1=0.999"])
    assert code == 1
    assert "RED" in capsys.readouterr().out


def test_cli_exits_two_on_usage_errors():
    assert gate_cli.main(["configs/nope.yaml", "--min", "recall@5=0.9"]) == 2   # missing config
    assert gate_cli.main(["configs/naive.yaml"]) == 2                            # no thresholds
    assert gate_cli.main(["configs/naive.yaml", "--min", "recall@5"]) == 2       # bad syntax
    assert gate_cli.main(["configs/naive.yaml", "--min", "recall@5=abc"]) == 2   # non-numeric
    assert gate_cli.main(["configs/naive.yaml", "--golden", "configs/nope.yaml"]) == 2
    assert gate_cli.main(["configs/naive.yaml", "--min", "nonexistent@5=0.9"]) == 2


def test_parse_thresholds_handles_the_at_sign_in_metric_names():
    assert gate_cli._parse_thresholds(["recall@5=0.9", "mrr=0.8"]) == {"recall@5": 0.9, "mrr": 0.8}


# ── Tracing ───────────────────────────────────────────────────────────────────


def test_trace_round_trip_preserves_query_answer_spans_and_retrieval_set():
    tracer = Tracer(":memory:")
    q = Query("q1", "what is bm25?", gold_answer="a ranking function")
    retrieved = [Scored(Chunk("d1::0", "d1", "bm25 text"), 0.91),
                 Scored(Chunk("d2::0", "d2", "other text"), 0.42)]
    result = PipelineResult(
        query=q, retrieved=retrieved, context="ctx", answer="a sparse ranking function",
        stage_latency_ms={"embed_query_ms": 0.5, "retrieve_ms": 2.0, "generate_ms": 7.5,
                          "total_ms": 10.0},
        tokens={"in": 120, "out": 30}, cost_usd=0.00042, extra={"has_citation": True},
    )
    tid = tracer.record(result, config="naive", dataset="builtin_mini")

    t = tracer.trace(tid)
    assert t["query_id"] == "q1" and t["query_text"] == "what is bm25?"
    assert t["answer"] == "a sparse ranking function"
    assert t["config"] == "naive" and t["dataset"] == "builtin_mini"
    assert t["total_ms"] == pytest.approx(10.0)
    assert t["tokens_in"] == 120 and t["tokens_out"] == 30
    assert t["cost_usd"] == pytest.approx(0.00042)
    assert t["n_retrieved"] == 2
    assert t["extra"] == {"has_citation": True}          # json round-trip

    # spans: every stage except the total rollup
    spans = {s["stage"]: s["latency_ms"] for s in t["spans"]}
    assert spans == {"embed_query_ms": 0.5, "retrieve_ms": 2.0, "generate_ms": 7.5}
    assert "total_ms" not in spans

    # the retrieval set is the whole point — ids, order and scores survive
    assert [r["chunk_id"] for r in t["retrieved"]] == ["d1::0", "d2::0"]
    assert [r["rank"] for r in t["retrieved"]] == [1, 2]
    assert t["retrieved"][0]["score"] == pytest.approx(0.91)
    tracer.close()


def test_tracer_summary_aggregates_latency_cost_and_stage_breakdown():
    tracer = Tracer(":memory:")
    for i in range(4):
        tracer.record(
            PipelineResult(
                query=Query(f"q{i}", "t"), retrieved=[], context="", answer="a",
                stage_latency_ms={"retrieve_ms": 1.0, "generate_ms": 3.0, "total_ms": 4.0},
                tokens={"in": 10, "out": 5}, cost_usd=0.001,
            ),
            config="c1",
        )
    s = tracer.summary(config="c1")
    assert s["n_traces"] == 4
    assert s["latency_p50_ms"] == pytest.approx(4.0)
    assert s["total_cost_usd"] == pytest.approx(0.004)
    assert s["cost_per_query_usd"] == pytest.approx(0.001)
    assert s["tokens_in"] == 40 and s["tokens_out"] == 20
    assert s["stage_mean_ms"]["generate_ms"] == pytest.approx(3.0)
    tracer.close()


def test_tracer_summary_is_empty_not_crashing_when_nothing_recorded():
    tracer = Tracer(":memory:")
    assert tracer.summary() == {"n_traces": 0}
    assert tracer.traces() == []
    assert tracer.trace("nope") is None
    tracer.close()


def test_traces_are_filtered_by_config():
    tracer = Tracer(":memory:")
    for cfg in ("a", "a", "b"):
        tracer.record(PipelineResult(query=Query("q", "t"), retrieved=[], context="", answer="x",
                                     stage_latency_ms={"total_ms": 1.0}), config=cfg)
    assert len(tracer.traces(config="a")) == 2
    assert len(tracer.traces(config="b")) == 1
    assert len(tracer.traces()) == 3
    tracer.close()


def test_traced_pipeline_records_every_query_and_stays_transparent():
    from harness import config as cfgmod
    from harness.data import load_dataset

    cfg, pipeline = cfgmod.load("configs/naive.yaml")
    tracer = Tracer(":memory:")
    traced = TracedPipeline(pipeline, tracer, config="naive", dataset=cfg["dataset"])

    docs, queries = load_dataset(cfg["dataset"])
    traced.build(docs)
    results = [traced.run_query(q) for q in queries]

    assert len(traced.trace_ids) == len(queries) == len(results)
    assert tracer.summary(config="naive")["n_traces"] == len(queries)
    # transparent: undefined attributes fall through to the wrapped pipeline
    assert traced.embedder.name == pipeline.embedder.name
    assert traced.build_stats["n_docs"] == len(docs)
    # and the wrapper does not alter the pipeline's output
    assert results[0].answer == pipeline.run_query(queries[0]).answer
    tracer.close()


def test_traced_pipeline_persists_to_a_file_db(tmp_path):
    from harness import config as cfgmod
    from harness.data import load_dataset

    db = tmp_path / "sub" / "traces.sqlite3"      # parent dir must be created
    cfg, pipeline = cfgmod.load("configs/naive.yaml")
    tracer = Tracer(db)
    traced = TracedPipeline(pipeline, tracer, config="naive")
    docs, queries = load_dataset(cfg["dataset"])
    traced.build(docs)
    traced.run_query(queries[0])
    tracer.close()

    assert db.exists()
    reopened = Tracer(db)                          # durable across connections
    assert reopened.summary()["n_traces"] == 1
    reopened.close()
