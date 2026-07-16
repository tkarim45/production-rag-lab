"""Per-request tracing (Phase 13) — every query becomes a durable, queryable row.

A RAG pipeline that only prints an aggregate score is unoperable: when a user says "it gave
me a bad answer at 3pm", you need *that request's* retrieval set, scores, stage latencies,
tokens and cost — not a leaderboard mean. This is the traces-and-spans model, minus the
vendor:

    traces      one row per request   (query, answer, total latency, tokens, cost)
    spans       one row per stage     (embed_query / retrieve / rerank / assemble / generate)
    retrievals  one row per hit       (rank, chunk_id, doc_id, score)

Deps: `sqlite3` from the stdlib. No OTel, no collector, no daemon — the schema *is* the
lesson, and it's the same schema an OTel exporter would land in.

Two ways in:
- `Tracer.record(result)` — the runner-side hook: hand it a `PipelineResult`, get a trace id.
- `TracedPipeline(pipeline, tracer)` — a transparent wrapper; `run_query` traces as a side
  effect, so any existing code path becomes instrumented without touching the pipeline.

CLI:
    python -m modules.ops.tracing --config configs/naive.yaml   # run traced, print dashboard
    python -m modules.ops.tracing --summary                     # dashboard from the stored db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.contract import PipelineResult
from harness.metrics import efficiency as E

# gitignored (`*.sqlite3`) — traces are runtime state, not a repo artifact
DEFAULT_DB = Path(__file__).resolve().parents[2] / "results" / "traces.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id    TEXT PRIMARY KEY,
    ts          REAL NOT NULL,
    config      TEXT NOT NULL,
    dataset     TEXT,
    query_id    TEXT,
    query_text  TEXT,
    answer      TEXT,
    n_retrieved INTEGER NOT NULL,
    total_ms    REAL NOT NULL,
    tokens_in   INTEGER NOT NULL DEFAULT 0,
    tokens_out  INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL NOT NULL DEFAULT 0.0,
    extra       TEXT
);
CREATE TABLE IF NOT EXISTS spans (
    trace_id   TEXT NOT NULL,
    stage      TEXT NOT NULL,
    latency_ms REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS retrievals (
    trace_id TEXT NOT NULL,
    rank     INTEGER NOT NULL,
    chunk_id TEXT NOT NULL,
    doc_id   TEXT NOT NULL,
    score    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_config ON traces(config);
CREATE INDEX IF NOT EXISTS idx_spans_trace   ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_retr_trace    ON retrievals(trace_id);
"""

# stage latencies that are rollups, not spans
_NOT_A_SPAN = ("total_ms",)


class Tracer:
    """Writes PipelineResults to SQLite and reads them back for a dashboard."""

    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        if str(db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ── write ────────────────────────────────────────────────────────────────
    def record(self, result: PipelineResult, config: str = "adhoc",
               dataset: str | None = None, trace_id: str | None = None) -> str:
        """Persist one request. Returns the trace id (the handle a support ticket quotes)."""
        tid = trace_id or uuid.uuid4().hex[:16]
        lat = result.stage_latency_ms
        self.conn.execute(
            "INSERT INTO traces (trace_id, ts, config, dataset, query_id, query_text, answer, "
            "n_retrieved, total_ms, tokens_in, tokens_out, cost_usd, extra) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (tid, time.time(), config, dataset, result.query.query_id, result.query.text,
             result.answer, len(result.retrieved), float(lat.get("total_ms", 0.0)),
             int(result.tokens.get("in", 0)), int(result.tokens.get("out", 0)),
             float(result.cost_usd), json.dumps(result.extra, default=str)),
        )
        self.conn.executemany(
            "INSERT INTO spans (trace_id, stage, latency_ms) VALUES (?,?,?)",
            [(tid, stage, float(ms)) for stage, ms in lat.items() if stage not in _NOT_A_SPAN],
        )
        self.conn.executemany(
            "INSERT INTO retrievals (trace_id, rank, chunk_id, doc_id, score) VALUES (?,?,?,?,?)",
            [(tid, i, s.chunk.chunk_id, s.chunk.doc_id, float(s.score))
             for i, s in enumerate(result.retrieved, start=1)],
        )
        self.conn.commit()
        return tid

    def record_all(self, results: list[PipelineResult], config: str = "adhoc",
                   dataset: str | None = None) -> list[str]:
        return [self.record(r, config=config, dataset=dataset) for r in results]

    # ── read ─────────────────────────────────────────────────────────────────
    def _rows(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def traces(self, config: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if config:
            return self._rows("SELECT * FROM traces WHERE config=? ORDER BY ts DESC LIMIT ?",
                              (config, limit))
        return self._rows("SELECT * FROM traces ORDER BY ts DESC LIMIT ?", (limit,))

    def trace(self, trace_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM traces WHERE trace_id=?", (trace_id,))
        if not rows:
            return None
        t = rows[0]
        t["extra"] = json.loads(t["extra"] or "{}")
        t["spans"] = self.spans_for(trace_id)
        t["retrieved"] = self.retrievals_for(trace_id)
        return t

    def spans_for(self, trace_id: str) -> list[dict[str, Any]]:
        return self._rows("SELECT stage, latency_ms FROM spans WHERE trace_id=?", (trace_id,))

    def retrievals_for(self, trace_id: str) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT rank, chunk_id, doc_id, score FROM retrievals WHERE trace_id=? ORDER BY rank",
            (trace_id,),
        )

    def summary(self, config: str | None = None) -> dict[str, Any]:
        """The text dashboard's data: volume, latency percentiles, cost, stage breakdown."""
        traces = self.traces(config=config, limit=10_000)
        if not traces:
            return {"n_traces": 0}
        totals = [t["total_ms"] for t in traces]
        out: dict[str, Any] = {"n_traces": len(traces)}
        out.update({f"latency_{k}": v for k, v in E.latency_summary(totals).items()})
        out["total_cost_usd"] = sum(t["cost_usd"] for t in traces)
        out["cost_per_query_usd"] = out["total_cost_usd"] / len(traces)
        out["tokens_in"] = sum(t["tokens_in"] for t in traces)
        out["tokens_out"] = sum(t["tokens_out"] for t in traces)
        # mean latency per stage — which stage owns the p50 budget
        where = "WHERE t.config=?" if config else ""
        params = (config,) if config else ()
        out["stage_mean_ms"] = {
            r["stage"]: r["mean_ms"] for r in self._rows(
                f"SELECT s.stage, AVG(s.latency_ms) AS mean_ms FROM spans s "
                f"JOIN traces t ON t.trace_id = s.trace_id {where} "
                f"GROUP BY s.stage ORDER BY mean_ms DESC", params)
        }
        return out

    def close(self) -> None:
        self.conn.close()


@dataclass
class TracedPipeline:
    """Transparent wrapper: same interface as `Pipeline`, every query lands in the tracer.

    `run_query` is the only overridden behaviour; everything else (build, build_stats,
    chunker, …) delegates, so a TracedPipeline is a drop-in wherever a Pipeline goes.
    """

    pipeline: Any
    tracer: Tracer
    config: str = "adhoc"
    dataset: str | None = None
    trace_ids: list[str] = field(default_factory=list)

    def build(self, docs) -> None:
        self.pipeline.build(docs)

    def run_query(self, query) -> PipelineResult:
        result = self.pipeline.run_query(query)
        self.trace_ids.append(
            self.tracer.record(result, config=self.config, dataset=self.dataset)
        )
        return result

    def __getattr__(self, name: str):
        # dataclass fields resolve normally; anything else is the wrapped pipeline's.
        if name in ("pipeline", "tracer"):
            raise AttributeError(name)
        return getattr(self.pipeline, name)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _print_dashboard(tracer: Tracer, config: str | None) -> None:
    s = tracer.summary(config=config)
    label = config or "ALL configs"
    print(f"\n=== trace dashboard: {label} ({tracer.db_path}) ===")
    if not s["n_traces"]:
        print("  no traces recorded")
        return
    print(f"  requests            {s['n_traces']}")
    print(f"  latency p50/p95/p99 {s['latency_p50_ms']:.2f} / {s['latency_p95_ms']:.2f} / "
          f"{s['latency_p99_ms']:.2f} ms")
    print(f"  tokens in/out       {s['tokens_in']} / {s['tokens_out']}")
    print(f"  cost total / query  ${s['total_cost_usd']:.6f} / ${s['cost_per_query_usd']:.6f}")
    print("  stage mean latency:")
    for stage, ms in s["stage_mean_ms"].items():
        share = 100 * ms / s["latency_mean_ms"] if s["latency_mean_ms"] else 0.0
        print(f"    {stage:<18} {ms:8.3f} ms  ({share:4.1f}% of mean)")
    recent = tracer.traces(config=config, limit=3)
    print("  most recent traces:")
    for t in recent:
        print(f"    {t['trace_id']}  {t['query_id']:<6} {t['total_ms']:7.2f} ms  "
              f"{t['n_retrieved']} hits  \"{t['query_text'][:48]}\"")
    drill = tracer.trace(recent[0]["trace_id"])
    print(f"  drill-down {drill['trace_id']} — retrieval set:")
    for r in drill["retrieved"][:5]:
        print(f"    #{r['rank']} {r['chunk_id']:<14} score={r['score']:.4f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-lab-tracing", description="Per-request RAG tracing.")
    ap.add_argument("--config", help="pipeline YAML to run under tracing (e.g. configs/naive.yaml)")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--summary", action="store_true", help="print the dashboard for stored traces")
    ap.add_argument("--label", help="config label to filter/record under (default: config stem)")
    args = ap.parse_args(argv)

    tracer = Tracer(args.db)
    label = args.label
    if args.config:
        from harness import config as cfgmod

        cfg, pipeline = cfgmod.load(args.config)
        label = label or Path(args.config).stem
        traced = TracedPipeline(pipeline, tracer, config=label, dataset=cfg["dataset"])
        from harness.data import load_dataset

        docs, queries = load_dataset(cfg["dataset"])
        traced.build(docs)
        for q in queries:
            traced.run_query(q)
        print(f"traced {len(traced.trace_ids)} requests → {tracer.db_path}")
    if args.config or args.summary:
        _print_dashboard(tracer, label)
    else:
        ap.error("pass --config to record traces and/or --summary to read them")
    tracer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
