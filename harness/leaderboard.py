"""Leaderboard — append a config's metrics as a JSON row and render a markdown table.

Every `bench` run appends one row to `results/leaderboard.jsonl` (append-only history) and
re-renders `results/leaderboard.md`. Rows carry the config name + the stage component names
so the table reads as "this combination scored this." Sorted by a headline column.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent.parent / "results"
JSONL = RESULTS / "leaderboard.jsonl"
MD = RESULTS / "leaderboard.md"

# columns shown in the rendered table (in order); others stay in the JSONL
_COLS = [
    ("config", "config"),
    ("recall@5", "recall@5"),
    ("mrr", "mrr"),
    ("ndcg@10", "ndcg@10"),
    ("token_f1", "token_f1"),
    ("em", "em"),
    ("latency_p50_ms", "latency_p50_ms"),
    ("cost_per_query_usd", "cost/q $"),
]


def append_row(row: dict[str, Any]) -> None:
    RESULTS.mkdir(exist_ok=True)
    with JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if v == float("inf"):
            return "∞"
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)


def render() -> str:
    if not JSONL.exists():
        return "_No runs yet._"
    rows = [json.loads(line) for line in JSONL.read_text().splitlines() if line.strip()]
    # latest row per config wins the table; full history stays in the jsonl
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r.get("config", "?")] = r
    ordered = sorted(latest.values(), key=lambda r: (-r.get("recall@5", 0), r.get("config", "")))

    header = "| " + " | ".join(label for _, label in _COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    lines = [
        "# Leaderboard",
        "",
        f"_{len(latest)} config(s); latest run each. Full history in `leaderboard.jsonl`._",
        "",
        header,
        sep,
    ]
    for r in ordered:
        lines.append("| " + " | ".join(_fmt(r.get(key, "—")) for key, _ in _COLS) + " |")
    md = "\n".join(lines) + "\n"
    MD.write_text(md)
    return md
