"""Optional FastAPI app (Phase 12) — the pipeline behind an HTTP endpoint.

    pip install -e ".[serve]"
    python -m modules.serving.api configs/naive.yaml

    GET  /health           → readiness + what was actually built (chunks, dim, build time)
    POST /ask {"query": …} → answer + citations + the per-stage latency breakdown + cost

`fastapi` is an optional dependency: this module imports cleanly without it (the core stays
numpy + pyyaml), and `create_app` raises with an install hint instead of an ImportError
traceback. The index is built once at app creation, not per request — the whole point of the
serving layer is that build cost is amortized and only the per-query stages are on the hot
path. Every response echoes `latency_ms`, so the thing you'd page on is in the payload.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:  # optional dep — absence must degrade, not explode
    import fastapi  # noqa: F401
    from pydantic import BaseModel, Field

    HAVE_FASTAPI = True
except Exception:  # pragma: no cover - exercised only where fastapi is absent
    HAVE_FASTAPI = False

_INSTALL_HINT = (
    "the serving API needs fastapi + uvicorn (optional deps). Install with: "
    'pip install -e ".[serve]"'
)

if HAVE_FASTAPI:
    # module scope, not inside create_app: FastAPI resolves these annotations against module
    # globals (this file uses postponed annotations), so a nested class reads as a query param.

    class AskRequest(BaseModel):
        query: str = Field(..., min_length=1)

    class AskResponse(BaseModel):
        answer: str
        citations: list[str]
        latency_ms: dict[str, float]
        cost_usd: float
        cache_hit: bool


def create_app(config_path: str | Path = "configs/naive.yaml"):
    """Build the pipeline from a config and wrap it in a FastAPI app."""
    if not HAVE_FASTAPI:
        raise RuntimeError(_INSTALL_HINT)

    from fastapi import FastAPI, HTTPException

    import modules.serving  # noqa: F401  registers the Phase 12 cache wrappers
    from harness import config as cfgmod
    from harness.contract import Query
    from harness.data import load_dataset

    cfg, pipeline = cfgmod.load(config_path)
    docs, _ = load_dataset(cfg["dataset"])
    pipeline.build(docs)  # once, at startup — never on the request path

    app = FastAPI(title="production-rag-lab", version="0.1.0")
    app.state.config_name = Path(str(config_path)).stem
    app.state.pipeline = pipeline
    app.state.n_requests = 0

    @app.get("/health")
    def health() -> dict[str, Any]:
        stats = pipeline.build_stats or {}
        return {
            "status": "ok" if pipeline.build_stats else "building",
            "config": app.state.config_name,
            "dataset": cfg["dataset"],
            "n_docs": stats.get("n_docs"),
            "n_chunks": stats.get("n_chunks"),
            "embed_dim": stats.get("embed_dim"),
            "build_ms": round(sum(v for k, v in stats.items() if k.endswith("_ms")), 2),
            "n_requests": app.state.n_requests,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        text = req.query.strip()
        if not text:
            raise HTTPException(status_code=422, detail="query must not be empty")
        app.state.n_requests += 1
        result = pipeline.run_query(Query(query_id=f"req-{app.state.n_requests}", text=text))
        return AskResponse(
            answer=result.answer,
            citations=result.retrieved_chunk_ids,
            latency_ms={k: round(v, 3) for k, v in result.stage_latency_ms.items()},
            cost_usd=result.cost_usd,
            cache_hit=bool(result.extra.get("cache_hit", False)),
        )

    return app


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config = argv[0] if argv else "configs/naive.yaml"
    if not HAVE_FASTAPI:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2
    try:
        import uvicorn
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2
    uvicorn.run(create_app(config), host="127.0.0.1", port=8000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
