.PHONY: install test bench bench-claude capstone gate ingest scale leaderboard components clean

# Personal-project env per docs/04-setup.md. Override: make PY=... <target>
PY ?= python
CONFIG ?= configs/naive.yaml

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

# `make bench` (naive) or `make bench CONFIG=configs/whatever.yaml`.
# Also supports `make bench configs/x.yaml` — the path goal is swallowed by the % rule below.
bench:
	$(PY) -m harness.runner $(filter-out bench,$(MAKECMDGOALS)) $(if $(filter-out bench,$(MAKECMDGOALS)),,$(CONFIG))

bench-claude:
	$(PY) -m harness.runner configs/naive_claude.yaml

# Phase 16 — the composed best-of-every-layer pipeline (needs .env for the Claude generator)
capstone:
	$(PY) -m harness.runner configs/capstone.yaml

# Phase 13 — the golden-eval regression gate CI runs (key-free). exit 1 = regression.
gate:
	$(PY) -m harness.gate --golden configs/golden_gate.yaml

# Phase 1 — ingestion quality report
ingest:
	$(PY) -m harness.ingest data/raw_samples

# Phase 15 — scaling study (1k -> 50k vectors). Slow: ~2 min.
scale:
	$(PY) -m harness.scale_bench --sizes 1000 10000 50000

leaderboard:
	$(PY) -c "from harness import leaderboard; print(leaderboard.render())"

components:
	$(PY) -c "import modules, json; from harness.registry import all_components; print(json.dumps(all_components(), indent=2))"

clean:
	rm -rf results/*.json results/*.jsonl results/*.md .pytest_cache **/__pycache__

# swallow a config path passed as a goal (so `make bench configs/x.yaml` works)
%:
	@:
