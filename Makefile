.PHONY: install test bench bench-claude leaderboard components clean

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

leaderboard:
	$(PY) -c "from harness import leaderboard; print(leaderboard.render())"

components:
	$(PY) -c "import modules, json; from harness.registry import all_components; print(json.dumps(all_components(), indent=2))"

clean:
	rm -rf results/*.json results/*.jsonl results/*.md .pytest_cache **/__pycache__

# swallow a config path passed as a goal (so `make bench configs/x.yaml` works)
%:
	@:
