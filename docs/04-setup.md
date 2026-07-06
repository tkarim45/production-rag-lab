# 04 — Setup (Apple M1, 8 GB)

## 0. Golden rules

- Never conda `base` / system Python (hook-enforced). Use `personal`; `claude` for scratch.
- Usable memory ≈ 4–5 GB after the OS. Corpora on the laptop = 10k–100k docs; the
  million-doc run is an optional cloud burst (the code path is the same).
- Cloud creds from global `~/.env`; never paste/commit secrets.

## 1. Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate personal
pip install -r requirements.txt
```

`requirements.txt` (grows per phase): `sentence-transformers faiss-cpu hnswlib rank-bm25
datasets beir ragas anthropic[bedrock] llama-cpp-python mlx mlx-lm networkx fastapi
uvicorn redis duckdb scikit-learn numpy pandas matplotlib mlflow dvc pytest ruff black
python-dotenv datasketch streamlit`.

Phase-specific extras: SPLADE (`transformers`), ColBERT (`ragatouille` or `colbert-ai`),
DiskANN (`diskannpy`), OCR (`pytesseract` + `brew install tesseract`), PDF layout
(`pymupdf`, `unstructured`).

## 2. Local generation model (cheap tier + offline)

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF','qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='models')"
```
Route: local model for cheap/bulk eval passes, Claude API for judged/hard queries.

## 3. Embeddings on the M1

- Default: a small sentence-transformer (e.g. `bge-small-en-v1.5`, `all-MiniLM-L6-v2`) —
  fast, low memory.
- Quantization module (Phase 3): int8 / binary embeddings for memory + speed.
- Embedder fine-tune (Phase 3): MLX-LoRA on a small base.
- Cache embeddings to disk (DuckDB/npz) — recomputing across benchmark runs is the biggest
  time sink; cache aggressively.

## 4. Datasets (Phase 0)

```bash
# BEIR subset (has qrels for retrieval metrics)
python -m harness.data.download --beir fiqa scifact nfcorpus
# QA set with gold answers (EM/F1 + multi-hop)
python -m harness.data.download --qa hotpotqa squad
# domain corpus + synthetic eval set
python -m harness.data.download --domain sec_filings
```
Keep corpora under `data/` (gitignored). Start small (10k docs) to iterate fast.

## 5. Cloud creds

```bash
set -a; source ~/.env; set +a   # ANTHROPIC_API_KEY / AWS_* for judge + generation
```

## 6. Running benchmarks

```bash
make bench configs/naive.yaml        # run one pipeline config → metrics JSON
make bench-all                       # run a phase's config grid → leaderboard
make leaderboard                     # render results/ tables + charts
make test                            # metric unit tests + pipeline smoke tests
```

## 7. Scaling notes (Phase 15)

- On the M1: prove the code path on 100k docs with HNSW / DiskANN (on-disk).
- Million-doc run: documented cloud burst (bigger instance or managed vector DB). Same
  adapters, larger data — never required on the laptop.
- Cache + memory-map indexes; stream ingestion; don't hold the whole corpus in RAM.

## 8. Troubleshooting

- **Benchmark runs are slow** → cache embeddings; use the local model for bulk passes,
  Claude only for judged metrics; subsample the eval set during development.
- **OOM building an index** → use IVF/HNSW with PQ or DiskANN (on-disk), not Flat, on
  large corpora.
- **Judge cost blowing up** → sample a fraction for LLM-judged metrics; run cheap metrics
  (EM/F1/retrieval) on the full set.
- **Metal not used** → rebuild `llama-cpp-python` with `CMAKE_ARGS="-DGGML_METAL=on"`.
