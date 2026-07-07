"""CLI: ingest a directory of raw files → Documents + a quality report.

    python -m harness.ingest data/raw_samples

Prints the ingestion quality report (parse fidelity, dedup rate, counts) and writes it to
results/ingestion_report.json. This is the Phase 1 deliverable; later phases consume the
cleaned corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from modules.ingestion import ingest_dir

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-lab-ingest")
    ap.add_argument("root", nargs="?", default="data/raw_samples")
    ap.add_argument("--threshold", type=float, default=0.8, help="near-dup Jaccard threshold")
    args = ap.parse_args(argv)

    if not Path(args.root).exists():
        print(f"path not found: {args.root}", file=sys.stderr)
        return 2

    docs, report = ingest_dir(args.root, dedup_threshold=args.threshold)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ingestion_report.json").write_text(json.dumps(report.to_dict(), indent=2))

    print(f"\n=== Ingestion report: {args.root} ===")
    print(f"  files seen         {report.n_files_seen}")
    print(f"  parsed             {report.n_parsed}  (formats: {report.formats})")
    print(f"  languages          {report.languages}")
    print(f"  skipped (no parser){report.n_skipped_unsupported}")
    print(f"  after dedup        {report.n_after_dedup}  (dedup rate {report.dedup_rate:.1%})")
    if report.exact_dropped:
        print(f"  exact dups dropped {report.exact_dropped}")
    if report.near_dropped:
        print(f"  near dups dropped  {report.near_dropped}")
    print(f"\nDocuments kept: {[d.doc_id for d in docs]}")
    print(f"Report → {RESULTS / 'ingestion_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
