"""Ingestion & parsing (Phase 1) — raw files → clean, deduped, metadata-rich Documents.

Deps-free for text/markdown/HTML/CSV; PDF/DOCX/OCR parsers register only if their optional
deps are installed. Exposes `ingest_dir` (the Phase 1 entrypoint) + a quality report.
"""

from modules.ingestion.ingestor import IngestionReport, ingest_dir  # noqa: F401
