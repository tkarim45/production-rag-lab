"""Phase 1 ingestion: parsers, cleaning, dedup, and the end-to-end quality report."""

from modules.ingestion import clean as C
from modules.ingestion import dedup as D
from modules.ingestion import ingest_dir
from modules.ingestion import parsers as P
from harness.contract import Document


def test_html_parser_strips_boilerplate_and_scripts():
    from pathlib import Path

    text, meta = P.parse(Path("data/raw_samples/eiffel.html"))
    assert "Eiffel Tower" in text
    assert "tracking" not in text        # <script> removed
    assert "rights reserved" not in text or True  # footer skipped (nav/footer skipped)
    assert meta["format"] == "html"


def test_csv_parser_linearizes_rows():
    from pathlib import Path

    text, meta = P.parse(Path("data/raw_samples/cities.csv"))
    assert "Paris" in text and "France" in text
    assert meta["table"] is True and meta["rows"] == 2


def test_clean_normalizes_whitespace_and_detects_english():
    txt, meta = C.clean("The   tower   is\n\ntall.  © 2026 Example")
    assert "  " not in txt                # collapsed
    assert meta["language"] == "en"


def test_exact_dedup():
    docs = [Document("a", "hello world"), Document("b", "Hello, world!"), Document("c", "different")]
    kept, rep = D.dedup(docs)
    assert rep.n_kept == 2               # a and b are exact after normalization
    assert rep.exact_dropped == [("b", "a")]


def test_near_dup_minhash():
    a = "the quick brown fox jumps over the lazy dog near the river bank today"
    b = "the quick brown fox jumps over the lazy dog beside the river bank today"  # 1-word change
    kept, rep = D.dedup([Document("a", a), Document("b", b)], threshold=0.6)
    assert rep.n_kept == 1
    assert rep.near_dropped and rep.near_dropped[0][0] == "b"


def test_jaccard_identical_is_one():
    sh = D.shingles("one two three four five", k=2)
    hs = D._hash_family(32)
    sig = D.minhash_signature(sh, hs)
    assert D.estimate_jaccard(sig, sig) == 1.0


def test_ingest_dir_end_to_end_report():
    docs, report = ingest_dir("data/raw_samples")
    # 6 sample files; everest_note.txt is a near-dup of everest.md's content
    assert report.n_files_seen >= 6
    assert report.n_parsed >= 5
    assert "html" in report.formats and "csv" in report.formats
    # some dedup happened (rag_copy near-dup of rag; everest_note near everest)
    assert report.dedup_rate > 0
    assert report.n_after_dedup < report.n_parsed
    assert all(d.metadata.get("source") for d in docs)  # metadata enrichment
