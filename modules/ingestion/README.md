# Module: Ingestion & parsing (Phase 1)

**Lesson.** Ingestion is where most RAG quality is silently won or lost. Garbage parsing
(dropped tables, boilerplate treated as content, undeduped near-copies) poisons every
downstream layer, and no reranker recovers from a corpus that never contained the answer
cleanly. The production concerns: parse fidelity across formats, cleaning without dropping
answerable text, and deduplication (exact **and** near-dup) so the index isn't dominated by
copies.

## What's implemented
- **Parsers** (`parsers.py`), deps-free for `.txt/.md/.html/.csv` (HTML strips
  script/style/nav/footer boilerplate; CSV linearizes rows to retrievable prose). `.pdf`
  (pymupdf) and `.docx` (python-docx) register only if the optional dep is installed.
- **Cleaning** (`clean.py`), whitespace normalization, boilerplate-line stripping, a cheap
  stopword language guess. Conservative on purpose: aggressive cleaning drops answers.
- **Dedup** (`dedup.py`), exact via normalized SHA-1; **near-dup via from-scratch MinHash**
  (k-shingles → n hash-min signatures → estimated Jaccard → threshold). No `datasketch` dep,
  so the mechanism is legible.
- **Ingestor** (`ingestor.py`), dir → clean, deduped, metadata-rich `Document`s + an
  `IngestionReport` (parse fidelity, dedup rate, format/language counts).

## Result (on `data/raw_samples/`, 6 files)

`python -m harness.ingest data/raw_samples`:

| metric | value |
|---|---|
| files seen | 6 |
| parsed | 6 (csv 1, html 1, markdown 1, txt 3) |
| languages | en ×5, unknown ×1 (the CSV table) |
| after dedup | 5 (**dedup rate 16.7%**) |
| dropped | `rag_copy` → exact dup of `rag` |

**Honest finding.** `rag_copy.txt` was authored as a *near*-dup (hyphenation changed:
"knowledge-intensive" → "knowledge intensive"), but the dedup normalizer folds punctuation,
so after normalization it became an **exact** dup and was caught by the cheaper hash before
MinHash ran. Lesson: your normalization policy determines whether a pair is "exact" or
"near", they aren't fixed categories, and the cheap exact check subsumes near-dups your
normalizer happens to collapse. `everest_note.txt` (a genuinely reworded Everest sentence)
was **kept**, below the Jaccard threshold, which is the correct, conservative call.

## Next
Phase 2 (chunking) consumes this cleaned corpus. The tiny sample set is a mechanism demo;
Phase 1's `.[data]` step swaps in a real multi-format corpus where parse-fidelity and
near-dup rates actually bite.
