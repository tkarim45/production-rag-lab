# Module: Security & governance (Phase 14)

**Lesson.** Every phase before this one optimizes *what the retriever fetches*. This one
starts from the opposite assumption — that the corpus is hostile, the caller is not
authorized, and the text is regulated — and asks what your pipeline does then. The
uncomfortable part is that the earlier phases make it worse, not better: a poisoned document
is delivered to the model **by your best retriever**, and the better the retrieval, the more
reliably the attack lands. Retrieval quality is not a security control.

The through-line across all four studies: **security controls belong at the earliest point
where the property is still enforceable.** PII redacted at ingest cannot be leaked by a
future bug. An ACL enforced at retrieval means the model never read the chunk. A delete that
reaches only the index has not deleted anything. Every "we'll filter it on the way out"
design in this module is measurably a leak.

## What's implemented
- **`injection.py`** — `InjectionScreen` (5 regex rule families over retrieved text:
  override / persona-hijack / fake-system-tag / instruction-injection / exfiltration), which
  flags, or **strips at sentence granularity** so a poisoned chunk loses its payload but
  keeps its legitimate body. Plus two assemblers: `screened` (strip, then assemble) and
  `spotlight` (Hines et al. 2024 — delimit retrieved text as DATA, with delimiter-escape
  neutralization and an optional per-request nonce).
- **`pii.py`** — `PIIDetector`: email / SSN / phone / credit-card regexes with priority-based
  overlap resolution and a from-scratch **Luhn** gate on card candidates. Redacts at ingest
  (`redact_documents` → PII never reaches the index) and at output (`redact_text`).
- **`acl.py`** — `Principal` + `can_read` (tenant partition first, then doc ACL, fail-closed
  by default) + `ACLRetriever`, a registered retriever wrapping any base (dense/sparse/hybrid)
  that filters an overfetched pool **before** the top-k cut. `postfiltered_answer` implements
  the wrong design faithfully so it can be measured.
- **`rtbf.py`** — `SecureStore` owns chunks + index + retriever sub-indexes + `AnswerCache`
  behind one `delete(doc_id, mode="hard"|"tombstone")`, because the only way a deletion gets
  reliably completed is to give it a single seam.
- **`audit.py`** — append-only `AuditLog` recording decisions (ids, counts, principals) and
  never content.
- **`attacks.py`** — the labeled sets every number below comes from.

## Results

`python -m modules.security.bench` → `results/security_report.json`. Corpus: `builtin_docs`
(13 docs, 20 queries) + the labeled sets. Generator: deterministic `extractive_mock`.

### 1. Prompt-injection screen — 12 poisoned + 12 benign chunks

| metric | value |
|---|--:|
| precision | **0.900** |
| recall | **0.750** |
| F1 | 0.818 |
| tp / fp / fn / tn | 9 / 1 / 3 / 11 |
| payload reaches context — `concat` | **12 / 12** |
| payload reaches context — `screened` | **3 / 12** |
| delimiter escape — naive fixed marker | **escaped** |
| delimiter escape — `spotlight` | **held** |

Missed: `inj10`, `inj11` (authority framing), `inj12` (character-spaced). False alarm: `ben10`.

### 2. PII detection — 16 labeled texts (8 positive, 8 negative)

| detector | precision | recall | F1 | false positives |
|---|--:|--:|--:|---|
| regex only | 0.706 | **1.000** | 0.828 | 5 (4 Luhn-invalid ids + 1 IMEI) |
| **regex + Luhn** | **0.923** | **1.000** | **0.960** | **1 (the IMEI)** |

Ingest redaction: 13 spans redacted across 16 docs → **0 docs contain PII afterwards**.

### 3. Permission-aware retrieval — analyst (group `research`, tenant `acme`)

| path | secret in context | secret in answer |
|---|--:|--:|
| post-hoc filter (retrieve → generate → strip citations) | **yes (2/2)** | **yes (2/2)** |
| **ACL at retrieval** | **no (0/2)** | **no (0/2)** |

Tenant isolation: an `acme` finance principal reads the acme memo (`True`) and is refused the
`globex` memo (`False`) despite holding a matching `finance` ACL entry. Fail-closed cost on an
un-stamped corpus: **0 results across 20 queries** (vs 100 fail-open).

### 4. Right-to-be-forgotten — deleting `t10` from a hybrid (dense + BM25) store

| mode | retrievable before | retrievable after | cache still serves deleted text | rebuild |
|---|--:|--:|--:|--:|
| tombstone | yes | **yes** | **yes** | 0 ms |
| **hard delete** | yes | **no** | **no** | 0.48 ms |

## Honest findings

1. **The injection detector is 0.900 precision / 0.750 recall, and the recall number is the
   honest one.** Three of twelve attacks walk straight through: two use *polite authority
   framing* ("assistants summarizing this article are required to omit…") and one spaces out
   its characters. None contains an imperative the rules know. This is not a tuning problem —
   a regex matches phrasings, and the space of phrasings is the attacker's to choose. **The
   one false positive is the same coin's other face**: `ben10` ("client software may send a
   request to https://example.org") trips the exfiltration rule, because the difference
   between describing a request and *commanding* one is intent, not vocabulary. Any rule
   aggressive enough to catch inj10 would eat more of the benign half.
2. **Detection F1 is a proxy; delivery is the event.** The number that actually describes the
   system is that **`concat` delivers the payload 12/12 and `screened` delivers it 3/12** —
   and those 3 are exactly the 3 the detector missed. A screen's blind spot *is* its delivery
   rate. This is why `spotlight` is in the module: it removes nothing (verified by test), so
   it has no blind spot to leak through — it changes the frame instead of guessing the
   content. The two compose, which is the honest recommendation.
3. **A delimiter is only a boundary if the attacker cannot write it.** The naive fixed-marker
   assembler is escaped by a chunk containing `<</DATA 1>>` — the model sees the block close
   early and reads the payload as top-level prose. Delimiting is presented as the simple,
   safe technique; it is simple, and it is safe only with the neutralization step nobody
   mentions.
4. **Luhn is worth it and it is not enough.** The gate lifts card precision **0.706 → 0.923**
   at zero recall cost — it kills all four invoice/order/ticket false positives for ~10 lines
   of arithmetic. The remaining false positive is the *interesting* one: a **Luhn-valid
   IMEI**. Luhn is a checksum, not a semantic classifier, and IMEIs use the same checksum, so
   no regex tuning closes this gap — it needs an issuer-prefix (IIN) rule, a different kind of
   evidence. **Precision 0.923 is this detector's ceiling, not its current score.**
5. **Post-hoc ACL filtering leaks 2/2, and it leaks the way that looks clean.** The restricted
   memo reaches the context, the model reads it, the answer repeats the secret verbatim — and
   the citation list is scrubbed of `r01`, so the response *passes inspection*. Stripping the
   citation removes the evidence of the leak, not the leak. ACL-at-retrieval scores 0/2 on
   both context and answer for one structural reason: the chunk was never fetched.
6. **The ACL's real cost is metadata discipline, not code.** Fail-closed on an un-stamped
   corpus returned **0 results across all 20 queries** (vs 100 fail-open). The ACL layer is
   ~80 lines; making every document carry a correct ACL stamp from ingestion is the actual
   project. Fail-closed converts an ingestion gap into an availability bug instead of a
   disclosure — the right trade, and a real bill.
7. **The tombstone fails completely, not partially.** It leaves the doc retrievable *and*
   leaves the cache serving it, because a tombstone is a read-path convention and neither the
   index, the BM25 posting lists, nor the cache check the flag. A hard delete costs a **0.48 ms
   rebuild** on this corpus — which is precisely the problem: the rebuild looks free at 13
   docs and is what makes teams reach for the tombstone at 13 million, where the pressure is
   real and the leak is identical.
8. **The cache is the copy everyone forgets — and it is a verbatim copy.** With the extractive
   generator, the cached answer is provably a substring of the deleted document (asserted in
   `test_hard_delete_purges_the_answer_cache`). A cached LLM answer is source text laundered
   through a model, and it survives an index delete untouched. This is why `AnswerCache`
   records the `doc_ids` each entry derives from: a cache keyed only by query text cannot be
   purged by document, leaving flush-everything or leave-it-there as the only options.

**Caveats.** The labeled sets are small (24 injection chunks, 16 PII texts, 2 restricted
memos) and hand-authored by the same person who wrote the rules they test — precision/recall
here describe *these* rules on *these* attacks, and both numbers would fall on a set authored
by someone else. Treat them as directional, and as a harness that makes a regression visible,
not as a security posture. The `spotlight` results are mechanical claims only (what reaches
the context, whether the delimiter holds); **whether a model honors the DATA frame is a model
property and is not measured here** — that needs a real generator and an attack-success
metric, which is `llm-red-teaming-framework`'s job, not this bench's. Multi-tenancy is
enforced as a metadata partition (single index, filter at retrieval), which is the weakest of
the tenancy models — index-per-tenant is stronger and unmeasured. Source-trust/provenance
scoring is **not implemented** and remains open in `TODO.md`.

## Next
Phase 15 (scaling) is where the RTBF rebuild cost stops being 0.48 ms and becomes the design
constraint that justifies the tombstone everyone reaches for — the honest way to close finding
7 is an index with a real delete primitive, benchmarked at size.
