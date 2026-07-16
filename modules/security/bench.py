"""Phase 14 security bench — every number in `modules/security/README.md` comes from here.

    python -m modules.security.bench

Runs four studies on `builtin_docs` + the labeled sets in `attacks.py`, prints the tables,
and writes `results/security_report.json`. Deps-light and offline: the generator is the
deterministic `extractive_mock`, which is exactly right for the ACL study — it quotes the
context verbatim, so "did the secret reach the answer" is a substring test with no model
judgment in the loop. It is exactly wrong for judging whether spotlighting *works* (that
needs a real model), which is why this bench measures what spotlighting mechanically does
(what reaches the context, whether the delimiter holds) and claims nothing about obedience.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from harness.contract import Chunk, Document, Query, Scored
from harness.data.datasets import load_dataset
from harness.registry import build
from modules.security.acl import ACLRetriever, Principal, can_read, postfiltered_answer
from modules.security.attacks import (
    ACL_DOCS,
    ACL_QUERIES,
    DELIMITER_ESCAPE_CHUNK,
    INJECTION_SET,
    LITIGATION_SECRET,
    PII_SET,
    SEVERANCE_SECRET,
)
from modules.security.audit import AuditLog
from modules.security.injection import InjectionScreen, SpotlightAssembler
from modules.security.pii import PIIDetector
from modules.security.rtbf import AnswerCache, SecureStore

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ── Study 1: prompt injection ─────────────────────────────────────────────────


def bench_injection() -> dict[str, Any]:
    screen = InjectionScreen()

    tp = fp = fn = tn = 0
    missed: list[str] = []
    false_alarms: list[str] = []
    for cid, text, poisoned in INJECTION_SET:
        flagged = screen.is_poisoned(text)
        if poisoned and flagged:
            tp += 1
        elif poisoned and not flagged:
            fn += 1
            missed.append(cid)
        elif not poisoned and flagged:
            fp += 1
            false_alarms.append(cid)
        else:
            tn += 1

    detection = _prf(tp, fp, fn)
    detection.update({"tn": tn, "missed": missed, "false_alarms": false_alarms})

    # What actually reaches the model: does the hostile sentence survive into the context?
    # This is the number that matters — a detector's F1 is a proxy, delivery is the event.
    poisoned_items = [(cid, text) for cid, text, p in INJECTION_SET if p]
    delivered_concat = 0
    delivered_screened = 0
    for _cid, text in poisoned_items:
        delivered_concat += 1                                # concat passes text through whole
        if screen.strip(text) == text:                       # nothing removed → payload lands
            delivered_screened += 1

    # Delimiter escape: can a chunk close its own data block?
    escape_chunk = _chunk("esc", DELIMITER_ESCAPE_CHUNK)
    naive_out = _naive_delimited([escape_chunk])
    spot_out = SpotlightAssembler().assemble(Query("q", "neutron stars"), [escape_chunk])
    naive_escaped = naive_out.count("<</DATA 1>>") > 1
    spot_escaped = spot_out.count("<</DATA 1>>") > 1

    return {
        "detection": detection,
        "delivery": {
            "n_poisoned": len(poisoned_items),
            "concat_delivered": delivered_concat,
            "screened_delivered": delivered_screened,
        },
        "delimiter_escape": {"naive_delimiter": naive_escaped, "spotlight": spot_escaped},
    }


def _chunk(cid: str, text: str) -> Scored:
    return Scored(chunk=Chunk(chunk_id=f"{cid}::0", doc_id=cid, text=text), score=1.0)


def _naive_delimited(scored) -> str:
    """A fixed-marker delimiter with no neutralization — the obvious implementation."""
    blocks = [f"<<DATA {i}>>\n{s.chunk.text}\n<</DATA {i}>>" for i, s in enumerate(scored, 1)]
    return "\n\n".join(blocks)


# ── Study 2: PII ──────────────────────────────────────────────────────────────


def bench_pii() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, luhn in (("regex_only", False), ("regex+luhn", True)):
        detector = PIIDetector(luhn=luhn)
        tp = fp = fn = 0
        fp_examples: list[str] = []
        for cid, text, gold in PII_SET:
            found = {(f.pii_type, f.text) for f in detector.scan(text)}
            want = set(gold)
            tp += len(found & want)
            extra = found - want
            fp += len(extra)
            fn += len(want - found)
            if extra:
                fp_examples.append(f"{cid}:{sorted(v for _, v in extra)[0]}")
        out[label] = {**_prf(tp, fp, fn), "fp_examples": fp_examples}

    # Ingest-time redaction: prove the PII never reaches the index.
    detector = PIIDetector(luhn=True)
    docs = [Document(doc_id=cid, text=text) for cid, text, _ in PII_SET]
    clean, counts = detector.redact_documents(docs)
    leaked = [d.doc_id for d in clean if detector.has_pii(d.text)]
    out["ingest_redaction"] = {
        "docs": len(docs),
        "spans_redacted": sum(counts.values()),
        "by_type": counts,
        "docs_with_pii_after": len(leaked),
    }
    return out


# ── Study 3: ACL ──────────────────────────────────────────────────────────────


def _build_acl_corpus() -> tuple[list[Document], list[Query]]:
    """builtin_docs (stamped public) + the restricted memos."""
    docs, queries = load_dataset("builtin_docs")
    stamped = [
        Document(
            doc_id=d.doc_id,
            text=d.text,
            metadata={**d.metadata, "acl": ["public"], "tenant": "acme"},
        )
        for d in docs
    ]
    return stamped + list(ACL_DOCS), list(queries)


def bench_acl() -> dict[str, Any]:
    docs, builtin_queries = _build_acl_corpus()
    audit = AuditLog()

    chunker = build("chunker", "fixed")
    embedder = build("embedder", "tfidf")
    index = build("index", "flat")
    assembler = build("assembler", "concat")
    generator = build("generator", "extractive_mock")

    chunks = embedder.encode_chunks(chunker.run(docs))
    index.build(chunks)

    # An analyst in the research group: no finance, no legal.
    analyst = Principal.of("analyst", ["research"], tenant="acme")

    acl_retriever = ACLRetriever(base="dense", principal=analyst, default="deny", audit=audit)
    acl_retriever.bind_corpus(chunks, index, embedder)
    plain = build("retriever", "dense")
    plain.bind(index)

    secrets = {"aq01": SEVERANCE_SECRET, "aq02": LITIGATION_SECRET}
    rows = []
    for q in ACL_QUERIES[:2]:                       # the two restricted queries
        q = embedder.encode_query(q)
        secret = secrets[q.query_id]

        # (a) post-hoc filtering: retrieve everything, generate, strip citations after.
        wide = plain.retrieve(q, 5)
        post = postfiltered_answer(q, wide, assembler, generator, analyst)

        # (b) ACL at retrieval: the restricted chunk is never fetched.
        narrow = acl_retriever.retrieve(q, 5)
        acl_context = assembler.assemble(q, narrow)
        acl_answer = generator.generate(q, acl_context)["answer"]

        rows.append(
            {
                "query_id": q.query_id,
                "posthoc_secret_in_context": secret.lower() in post["context"].lower(),
                "posthoc_secret_in_answer": secret.lower() in post["answer"].lower(),
                "posthoc_hidden_citations": post["hidden_citations"],
                "acl_secret_in_context": secret.lower() in acl_context.lower(),
                "acl_secret_in_answer": secret.lower() in acl_answer.lower(),
                "acl_restricted_retrieved": [
                    s.chunk.doc_id for s in narrow if s.chunk.doc_id in ("r01", "r02", "x01")
                ],
            }
        )

    # Tenant isolation: an acme finance principal must not see the globex finance memo,
    # even though the ACL entry matches exactly.
    acme_finance = Principal.of("cfo", ["finance"], tenant="acme")
    globex_chunk = next(c for c in chunks if c.doc_id == "x01")
    acme_chunk = next(c for c in chunks if c.doc_id == "r01")
    tenant_row = {
        "acme_finance_reads_acme_memo": can_read(acme_finance, acme_chunk),
        "acme_finance_reads_globex_memo": can_read(acme_finance, globex_chunk),
    }

    # Fail-closed cost: the same corpus with the ACL stamps missing. This is what an
    # ingestion gap costs you under each default policy.
    bare_docs, _ = load_dataset("builtin_docs")
    bare_embedder = build("embedder", "tfidf")
    bare_index = build("index", "flat")
    bare_chunks = bare_embedder.encode_chunks(build("chunker", "fixed").run(bare_docs))
    bare_index.build(bare_chunks)
    closed = ACLRetriever(base="dense", principal=analyst, default="deny")
    closed.bind_corpus(bare_chunks, bare_index, bare_embedder)
    opened = ACLRetriever(base="dense", principal=analyst, default="allow")
    opened.bind_corpus(bare_chunks, bare_index, bare_embedder)
    n_closed = n_opened = 0
    for q in builtin_queries:
        q = bare_embedder.encode_query(q)
        n_closed += len(closed.retrieve(q, 5))
        n_opened += len(opened.retrieve(q, 5))

    return {
        "leak_table": rows,
        "tenant_isolation": tenant_row,
        "unstamped_corpus": {
            "n_queries": len(builtin_queries),
            "results_fail_closed": n_closed,
            "results_fail_open": n_opened,
        },
        "audit_events": len(audit),
        "acl_denials_logged": len(audit.of_type("acl_deny")),
    }


# ── Study 4: right-to-be-forgotten ────────────────────────────────────────────


def bench_rtbf() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in ("tombstone", "hard"):
        docs, queries = load_dataset("builtin_docs")
        embedder = build("embedder", "tfidf")
        index = build("index", "flat")
        # hybrid → the store must also rebuild the BM25 side-index, which is the read path a
        # naive delete forgets about entirely.
        retriever = build("retriever", "hybrid")
        audit = AuditLog()
        store = SecureStore(index=index, embedder=embedder, retriever=retriever,
                            cache=AnswerCache(), audit=audit)
        store.build(embedder.encode_chunks(build("chunker", "fixed").run(docs)))

        target = "t10"                                  # the DNA doc
        target_query = next(q for q in queries if target in q.relevant_doc_ids)
        target_text = next(d for d in docs if d.doc_id == target).text

        before = store.is_retrievable(target, queries, k=10)

        # Warm the cache the way serving would: an answer derived from the doc. With the
        # extractive generator the cached answer is a *verbatim span of the source*, which
        # makes "the cache is a copy of the document" literally true and checkable.
        generator = build("generator", "extractive_mock")
        assembler = build("assembler", "concat")
        eq = embedder.encode_query(target_query)
        scored = store.search(eq, 5)
        answer = generator.generate(eq, assembler.assemble(eq, scored))["answer"]
        store.cache.set(target_query.text, answer, [s.chunk.doc_id for s in scored])
        cached_answer_is_copy = answer in target_text

        report = store.delete(target, mode=mode)

        after = store.is_retrievable(target, queries, k=10)
        cached = store.cache.get(target_query.text)
        out[mode] = {
            "retrievable_before": before,
            "retrievable_after": after,
            "chunks_after": report["chunks_after"],
            "cache_entries_after": len(store.cache),
            # the honest probe: is the deleted document's text still recoverable verbatim
            # from the cache, regardless of what the index now says?
            "cached_answer_was_verbatim_source": cached_answer_is_copy,
            "cache_still_serves_deleted_text": bool(cached is not None and cached in target_text),
            "rebuild_ms": round(report["rebuild_ms"], 2),
            "delete_events_logged": len(audit.of_type("delete")),
        }
    return out


# ── Entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rag-lab-security")
    ap.add_argument("--json-only", action="store_true")
    args = ap.parse_args(argv)

    report = {
        "injection": bench_injection(),
        "pii": bench_pii(),
        "acl": bench_acl(),
        "rtbf": bench_rtbf(),
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "security_report.json").write_text(json.dumps(report, indent=2, default=str))

    if not args.json_only:
        inj = report["injection"]
        d = inj["detection"]
        print("\n=== Phase 14: Security & governance ===")
        print("\n-- Injection screen (12 poisoned + 12 benign) --")
        print(f"  precision {d['precision']:.3f}  recall {d['recall']:.3f}  F1 {d['f1']:.3f}")
        print(f"  tp {d['tp']}  fp {d['fp']}  fn {d['fn']}  tn {d['tn']}")
        print(f"  missed:       {d['missed']}")
        print(f"  false alarms: {d['false_alarms']}")
        dl = inj["delivery"]
        print(f"  payload reaches context — concat {dl['concat_delivered']}/{dl['n_poisoned']}"
              f"  screened {dl['screened_delivered']}/{dl['n_poisoned']}")
        print(f"  delimiter escape — naive {inj['delimiter_escape']['naive_delimiter']}"
              f"  spotlight {inj['delimiter_escape']['spotlight']}")

        print("\n-- PII detection --")
        for label in ("regex_only", "regex+luhn"):
            p = report["pii"][label]
            print(f"  {label:12s} precision {p['precision']:.3f}  recall {p['recall']:.3f}  "
                  f"F1 {p['f1']:.3f}  (fp {p['fp']}: {p['fp_examples']})")
        ing = report["pii"]["ingest_redaction"]
        print(f"  ingest redaction: {ing['spans_redacted']} spans over {ing['docs']} docs; "
              f"docs with PII after = {ing['docs_with_pii_after']}")

        print("\n-- ACL --")
        for r in report["acl"]["leak_table"]:
            print(f"  {r['query_id']}  post-hoc: secret in context {r['posthoc_secret_in_context']}"
                  f" / in answer {r['posthoc_secret_in_answer']}"
                  f"  ||  acl@retrieval: in context {r['acl_secret_in_context']}"
                  f" / in answer {r['acl_secret_in_answer']}")
        print(f"  tenant isolation: {report['acl']['tenant_isolation']}")
        print(f"  unstamped corpus: {report['acl']['unstamped_corpus']}")

        print("\n-- Right to be forgotten --")
        for mode, r in report["rtbf"].items():
            print(f"  {mode:10s} retrievable before {r['retrievable_before']} → after "
                  f"{r['retrievable_after']}   cache still serves deleted text: "
                  f"{r['cache_still_serves_deleted_text']}   rebuild {r['rebuild_ms']} ms")

        print(f"\nReport → {RESULTS / 'security_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
