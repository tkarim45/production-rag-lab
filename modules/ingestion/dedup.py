"""Deduplication: exact (content hash) + near-dup (from-scratch MinHash / Jaccard).

- Exact: SHA-1 of normalized text → drop identical docs.
- Near-dup: k-shingle MinHash signatures, estimate Jaccard by signature agreement, drop
  docs above a similarity threshold. Implemented from scratch (no datasketch dep) so it
  runs on the stdlib and the mechanism is legible — this is a teaching repo.

Returns the kept docs plus a report of what was dropped and why.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

_WORD = re.compile(r"[a-z0-9]+")


def _normalize_for_hash(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def content_hash(text: str) -> str:
    return hashlib.sha1(_normalize_for_hash(text).encode()).hexdigest()


def shingles(text: str, k: int = 3) -> set[int]:
    """k-word shingles as hashed ints (stable FNV-1a; process-independent)."""
    words = _WORD.findall(text.lower())
    if len(words) < k:
        grams = [" ".join(words)] if words else []
    else:
        grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    out = set()
    for g in grams:
        h = 2166136261
        for ch in g:
            h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
        out.add(h)
    return out


def _hash_family(n: int) -> list[tuple[int, int]]:
    """n (a, b) pairs for hashes h(x) = (a*x + b) mod P, deterministic (no RNG)."""
    P = (1 << 31) - 1
    pairs = []
    a, b = 1, 0
    for i in range(n):
        a = (a * 6364136223846793005 + 1) & 0xFFFFFFFF or 1
        b = (b * 1442695040888963407 + 12345) & 0xFFFFFFFF
        pairs.append((a, b))
    return pairs


_MOD = (1 << 31) - 1


def minhash_signature(shingle_set: set[int], hashes: list[tuple[int, int]]) -> list[int]:
    if not shingle_set:
        return [0] * len(hashes)
    sig = []
    for a, b in hashes:
        sig.append(min(((a * x + b) % _MOD) for x in shingle_set))
    return sig


def estimate_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a:
        return 0.0
    agree = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return agree / len(sig_a)


@dataclass
class DedupReport:
    n_input: int
    n_kept: int
    exact_dropped: list[tuple[str, str]]  # (dropped_id, kept_id)
    near_dropped: list[tuple[str, str, float]]  # (dropped_id, kept_id, jaccard)


def dedup(
    docs: Iterable, threshold: float = 0.8, num_hashes: int = 64, k: int = 3
) -> tuple[list, DedupReport]:
    """docs must have `.doc_id` and `.text`. Returns (kept_docs, report)."""
    docs = list(docs)
    hashes = _hash_family(num_hashes)

    kept: list = []
    kept_hashes: dict[str, str] = {}       # content_hash → kept doc_id
    kept_sigs: list[tuple[str, list[int]]] = []
    exact_dropped, near_dropped = [], []

    for d in docs:
        ch = content_hash(d.text)
        if ch in kept_hashes:
            exact_dropped.append((d.doc_id, kept_hashes[ch]))
            continue
        sig = minhash_signature(shingles(d.text, k), hashes)
        dup_of = None
        for kid, ksig in kept_sigs:
            j = estimate_jaccard(sig, ksig)
            if j >= threshold:
                dup_of = (kid, j)
                break
        if dup_of is not None:
            near_dropped.append((d.doc_id, dup_of[0], round(dup_of[1], 3)))
            continue
        kept.append(d)
        kept_hashes[ch] = d.doc_id
        kept_sigs.append((d.doc_id, sig))

    report = DedupReport(
        n_input=len(docs), n_kept=len(kept),
        exact_dropped=exact_dropped, near_dropped=near_dropped,
    )
    return kept, report
