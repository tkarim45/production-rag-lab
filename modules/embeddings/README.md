# Module: Embeddings (Phase 3)

**Lesson.** The embedder decides what "similar" means, and its footprint decides whether the
index fits in memory at scale. The two production levers are **model choice** (quality) and
**compression**, quantization (int8/binary) and dimensionality truncation (Matryoshka), 
which trade recall for memory/speed. You only know the exchange rate by measuring it.

## Implemented (deps-light)
- `tfidf`, corpus-fit TF-IDF, a strong lexical embedder (the floor a neural model must beat).
- `quantized`, wrap any base embedder, quantize to **int8** (1 byte/dim) or **binary**
  (1 bit/dim); float cosine on the dequantized vector so the metric reflects *information*
  lost, not index mechanics.
- `matryoshka`, truncate any base embedder to the first `dim` components + renormalize.
- The real neural `sentence_transformer` embedder registers from `modules.baseline` when
  `.[embed]` is installed (MTEB-grade, Phase 3's optional upgrade).

## Result: `builtin_docs`, fixed chunker + flat index

`python -m harness.sweep --vary embedder --options hashing tfidf quantized_int8 quantized_binary matryoshka_64 matryoshka_128 matryoshka_256`

| embedder | recall@1 | recall@5 | MRR | NDCG@10 | mem/vec | note |
|---|--:|--:|--:|--:|--:|---|
| **tfidf** | **0.950** | 1.000 | **1.000** | **1.000** | dim×4 B | strong lexical baseline |
| **quantized int8** | **0.950** | 1.000 | **1.000** | **1.000** | **dim×1 B (4×↓)** | **lossless here** |
| matryoshka 256 | 0.800 | 0.950 | 0.888 | 0.897 | 256×4 B | graceful |
| matryoshka 128 | 0.650 | 0.825 | 0.743 | 0.750 | 128×4 B | graceful |
| hashing (Phase 0) | 0.550 | 0.950 | 0.750 | 0.797 | 512×4 B | weak baseline |
| matryoshka 64 | 0.375 | 0.725 | 0.529 | 0.572 | 64×4 B | too aggressive |
| quantized binary | 0.050 | 0.250 | 0.133 | 0.163 | dim/8 B (32×↓) | **collapses** |

**Honest findings.**
1. **Embedder choice dominates everything downstream so far.** Swapping the Phase 0 `hashing`
   embedder for `tfidf` lifts recall@1 0.55 → 0.95 and MRR 0.75 → 1.00, a bigger gain than any
   chunker delivered. Retrieval quality is embedder-first.
2. **int8 quantization is free money here**, identical retrieval to full float at **4× less
   memory**. On real deployments int8 is the default; this shows why.
3. **Binary quantization is catastrophic on sparse lexical vectors** (recall@1 0.05). A 1-bit
   sign is meaningless when most dims are exactly zero, binary is a dense-neural-embedding
   technique, not a universal one. The leaderboard makes the failure obvious instead of the
   "32× smaller!" headline hiding it.
4. **Matryoshka is a smooth dial:** 256 dims keeps 84% of full recall@1 at half the memory; 64
   dims (an 8× cut) drops to 0.375. Truncation works *because* TF-IDF dims here are
   frequency-ordered, a real Matryoshka model is trained for this; naive truncation of an
   unordered embedder would not degrade so gracefully (noted, not claimed as general).

Caveat: TF-IDF saturates this tiny lexical corpus (MRR 1.0), so it can't separate from a real
neural embedder here, that separation needs the `.[embed]` model + a harder/larger corpus.
