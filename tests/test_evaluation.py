"""Phase 11: faithfulness/context metrics + bootstrap significance (deps-free paths)."""

import numpy as np

from harness.contract import Chunk, PipelineResult, Query, Scored
from modules.evaluation import faithfulness as F
from modules.evaluation.significance import bootstrap_ci, paired_delta_ci


def _result(answer, context, gold, retrieved_docs, relevant_docs):
    q = Query("q", "what is the capital", gold_answer=gold, relevant_doc_ids=set(relevant_docs))
    scored = [Scored(Chunk(f"{d}::0", d, "text"), 1.0) for d in retrieved_docs]
    return PipelineResult(query=q, retrieved=scored, context=context, answer=answer)


def test_groundedness_full_and_zero():
    assert F.answer_groundedness("paris france", "the capital is paris france") == 1.0
    assert F.answer_groundedness("tokyo japan", "the capital is paris france") == 0.0


def test_groundedness_partial():
    # content words {paris, japan}; context has paris only → 0.5
    assert F.answer_groundedness("paris japan", "paris is in france") == 0.5


def test_groundedness_empty_answer_is_vacuously_grounded():
    assert F.answer_groundedness("", "anything") == 1.0
    assert F.answer_groundedness("the a of", "anything") == 1.0   # stopwords only


def test_context_precision_and_recall():
    r = _result(answer="x", context="the capital is paris france",
                gold="paris france", retrieved_docs=["a", "b", "c", "d"], relevant_docs=["a", "b"])
    assert F.context_precision(r) == 0.5          # 2 of 4 retrieved are relevant
    assert F.context_recall(r) == 1.0             # gold words all in context


def test_context_recall_partial():
    r = _result(answer="x", context="paris is nice", gold="paris france",
                retrieved_docs=["a"], relevant_docs=["a"])
    assert F.context_recall(r) == 0.5             # 'france' missing from context


def test_score_batch_keys():
    r = _result("paris", "paris france", "paris", ["a"], ["a"])
    m = F.score_batch([r])
    assert set(m) == {"groundedness", "context_precision", "context_recall"}


def test_bootstrap_ci_brackets_the_mean():
    vals = [0.5] * 20
    mean, lo, hi = bootstrap_ci(vals)
    assert mean == 0.5 and lo == 0.5 and hi == 0.5   # zero variance → degenerate CI


def test_paired_delta_ci_detects_a_real_shift():
    a = [1.0] * 20
    b = [0.0] * 20
    d = paired_delta_ci(a, b)
    assert d["mean_delta"] == 1.0 and d["significant"] == 1.0


def test_paired_delta_ci_noise_is_not_significant():
    rng = np.random.RandomState(0)
    a = list(rng.rand(30))
    b = list(rng.rand(30))       # same distribution → CI should straddle 0
    d = paired_delta_ci(a, b)
    assert d["significant"] == 0.0


def test_paired_delta_requires_aligned_lists():
    import pytest

    with pytest.raises(ValueError):
        paired_delta_ci([1.0, 2.0], [1.0])
