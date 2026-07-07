"""EM / token-F1 vs hand-computed cases + efficiency helpers."""

import math

from harness.metrics import answer as A
from harness.metrics import efficiency as E


def test_normalize_drops_articles_punct_case():
    assert A.normalize("The Eiffel Tower!") == "eiffel tower"


def test_exact_match():
    assert A.exact_match("the cat", "cat") == 1.0        # article dropped
    assert A.exact_match("a dog", "cat") == 0.0


def test_token_f1_partial():
    # "the" is dropped as an article → gold tokens {brown, fox}; pred {quick, brown, fox}.
    # overlap {brown, fox}=2 → precision 2/3, recall 2/2=1 → f1 = 2*(2/3)/(2/3+1) = 0.8
    assert math.isclose(A.token_f1("quick brown fox", "the brown fox"), 0.8)


def test_token_f1_edge_cases():
    assert A.token_f1("", "") == 1.0
    assert A.token_f1("something", "") == 0.0
    assert A.token_f1("cat", "dog") == 0.0


def test_percentile_nearest_rank():
    vals = [10, 20, 30, 40, 50]
    assert E.percentile(vals, 50) == 30
    assert E.percentile(vals, 100) == 50
    assert E.percentile(vals, 0) == 10


def test_cost_per_correct():
    # 10 queries, mean correctness 0.5 → 5 correct; $1 total → $0.20 each
    assert E.cost_per_correct_answer(1.0, 10, 0.5) == 0.2
    assert E.cost_per_correct_answer(1.0, 10, 0.0) == float("inf")
