"""Tests for pure ranking, answer, and latency metrics."""

import math
import unicodedata

import pytest

from udsc2026.evaluation.metrics import (
    aggregate_latencies,
    mean_recall_at_k,
    mean_reciprocal_rank,
    mean_rouge_l,
    recall_at_k,
    reciprocal_rank,
    rouge_l_score,
)


def test_reciprocal_rank_counts_duplicate_rank_positions() -> None:
    assert reciprocal_rank(["wrong", "wrong", "gold"], {"gold"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_zero_when_no_gold_is_retrieved() -> None:
    assert reciprocal_rank([], {"gold"}) == 0.0
    assert reciprocal_rank(["other"], {"gold"}) == 0.0


@pytest.mark.parametrize(
    ("predictions", "gold"),
    [([""], ["gold"]), (["hit"], []), (["hit"], ["  "])],
)
def test_reciprocal_rank_rejects_invalid_ids(
    predictions: list[str], gold: list[str]
) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank(predictions, gold)


def test_mean_reciprocal_rank_is_macro_average() -> None:
    score = mean_reciprocal_rank(
        [["gold-a"], ["wrong", "gold-b"], ["wrong"]],
        [["gold-a"], ["gold-b"], ["gold-c"]],
    )
    assert score == pytest.approx(0.5)


def test_recall_at_k_uses_fraction_of_all_unique_gold_chunks() -> None:
    assert recall_at_k(["a", "a", "b", "c"], {"a", "b", "d"}, 2) == pytest.approx(1 / 3)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_recall_at_k_rejects_invalid_cutoffs(k: object) -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k)  # type: ignore[arg-type]


def test_macro_metrics_reject_empty_or_misaligned_samples() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mean_reciprocal_rank([], [])
    with pytest.raises(ValueError, match="counts must match"):
        mean_recall_at_k([["a"]], [["a"], ["b"]], 1)
    with pytest.raises(ValueError, match="counts must match"):
        mean_rouge_l(["a"], ["a", "b"])


def test_rouge_l_normalizes_vietnamese_unicode_case_and_punctuation() -> None:
    composed = "Người lao động được bảo vệ."
    decomposed = unicodedata.normalize("NFD", "NGƯỜI LAO ĐỘNG ĐƯỢC BẢO VỆ")
    assert rouge_l_score(composed, decomposed) == 1.0


def test_rouge_l_returns_expected_f1_and_handles_empty_text() -> None:
    assert rouge_l_score("người lao động", "người lao động Việt Nam") == pytest.approx(
        0.75
    )
    assert rouge_l_score("", "") == 1.0
    assert rouge_l_score("", "pháp luật") == 0.0
    assert rouge_l_score("pháp luật", "") == 0.0


def test_rouge_l_rejects_non_string_values() -> None:
    with pytest.raises(TypeError):
        rouge_l_score(123, "reference")  # type: ignore[arg-type]


def test_latency_aggregation_is_deterministic_and_interpolated() -> None:
    stats = aggregate_latencies([40, 10, 30, 20])
    assert stats.count == 4
    assert stats.total_ms == 100
    assert stats.min_ms == 10
    assert stats.max_ms == 40
    assert stats.mean_ms == 25
    assert stats.median_ms == 25
    assert stats.p95_ms == pytest.approx(38.5)
    assert stats.p99_ms == pytest.approx(39.7)


@pytest.mark.parametrize("values", [[], [-1], [math.nan], [math.inf]])
def test_latency_aggregation_rejects_invalid_values(values: list[float]) -> None:
    with pytest.raises(ValueError):
        aggregate_latencies(values)


def test_latency_aggregation_rejects_non_numeric_and_boolean_values() -> None:
    with pytest.raises(TypeError):
        aggregate_latencies(["10"])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        aggregate_latencies([True])
