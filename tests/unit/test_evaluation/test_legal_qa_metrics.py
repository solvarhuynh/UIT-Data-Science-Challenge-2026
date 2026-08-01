"""Tests for corpus-free Task 2 METEOR and ROUGE-L diagnostics."""

import unicodedata
from dataclasses import FrozenInstanceError

import pytest

from udsc2026.evaluation.legal_qa_metrics import (
    METEOR_ALPHA,
    METEOR_BETA,
    METEOR_DIAGNOSTIC_PROFILE,
    METEOR_GAMMA,
    ROUGE_L_DIAGNOSTIC_PROFILE,
    TOKENIZATION_PROFILE,
    meteor_diagnostic,
    meteor_diagnostic_score,
    normalize_legal_qa_text,
    rouge_l_diagnostic,
    rouge_l_f1_score,
    tokenize_legal_qa_text,
)


def test_profile_constants_are_explicit_and_versioned() -> None:
    assert TOKENIZATION_PROFILE.endswith("-v1")
    assert METEOR_DIAGNOSTIC_PROFILE.endswith("-v1")
    assert ROUGE_L_DIAGNOSTIC_PROFILE.endswith("-v1")
    assert (METEOR_ALPHA, METEOR_BETA, METEOR_GAMMA) == (0.9, 3.0, 0.5)


def test_tokenizer_normalizes_vietnamese_unicode_case_and_multiline_format() -> None:
    decomposed = unicodedata.normalize(
        "NFD",
        "  TÀI\nLIỆU\u00a0được-bảo vệ  ",
    )
    assert tokenize_legal_qa_text(decomposed) == [
        "tài",
        "liệu",
        "được",
        "bảo",
        "vệ",
    ]


def test_normalizer_removes_format_artifacts_without_splitting_words() -> None:
    source = "đ\u00adược\ufeff bảo\u00a0vệ"
    normalized = normalize_legal_qa_text(source)
    assert "\u00ad" not in normalized
    assert "\ufeff" not in normalized
    assert tokenize_legal_qa_text(source) == ["được", "bảo", "vệ"]


def test_meteor_identical_sentence_uses_declared_fragmentation_formula() -> None:
    result = meteor_diagnostic("a b c d", "a b c d")
    assert result.matches == 4
    assert result.chunks == 1
    assert result.precision == result.recall == result.harmonic_mean == 1.0
    assert result.fragmentation_penalty == pytest.approx(0.5 * (1 / 4) ** 3)
    assert result.score == pytest.approx(0.9921875)
    assert meteor_diagnostic_score("a b c d", "a b c d") == result.score


def test_meteor_fragmentation_penalizes_reordered_exact_matches() -> None:
    ordered = meteor_diagnostic("a b c d", "a b c d")
    reordered = meteor_diagnostic("a c b d", "a b c d")
    assert reordered.matches == 4
    assert reordered.chunks == 4
    assert reordered.fragmentation_penalty == 0.5
    assert reordered.score == 0.5
    assert reordered.score < ordered.score


def test_meteor_repetition_reduces_precision_and_breaks_a_chunk() -> None:
    result = meteor_diagnostic("a a b", "a b")
    assert result.matches == 2
    assert result.chunks == 2
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == 1.0
    assert result.score == pytest.approx(0.4761904761904762)
    assert result.score < meteor_diagnostic_score("a b", "a b")


def test_meteor_disjoint_and_empty_inputs_have_explicit_zero_behavior() -> None:
    for prediction, reference in [
        ("khác", "đúng"),
        ("", "đúng"),
        ("đúng", ""),
        ("", ""),
        ("...", "!!!"),
    ]:
        result = meteor_diagnostic(prediction, reference)
        assert result.score == 0.0
        assert result.matches == 0
        assert result.chunks == 0


def test_meteor_is_invariant_to_nfc_nfd_case_punctuation_and_line_breaks() -> None:
    reference = "Người lao động được bảo vệ"
    prediction = unicodedata.normalize("NFD", "NGƯỜI\nLAO ĐỘNG, ĐƯỢC BẢO VỆ!")
    assert meteor_diagnostic(prediction, reference) == meteor_diagnostic(
        reference,
        reference,
    )


def test_rouge_l_identical_disjoint_and_empty_behavior() -> None:
    identical = rouge_l_diagnostic("một hai ba", "một hai ba")
    assert identical.lcs_length == 3
    assert identical.precision == identical.recall == identical.f1 == 1.0
    assert rouge_l_f1_score("một hai ba", "một hai ba") == 1.0

    assert rouge_l_diagnostic("một", "khác").f1 == 0.0
    assert rouge_l_diagnostic("", "tham chiếu").f1 == 0.0
    assert rouge_l_diagnostic("dự đoán", "").f1 == 0.0
    assert rouge_l_diagnostic("", "").f1 == 1.0


def test_rouge_l_repetition_lowers_precision_but_preserves_recall() -> None:
    result = rouge_l_diagnostic("a a b", "a b")
    assert result.lcs_length == 2
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == 1.0
    assert result.f1 == pytest.approx(0.8)


def test_rouge_l_preserves_sequence_information() -> None:
    ordered = rouge_l_diagnostic("a b c d", "a b c d")
    reordered = rouge_l_diagnostic("a c b d", "a b c d")
    assert reordered.lcs_length == 3
    assert reordered.f1 == pytest.approx(0.75)
    assert reordered.f1 < ordered.f1


def test_metric_results_are_immutable_diagnostics() -> None:
    meteor = meteor_diagnostic("a", "a")
    rouge = rouge_l_diagnostic("a", "a")
    with pytest.raises(FrozenInstanceError):
        meteor.score = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        rouge.f1 = 0.0  # type: ignore[misc]


@pytest.mark.parametrize("function", [normalize_legal_qa_text, tokenize_legal_qa_text])
def test_text_boundaries_reject_non_strings(function: object) -> None:
    with pytest.raises(TypeError, match="must be strings"):
        function(123)  # type: ignore[operator]


def test_metric_entry_points_reject_non_strings() -> None:
    with pytest.raises(TypeError):
        meteor_diagnostic(123, "reference")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        rouge_l_diagnostic("prediction", None)  # type: ignore[arg-type]


def test_metric_entry_points_reject_lone_unicode_surrogates() -> None:
    with pytest.raises(ValueError, match="Unicode scalar"):
        meteor_diagnostic("dự đoán\ud800", "tham chiếu")
    with pytest.raises(ValueError, match="Unicode scalar"):
        rouge_l_diagnostic("dự đoán", "tham chiếu\udfff")
