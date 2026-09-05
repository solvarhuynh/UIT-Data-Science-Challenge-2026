"""Tests for label-free Task 2 P15 answer candidates."""

from __future__ import annotations

import pytest

from udsc2026.evaluation.legal_qa_candidates import (
    answer_features,
    build_answer_candidates,
    select_evidence_sentences,
    trim_at_sentence_boundary,
    validate_candidate_bank,
)


def test_trim_prefers_complete_sentence() -> None:
    answer = "Câu đầu có đủ ý. Câu thứ hai dài hơn giới hạn cho phép."

    assert trim_at_sentence_boundary(answer, 6) == "Câu đầu có đủ ý."


def test_evidence_selector_deduplicates_and_respects_limit() -> None:
    evidence = (
        "Điều 3 quy định mức phạt là 10 triệu đồng.\n"
        "Điều 3 quy định mức phạt là 10 triệu đồng.\n"
        "Nội dung không liên quan đến câu hỏi này."
    )
    selected = select_evidence_sentences(
        "Mức phạt theo Điều 3 là bao nhiêu?",
        evidence,
        "Người vi phạm bị phạt.",
        word_limit=12,
    )

    assert selected.count("10 triệu") == 1
    assert len(selected.split()) <= 12


def test_candidate_family_has_stable_profiles_and_keeps_qwen() -> None:
    candidates = build_answer_candidates(
        "Điều kiện là gì?",
        "Điều kiện thứ nhất. Điều kiện thứ hai.",
        "Điều 1 quy định điều kiện thứ nhất. Điều 2 quy định điều kiện thứ hai.",
    )

    assert candidates[0].profile == "qwen"
    assert len(candidates) == 25
    assert len({candidate.profile for candidate in candidates}) == len(candidates)
    assert any(candidate.profile.startswith("smart_prefix") for candidate in candidates)
    assert any(candidate.profile == "raw_prefix_352" for candidate in candidates)


def test_features_are_fixed_width() -> None:
    profiles = ["qwen", "smart_prefix_48"]
    first = answer_features(
        "Mức phạt 10 triệu?",
        "Phạt 10 triệu đồng.",
        "Phạt 10 triệu đồng.",
        "Điều 1 phạt 10 triệu đồng.",
        profile="qwen",
        profile_names=profiles,
    )
    second = answer_features(
        "Mức phạt 10 triệu?",
        "Điều 1 phạt 10 triệu đồng.",
        "Phạt 10 triệu đồng.",
        "Điều 1 phạt 10 triệu đồng.",
        profile="smart_prefix_48",
        profile_names=profiles,
    )

    assert len(first) == len(second) == len(profiles) + 10
    assert first[:2] == [1.0, 0.0]
    assert second[:2] == [0.0, 1.0]


def test_candidate_bank_requires_rectangular_profiles() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_candidate_bank(
            [
                {"id": "q1", "profile": "a", "answer": "one"},
                {"id": "q1", "profile": "b", "answer": "two"},
                {"id": "q2", "profile": "a", "answer": "three"},
            ]
        )
