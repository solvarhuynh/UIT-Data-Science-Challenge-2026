from scripts.task1.gemini_warmup_overlay import (
    _accept_api_matches,
    _build_exact_matches,
    _candidate_map,
)


def test_exact_question_match_can_use_different_ids() -> None:
    warmup = {"warmup-1": {"question": "Điều kiện cấp phép?", "answer": ["101"]}}
    questions = {"target-9": " điều kiện CẤP PHÉP? "}

    matches = _build_exact_matches(warmup, questions)

    assert matches["target-9"]["warmup_id"] == "warmup-1"
    assert matches["target-9"]["source"] == "exact_normalized_question"


def test_api_match_accepts_only_allowed_high_confidence_identity() -> None:
    response = {
        "matches": [
            {
                "target_id": "target-1",
                "warmup_id": "warmup-1",
                "same_question": True,
                "confidence": 0.96,
            },
            {
                "target_id": "target-2",
                "warmup_id": "not-a-candidate",
                "same_question": True,
                "confidence": 0.99,
            },
        ]
    }

    accepted = _accept_api_matches(
        response,
        {"target-1": {"warmup-1"}, "target-2": {"warmup-2"}},
        0.92,
    )

    assert set(accepted) == {"target-1"}


def test_candidate_map_excludes_already_exactly_matched_target() -> None:
    warmup = {
        "warmup-1": {"question": "Một câu hỏi", "answer": ["1"]},
        "warmup-2": {"question": "Câu hỏi khác", "answer": ["2"]},
    }
    questions = {"target-1": "Một câu hỏi", "target-2": "Câu hỏi mới"}

    candidates = _candidate_map(warmup, questions, {"target-1"}, 1)

    assert set(candidates) == {"target-2"}
    assert len(candidates["target-2"]) == 1
