"""Tests for strict Task 2 loading, alignment, and reporting."""

import json
import unicodedata
from pathlib import Path

import pytest
from pydantic import ValidationError

from udsc2026.evaluation.legal_qa import (
    LegalQAAggregate,
    LegalQAEvaluationReport,
    LegalQAPrediction,
    LegalQAPredictionSet,
    LegalQAWarmupDataset,
    LegalQAWarmupSample,
    evaluate_legal_qa,
    load_legal_qa_predictions,
    load_legal_qa_question_ids,
    load_legal_qa_warmup,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ACTUAL_WARMUP = REPOSITORY_ROOT / "data" / "task2" / "warmup.json"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _reference(
    question_id: str,
    answer: str,
    question: str = "Câu hỏi?",
) -> LegalQAWarmupSample:
    return LegalQAWarmupSample(
        id=question_id,
        question=question,
        answer=answer,
    )


def test_warmup_loader_preserves_raw_unicode_whitespace_and_format_characters(
    tmp_path: Path,
) -> None:
    raw_question = unicodedata.normalize("NFD", "  Câu hỏi?  ")
    raw_answer = "Dòng một\nDòng hai\u00a0đ\u00adược\ufeff"
    path = tmp_path / "warmup.json"
    _write_json(
        path,
        {"001": {"question": raw_question, "answer": raw_answer}},
    )

    samples = load_legal_qa_warmup(path)

    assert samples == [
        LegalQAWarmupSample(
            id="001",
            question=raw_question,
            answer=raw_answer,
        )
    ]
    assert samples[0].question != unicodedata.normalize("NFC", raw_question)
    assert samples[0].answer.endswith("\ufeff")


def test_warmup_root_contract_is_strict_and_forbids_unknown_fields() -> None:
    valid = LegalQAWarmupDataset.model_validate(
        {"q": {"question": "Hỏi?", "answer": "Đáp."}},
        strict=True,
    )
    assert valid.root["q"].answer == "Đáp."

    invalid_payloads = [
        [],
        {},
        {"q": {"question": "Hỏi?", "answer": "Đáp.", "extra": True}},
        {"q": {"question": 1, "answer": "Đáp."}},
        {"q": {"question": "Hỏi?", "answer": ["Đáp."]}},
        {"q": {"question": " ", "answer": "Đáp."}},
        {"q": {"question": "Hỏi?", "answer": ""}},
        {" q": {"question": "Hỏi?", "answer": "Đáp."}},
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            LegalQAWarmupDataset.model_validate(payload, strict=True)


def test_warmup_loader_rejects_duplicate_keys_and_nonstandard_constants(
    tmp_path: Path,
) -> None:
    root_duplicate = tmp_path / "root-duplicate.json"
    root_duplicate.write_text(
        '{"q":{"question":"Một?","answer":"Một"},'
        '"q":{"question":"Hai?","answer":"Hai"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key: q"):
        load_legal_qa_warmup(root_duplicate)

    nested_duplicate = tmp_path / "nested-duplicate.json"
    nested_duplicate.write_text(
        '{"q":{"question":"Một?","answer":"Một","answer":"Hai"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key: answer"):
        load_legal_qa_warmup(nested_duplicate)

    nonstandard = tmp_path / "nan.json"
    nonstandard.write_text('{"q":{"question":"Hỏi?","answer":NaN}}')
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        load_legal_qa_warmup(nonstandard)


def test_prediction_loader_requires_strict_array_and_preserves_empty_answer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.json"
    payload = [
        {"id": "q1", "answer": " Dòng một\nDòng hai "},
        {"id": "q2", "answer": ""},
    ]
    _write_json(path, payload)

    predictions = load_legal_qa_predictions(path)

    assert predictions[0].answer == " Dòng một\nDòng hai "
    assert predictions[1].answer == ""


def test_question_manifest_loader_supports_phase_shapes_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "mapping.json"
    _write_json(
        mapping,
        {
            "001": {"question": "Câu một?"},
            "002": {"question": "Câu hai?", "answer": "Tham chiếu"},
        },
    )
    assert load_legal_qa_question_ids(mapping) == ["001", "002"]

    string_ids = tmp_path / "ids.json"
    _write_json(string_ids, ["002", "001"])
    assert load_legal_qa_question_ids(string_ids) == ["002", "001"]

    object_ids = tmp_path / "objects.json"
    _write_json(
        object_ids,
        [
            {"id": "001", "question": "Câu một?"},
            {"id": "002", "answer": "Metadata tùy chọn"},
        ],
    )
    assert load_legal_qa_question_ids(object_ids) == ["001", "002"]

    invalid_payloads = (
        [],
        ["001", {"id": "002"}],
        [{"id": "001"}, {"id": "001"}],
        {"001": {"answer": "Thiếu question"}},
        [{"id": "001", "extra": True}],
    )
    for index, payload in enumerate(invalid_payloads):
        invalid = tmp_path / f"invalid-manifest-{index}.json"
        _write_json(invalid, payload)
        with pytest.raises((TypeError, ValueError)):
            load_legal_qa_question_ids(invalid)


@pytest.mark.parametrize(
    "payload",
    [
        {"q": {"answer": "Đáp"}},
        [],
        [{"id": "q", "answer": "Đáp", "extra": 1}],
        [{"id": 1, "answer": "Đáp"}],
        [{"id": "q", "answer": None}],
        [{"id": " q", "answer": "Đáp"}],
        [{"id": "q\u0085log", "answer": "Đáp"}],
        [{"id": "q", "answer": "Một"}, {"id": "q", "answer": "Hai"}],
    ],
)
def test_prediction_loader_rejects_invalid_schema(
    tmp_path: Path,
    payload: object,
) -> None:
    path = tmp_path / "predictions.json"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="invalid LegalQA prediction schema"):
        load_legal_qa_predictions(path)


def test_prediction_set_and_model_reject_duplicates_extra_and_coercion() -> None:
    with pytest.raises(ValidationError, match="duplicate question ID"):
        LegalQAPredictionSet(
            root=[
                LegalQAPrediction(id="q", answer="Một"),
                LegalQAPrediction(id="q", answer="Hai"),
            ]
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        LegalQAPrediction.model_validate({"id": "q", "answer": "Đáp", "score": 1.0})
    with pytest.raises(ValidationError):
        LegalQAPrediction.model_validate({"id": "q", "answer": 123})


def test_contracts_reject_lone_surrogates_but_keep_valid_format_characters(
    tmp_path: Path,
) -> None:
    invalid_reference = tmp_path / "invalid-reference.json"
    invalid_reference.write_text(
        json.dumps(
            {"q": {"question": "Hỏi?", "answer": "Đáp\ud800"}},
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unicode scalar"):
        load_legal_qa_warmup(invalid_reference)

    invalid_prediction = tmp_path / "invalid-prediction.json"
    invalid_prediction.write_text(
        json.dumps(
            [{"id": "q", "answer": "Đáp\udfff"}],
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unicode scalar"):
        load_legal_qa_predictions(invalid_prediction)

    valid = LegalQAPrediction(id="q", answer="đ\u00adược\ufeff\n")
    assert valid.answer == "đ\u00adược\ufeff\n"


def test_loaders_reject_missing_malformed_and_non_utf8_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_legal_qa_warmup(tmp_path / "missing.json")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json}", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_legal_qa_warmup(malformed)

    non_utf8 = tmp_path / "non-utf8.json"
    non_utf8.write_bytes(b"\xff\xfe")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_legal_qa_predictions(non_utf8)


def test_evaluation_aligns_by_id_and_reports_expected_macro_scores() -> None:
    references = [
        _reference("q1", "a b c d"),
        _reference("q2", "x y"),
        _reference("q3", "legal"),
    ]
    predictions = [
        LegalQAPrediction(id="q3", answer=""),
        LegalQAPrediction(id="q1", answer="a b c d"),
        LegalQAPrediction(id="q2", answer="x z"),
    ]

    report = evaluate_legal_qa(references, predictions)

    assert report.evaluation_scope == "local_diagnostic"
    assert report.official_scorer_parity is False
    assert report.profile.official_scorer_parity is False
    assert report.profile.external_corpora_used is False
    assert report.aggregate.sample_count == 3
    assert report.aggregate.empty_prediction_count == 1
    assert report.aggregate.meteor == pytest.approx((0.9921875 + 0.25) / 3)
    assert report.aggregate.rouge_l == pytest.approx((1.0 + 0.5) / 3)
    assert [item.id for item in report.per_query] == ["q1", "q2", "q3"]
    assert report.per_query[0].meteor.matches == 4
    assert report.per_query[1].rouge_l.lcs_length == 1
    assert report.per_query[2].meteor.prediction_token_count == 0


def test_evaluation_fingerprints_raw_data_and_ignores_prediction_file_order() -> None:
    references = [_reference("q1", "Đáp án"), _reference("q2", "Khác")]
    predictions = [
        LegalQAPrediction(id="q1", answer="Đáp án"),
        LegalQAPrediction(id="q2", answer="Khác"),
    ]
    original = evaluate_legal_qa(references, predictions)
    reordered = evaluate_legal_qa(references, list(reversed(predictions)))
    changed_reference = evaluate_legal_qa(
        [
            _reference("q1", "Đáp án", question="Câu hỏi? "),
            references[1],
        ],
        predictions,
    )
    whitespace_prediction = evaluate_legal_qa(
        references,
        [
            LegalQAPrediction(id="q1", answer="Đáp  án"),
            predictions[1],
        ],
    )

    assert reordered == original
    assert len(original.dataset_fingerprint) == 64
    assert changed_reference.dataset_fingerprint != original.dataset_fingerprint
    assert (
        whitespace_prediction.prediction_fingerprint != original.prediction_fingerprint
    )
    assert whitespace_prediction.aggregate == original.aggregate


def test_evaluation_rejects_duplicate_missing_unexpected_and_wrong_types() -> None:
    reference = _reference("q1", "Đáp")
    prediction = LegalQAPrediction(id="q1", answer="Đáp")
    with pytest.raises(ValueError, match="references must not be empty"):
        evaluate_legal_qa([], [])
    with pytest.raises(ValueError, match="duplicate question ID in references"):
        evaluate_legal_qa([reference, reference], [prediction])
    with pytest.raises(ValueError, match="duplicate question ID in predictions"):
        evaluate_legal_qa(
            [reference],
            [prediction, LegalQAPrediction(id="q1", answer="Khác")],
        )
    with pytest.raises(ValueError, match="missing IDs: q1.*unexpected IDs: q2"):
        evaluate_legal_qa(
            [reference],
            [LegalQAPrediction(id="q2", answer="Đáp")],
        )
    with pytest.raises(TypeError, match=r"references\[0\]"):
        evaluate_legal_qa([object()], [prediction])  # type: ignore[list-item]
    with pytest.raises(TypeError, match=r"predictions\[0\]"):
        evaluate_legal_qa([reference], [object()])  # type: ignore[list-item]


def test_evaluation_rejects_reference_with_no_metric_tokens() -> None:
    reference = _reference("q", "... !!!")
    prediction = LegalQAPrediction(id="q", answer="")
    with pytest.raises(ValueError, match="has no metric tokens for ID q"):
        evaluate_legal_qa([reference], [prediction])


def test_evaluation_revalidates_model_copy_payloads_at_its_boundary() -> None:
    reference = _reference("q", "Đáp")
    prediction = LegalQAPrediction(id="q", answer="Đáp")
    forged_reference = reference.model_copy(update={"answer": "bad\ud800"})
    forged_prediction = prediction.model_copy(update={"id": " q"})

    with pytest.raises(ValidationError, match="Unicode scalar"):
        evaluate_legal_qa([forged_reference], [prediction])
    with pytest.raises(ValidationError, match="surrounding whitespace"):
        evaluate_legal_qa([reference], [forged_prediction])


def test_report_contract_rejects_forged_aggregate() -> None:
    report = evaluate_legal_qa(
        [_reference("q", "a b")],
        [LegalQAPrediction(id="q", answer="a b")],
    )
    payload = report.model_dump(mode="python")
    payload["aggregate"]["meteor"] = 0.0
    with pytest.raises(ValidationError, match="aggregate METEOR"):
        LegalQAEvaluationReport.model_validate(payload)

    with pytest.raises(ValidationError, match="cannot exceed"):
        LegalQAAggregate(
            sample_count=1,
            empty_prediction_count=2,
            meteor=0.0,
            rouge_l=0.0,
        )


def test_report_round_trip_is_deterministic() -> None:
    report = evaluate_legal_qa(
        [_reference("câu-hỏi", "Người lao động được bảo vệ.")],
        [LegalQAPrediction(id="câu-hỏi", answer="Người lao động được bảo vệ.")],
    )
    encoded = report.model_dump_json()
    restored = LegalQAEvaluationReport.model_validate_json(encoded, strict=True)
    assert restored == report
    assert restored.model_dump_json() == encoded


def test_report_and_nested_profile_are_deeply_immutable() -> None:
    report = evaluate_legal_qa(
        [_reference("q", "Đáp")],
        [LegalQAPrediction(id="q", answer="Đáp")],
    )
    assert isinstance(report.per_query, tuple)
    with pytest.raises(AttributeError):
        report.per_query.append(report.per_query[0])  # type: ignore[attr-defined]
    with pytest.raises(ValidationError, match="frozen"):
        report.aggregate.meteor = 0.0
    with pytest.raises(ValidationError, match="frozen"):
        report.profile.official_scorer_parity = True  # type: ignore[assignment]


@pytest.mark.skipif(not ACTUAL_WARMUP.is_file(), reason="Task 2 warm-up not present")
def test_actual_task2_warmup_schema_and_known_unicode_audit() -> None:
    samples = load_legal_qa_warmup(ACTUAL_WARMUP)
    assert len(samples) == 500
    assert len({sample.id for sample in samples}) == 500
    assert (
        sum(
            sample.answer != unicodedata.normalize("NFC", sample.answer)
            for sample in samples
        )
        == 18
    )
    assert sum("\u00a0" in sample.answer for sample in samples) == 8
    assert sum("\u00ad" in sample.answer for sample in samples) == 1
    assert sum("\ufeff" in sample.answer for sample in samples) == 3

    report = evaluate_legal_qa(
        samples,
        [LegalQAPrediction(id=sample.id, answer=sample.answer) for sample in samples],
    )
    assert report.aggregate.sample_count == 500
    assert report.aggregate.rouge_l == 1.0
    assert report.aggregate.meteor > 0.999999
