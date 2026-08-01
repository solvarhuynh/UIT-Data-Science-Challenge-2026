"""Subprocess coverage for Task 2 LegalQA audit/evaluation/submission CLIs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = PROJECT_ROOT / "scripts"
ACTUAL_WARMUP = PROJECT_ROOT / "data" / "task2" / "warmup.json"
ACTUAL_WARMUP_SHA256 = (
    "b824e4f18bd9181c021498a28e402b7374d6d559f2ff28caa4120a9d932f82c5"
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_submission_zip(
    path: Path,
    payload: object,
    *,
    member_name: str = "submission.json",
) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(member_name, encoded)


def _run(script: str, *arguments: object) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPTS / script)]
    command.extend(str(argument) for argument in arguments)
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _warmup_payload() -> dict[str, dict[str, str]]:
    return {
        "001": {
            "question": "  Người lao động được nghỉ?  ",
            "answer": "Dòng một\n\nDòng hai  ",
        },
        "002": {
            "question": unicodedata.normalize("NFD", "Bảo hiểm áp dụng thế nào?"),
            "answer": (
                unicodedata.normalize("NFD", "Bảo hiểm") + "\u00a0mềm\u00ad" + "\ufeff"
            ),
        },
        "003": {
            "question": "Quy định khác?",
            "answer": "Câu trả lời thứ ba.",
        },
    }


def _prediction_payload(
    warmup: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    return [
        {"id": question_id, "answer": record["answer"]}
        for question_id, record in warmup.items()
    ]


def _official_submission_payload(
    warmup: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    return {
        question_id: {"answer": record["answer"]}
        for question_id, record in warmup.items()
    }


@pytest.mark.parametrize(
    ("script", "required_fragments"),
    [
        (
            "write_legal_qa_submission.py",
            ("internal Task 2 prediction JSON array", "object keyed by question ID"),
        ),
        (
            "validate_legal_qa_submission.py",
            ("official Task 2 submission ZIP", "object keyed by question ID"),
        ),
        (
            "evaluate_legal_qa.py",
            (
                "internal Task 2 prediction JSON array",
                "official object-shaped submission ZIP",
            ),
        ),
        (
            "make_legal_qa_warmup_smoke_submission.py",
            ("official object-shaped ZIP", "must never be uploaded"),
        ),
    ],
)
def test_cli_help_distinguishes_internal_json_from_official_zip(
    script: str,
    required_fragments: tuple[str, str],
) -> None:
    result = _run(script, "--help")

    assert result.returncode == 0, result.stderr
    normalized_help = " ".join(result.stdout.split())
    for fragment in required_fragments:
        assert fragment in normalized_help


def test_audit_reports_deterministic_unicode_whitespace_and_checksum(
    tmp_path: Path,
) -> None:
    warmup = tmp_path / "warmup.json"
    output = tmp_path / "audit.json"
    _write_json(warmup, _warmup_payload())

    first = _run(
        "audit_legal_qa_warmup.py",
        "--input",
        warmup,
        "--output",
        output,
    )
    assert first.returncode == 0, first.stderr
    summary = json.loads(first.stdout)
    assert summary["status"] == "valid"
    assert summary["question_count"] == 3
    first_bytes = output.read_bytes()

    second = _run(
        "audit_legal_qa_warmup.py",
        "--input",
        warmup,
        "--output",
        output,
    )
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes
    report = json.loads(first_bytes)
    assert report["dataset_sha256"] == hashlib.sha256(warmup.read_bytes()).hexdigest()
    assert report["question_count"] == 3
    assert report["reference_answer_count"] == 3
    assert report["question"]["surrounding_whitespace_ids"] == ["001"]
    assert report["question"]["non_nfc_ids"] == ["002"]
    assert report["answer"]["multiline_ids"] == ["001"]
    assert report["answer"]["blank_line"] == {
        "ids": ["001"],
        "occurrence_count": 1,
        "record_count": 1,
    }
    assert report["answer"]["non_breaking_space"]["ids"] == ["002"]
    assert report["answer"]["soft_hyphen"]["occurrence_count"] == 1
    assert report["answer"]["embedded_bom_feff"]["occurrence_count"] == 1
    assert report["answer"]["format_character_cf"]["occurrence_count"] == 2
    assert report["answer"]["trailing_line_whitespace"]["ids"] == ["001"]


def test_audit_rejects_schema_error_and_input_output_collision(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    _write_json(invalid, {"q": {"question": "Hỏi?", "answer": ["not-string"]}})
    schema_error = _run(
        "audit_legal_qa_warmup.py",
        "--input",
        invalid,
        "--output",
        tmp_path / "audit.json",
    )
    assert schema_error.returncode == 2
    assert "schema" in schema_error.stderr

    warmup = tmp_path / "warmup.json"
    _write_json(warmup, _warmup_payload())
    original = warmup.read_bytes()
    collision = _run(
        "audit_legal_qa_warmup.py",
        "--input",
        warmup,
        "--output",
        warmup,
    )
    assert collision.returncode == 2
    assert "must not overwrite" in collision.stderr
    assert warmup.read_bytes() == original


@pytest.mark.skipif(not ACTUAL_WARMUP.is_file(), reason="Task 2 warm-up unavailable")
def test_actual_warmup_audit_matches_locked_regression_counts(tmp_path: Path) -> None:
    output = tmp_path / "actual-audit.json"
    result = _run(
        "audit_legal_qa_warmup.py",
        "--input",
        ACTUAL_WARMUP,
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset_sha256"] == ACTUAL_WARMUP_SHA256
    assert report["question_count"] == 500
    assert report["reference_answer_count"] == 500
    assert report["question"]["collapsed_whitespace_change_count"] == 31
    assert report["question"]["non_nfc_count"] == 3
    assert report["answer"]["multiline_count"] == 498
    assert report["answer"]["non_nfc_count"] == 18
    assert report["answer"]["character_length"]["min"] == 176
    assert report["answer"]["character_length"]["median"] == 1373.0
    assert report["answer"]["character_length"]["max"] == 8089
    assert report["answer"]["non_breaking_space"]["record_count"] == 8
    assert report["answer"]["non_breaking_space"]["occurrence_count"] == 42
    assert report["answer"]["soft_hyphen"]["record_count"] == 1
    assert report["answer"]["soft_hyphen"]["occurrence_count"] == 4
    assert report["answer"]["embedded_bom_feff"]["record_count"] == 3
    assert report["answer"]["format_character_cf"]["record_count"] == 4
    assert report["answer"]["blank_line"]["record_count"] == 161
    assert report["answer"]["trailing_line_whitespace"]["record_count"] == 20


@pytest.mark.parametrize("prediction_suffix", [".json", ".zip"])
def test_evaluate_diagnostic_json_and_zip_preserves_profile_and_is_deterministic(
    tmp_path: Path,
    prediction_suffix: str,
) -> None:
    warmup_payload = _warmup_payload()
    predictions_payload = _prediction_payload(warmup_payload)
    warmup = tmp_path / "warmup.json"
    predictions = tmp_path / f"predictions{prediction_suffix}"
    output = tmp_path / "report.json"
    _write_json(warmup, warmup_payload)
    if prediction_suffix == ".json":
        _write_json(predictions, predictions_payload)
    else:
        _write_submission_zip(
            predictions,
            _official_submission_payload(warmup_payload),
        )

    first = _run(
        "evaluate_legal_qa.py",
        "--references",
        warmup,
        "--predictions",
        predictions,
        "--output",
        output,
    )
    assert first.returncode == 0, first.stderr
    summary = json.loads(first.stdout)
    assert summary["evaluation_scope"] == "local_diagnostic"
    assert summary["official_scorer_parity"] is False
    assert summary["question_count"] == 3
    assert summary["rouge_l"] == 1.0
    assert 0.9 < summary["meteor"] < 1.0
    first_bytes = output.read_bytes()

    second = _run(
        "evaluate_legal_qa.py",
        "--references",
        warmup,
        "--predictions",
        predictions,
        "--output",
        output,
    )
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes
    report = json.loads(first_bytes)
    assert report["official_scorer_parity"] is False
    assert report["profile"]["exact_token_matches_only"] is True
    assert report["profile"]["external_corpora_used"] is False
    assert len(report["per_query"]) == 3
    assert all(item["rouge_l"]["f1"] == 1.0 for item in report["per_query"])


def test_evaluate_accepts_empty_for_diagnostics_and_reports_zero_contribution(
    tmp_path: Path,
) -> None:
    warmup_payload = _warmup_payload()
    predictions_payload = _prediction_payload(warmup_payload)
    predictions_payload[1]["answer"] = ""
    warmup = tmp_path / "warmup.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "report.json"
    _write_json(warmup, warmup_payload)
    _write_json(predictions, predictions_payload)

    result = _run(
        "evaluate_legal_qa.py",
        "--references",
        warmup,
        "--predictions",
        predictions,
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["aggregate"]["empty_prediction_count"] == 1
    diagnostic = next(item for item in report["per_query"] if item["id"] == "002")
    assert diagnostic["meteor"]["score"] == 0.0
    assert diagnostic["rouge_l"]["f1"] == 0.0


def test_evaluate_rejects_coverage_unsafe_zip_and_output_collision(
    tmp_path: Path,
) -> None:
    warmup = tmp_path / "warmup.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "report.json"
    _write_json(warmup, _warmup_payload())
    _write_json(predictions, [{"id": "001", "answer": "Thiếu hai câu"}])

    coverage = _run(
        "evaluate_legal_qa.py",
        "--references",
        warmup,
        "--predictions",
        predictions,
        "--output",
        output,
    )
    assert coverage.returncode == 2
    assert "prediction IDs do not match references" in coverage.stderr
    assert not output.exists()

    unsafe_zip = tmp_path / "unsafe.zip"
    _write_submission_zip(
        unsafe_zip,
        _official_submission_payload(_warmup_payload()),
        member_name="../submission.json",
    )
    unsafe = _run(
        "evaluate_legal_qa.py",
        "--references",
        warmup,
        "--predictions",
        unsafe_zip,
        "--output",
        output,
    )
    assert unsafe.returncode == 2
    assert "named exactly" in unsafe.stderr

    original = warmup.read_bytes()
    collision = _run(
        "evaluate_legal_qa.py",
        "--references",
        warmup,
        "--predictions",
        predictions,
        "--output",
        warmup,
    )
    assert collision.returncode == 2
    assert "must not overwrite" in collision.stderr
    assert warmup.read_bytes() == original


def test_writer_supports_question_only_mapping_preserves_raw_answer_and_is_stable(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "questions.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "submission.zip"
    _write_json(
        questions,
        {
            "q2": {"question": "Hai?"},
            "q1": {"question": "Một?"},
        },
    )
    payload = [
        {"id": "q1", "answer": "  Giữ đầu cuối\nnguyên dạng  "},
        {"id": "q2", "answer": "Câu trả lời hai"},
    ]
    _write_json(predictions, payload)

    first = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        questions,
        "--output",
        output,
    )
    assert first.returncode == 0, first.stderr
    summary = json.loads(first.stdout)
    assert summary["question_count"] == 2
    assert summary["empty_answer_count"] == 0
    assert summary["member"] == "submission.json"
    first_bytes = output.read_bytes()
    assert summary["artifact_sha256"] == hashlib.sha256(first_bytes).hexdigest()

    second = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        questions,
        "--output",
        output,
    )
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes
    with ZipFile(output) as archive:
        assert archive.namelist() == ["submission.json"]
        assert json.loads(archive.read("submission.json")) == {
            "q1": {"answer": "  Giữ đầu cuối\nnguyên dạng  "},
            "q2": {"answer": "Câu trả lời hai"},
        }


def test_writer_supports_id_and_object_arrays_for_exact_coverage(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "submission.zip"
    payload = [
        {"id": "q1", "answer": "Một"},
        {"id": "q2", "answer": "Hai"},
    ]
    _write_json(predictions, payload)
    for index, questions_payload in enumerate(
        (["q1", "q2"], [{"id": "q2", "question": "Hai?"}, {"id": "q1"}])
    ):
        questions = tmp_path / f"questions-{index}.json"
        _write_json(questions, questions_payload)
        result = _run(
            "write_legal_qa_submission.py",
            "--input",
            predictions,
            "--questions",
            questions,
            "--output",
            output,
        )
        assert result.returncode == 0, result.stderr


def test_writer_rejects_official_mapping_as_internal_prediction_input(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "questions.json"
    predictions = tmp_path / "predictions.json"
    output = tmp_path / "submission.zip"
    _write_json(questions, ["q"])
    _write_json(predictions, {"q": {"answer": "Đáp án"}})

    result = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        questions,
        "--output",
        output,
    )

    assert result.returncode == 2
    assert "invalid LegalQA prediction schema" in result.stderr
    assert not output.exists()


def test_writer_enforces_coverage_empty_policy_and_manifest_strictness(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.json"
    questions = tmp_path / "questions.json"
    output = tmp_path / "submission.zip"
    _write_json(predictions, [{"id": "q1", "answer": ""}])
    _write_json(questions, ["q1", "q2"])

    coverage = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        questions,
        "--output",
        output,
    )
    assert coverage.returncode == 2
    assert "coverage mismatch" in coverage.stderr

    _write_json(questions, ["q1"])
    empty = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        questions,
        "--output",
        output,
    )
    assert empty.returncode == 2
    assert "blank answers" in empty.stderr

    allowed = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        questions,
        "--empty-answer-policy",
        "allow",
        "--output",
        output,
    )
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(allowed.stdout)["empty_answer_count"] == 1

    duplicate_keys = tmp_path / "duplicate-keys.json"
    duplicate_keys.write_text(
        '{"q":{"question":"Một?"},"q":{"question":"Hai?"}}',
        encoding="utf-8",
    )
    strict = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        duplicate_keys,
        "--output",
        output,
    )
    assert strict.returncode == 2
    assert "duplicate JSON object key" in strict.stderr


def test_writer_honors_environment_default_and_rejects_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = tmp_path / "questions.json"
    predictions = tmp_path / "predictions.json"
    environment_output = tmp_path / "from-environment.zip"
    _write_json(questions, ["q"])
    _write_json(predictions, [{"id": "q", "answer": "Đáp án"}])
    monkeypatch.setenv("LEGAL_QA_SUBMISSION_PATH", str(environment_output))

    environment = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        questions,
    )
    assert environment.returncode == 0, environment.stderr
    assert environment_output.is_file()

    collision_output = tmp_path / "collision.zip"
    try:
        collision_output.hardlink_to(predictions)
    except OSError:
        pytest.skip("hard-link creation unavailable")
    original = predictions.read_bytes()
    collision = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        questions,
        "--output",
        collision_output,
    )
    assert collision.returncode == 2
    assert "must not overwrite" in collision.stderr
    assert predictions.read_bytes() == original
    assert collision_output.read_bytes() == original


def test_writer_rejects_non_zip_output_and_symlink_target(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    predictions = tmp_path / "predictions.json"
    _write_json(questions, ["q"])
    _write_json(predictions, [{"id": "q", "answer": "Đáp án"}])
    wrong_suffix = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        questions,
        "--output",
        tmp_path / "submission.json",
    )
    assert wrong_suffix.returncode == 2
    assert "must use .zip" in wrong_suffix.stderr

    target = tmp_path / "target.zip"
    target.write_bytes(b"unchanged")
    link = tmp_path / "linked.zip"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    symlink = _run(
        "write_legal_qa_submission.py",
        "--input",
        predictions,
        "--questions",
        questions,
        "--output",
        link,
    )
    assert symlink.returncode == 2
    assert "symbolic link" in symlink.stderr
    assert target.read_bytes() == b"unchanged"


def test_validator_reports_valid_zip_and_accepts_empty_only_when_requested(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission.zip"
    questions = tmp_path / "questions.json"
    payload = {
        "q1": {"answer": "Một"},
        "q2": {"answer": ""},
    }
    _write_submission_zip(submission, payload)
    _write_json(questions, [{"id": "q2"}, {"id": "q1", "question": "Một?"}])

    default = _run(
        "validate_legal_qa_submission.py",
        "--input",
        submission,
        "--questions",
        questions,
    )
    assert default.returncode == 2
    assert "blank answers" in default.stderr

    allowed = _run(
        "validate_legal_qa_submission.py",
        "--input",
        submission,
        "--questions",
        questions,
        "--empty-answer-policy",
        "allow",
    )
    assert allowed.returncode == 0, allowed.stderr
    summary = json.loads(allowed.stdout)
    assert summary["status"] == "valid"
    assert summary["question_count"] == 2
    assert summary["empty_answer_count"] == 1
    assert (
        summary["artifact_sha256"]
        == hashlib.sha256(submission.read_bytes()).hexdigest()
    )


def test_validator_rejects_json_and_unsafe_zip_shapes(tmp_path: Path) -> None:
    questions = tmp_path / "questions.json"
    _write_json(questions, ["q"])
    standalone = tmp_path / "submission.json"
    _write_json(standalone, [{"id": "q", "answer": "Đáp án"}])
    json_result = _run(
        "validate_legal_qa_submission.py",
        "--input",
        standalone,
        "--questions",
        questions,
    )
    assert json_result.returncode == 2
    assert "must use .zip" in json_result.stderr

    extra = tmp_path / "extra.zip"
    with ZipFile(extra, "w") as archive:
        archive.writestr("submission.json", standalone.read_bytes())
        archive.writestr("extra.txt", "forbidden")
    extra_result = _run(
        "validate_legal_qa_submission.py",
        "--input",
        extra,
        "--questions",
        questions,
    )
    assert extra_result.returncode == 2
    assert "exactly one" in extra_result.stderr

    traversal = tmp_path / "traversal.zip"
    _write_submission_zip(
        traversal,
        {"q": {"answer": "Đáp án"}},
        member_name="../submission.json",
    )
    traversal_result = _run(
        "validate_legal_qa_submission.py",
        "--input",
        traversal,
        "--questions",
        questions,
    )
    assert traversal_result.returncode == 2
    assert "named exactly" in traversal_result.stderr


def test_validator_rejects_legacy_root_array_inside_official_zip(
    tmp_path: Path,
) -> None:
    questions = tmp_path / "questions.json"
    submission = tmp_path / "legacy-array.zip"
    _write_json(questions, ["q"])
    _write_submission_zip(submission, [{"id": "q", "answer": "Đáp án"}])

    result = _run(
        "validate_legal_qa_submission.py",
        "--input",
        submission,
        "--questions",
        questions,
    )

    assert result.returncode == 2
    assert "object" in result.stderr
    assert "array" in result.stderr


def test_oracle_smoke_guards_acknowledgement_and_filename(tmp_path: Path) -> None:
    warmup = tmp_path / "warmup.json"
    output = tmp_path / "oracle_DO_NOT_SUBMIT.zip"
    _write_json(warmup, _warmup_payload())

    no_ack = _run(
        "make_legal_qa_warmup_smoke_submission.py",
        "--warmup",
        warmup,
        "--output",
        output,
    )
    assert no_ack.returncode == 2
    assert "--acknowledge-label-leakage" in no_ack.stderr
    assert not output.exists()

    safe_looking_name = _run(
        "make_legal_qa_warmup_smoke_submission.py",
        "--warmup",
        warmup,
        "--output",
        tmp_path / "submission.zip",
        "--acknowledge-label-leakage",
    )
    assert safe_looking_name.returncode == 2
    assert "filename must contain DO_NOT_SUBMIT" in safe_looking_name.stderr


def test_oracle_smoke_is_deterministic_loud_and_preserves_references(
    tmp_path: Path,
) -> None:
    warmup_payload = _warmup_payload()
    warmup = tmp_path / "warmup.json"
    output = tmp_path / "oracle_DO_NOT_SUBMIT.zip"
    _write_json(warmup, warmup_payload)

    first = _run(
        "make_legal_qa_warmup_smoke_submission.py",
        "--warmup",
        warmup,
        "--output",
        output,
        "--acknowledge-label-leakage",
    )
    assert first.returncode == 0, first.stderr
    summary = json.loads(first.stdout)
    assert summary["label_leakage"] is True
    assert summary["status"] == "local_only_do_not_submit"
    assert "DIRECTLY COPIES REFERENCE ANSWERS" in first.stderr
    assert "MUST NEVER BE UPLOADED OR SUBMITTED" in first.stderr
    first_bytes = output.read_bytes()

    second = _run(
        "make_legal_qa_warmup_smoke_submission.py",
        "--warmup",
        warmup,
        "--output",
        output,
        "--acknowledge-label-leakage",
    )
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes
    with ZipFile(output) as archive:
        payload = json.loads(archive.read("submission.json"))
    assert payload == _official_submission_payload(warmup_payload)


@pytest.mark.skipif(not ACTUAL_WARMUP.is_file(), reason="Task 2 warm-up unavailable")
def test_actual_warmup_oracle_round_trip_is_local_only(tmp_path: Path) -> None:
    output = tmp_path / "actual_oracle_DO_NOT_SUBMIT.zip"
    result = _run(
        "make_legal_qa_warmup_smoke_submission.py",
        "--warmup",
        ACTUAL_WARMUP,
        "--output",
        output,
        "--acknowledge-label-leakage",
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["question_count"] == 500
    assert summary["label_leakage"] is True
    with ZipFile(output) as archive:
        assert archive.namelist() == ["submission.json"]
        payload = json.loads(archive.read("submission.json"))
    warmup = json.loads(ACTUAL_WARMUP.read_text(encoding="utf-8"))
    assert len(payload) == 500
    first_question_id = next(iter(warmup))
    assert payload[first_question_id]["answer"] == warmup[first_question_id]["answer"]

    validation = _run(
        "validate_legal_qa_submission.py",
        "--input",
        output,
        "--questions",
        ACTUAL_WARMUP,
    )
    assert validation.returncode == 0, validation.stderr
    assert json.loads(validation.stdout)["question_count"] == 500

    report = tmp_path / "actual-oracle-diagnostic.json"
    evaluation = _run(
        "evaluate_legal_qa.py",
        "--references",
        ACTUAL_WARMUP,
        "--predictions",
        output,
        "--output",
        report,
    )
    assert evaluation.returncode == 0, evaluation.stderr
    evaluation_summary = json.loads(evaluation.stdout)
    assert evaluation_summary["question_count"] == 500
    assert evaluation_summary["rouge_l"] == 1.0
    assert evaluation_summary["official_scorer_parity"] is False
