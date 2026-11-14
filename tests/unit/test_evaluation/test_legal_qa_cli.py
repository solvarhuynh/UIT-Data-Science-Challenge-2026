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


def _resolve_script(script: str) -> Path:
    for candidate in (
        SCRIPTS / script,
        SCRIPTS / "evaluation" / script,
        SCRIPTS / "inference" / script,
    ):
        if candidate.is_file():
            return candidate
    return SCRIPTS / script


def _run(script: str, *arguments: object) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(_resolve_script(script))]
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


