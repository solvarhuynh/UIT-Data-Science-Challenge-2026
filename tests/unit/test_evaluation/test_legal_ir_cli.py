"""End-to-end subprocess tests for the LegalIR warm-up command-line tools."""

from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = PROJECT_ROOT / "scripts"
ACTUAL_WARMUP = PROJECT_ROOT / "data" / "task1" / "warmup.json"
ACTUAL_WARMUP_SHA256 = (
    "fadfbcab2923c085980239396d5564c16c232714f42004a36ed2a3d93962c0f9"
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_submission_zip(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", encoded)


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


def _official_references() -> list[dict[str, object]]:
    return [
        {"id": "q1", "gold_documents": ["gold-1", "gold-1b"]},
        {"id": "q2", "gold_documents": ["gold-2"]},
        {"id": "q3", "gold_documents": ["gold-3", "gold-3b"]},
        {"id": "q4", "gold_documents": ["gold-4"]},
    ]


def _official_predictions() -> list[dict[str, object]]:
    return [
        {"id": "q1", "documents": ["gold-1", "a", "gold-1b"]},
        {"id": "q2", "documents": ["a", "gold-2", "b"]},
        {"id": "q3", "documents": ["a", "b", "c", "gold-3"]},
        {"id": "q4", "documents": ["a", "b", "c"]},
    ]


def _official_wire(
    predictions: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    wire: dict[str, dict[str, object]] = {}
    for prediction in predictions:
        question_id = prediction["id"]
        documents = prediction["documents"]
        assert isinstance(question_id, str)
        assert isinstance(documents, list)
        wire[question_id] = {"answer": documents}
    return wire


@pytest.mark.parametrize(
    "prediction_format",
    ["internal-json", "official-json", "official-zip"],
)
def test_evaluate_internal_and_official_inputs_are_exact_and_deterministic(
    tmp_path: Path,
    prediction_format: str,
) -> None:
    references = tmp_path / "references.json"
    prediction_suffix = ".zip" if prediction_format == "official-zip" else ".json"
    predictions = tmp_path / f"{prediction_format}{prediction_suffix}"
    report = tmp_path / "report.json"
    _write_json(references, _official_references())
    if prediction_format == "internal-json":
        _write_json(predictions, _official_predictions())
    elif prediction_format == "official-json":
        _write_json(predictions, _official_wire(_official_predictions()))
    else:
        _write_submission_zip(
            predictions,
            _official_wire(_official_predictions()),
        )

    first = _run(
        "evaluate_legal_ir.py",
        "--references",
        references,
        "--predictions",
        predictions,
        "--output",
        report,
    )
    assert first.returncode == 0, first.stderr
    assert "Recall(primary)=0.625000000000" in first.stdout
    assert "Precision(secondary)=0.312500000000" in first.stdout
    first_bytes = report.read_bytes()

    second = _run(
        "evaluate_legal_ir.py",
        "--references",
        references,
        "--predictions",
        predictions,
        "--output",
        report,
    )
    assert second.returncode == 0, second.stderr
    assert report.read_bytes() == first_bytes
    decoded = json.loads(first_bytes)
    assert decoded["schema_version"] == "legal-ir-evaluation-v2"
    assert decoded["evaluation_mode"] == "official_set"
    assert decoded["metric_priority"] == ["recall", "precision"]
    assert decoded["aggregate"] == {
        "multi_gold_sample_count": 2,
        "precision": 0.3125,
        "recall": 0.625,
        "sample_count": 4,
    }
    assert [item["relevant_retrieved_count"] for item in decoded["per_query"]] == [
        2,
        1,
        1,
        0,
    ]


def test_evaluate_warmup_multi_gold_is_official_by_default(
    tmp_path: Path,
) -> None:
    warmup = tmp_path / "warmup.json"
    predictions = tmp_path / "submission.json"
    report = tmp_path / "report.json"
    _write_json(
        warmup,
        {"q": {"question": "Văn bản nào?", "answer": ["gold-a", "gold-b"]}},
    )
    _write_json(
        predictions,
        [{"id": "q", "documents": ["gold-b", "x", "y"]}],
    )

    evaluated = _run(
        "evaluate_legal_ir.py",
        "--references",
        warmup,
        "--predictions",
        predictions,
        "--output",
        report,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    decoded = json.loads(report.read_text(encoding="utf-8"))
    assert decoded["evaluation_mode"] == "official_set"
    assert decoded["aggregate"]["recall"] == 0.5
    assert decoded["aggregate"]["precision"] == pytest.approx(1 / 3)
    assert decoded["aggregate"]["multi_gold_sample_count"] == 1
    assert decoded["per_query"][0]["matched_documents"] == ["gold-b"]

    retired_mode = _run(
        "evaluate_legal_ir.py",
        "--references",
        warmup,
        "--predictions",
        predictions,
        "--mode",
        "warmup-any-gold",
        "--output",
        report,
    )
    assert retired_mode.returncode == 2
    assert "invalid choice" in retired_mode.stderr


def test_evaluate_rejects_coverage_errors_and_malicious_zip(tmp_path: Path) -> None:
    references = tmp_path / "references.json"
    predictions = tmp_path / "bad.zip"
    report = tmp_path / "report.json"
    _write_json(references, [{"id": "q", "gold_documents": ["gold"]}])
    with ZipFile(predictions, "w") as archive:
        archive.writestr(
            "../submission.json",
            '[{"id":"q","documents":["gold","a","b"]}]',
        )

    malicious = _run(
        "evaluate_legal_ir.py",
        "--references",
        references,
        "--predictions",
        predictions,
        "--output",
        report,
    )
    assert malicious.returncode == 2
    assert "named exactly" in malicious.stderr
    assert not report.exists()

    _write_json(
        predictions.with_suffix(".json"),
        [{"id": "unexpected", "documents": ["gold", "a", "b"]}],
    )
    coverage = _run(
        "evaluate_legal_ir.py",
        "--references",
        references,
        "--predictions",
        predictions.with_suffix(".json"),
        "--output",
        report,
    )
    assert coverage.returncode == 2
    assert "do not match references" in coverage.stderr


def test_evaluate_refuses_input_output_collision_without_modifying_input(
    tmp_path: Path,
) -> None:
    references = tmp_path / "references.json"
    predictions = tmp_path / "submission.json"
    _write_json(references, [{"id": "q", "gold_documents": ["gold"]}])
    _write_json(
        predictions,
        [{"id": "q", "documents": ["gold", "a", "b"]}],
    )
    original = references.read_bytes()

    result = _run(
        "evaluate_legal_ir.py",
        "--references",
        references,
        "--predictions",
        predictions,
        "--output",
        references,
    )
    assert result.returncode == 2
    assert "must not overwrite" in result.stderr
    assert references.read_bytes() == original


def test_write_submission_enforces_coverage_and_can_append_full_corpus(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    questions = tmp_path / "questions.json"
    corpus = tmp_path / "corpus.json"
    output = tmp_path / "submission.zip"
    full_output = tmp_path / "full.json"
    payload = [
        {"id": "q2", "documents": ["d3", "d1", "d2"]},
        {"id": "q1", "documents": ["d2", "d3", "d1"]},
    ]
    _write_json(candidate, payload)
    _write_json(questions, ["q1", "q2"])
    _write_json(corpus, ["d1", "d2", "d3", "d4", "d5"])

    written = _run(
        "write_legal_ir_submission.py",
        "--input",
        candidate,
        "--questions",
        questions,
        "--corpus-manifest",
        corpus,
        "--output",
        output,
    )
    assert written.returncode == 0, written.stderr
    first_bytes = output.read_bytes()
    repeated = _run(
        "write_legal_ir_submission.py",
        "--input",
        candidate,
        "--questions",
        questions,
        "--corpus-manifest",
        corpus,
        "--output",
        output,
    )
    assert repeated.returncode == 0, repeated.stderr
    assert output.read_bytes() == first_bytes
    with ZipFile(output) as archive:
        assert archive.namelist() == ["submission.json"]
        assert json.loads(archive.read("submission.json")) == _official_wire(payload)

    completed = _run(
        "write_legal_ir_submission.py",
        "--input",
        candidate,
        "--questions",
        questions,
        "--corpus-manifest",
        corpus,
        "--append-missing-corpus",
        "--output",
        full_output,
    )
    assert completed.returncode == 0, completed.stderr
    full_payload = json.loads(full_output.read_text(encoding="utf-8"))
    assert full_payload["q2"]["answer"] == ["d3", "d1", "d2", "d4", "d5"]
    assert full_payload["q1"]["answer"] == ["d2", "d3", "d1", "d4", "d5"]


def test_write_submission_accepts_warmup_as_question_source_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    warmup = tmp_path / "warmup.json"
    output = tmp_path / "submission.zip"
    _write_json(
        warmup,
        {
            "q1": {"question": "Một?", "answer": ["d1"]},
            "q2": {"question": "Hai?", "answer": ["d2"]},
        },
    )
    _write_json(candidate, [{"id": "q1", "documents": ["d1", "d2", "d3"]}])

    result = _run(
        "write_legal_ir_submission.py",
        "--input",
        candidate,
        "--questions",
        warmup,
        "--output",
        output,
    )
    assert result.returncode == 2
    assert "coverage mismatch" in result.stderr
    assert "q2" in result.stderr
    assert not output.exists()


def test_writer_and_validator_accept_question_only_phase_mapping(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    questions = tmp_path / "phase-questions.json"
    output = tmp_path / "submission.zip"
    _write_json(
        questions,
        {
            "q1": {"question": "Một?"},
            "q2": {"question": "Hai?"},
        },
    )
    _write_json(
        candidate,
        [
            {"id": "q2", "documents": ["d2", "d1", "d3"]},
            {"id": "q1", "documents": ["d1", "d2", "d3"]},
        ],
    )

    written = _run(
        "write_legal_ir_submission.py",
        "--input",
        candidate,
        "--questions",
        questions,
        "--output",
        output,
    )
    assert written.returncode == 0, written.stderr

    validated = _run(
        "validate_legal_ir_submission.py",
        "--input",
        output,
        "--questions",
        questions,
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["question_count"] == 2


def test_writer_and_validator_accept_empty_answer_array(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    questions = tmp_path / "questions.json"
    corpus = tmp_path / "corpus.json"
    output = tmp_path / "submission.zip"
    _write_json(candidate, [{"id": "q", "documents": []}])
    _write_json(questions, ["q"])
    _write_json(corpus, ["d1", "d2"])

    written = _run(
        "write_legal_ir_submission.py",
        "--input",
        candidate,
        "--questions",
        questions,
        "--corpus-manifest",
        corpus,
        "--output",
        output,
    )
    assert written.returncode == 0, written.stderr
    assert "min_answer_documents=0" in written.stdout

    validated = _run(
        "validate_legal_ir_submission.py",
        "--input",
        output,
        "--questions",
        questions,
        "--corpus-manifest",
        corpus,
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["min_documents_per_question"] == 0
    with ZipFile(output) as archive:
        assert json.loads(archive.read("submission.json")) == {"q": {"answer": []}}


def test_write_submission_guards_completion_and_all_input_collisions(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    questions = tmp_path / "questions.json"
    _write_json(
        candidate,
        [{"id": "q", "documents": ["d1", "d2", "d3"]}],
    )
    _write_json(questions, ["q"])
    original = candidate.read_bytes()

    no_manifest = _run(
        "write_legal_ir_submission.py",
        "--input",
        candidate,
        "--questions",
        questions,
        "--append-missing-corpus",
        "--output",
        tmp_path / "submission.zip",
    )
    assert no_manifest.returncode == 2
    assert "requires --corpus-manifest" in no_manifest.stderr

    collision = _run(
        "write_legal_ir_submission.py",
        "--input",
        candidate,
        "--questions",
        questions,
        "--output",
        candidate,
    )
    assert collision.returncode == 2
    assert "must not overwrite" in collision.stderr
    assert candidate.read_bytes() == original


def test_write_submission_honors_environment_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.json"
    questions = tmp_path / "questions.json"
    output = tmp_path / "from-environment.zip"
    _write_json(
        candidate,
        [{"id": "q", "documents": ["d1", "d2", "d3"]}],
    )
    _write_json(questions, ["q"])
    monkeypatch.setenv("LEGAL_IR_SUBMISSION_PATH", str(output))

    result = _run(
        "write_legal_ir_submission.py",
        "--input",
        candidate,
        "--questions",
        questions,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert str(output) in result.stdout


def test_validate_submission_checks_optional_manifests_and_full_ranking(
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission.zip"
    questions = tmp_path / "questions.json"
    corpus = tmp_path / "corpus.json"
    _write_submission_zip(
        submission,
        {"q": {"answer": ["d3", "d1", "d2"]}},
    )
    _write_json(questions, ["q"])
    _write_json(corpus, ["d1", "d2", "d3"])

    valid = _run(
        "validate_legal_ir_submission.py",
        "--input",
        submission,
        "--questions",
        questions,
        "--corpus-manifest",
        corpus,
        "--require-complete-ranking",
    )
    assert valid.returncode == 0, valid.stderr
    assert json.loads(valid.stdout) == {
        "max_documents_per_question": 3,
        "min_documents_per_question": 3,
        "question_count": 1,
        "status": "valid",
    }

    _write_json(corpus, ["d1", "d2", "d3", "d4"])
    incomplete = _run(
        "validate_legal_ir_submission.py",
        "--input",
        submission,
        "--corpus-manifest",
        corpus,
        "--require-complete-ranking",
    )
    assert incomplete.returncode == 2
    assert "complete corpus ranking" in incomplete.stderr


@pytest.mark.parametrize("suffix", [".json", ".zip"])
def test_official_validator_rejects_legacy_array_contract(
    tmp_path: Path,
    suffix: str,
) -> None:
    submission = tmp_path / f"legacy{suffix}"
    legacy = [{"id": "q", "documents": ["d1", "d2", "d3"]}]
    if suffix == ".json":
        _write_json(submission, legacy)
    else:
        _write_submission_zip(submission, legacy)

    result = _run("validate_legal_ir_submission.py", "--input", submission)

    assert result.returncode == 2
    assert "root must be an object" in result.stderr


def test_validate_submission_rejects_unsafe_archive_and_bad_manifest(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "submission.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "submission.json",
            '[{"id":"q","documents":["d1","d2","d3"]}]',
        )
        archive.writestr("extra.txt", "not allowed")
    corpus = tmp_path / "corpus.json"
    _write_json(corpus, {"documents": ["d1", "d2", "d3"]})

    unsafe = _run("validate_legal_ir_submission.py", "--input", archive_path)
    assert unsafe.returncode == 2
    assert "exactly one" in unsafe.stderr

    bad_manifest = _run(
        "validate_legal_ir_submission.py",
        "--input",
        archive_path,
        "--corpus-manifest",
        corpus,
    )
    assert bad_manifest.returncode == 2
    assert "root must be a JSON array" in bad_manifest.stderr


def test_audit_warmup_reports_deterministic_label_and_unicode_diagnostics(
    tmp_path: Path,
) -> None:
    warmup = tmp_path / "warmup.json"
    output = tmp_path / "audit.json"
    decomposed = unicodedata.normalize("NFD", "Bảo hiểm?")
    _write_json(
        warmup,
        {
            "q1": {"question": "  Hỏi gì?  ", "answer": ["d1"]},
            "q2": {"question": decomposed, "answer": ["d2", "d3"]},
            "q3": {"question": "Khác?", "answer": ["d1"]},
        },
    )

    first = _run(
        "audit_legal_ir_warmup.py",
        "--input",
        warmup,
        "--output",
        output,
    )
    assert first.returncode == 0, first.stderr
    first_bytes = output.read_bytes()
    second = _run(
        "audit_legal_ir_warmup.py",
        "--input",
        warmup,
        "--output",
        output,
    )
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes
    report = json.loads(first_bytes)
    assert report["question_count"] == 3
    assert report["total_answer_label_count"] == 4
    assert report["unique_labeled_document_count"] == 3
    assert report["single_gold_question_count"] == 2
    assert report["multi_gold_question_count"] == 1
    assert report["answer_count_distribution"] == {"1": 2, "2": 1}
    assert report["whitespace_normalization_question_ids"] == ["q1"]
    assert report["non_nfc_question_ids"] == ["q2"]
    assert report["normalized_question_change_ids"] == ["q1", "q2"]


def test_audit_refuses_to_overwrite_warmup(tmp_path: Path) -> None:
    warmup = tmp_path / "warmup.json"
    _write_json(warmup, {"q": {"question": "Hỏi?", "answer": ["d1"]}})
    original = warmup.read_bytes()

    result = _run(
        "audit_legal_ir_warmup.py",
        "--input",
        warmup,
        "--output",
        warmup,
    )
    assert result.returncode == 2
    assert "must not overwrite" in result.stderr
    assert warmup.read_bytes() == original


@pytest.mark.skipif(not ACTUAL_WARMUP.is_file(), reason="local warmup.json unavailable")
def test_actual_warmup_audit_and_official_set_evaluation(tmp_path: Path) -> None:
    output = tmp_path / "actual-audit.json"
    audit = _run(
        "audit_legal_ir_warmup.py",
        "--input",
        ACTUAL_WARMUP,
        "--output",
        output,
    )
    assert audit.returncode == 0, audit.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset_sha256"] == ACTUAL_WARMUP_SHA256
    assert report["question_count"] == 500
    assert report["total_answer_label_count"] == 542
    assert report["unique_labeled_document_count"] == 426
    assert report["multi_gold_question_count"] == 37
    assert report["whitespace_normalization_question_count"] == 22
    assert report["non_nfc_question_count"] == 5

    warmup_payload = json.loads(ACTUAL_WARMUP.read_text(encoding="utf-8"))
    predictions = tmp_path / "oracle-predictions.json"
    _write_json(
        predictions,
        [
            {"id": question_id, "documents": record["answer"]}
            for question_id, record in warmup_payload.items()
        ],
    )
    evaluation_output = tmp_path / "evaluation.json"
    evaluated = _run(
        "evaluate_legal_ir.py",
        "--references",
        ACTUAL_WARMUP,
        "--predictions",
        predictions,
        "--output",
        evaluation_output,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    evaluation = json.loads(evaluation_output.read_text(encoding="utf-8"))
    assert evaluation["aggregate"]["multi_gold_sample_count"] == 37
    assert evaluation["aggregate"]["recall"] == 1.0
    assert evaluation["aggregate"]["precision"] == 1.0


def test_oracle_smoke_requires_acknowledgement_and_dangerous_filename(
    tmp_path: Path,
) -> None:
    warmup = tmp_path / "warmup.json"
    output = tmp_path / "oracle_DO_NOT_SUBMIT.zip"
    _write_json(
        warmup,
        {
            "q1": {"question": "Một?", "answer": ["d2", "d1"]},
            "q2": {"question": "Hai?", "answer": ["d3"]},
            "q3": {"question": "Ba?", "answer": ["d4"]},
        },
    )

    guarded = _run(
        "make_warmup_smoke_submission.py",
        "--warmup",
        warmup,
        "--output",
        output,
    )
    assert guarded.returncode == 2
    assert "--acknowledge-label-leakage" in guarded.stderr
    assert not output.exists()

    unsafe_name = _run(
        "make_warmup_smoke_submission.py",
        "--warmup",
        warmup,
        "--output",
        tmp_path / "looks-safe.zip",
        "--acknowledge-label-leakage",
    )
    assert unsafe_name.returncode == 2
    assert "filename must contain DO_NOT_SUBMIT" in unsafe_name.stderr

    created = _run(
        "make_warmup_smoke_submission.py",
        "--warmup",
        warmup,
        "--output",
        output,
        "--acknowledge-label-leakage",
    )
    assert created.returncode == 0, created.stderr
    assert "DIRECTLY LEAKS" in created.stderr
    assert "NEVER UPLOAD OR SUBMIT" in created.stderr
    first_bytes = output.read_bytes()
    repeated = _run(
        "make_warmup_smoke_submission.py",
        "--warmup",
        warmup,
        "--output",
        output,
        "--acknowledge-label-leakage",
    )
    assert repeated.returncode == 0, repeated.stderr
    assert output.read_bytes() == first_bytes
    with ZipFile(output) as archive:
        payload = json.loads(archive.read("submission.json"))
    assert payload == {
        "q1": {"answer": ["d2", "d1"]},
        "q2": {"answer": ["d3"]},
        "q3": {"answer": ["d4"]},
    }


def test_oracle_smoke_accepts_fewer_than_three_labeled_documents(
    tmp_path: Path,
) -> None:
    warmup = tmp_path / "warmup.json"
    output = tmp_path / "oracle_DO_NOT_SUBMIT.zip"
    _write_json(
        warmup,
        {
            "q1": {"question": "Một?", "answer": ["d1"]},
            "q2": {"question": "Hai?", "answer": ["d2"]},
        },
    )

    result = _run(
        "make_warmup_smoke_submission.py",
        "--warmup",
        warmup,
        "--output",
        output,
        "--acknowledge-label-leakage",
    )
    assert result.returncode == 0, result.stderr
    with ZipFile(output) as archive:
        assert json.loads(archive.read("submission.json")) == {
            "q1": {"answer": ["d1"]},
            "q2": {"answer": ["d2"]},
        }
