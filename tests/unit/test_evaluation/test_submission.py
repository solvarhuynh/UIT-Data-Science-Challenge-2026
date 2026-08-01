"""Tests for generic schema-configured QAResponse CSV generation."""

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from udsc2026.contracts.qa import QAResponse
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.loaders import load_qa_responses
from udsc2026.evaluation.submission import (
    SubmissionColumn,
    SubmissionSchema,
    build_submission_row,
    write_submission,
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "tv5"


def _schema_from_fixture() -> SubmissionSchema:
    return SubmissionSchema.model_validate_json(
        (FIXTURE_DIR / "submission_schema.json").read_text(encoding="utf-8")
    )


def test_write_submission_uses_configured_columns_and_utf8_bom(
    tmp_path: Path,
) -> None:
    results = load_qa_responses(FIXTURE_DIR / "qa_responses.jsonl")
    output = tmp_path / "submission.csv"
    write_submission(results, output, _schema_from_fixture())

    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    with output.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == ["id", "answer", "citations_json", "citation_count"]
    assert rows[0]["id"] == "tv5_dev_002"
    assert rows[0]["citation_count"] == "1"
    assert json.loads(rows[0]["citations_json"])[0]["article"] == "Điều 1"


def test_submission_output_is_deterministic(tmp_path: Path) -> None:
    results = load_qa_responses(FIXTURE_DIR / "qa_responses.jsonl")
    output = tmp_path / "submission.csv"
    write_submission(results, output, _schema_from_fixture())
    first = output.read_bytes()
    write_submission(results, output, _schema_from_fixture())
    assert output.read_bytes() == first


def test_default_schema_is_generic_and_based_only_on_qa_response() -> None:
    schema = SubmissionSchema()
    assert schema.schema_name == "qa_response_v1"
    assert [column.source for column in schema.columns] == [
        "trace_id",
        "answer",
        "citations",
        "confidence",
    ]


def test_missing_required_submission_value_is_rejected() -> None:
    result = QAResponse(
        answer="Trả lời",
        used_prompt_version="v1",
        trace_id=None,
    )
    schema = SubmissionSchema(
        columns=[
            SubmissionColumn(name="id", source="trace_id", required=True),
        ]
    )
    with pytest.raises(ValueError, match="required submission source"):
        build_submission_row(result, schema)


def test_submission_schema_rejects_duplicate_columns_and_bad_serializers() -> None:
    duplicate = SubmissionColumn(name="answer", source="answer")
    with pytest.raises(ValidationError, match="unique"):
        SubmissionSchema(columns=[duplicate, duplicate])
    with pytest.raises(ValidationError, match="requires a list"):
        SubmissionColumn(name="bad", source="answer", serializer="count")
    with pytest.raises(ValidationError, match="must use the json"):
        SubmissionColumn(name="bad", source="citations", serializer="join")
    with pytest.raises(ValidationError, match="cannot use the text"):
        SubmissionColumn(name="bad", source="warnings", serializer="text")


def test_submission_columns_are_immutable_after_validation() -> None:
    column = SubmissionColumn(name="answer", source="answer")

    with pytest.raises(ValidationError, match="frozen"):
        column.serializer = "count"


def test_submission_schema_is_deeply_immutable_after_validation() -> None:
    schema = SubmissionSchema()

    with pytest.raises(ValidationError, match="frozen"):
        schema.delimiter = ";"
    with pytest.raises(AttributeError):
        schema.columns.append(  # type: ignore[attr-defined]
            SubmissionColumn(name="extra", source="answer")
        )


def test_submission_rejects_non_finite_scalar_and_nested_json_numbers() -> None:
    # Bypass the stricter shared contract to prove the output boundary remains safe
    # when handed an object created by an unvalidated legacy/internal path.
    non_finite_confidence = QAResponse.model_construct(
        answer="Trả lời",
        used_prompt_version="v1",
        confidence=float("nan"),
    )
    confidence_schema = SubmissionSchema(
        columns=[
            SubmissionColumn(
                name="confidence",
                source="confidence",
                required=False,
            )
        ]
    )
    with pytest.raises(ValueError, match="finite number"):
        build_submission_row(non_finite_confidence, confidence_schema)

    non_finite_hit = QAResponse.model_construct(
        answer="Trả lời",
        used_prompt_version="v1",
        retrieval_hits=[
            RetrievalHit.model_construct(
                chunk_id="chunk",
                doc_id="doc",
                text="Nội dung",
                score=float("inf"),
                metadata={},
            )
        ],
    )
    hit_schema = SubmissionSchema(
        columns=[
            SubmissionColumn(
                name="hits",
                source="retrieval_hits",
                serializer="json",
            )
        ]
    )
    with pytest.raises(ValueError):
        build_submission_row(non_finite_hit, hit_schema)


@pytest.mark.parametrize(
    "dangerous_value",
    ["=2+2", "+1+1", "-1+1", "@SUM(A1:A2)", "\t=2+2", "\r=2+2", "\n=2+2"],
)
def test_submission_neutralizes_spreadsheet_formulas(
    dangerous_value: str,
    tmp_path: Path,
) -> None:
    # model_construct also proves the CSV boundary remains safe if a legacy
    # internal caller bypasses QAResponse's whitespace normalization.
    result = QAResponse.model_construct(
        answer=dangerous_value,
        used_prompt_version="v1",
        trace_id=dangerous_value,
    )
    schema = SubmissionSchema(
        columns=[
            SubmissionColumn(name="trace_id", source="trace_id"),
            SubmissionColumn(name="answer", source="answer"),
        ]
    )

    row = build_submission_row(result, schema)
    assert row.values["trace_id"] == f"'{dangerous_value}"
    assert row.values["answer"] == f"'{dangerous_value}"

    output = tmp_path / "submission.csv"
    write_submission([result], output, schema)
    with output.open("r", encoding="utf-8-sig", newline="") as stream:
        written = next(csv.DictReader(stream))
    assert written == row.values


def test_write_submission_rejects_empty_results_and_non_csv_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        write_submission([], tmp_path / "submission.csv")
    results = load_qa_responses(FIXTURE_DIR / "qa_responses.jsonl")
    with pytest.raises(ValueError, match=".csv"):
        write_submission(results, tmp_path / "submission.txt")


def test_write_submission_rejects_directory_or_symlink_target(
    tmp_path: Path,
) -> None:
    results = load_qa_responses(FIXTURE_DIR / "qa_responses.jsonl")
    directory = tmp_path / "submission.csv"
    directory.mkdir()
    with pytest.raises(ValueError, match="directory"):
        write_submission(results, directory)

    link = tmp_path / "linked.csv"
    target = tmp_path / "target.csv"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    with pytest.raises(ValueError, match="symbolic link"):
        write_submission(results, link)


def test_write_submission_rejects_untyped_results(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="QAResponse"):
        write_submission(  # type: ignore[list-item]
            [{"answer": "not validated"}],
            tmp_path / "submission.csv",
        )


def test_write_submission_uses_mkstemp_descriptor_without_reopening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = load_qa_responses(FIXTURE_DIR / "qa_responses.jsonl")
    output = tmp_path / "submission.csv"

    def reject_named_open(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("temporary output must not be reopened by name")

    monkeypatch.setattr("builtins.open", reject_named_open)
    write_submission(results, output, _schema_from_fixture())

    assert output.is_file()
