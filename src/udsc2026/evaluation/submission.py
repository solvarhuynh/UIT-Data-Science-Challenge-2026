"""Configurable, schema-safe CSV writer over the shared ``QAResponse`` contract."""

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from udsc2026.contracts.qa import QAResponse

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SubmissionSource = Literal[
    "answer",
    "citations",
    "used_prompt_version",
    "retrieval_hits",
    "confidence",
    "warnings",
    "cache_hit",
    "trace_id",
]
Serializer = Literal["text", "json", "count", "join"]

_LIST_SOURCES = {"citations", "retrieval_hits", "warnings"}
_MODEL_LIST_SOURCES = {"citations", "retrieval_hits"}
_SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@")
_SPREADSHEET_CONTROL_PREFIXES = ("\t", "\r", "\n")


def _protect_spreadsheet_text(value: str) -> str:
    """Force formula-looking untrusted text to remain literal in spreadsheets."""

    trimmed_prefix = value.lstrip(" \t\r\n")
    if value.startswith(_SPREADSHEET_CONTROL_PREFIXES) or trimmed_prefix.startswith(
        _SPREADSHEET_FORMULA_PREFIXES
    ):
        return f"'{value}"
    return value


class SubmissionColumn(BaseModel):
    """One output CSV column mapped to a supported ``QAResponse`` field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonEmptyString
    source: SubmissionSource
    serializer: Serializer = "text"
    required: bool = True
    default: Optional[str] = None
    separator: str = " | "

    @field_validator("separator")
    @classmethod
    def reject_line_breaks_in_separator(cls, value: str) -> str:
        """Keep joined values confined to one CSV cell."""

        if "\r" in value or "\n" in value:
            raise ValueError("separator cannot contain line breaks")
        return value

    def model_post_init(self, __context: Any) -> None:
        """Validate serializer/source compatibility after field parsing."""

        if self.serializer in {"count", "join"} and self.source not in _LIST_SOURCES:
            raise ValueError(
                f"serializer {self.serializer!r} requires a list-valued source"
            )
        if self.serializer == "text" and self.source in _LIST_SOURCES:
            raise ValueError("list-valued sources cannot use the text serializer")
        if self.serializer == "join" and self.source in _MODEL_LIST_SOURCES:
            raise ValueError("model-valued lists must use the json serializer")


class SubmissionSchema(BaseModel):
    """Versioned generic CSV layout; replace it when CodaLab publishes its schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: NonEmptyString = "qa_response_v1"
    columns: Tuple[SubmissionColumn, ...] = Field(
        default_factory=lambda: (
            SubmissionColumn(name="trace_id", source="trace_id", required=False),
            SubmissionColumn(name="answer", source="answer"),
            SubmissionColumn(name="citations", source="citations", serializer="json"),
            SubmissionColumn(name="confidence", source="confidence", required=False),
        ),
        min_length=1,
    )
    delimiter: Literal[",", ";", "\t"] = ","
    include_header: bool = True
    encoding: Literal["utf-8", "utf-8-sig"] = "utf-8-sig"

    @field_validator("columns")
    @classmethod
    def reject_duplicate_column_names(
        cls, columns: Tuple[SubmissionColumn, ...]
    ) -> Tuple[SubmissionColumn, ...]:
        """Prevent silent overwrites while materializing dynamic rows."""

        names = [column.name for column in columns]
        if len(names) != len(set(names)):
            raise ValueError("submission column names must be unique")
        return columns


class SubmissionRow(BaseModel):
    """Validated serialized values for one dynamic submission row."""

    model_config = ConfigDict(extra="forbid")

    values: Dict[NonEmptyString, str] = Field(min_length=1)


def _serialize_value(value: Any, column: SubmissionColumn) -> str:
    """Serialize one supported contract value without locale dependence."""

    if value is None:
        if column.default is not None:
            return _protect_spreadsheet_text(column.default)
        if column.required:
            raise ValueError(f"required submission source {column.source!r} is missing")
        return ""

    if column.serializer == "count":
        return str(len(value))
    if column.serializer == "join":
        return _protect_spreadsheet_text(column.separator.join(value))
    if column.serializer == "json":
        serialized = (
            [
                item.model_dump(mode="json") if isinstance(item, BaseModel) else item
                for item in value
            ]
            if isinstance(value, list)
            else value
        )
        return json.dumps(
            serialized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"submission source {column.source!r} must be a finite number"
            )
        return format(value, ".15g")
    if isinstance(value, str):
        if column.required and not value:
            raise ValueError(
                f"required submission source {column.source!r} is an empty string"
            )
        return _protect_spreadsheet_text(value)
    return str(value)


def build_submission_row(
    result: QAResponse,
    schema: SubmissionSchema,
) -> SubmissionRow:
    """Map one shared QA contract to a validated dynamic CSV row."""

    if not isinstance(result, QAResponse):
        raise TypeError("submission results must contain QAResponse objects")
    values = {
        column.name: _serialize_value(getattr(result, column.source), column)
        for column in schema.columns
    }
    return SubmissionRow(values=values)


def write_submission(
    results: List[QAResponse],
    output_path: str | Path,
    schema: Optional[SubmissionSchema] = None,
) -> None:
    """Write deterministic CSV without assuming an unpublished competition schema."""

    if not results:
        raise ValueError("at least one QAResponse is required")
    resolved_schema = schema if schema is not None else SubmissionSchema()
    if not isinstance(resolved_schema, SubmissionSchema):
        raise TypeError("schema must be a SubmissionSchema")
    path = Path(output_path)
    if path.suffix.casefold() != ".csv":
        raise ValueError("submission output path must use the .csv extension")
    if path.is_symlink():
        raise ValueError("submission output target must not be a symbolic link")
    if path.is_dir():
        raise ValueError("submission output target must not be a directory")
    if path.exists() and not path.is_file():
        raise ValueError("submission output target must be a regular file")
    rows = [build_submission_row(result, resolved_schema) for result in results]

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        stream = os.fdopen(
            descriptor,
            "w",
            encoding=resolved_schema.encoding,
            newline="",
        )
        descriptor = -1
        with stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[column.name for column in resolved_schema.columns],
                delimiter=resolved_schema.delimiter,
                lineterminator="\n",
                extrasaction="raise",
            )
            if resolved_schema.include_header:
                writer.writeheader()
            writer.writerows(row.values for row in rows)
        os.replace(temporary_name, path)
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
