"""Strict JSON and JSONL loaders for evaluation and submission inputs."""

import json
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from udsc2026.contracts.qa import QAResponse
from udsc2026.evaluation.models import BenchmarkSample, PredictionSample

ModelT = TypeVar("ModelT", bound=BaseModel)


def _reject_non_standard_json_constant(value: str) -> NoReturn:
    """Reject NaN and infinity tokens that RFC-compliant JSON does not allow."""

    raise ValueError(f"non-standard JSON constant {value!r} is not permitted")


def _read_records(path: Path, container_key: str) -> List[Dict[str, Any]]:
    """Read object records from JSON array/wrapper or newline-delimited JSON."""

    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError(
            f"unsupported input format {path.suffix!r}; use .json or .jsonl"
        )

    if suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(
                        line,
                        parse_constant=_reject_non_standard_json_constant,
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} at line {line_number}: {exc.msg}"
                    ) from exc
                except ValueError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path} line {line_number} must contain a JSON object"
                    )
                records.append(value)
    else:
        try:
            with path.open("r", encoding="utf-8-sig") as stream:
                payload = json.load(
                    stream,
                    parse_constant=_reject_non_standard_json_constant,
                )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc
        except ValueError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
        if isinstance(payload, dict):
            unexpected = set(payload).difference({container_key})
            if unexpected or container_key not in payload:
                raise ValueError(
                    f"{path} wrapper must contain only the key {container_key!r}"
                )
            payload = payload[container_key]
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON array")
        records = []
        for index, value in enumerate(payload):
            if not isinstance(value, dict):
                raise ValueError(f"{path} record {index} must be a JSON object")
            records.append(value)

    if not records:
        raise ValueError(f"{path} does not contain any records")
    return records


def _validate_records(
    records: Sequence[Dict[str, Any]],
    model: Type[ModelT],
    path: Path,
) -> List[ModelT]:
    """Validate records and annotate validation errors with their source index."""

    result: List[ModelT] = []
    for index, record in enumerate(records):
        try:
            result.append(model.model_validate(record, strict=True))
        except ValidationError as exc:
            raise ValueError(f"invalid record {index} in {path}: {exc}") from exc
    return result


def _reject_duplicate_question_ids(
    samples: Sequence[BenchmarkSample] | Sequence[PredictionSample],
    path: Path,
) -> None:
    """Reject duplicate IDs at the file boundary."""

    seen: set[str] = set()
    for sample in samples:
        if sample.question_id in seen:
            raise ValueError(f"duplicate question_id in {path}: {sample.question_id}")
        seen.add(sample.question_id)


def load_benchmark(path: str | Path) -> List[BenchmarkSample]:
    """Load and strictly validate a benchmark from JSON or JSONL."""

    resolved = Path(path)
    samples = _validate_records(
        _read_records(resolved, "samples"), BenchmarkSample, resolved
    )
    _reject_duplicate_question_ids(samples, resolved)
    return samples


def load_predictions(path: str | Path) -> List[PredictionSample]:
    """Load and strictly validate file-based retrieval/QA predictions."""

    resolved = Path(path)
    samples = _validate_records(
        _read_records(resolved, "predictions"), PredictionSample, resolved
    )
    _reject_duplicate_question_ids(samples, resolved)
    return samples


def load_qa_responses(path: str | Path) -> List[QAResponse]:
    """Load strict shared ``QAResponse`` records for submission generation."""

    resolved = Path(path)
    records = _read_records(resolved, "results")
    allowed_fields = set(QAResponse.model_fields)
    for index, record in enumerate(records):
        unexpected = sorted(set(record).difference(allowed_fields))
        if unexpected:
            raise ValueError(
                f"invalid record {index} in {resolved}: unexpected fields "
                + ", ".join(unexpected)
            )
    return _validate_records(records, QAResponse, resolved)
