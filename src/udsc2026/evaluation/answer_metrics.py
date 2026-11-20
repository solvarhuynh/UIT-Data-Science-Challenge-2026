"""Strict LegalQA data contracts and organizer-compatible score reports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Any, List, Literal, NoReturn, Sequence, Tuple

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    Strict,
    StringConstraints,
    ValidationError,
    model_validator,
)

from udsc2026.evaluation.answer_scoring import official_meteor, official_rouge_l


def _validate_identifier(value: str) -> str:
    """Reject ambiguous IDs without rewriting their identity."""

    if not value:
        raise ValueError("identifier must not be empty")
    if value != value.strip():
        raise ValueError("identifier must not contain surrounding whitespace")
    if any(
        ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F for character in value
    ):
        raise ValueError("identifier must not contain control characters")
    _validate_unicode_scalar_string(value)
    return value


Identifier = Annotated[str, Strict(), AfterValidator(_validate_identifier)]
Fingerprint = Annotated[
    str,
    Strict(),
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


def _validate_unicode_scalar_string(value: str) -> str:
    """Reject lone UTF-16 surrogates while preserving every valid code point."""

    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("text must contain valid Unicode scalar values") from exc
    return value


def _validate_nonblank_raw_text(value: str) -> str:
    """Require substantive source text while retaining it byte-for-byte in memory."""

    if not value.strip():
        raise ValueError("reference text must not be blank")
    return _validate_unicode_scalar_string(value)


ReferenceText = Annotated[
    str,
    Strict(),
    AfterValidator(_validate_nonblank_raw_text),
]
PredictionText = Annotated[
    str,
    Strict(),
    AfterValidator(_validate_unicode_scalar_string),
]


class LegalQAModel(BaseModel):
    """Strict base contract for Task 2 evaluation artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_assignment=True,
    )


class LegalQAWarmupRecord(LegalQAModel):
    """Raw value associated with one ID in the organizer warm-up mapping."""

    question: ReferenceText
    answer: ReferenceText


class LegalQAWarmupDataset(RootModel[dict[Identifier, LegalQAWarmupRecord]]):
    """Exact root-mapping schema ``id -> {question, answer}``."""

    model_config = ConfigDict(strict=True, validate_assignment=True)

    @model_validator(mode="after")
    def reject_empty_dataset(self) -> "LegalQAWarmupDataset":
        """Require at least one LegalQA reference."""

        if not self.root:
            raise ValueError("LegalQA warm-up dataset must not be empty")
        return self


class LegalQAWarmupSample(LegalQAModel):
    """One raw question/reference pair with its opaque competition ID."""

    id: Identifier
    question: ReferenceText
    answer: ReferenceText


class LegalQAPrediction(LegalQAModel):
    """One raw generated answer; an empty answer is valid and scores zero."""

    id: Identifier
    answer: PredictionText


def _reject_duplicate_sample_ids(
    samples: Sequence[LegalQAPrediction] | Sequence[LegalQAWarmupSample],
    *,
    label: str,
) -> None:
    """Reject duplicate IDs at collection and evaluation boundaries."""

    seen: set[str] = set()
    for sample in samples:
        if sample.id in seen:
            raise ValueError(f"duplicate question ID in {label}: {sample.id}")
        seen.add(sample.id)


class LegalQAPredictionSet(RootModel[List[LegalQAPrediction]]):
    """Strict prediction JSON-array contract ``[{id, answer}, ...]``."""

    model_config = ConfigDict(strict=True, validate_assignment=True)

    @model_validator(mode="after")
    def validate_collection(self) -> "LegalQAPredictionSet":
        """Reject empty or duplicate-ID prediction arrays."""

        if not self.root:
            raise ValueError("LegalQA prediction set must not be empty")
        _reject_duplicate_sample_ids(self.root, label="predictions")
        return self


class LegalQAMetricProfile(LegalQAModel):
    """Names and parameters of the organizer-compatible answer metrics."""

    meteor: Literal["nltk-meteor-wordnet-omw-v1"] = "nltk-meteor-wordnet-omw-v1"
    rouge_l: Literal["rouge-score-rougeL-no-stem-v1"] = (
        "rouge-score-rougeL-no-stem-v1"
    )
    tokenization: Literal["organizer-whitespace-and-rouge-score-v1"] = (
        "organizer-whitespace-and-rouge-score-v1"
    )
    official_scorer_parity: Literal[True] = True


class LegalQAMeteorQueryMetrics(LegalQAModel):
    """Organizer-compatible METEOR score for one answer."""

    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class LegalQARougeLQueryMetrics(LegalQAModel):
    """Organizer-compatible ROUGE-L F1 for one answer."""

    f1: float = Field(ge=0, le=1, allow_inf_nan=False)


class LegalQAQueryDiagnostic(LegalQAModel):
    """Per-question scores and answer lengths for audit reports."""

    id: Identifier
    prediction_character_count: int = Field(ge=0)
    reference_character_count: int = Field(ge=1)
    empty_prediction: bool
    meteor: LegalQAMeteorQueryMetrics
    rouge_l: LegalQARougeLQueryMetrics


class LegalQAAggregate(LegalQAModel):
    """Macro-average diagnostic scores for one complete prediction set."""

    sample_count: int = Field(ge=1)
    empty_prediction_count: int = Field(ge=0)
    meteor: float = Field(ge=0, le=1, allow_inf_nan=False)
    rouge_l: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_empty_count(self) -> "LegalQAAggregate":
        """Bound the empty-token prediction count by the evaluated sample count."""

        if self.empty_prediction_count > self.sample_count:
            raise ValueError("empty_prediction_count cannot exceed sample_count")
        return self


class LegalQAEvaluationReport(LegalQAModel):
    """Per-question answer metrics with exact organizer scoring behavior."""

    schema_version: Literal["legal-qa-evaluation-v2"] = "legal-qa-evaluation-v2"
    evaluation_scope: Literal["reference_evaluation"] = "reference_evaluation"
    official_scorer_parity: Literal[True] = True
    dataset_fingerprint: Fingerprint
    prediction_fingerprint: Fingerprint
    profile: LegalQAMetricProfile = Field(default_factory=LegalQAMetricProfile)
    aggregate: LegalQAAggregate
    per_query: Tuple[LegalQAQueryDiagnostic, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aggregate(self) -> "LegalQAEvaluationReport":
        """Require the report means and answer coverage to match its rows."""

        if self.aggregate.sample_count != len(self.per_query):
            raise ValueError("aggregate sample_count must match per_query")
        ids = [item.id for item in self.per_query]
        if len(ids) != len(set(ids)):
            raise ValueError("per_query IDs must be unique")
        empty_count = sum(item.empty_prediction for item in self.per_query)
        if self.aggregate.empty_prediction_count != empty_count:
            raise ValueError("empty_prediction_count must match per_query")
        meteor = sum(item.meteor.score for item in self.per_query) / len(self.per_query)
        rouge_l = sum(item.rouge_l.f1 for item in self.per_query) / len(self.per_query)
        if not math.isclose(self.aggregate.meteor, meteor, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("aggregate METEOR must equal the per-query mean")
        if not math.isclose(self.aggregate.rouge_l, rouge_l, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("aggregate ROUGE-L must equal the per-query mean")
        return self


def _reject_non_standard_json_constant(value: str) -> NoReturn:
    """Reject JSON extensions such as NaN and Infinity."""

    raise ValueError(f"non-standard JSON constant {value!r} is not permitted")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Decode a JSON object without silently overwriting duplicate keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_strict_json(path: str | Path) -> tuple[Path, Any]:
    """Read RFC-compatible UTF-8 JSON while retaining precise source errors."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"LegalQA input file does not exist: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8-sig") as stream:
            payload = json.load(
                stream,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_non_standard_json_constant,
            )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {resolved}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid JSON in {resolved}: {exc}") from exc
    return resolved, payload


def load_legal_qa_warmup(path: str | Path) -> List[LegalQAWarmupSample]:
    """Load the strict Task 2 root mapping without modifying raw text."""

    resolved, payload = _read_strict_json(path)
    try:
        dataset = LegalQAWarmupDataset.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise ValueError(
            f"invalid LegalQA warm-up schema in {resolved}: {exc}"
        ) from exc
    return [
        LegalQAWarmupSample(
            id=question_id,
            question=record.question,
            answer=record.answer,
        )
        for question_id, record in dataset.root.items()
    ]


def load_legal_qa_predictions(path: str | Path) -> List[LegalQAPrediction]:
    """Load a strict Task 2 prediction JSON array without rewriting answers."""

    resolved, payload = _read_strict_json(path)
    try:
        dataset = LegalQAPredictionSet.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise ValueError(
            f"invalid LegalQA prediction schema in {resolved}: {exc}"
        ) from exc
    return list(dataset.root)


def _validate_question_manifest_record(
    record: object,
    *,
    allowed_fields: set[str],
    require_question: bool,
    location: str,
) -> dict[str, Any]:
    """Validate one question-source object without changing its text."""

    if not isinstance(record, dict):
        raise TypeError(f"{location} must be an object")
    unexpected = set(record).difference(allowed_fields)
    if unexpected:
        raise ValueError(
            f"{location} has unexpected fields: {', '.join(sorted(unexpected))}"
        )
    if require_question and "question" not in record:
        raise ValueError(f"{location} requires field 'question'")
    if "question" in record:
        question = record["question"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{location} question must be a non-blank string")
        _validate_unicode_scalar_string(question)
    if "answer" in record:
        answer = record["answer"]
        # Organizer public/test manifests retain this field but encode
        # unreleased references as JSON null. This loader derives IDs only, so
        # null is valid metadata here; prediction and warm-up loaders remain
        # strict and continue to require strings.
        if answer is not None:
            if not isinstance(answer, str):
                raise TypeError(f"{location} answer must be a string or null")
            _validate_unicode_scalar_string(answer)
    return record


def _validate_manifest_identifier(value: object, *, location: str) -> str:
    """Validate one opaque manifest ID without coercion or normalization."""

    if not isinstance(value, str):
        raise TypeError(f"{location} must be a string")
    try:
        return _validate_identifier(value)
    except ValueError as exc:
        raise ValueError(f"invalid {location}: {exc}") from exc


def load_legal_qa_question_ids(path: str | Path) -> List[str]:
    """Load exact coverage from Warm-up, phase input, or an ID manifest.

    Supported JSON roots are the organizer mapping
    ``id -> {question, optional nullable answer}``, an array of opaque string
    IDs, or an array of objects with ``id`` and optional nullable
    ``question``/``answer`` fields.
    This loader only derives coverage; it never rewrites question or answer
    text and does not expose reference answers as predictions.
    """

    resolved, payload = _read_strict_json(path)
    question_ids: List[str] = []
    if isinstance(payload, dict):
        if not payload:
            raise ValueError(f"question mapping in {resolved} must not be empty")
        for raw_id, raw_record in payload.items():
            question_id = _validate_manifest_identifier(
                raw_id,
                location="question ID",
            )
            _validate_question_manifest_record(
                raw_record,
                allowed_fields={"question", "answer"},
                require_question=True,
                location=f"question mapping record {question_id!r}",
            )
            question_ids.append(question_id)
    elif isinstance(payload, list):
        if not payload:
            raise ValueError(f"question manifest in {resolved} must not be empty")
        if all(isinstance(item, str) for item in payload):
            question_ids = [
                _validate_manifest_identifier(item, location="question ID")
                for item in payload
            ]
        elif all(isinstance(item, dict) for item in payload):
            for index, raw_record in enumerate(payload):
                record = _validate_question_manifest_record(
                    raw_record,
                    allowed_fields={"id", "question", "answer"},
                    require_question=False,
                    location=f"question manifest record {index}",
                )
                if "id" not in record:
                    raise ValueError(
                        f"question manifest record {index} requires field 'id'"
                    )
                question_ids.append(
                    _validate_manifest_identifier(
                        record["id"],
                        location=f"question manifest record {index} ID",
                    )
                )
        else:
            raise TypeError(
                "question manifest must contain only string IDs or only objects"
            )
    else:
        raise TypeError("question manifest root must be an object or array")

    seen: set[str] = set()
    for question_id in question_ids:
        if question_id in seen:
            raise ValueError(f"duplicate question ID in manifest: {question_id}")
        seen.add(question_id)
    return question_ids


def _fingerprint(kind: str, records: Sequence[dict[str, Any]]) -> str:
    """Hash an ordered raw payload using canonical UTF-8 JSON."""

    serialized = json.dumps(
        {
            "schema_version": "legal-qa-evaluation-v2",
            "kind": kind,
            "records": records,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _index_predictions(
    predictions: Sequence[LegalQAPrediction],
) -> dict[str, LegalQAPrediction]:
    """Index typed predictions while rejecting duplicate IDs."""

    indexed: dict[str, LegalQAPrediction] = {}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, LegalQAPrediction):
            raise TypeError(f"predictions[{index}] must be a LegalQAPrediction")
        validated = LegalQAPrediction.model_validate(
            prediction.model_dump(mode="python"),
            strict=True,
        )
        if validated.id in indexed:
            raise ValueError(f"duplicate question ID in predictions: {validated.id}")
        indexed[validated.id] = validated
    return indexed


def _validate_coverage(
    reference_ids: Sequence[str],
    predictions: dict[str, LegalQAPrediction],
) -> None:
    """Require exact question-ID coverage before calculating metrics."""

    expected = set(reference_ids)
    actual = set(predictions)
    missing = sorted(expected.difference(actual))
    unexpected = sorted(actual.difference(expected))
    if missing or unexpected:
        details: List[str] = []
        if missing:
            details.append("missing IDs: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected IDs: " + ", ".join(unexpected))
        raise ValueError(
            "prediction IDs do not match references (" + "; ".join(details) + ")"
        )


def evaluate_legal_qa(
    references: Sequence[LegalQAWarmupSample],
    predictions: Sequence[LegalQAPrediction],
) -> LegalQAEvaluationReport:
    """Score a complete prediction set with the organizer's metric behavior."""

    if not references:
        raise ValueError("LegalQA references must not be empty")
    typed_references: List[LegalQAWarmupSample] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, LegalQAWarmupSample):
            raise TypeError(f"references[{index}] must be a LegalQAWarmupSample")
        typed_references.append(
            LegalQAWarmupSample.model_validate(
                reference.model_dump(mode="python"), strict=True
            )
        )
    _reject_duplicate_sample_ids(typed_references, label="references")

    prediction_by_id = _index_predictions(predictions)
    reference_ids = [reference.id for reference in typed_references]
    _validate_coverage(reference_ids, prediction_by_id)

    rows: List[LegalQAQueryDiagnostic] = []
    aligned_predictions: List[LegalQAPrediction] = []
    for reference in typed_references:
        prediction = prediction_by_id[reference.id]
        aligned_predictions.append(prediction)
        rows.append(
            LegalQAQueryDiagnostic(
                id=reference.id,
                prediction_character_count=len(prediction.answer),
                reference_character_count=len(reference.answer),
                empty_prediction=not prediction.answer.strip(),
                meteor=LegalQAMeteorQueryMetrics(
                    score=official_meteor(reference.answer, prediction.answer)
                ),
                rouge_l=LegalQARougeLQueryMetrics(
                    f1=official_rouge_l(reference.answer, prediction.answer)
                ),
            )
        )

    count = len(rows)
    return LegalQAEvaluationReport(
        dataset_fingerprint=_fingerprint(
            "reference",
            [reference.model_dump(mode="json") for reference in typed_references],
        ),
        prediction_fingerprint=_fingerprint(
            "prediction",
            [prediction.model_dump(mode="json") for prediction in aligned_predictions],
        ),
        aggregate=LegalQAAggregate(
            sample_count=count,
            empty_prediction_count=sum(
                not prediction.answer.strip() for prediction in aligned_predictions
            ),
            meteor=sum(row.meteor.score for row in rows) / count,
            rouge_l=sum(row.rouge_l.f1 for row in rows) / count,
        ),
        per_query=tuple(rows),
    )


__all__ = [
    "LegalQAAggregate",
    "LegalQAEvaluationReport",
    "LegalQAMeteorQueryMetrics",
    "LegalQAMetricProfile",
    "LegalQAPrediction",
    "LegalQAPredictionSet",
    "LegalQAQueryDiagnostic",
    "LegalQARougeLQueryMetrics",
    "LegalQAWarmupDataset",
    "LegalQAWarmupRecord",
    "LegalQAWarmupSample",
    "evaluate_legal_qa",
    "load_legal_qa_predictions",
    "load_legal_qa_question_ids",
    "load_legal_qa_warmup",
]
