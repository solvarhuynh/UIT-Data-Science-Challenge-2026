"""Strict Task 2 LegalQA loading, alignment, and diagnostic reporting.

Raw questions, references, and predictions are preserved exactly at file
boundaries.  Text normalization is confined to the explicitly versioned local
metric profile in :mod:`udsc2026.evaluation.legal_qa_metrics`.  The organizer's
scoring implementation and parameters are not public, so reports always mark
official scorer parity as false.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
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

from udsc2026.evaluation.legal_qa_metrics import (
    METEOR_ALPHA,
    METEOR_BETA,
    METEOR_DIAGNOSTIC_PROFILE,
    METEOR_GAMMA,
    ROUGE_L_DIAGNOSTIC_PROFILE,
    TOKENIZATION_PROFILE,
    MeteorDiagnostic,
    RougeLDiagnostic,
    meteor_diagnostic,
    rouge_l_diagnostic,
)


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
    """Machine-readable disclosure of the local, non-official metric profile."""

    tokenization: Literal["legalqa-nfc-casefold-cf-strip-word-regex-v1"] = (
        "legalqa-nfc-casefold-cf-strip-word-regex-v1"
    )
    meteor: Literal["legalqa-meteor-exact-diagnostic-v1"] = (
        "legalqa-meteor-exact-diagnostic-v1"
    )
    rouge_l: Literal["legalqa-rouge-l-f1-diagnostic-v1"] = (
        "legalqa-rouge-l-f1-diagnostic-v1"
    )
    meteor_alpha: float = Field(default=METEOR_ALPHA, allow_inf_nan=False)
    meteor_beta: float = Field(default=METEOR_BETA, allow_inf_nan=False)
    meteor_gamma: float = Field(default=METEOR_GAMMA, allow_inf_nan=False)
    exact_token_matches_only: Literal[True] = True
    external_corpora_used: Literal[False] = False
    official_scorer_parity: Literal[False] = False

    @model_validator(mode="after")
    def validate_fixed_profile(self) -> "LegalQAMetricProfile":
        """Prevent a report label from being paired with different parameters."""

        if (
            self.tokenization != TOKENIZATION_PROFILE
            or self.meteor != METEOR_DIAGNOSTIC_PROFILE
            or self.rouge_l != ROUGE_L_DIAGNOSTIC_PROFILE
            or self.meteor_alpha != METEOR_ALPHA
            or self.meteor_beta != METEOR_BETA
            or self.meteor_gamma != METEOR_GAMMA
        ):
            raise ValueError("LegalQA metric profile parameters are fixed")
        return self


class LegalQAMeteorQueryMetrics(LegalQAModel):
    """Validated local METEOR contribution for one question."""

    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    precision: float = Field(ge=0, le=1, allow_inf_nan=False)
    recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    harmonic_mean: float = Field(ge=0, le=1, allow_inf_nan=False)
    fragmentation_penalty: float = Field(ge=0, le=1, allow_inf_nan=False)
    matches: int = Field(ge=0)
    chunks: int = Field(ge=0)
    prediction_token_count: int = Field(ge=0)
    reference_token_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_contribution(self) -> "LegalQAMeteorQueryMetrics":
        """Ensure declared components satisfy the versioned metric formula."""

        if self.matches > min(
            self.prediction_token_count,
            self.reference_token_count,
        ):
            raise ValueError("METEOR matches exceed available tokens")
        if self.matches == 0:
            if self.chunks != 0 or any(
                value != 0
                for value in (
                    self.score,
                    self.precision,
                    self.recall,
                    self.harmonic_mean,
                    self.fragmentation_penalty,
                )
            ):
                raise ValueError("zero-match METEOR diagnostics must be all zero")
            return self
        if not 1 <= self.chunks <= self.matches:
            raise ValueError("METEOR chunks must lie between one and matches")

        precision = self.matches / self.prediction_token_count
        recall = self.matches / self.reference_token_count
        denominator = METEOR_ALPHA * precision + (1 - METEOR_ALPHA) * recall
        harmonic_mean = precision * recall / denominator
        penalty = METEOR_GAMMA * math.pow(
            self.chunks / self.matches,
            METEOR_BETA,
        )
        score = harmonic_mean * (1 - penalty)
        expected = (
            ("precision", self.precision, precision),
            ("recall", self.recall, recall),
            ("harmonic_mean", self.harmonic_mean, harmonic_mean),
            ("fragmentation_penalty", self.fragmentation_penalty, penalty),
            ("score", self.score, score),
        )
        for label, actual, calculated in expected:
            if not math.isclose(actual, calculated, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"METEOR {label} does not match declared components")
        return self


class LegalQARougeLQueryMetrics(LegalQAModel):
    """Validated token-level ROUGE-L F1 contribution for one question."""

    f1: float = Field(ge=0, le=1, allow_inf_nan=False)
    precision: float = Field(ge=0, le=1, allow_inf_nan=False)
    recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    lcs_length: int = Field(ge=0)
    prediction_token_count: int = Field(ge=0)
    reference_token_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_contribution(self) -> "LegalQARougeLQueryMetrics":
        """Ensure precision, recall, and F1 agree with the declared LCS."""

        if self.lcs_length > min(
            self.prediction_token_count,
            self.reference_token_count,
        ):
            raise ValueError("ROUGE-L LCS exceeds available tokens")
        if self.prediction_token_count == 0 and self.reference_token_count == 0:
            expected_precision = expected_recall = expected_f1 = 1.0
        elif self.prediction_token_count == 0 or self.reference_token_count == 0:
            expected_precision = expected_recall = expected_f1 = 0.0
        else:
            expected_precision = self.lcs_length / self.prediction_token_count
            expected_recall = self.lcs_length / self.reference_token_count
            expected_f1 = (
                2
                * expected_precision
                * expected_recall
                / (expected_precision + expected_recall)
                if self.lcs_length
                else 0.0
            )
        expected = (
            ("precision", self.precision, expected_precision),
            ("recall", self.recall, expected_recall),
            ("f1", self.f1, expected_f1),
        )
        for label, actual, calculated in expected:
            if not math.isclose(actual, calculated, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"ROUGE-L {label} does not match declared LCS")
        return self


class LegalQAQueryDiagnostic(LegalQAModel):
    """Auditable per-query local metric contributions."""

    id: Identifier
    prediction_character_count: int = Field(ge=0)
    reference_character_count: int = Field(ge=1)
    meteor: LegalQAMeteorQueryMetrics
    rouge_l: LegalQARougeLQueryMetrics

    @model_validator(mode="after")
    def validate_shared_token_counts(self) -> "LegalQAQueryDiagnostic":
        """Require both metrics to operate on the same tokenized texts."""

        if (
            self.meteor.prediction_token_count != self.rouge_l.prediction_token_count
            or self.meteor.reference_token_count != self.rouge_l.reference_token_count
        ):
            raise ValueError("LegalQA metrics must share token counts")
        return self


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
    """Deterministic Task 2 report explicitly scoped to local diagnostics."""

    schema_version: Literal["legal-qa-evaluation-v1"] = "legal-qa-evaluation-v1"
    evaluation_scope: Literal["local_diagnostic"] = "local_diagnostic"
    official_scorer_parity: Literal[False] = False
    dataset_fingerprint: Fingerprint
    prediction_fingerprint: Fingerprint
    profile: LegalQAMetricProfile = Field(default_factory=LegalQAMetricProfile)
    aggregate: LegalQAAggregate
    per_query: Tuple[LegalQAQueryDiagnostic, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aggregate(self) -> "LegalQAEvaluationReport":
        """Reject forged aggregates or duplicate per-query diagnostics."""

        if self.aggregate.sample_count != len(self.per_query):
            raise ValueError("aggregate sample_count must match per_query")
        ids = [item.id for item in self.per_query]
        if len(ids) != len(set(ids)):
            raise ValueError("per_query IDs must be unique")
        empty_count = sum(
            item.meteor.prediction_token_count == 0 for item in self.per_query
        )
        if self.aggregate.empty_prediction_count != empty_count:
            raise ValueError("aggregate empty_prediction_count must match per_query")
        meteor = sum(item.meteor.score for item in self.per_query) / len(self.per_query)
        rouge_l = sum(item.rouge_l.f1 for item in self.per_query) / len(self.per_query)
        if not math.isclose(
            self.aggregate.meteor,
            meteor,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("aggregate METEOR must equal the per-query mean")
        if not math.isclose(
            self.aggregate.rouge_l,
            rouge_l,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
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
        if not isinstance(answer, str):
            raise TypeError(f"{location} answer must be a string")
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
    ``id -> {question, optional answer}``, an array of opaque string IDs, or an
    array of objects with ``id`` and optional ``question``/``answer`` fields.
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
            "schema_version": "legal-qa-evaluation-v1",
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


def _meteor_model(value: MeteorDiagnostic) -> LegalQAMeteorQueryMetrics:
    """Convert the immutable pure-metric result into a strict report model."""

    return LegalQAMeteorQueryMetrics(**asdict(value))


def _rouge_l_model(value: RougeLDiagnostic) -> LegalQARougeLQueryMetrics:
    """Convert the immutable pure-metric result into a strict report model."""

    return LegalQARougeLQueryMetrics(**asdict(value))


def evaluate_legal_qa(
    references: Sequence[LegalQAWarmupSample],
    predictions: Sequence[LegalQAPrediction],
) -> LegalQAEvaluationReport:
    """Evaluate a complete Task 2 run under the declared local profiles."""

    if not references:
        raise ValueError("LegalQA references must not be empty")
    typed_references: List[LegalQAWarmupSample] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, LegalQAWarmupSample):
            raise TypeError(f"references[{index}] must be a LegalQAWarmupSample")
        typed_references.append(
            LegalQAWarmupSample.model_validate(
                reference.model_dump(mode="python"),
                strict=True,
            )
        )
    _reject_duplicate_sample_ids(typed_references, label="references")

    prediction_by_id = _index_predictions(predictions)
    reference_ids = [reference.id for reference in typed_references]
    _validate_coverage(reference_ids, prediction_by_id)

    diagnostics: List[LegalQAQueryDiagnostic] = []
    aligned_predictions: List[LegalQAPrediction] = []
    for reference in typed_references:
        prediction = prediction_by_id[reference.id]
        aligned_predictions.append(prediction)
        meteor = meteor_diagnostic(prediction.answer, reference.answer)
        rouge_l = rouge_l_diagnostic(prediction.answer, reference.answer)
        if meteor.reference_token_count == 0:
            raise ValueError(
                f"reference answer has no metric tokens for ID {reference.id}"
            )
        diagnostics.append(
            LegalQAQueryDiagnostic(
                id=reference.id,
                prediction_character_count=len(prediction.answer),
                reference_character_count=len(reference.answer),
                meteor=_meteor_model(meteor),
                rouge_l=_rouge_l_model(rouge_l),
            )
        )

    sample_count = len(diagnostics)
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
            sample_count=sample_count,
            empty_prediction_count=sum(
                item.meteor.prediction_token_count == 0 for item in diagnostics
            ),
            meteor=sum(item.meteor.score for item in diagnostics) / sample_count,
            rouge_l=sum(item.rouge_l.f1 for item in diagnostics) / sample_count,
        ),
        per_query=tuple(diagnostics),
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
