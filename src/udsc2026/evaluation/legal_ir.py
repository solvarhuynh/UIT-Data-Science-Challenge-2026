"""Strict document-set evaluation contracts for DSC2026 LegalIR.

The official Task 1 metrics are macro Recall (primary) and macro Precision
(secondary).  Each question may have more than one relevant document, and the
metric treats both gold and predicted document IDs as sets.  Ranked retrieval
order is retained in prediction artifacts for submission compatibility, but it
    does not directly affect either official score.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Annotated, Any, List, Literal, NoReturn, Sequence

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    Strict,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from udsc2026.contracts.retrieval import RetrievalHit

OFFICIAL_MAX_DOCUMENTS = 5
_MATCHING_NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def _validate_identifier(value: str) -> str:
    """Reject blank or padded IDs without silently changing their identity."""

    if not value.strip():
        raise ValueError("identifier must not be blank")
    if value != value.strip():
        raise ValueError("identifier must not contain surrounding whitespace")
    if any(
        ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F for character in value
    ):
        raise ValueError("identifier must not contain control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("identifier must contain valid Unicode scalar values") from exc
    return value


Identifier = Annotated[str, Strict(), AfterValidator(_validate_identifier)]
Fingerprint = Annotated[
    str,
    Strict(),
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]


def normalize_legal_ir_query(value: str) -> str:
    """Return NFC text with leading, trailing, and repeated whitespace removed."""

    if not isinstance(value, str):
        raise TypeError("query must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("query must contain valid Unicode scalar values") from exc
    return " ".join(unicodedata.normalize("NFC", value).split())


def normalize_legal_ir_matching_question(value: str) -> str:
    """Normalize questions for supervised matching and grouped CV.

    This intentionally matches the LegalIR ensemble's exact-label/KNN
    normalization: NFKC, case-folding, punctuation collapse, then whitespace
    collapse.  It is separate from :func:`normalize_legal_ir_query`, whose
    narrower NFC normalization preserves retrieval-facing text.
    """

    if not isinstance(value, str):
        raise TypeError("question must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("question must contain valid Unicode scalar values") from exc
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_MATCHING_NON_WORD_RE.sub(" ", normalized).split())


def _validate_raw_question(value: str) -> str:
    """Validate raw question text while preserving every original code point."""

    if not normalize_legal_ir_query(value):
        raise ValueError("question must not be blank")
    return value


def _validate_normalized_question(value: str) -> str:
    """Require callers constructing normalized samples to use one canonical form."""

    normalized = normalize_legal_ir_query(value)
    if not normalized:
        raise ValueError("question must not be blank")
    if value != normalized:
        raise ValueError("question must be NFC-normalized with collapsed whitespace")
    return value


RawQuestion = Annotated[str, Strict(), AfterValidator(_validate_raw_question)]
NormalizedQuestion = Annotated[
    str,
    Strict(),
    AfterValidator(_validate_normalized_question),
]


def _reject_duplicate_values(values: Sequence[str], *, label: str) -> None:
    """Reject duplicate IDs instead of changing rank or relevance semantics."""

    if len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique IDs")


class LegalIRModel(BaseModel):
    """Strict base for evaluation-owned object schemas."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
    )


class WarmupRecord(LegalIRModel):
    """One value in the official warm-up root mapping."""

    question: RawQuestion
    answer: List[Identifier] = Field(min_length=1)

    @field_validator("answer")
    @classmethod
    def validate_unique_answers(cls, values: List[str]) -> List[str]:
        """Ensure one document cannot occur twice in an answer list."""

        _reject_duplicate_values(values, label="answer")
        return values


class WarmupDataset(RootModel[dict[Identifier, WarmupRecord]]):
    """Exact Task 1 root-mapping contract used by the Warm-up dataset."""

    model_config = ConfigDict(strict=True, validate_assignment=True)

    @model_validator(mode="after")
    def reject_empty_dataset(self) -> "WarmupDataset":
        """Require at least one question in an evaluation dataset."""

        if not self.root:
            raise ValueError("warm-up dataset must not be empty")
        return self


class WarmupSample(LegalIRModel):
    """Normalized internal warm-up sample that also preserves the source query."""

    id: Identifier
    raw_question: RawQuestion
    question: NormalizedQuestion
    gold_documents: List[Identifier] = Field(min_length=1)

    @field_validator("gold_documents")
    @classmethod
    def validate_unique_gold_documents(cls, values: List[str]) -> List[str]:
        """Reject ambiguous duplicate relevance labels."""

        _reject_duplicate_values(values, label="gold_documents")
        return values

    @model_validator(mode="after")
    def validate_question_pair(self) -> "WarmupSample":
        """Prevent normalized question text from drifting from its raw source."""

        if self.question != normalize_legal_ir_query(self.raw_question):
            raise ValueError("question must be the normalized form of raw_question")
        return self


class LegalIRPrediction(LegalIRModel):
    """Ranked document IDs returned for one competition question."""

    id: Identifier
    documents: List[Identifier] = Field(default_factory=list)

    @field_validator("documents")
    @classmethod
    def validate_unique_documents(cls, values: List[str]) -> List[str]:
        """Forbid duplicates because list position defines document rank."""

        _reject_duplicate_values(values, label="documents")
        return values


def legal_ir_prediction_from_hits(
    question_id: str,
    hits: Sequence[RetrievalHit],
    *,
    max_documents: int | None = None,
) -> LegalIRPrediction:
    """Collapse an ordered chunk ranking to unique document IDs.

    The first occurrence of a document is its best-ranked chunk, so retaining
    that occurrence preserves reranker order. Explicit hit ranks, when present,
    must agree with list position. The returned ranking can be empty for local
    evaluation and submission. Under the published metric, an empty prediction
    contributes zero Recall and zero Precision for that question.
    """

    if max_documents is not None and (
        isinstance(max_documents, bool)
        or not isinstance(max_documents, int)
        or max_documents <= 0
    ):
        raise ValueError("max_documents must be a positive integer or None")

    documents: List[str] = []
    seen_documents: set[str] = set()
    seen_chunks: set[str] = set()
    for index, hit in enumerate(hits):
        if not isinstance(hit, RetrievalHit):
            raise TypeError(f"hits[{index}] must be a RetrievalHit")
        _validate_identifier(hit.chunk_id)
        _validate_identifier(hit.doc_id)
        if hit.chunk_id in seen_chunks:
            raise ValueError(f"duplicate chunk ID in hits: {hit.chunk_id}")
        seen_chunks.add(hit.chunk_id)
        expected_rank = index + 1
        if hit.rank is not None and hit.rank != expected_rank:
            raise ValueError("explicit hit ranks must match one-based list positions")
        if hit.doc_id not in seen_documents:
            seen_documents.add(hit.doc_id)
            if max_documents is None or len(documents) < max_documents:
                documents.append(hit.doc_id)

    return LegalIRPrediction(id=question_id, documents=documents)


class LegalIRReference(LegalIRModel):
    """Official one-or-more-gold reference for one competition question.

    ``gold_document`` remains a validation-only alias for old programmatic
    callers.  Reports and serialized references always use the canonical
    plural ``gold_documents`` field.
    """

    id: Identifier
    gold_documents: List[Identifier] = Field(
        min_length=1,
        validation_alias=AliasChoices("gold_documents", "gold_document"),
    )

    @field_validator("gold_documents", mode="before")
    @classmethod
    def accept_legacy_single_gold(cls, value: object) -> object:
        """Canonicalize the former scalar field without weakening ID checks."""

        if isinstance(value, str):
            return [value]
        return value

    @field_validator("gold_documents")
    @classmethod
    def validate_unique_gold_documents(cls, values: List[str]) -> List[str]:
        """Reject duplicate relevance labels before applying set metrics."""

        _reject_duplicate_values(values, label="gold_documents")
        return values


def _validate_unique_sample_ids(
    samples: Sequence[LegalIRPrediction] | Sequence[LegalIRReference],
    *,
    label: str,
) -> None:
    """Validate uniqueness at collection boundaries."""

    seen: set[str] = set()
    for sample in samples:
        if sample.id in seen:
            raise ValueError(f"duplicate question ID in {label}: {sample.id}")
        seen.add(sample.id)


class LegalIRPredictionSet(RootModel[List[LegalIRPrediction]]):
    """Strict JSON-array contract for a complete prediction file."""

    model_config = ConfigDict(strict=True, validate_assignment=True)

    @model_validator(mode="after")
    def validate_collection(self) -> "LegalIRPredictionSet":
        """Reject empty or duplicate-ID prediction collections."""

        if not self.root:
            raise ValueError("prediction set must not be empty")
        _validate_unique_sample_ids(self.root, label="predictions")
        return self


class LegalIRReferenceSet(RootModel[List[LegalIRReference]]):
    """Strict JSON-array contract for official multi-gold references."""

    model_config = ConfigDict(strict=True, validate_assignment=True)

    @model_validator(mode="after")
    def validate_collection(self) -> "LegalIRReferenceSet":
        """Reject empty or duplicate-ID reference collections."""

        if not self.root:
            raise ValueError("reference set must not be empty")
        _validate_unique_sample_ids(self.root, label="references")
        return self


class LegalIRAggregate(LegalIRModel):
    """Competition-level macro metrics."""

    sample_count: int = Field(ge=1)
    multi_gold_sample_count: int = Field(default=0, ge=0)
    recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    precision: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_gold_count(self) -> "LegalIRAggregate":
        """Keep the warm-up data-quality counter within the sample count."""

        if self.multi_gold_sample_count > self.sample_count:
            raise ValueError("multi_gold_sample_count cannot exceed sample_count")
        return self


class LegalIRQueryDiagnostic(LegalIRModel):
    """Auditable set counts and score contributions for one question."""

    id: Identifier
    gold_documents: List[Identifier] = Field(min_length=1)
    predicted_documents: List[Identifier] = Field(default_factory=list)
    matched_documents: List[Identifier] = Field(default_factory=list)
    missed_gold_documents: List[Identifier] = Field(default_factory=list)
    false_positive_documents: List[Identifier] = Field(default_factory=list)
    relevant_count: int = Field(ge=1)
    predicted_count: int = Field(ge=0)
    relevant_retrieved_count: int = Field(ge=0)
    recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    precision: float = Field(ge=0, le=1, allow_inf_nan=False)

    @field_validator(
        "gold_documents",
        "predicted_documents",
        "matched_documents",
        "missed_gold_documents",
        "false_positive_documents",
    )
    @classmethod
    def validate_unique_document_lists(cls, values: List[str]) -> List[str]:
        """Keep diagnostic set members unambiguous."""

        _reject_duplicate_values(values, label="document list")
        return values

    @model_validator(mode="after")
    def validate_set_contribution(self) -> "LegalIRQueryDiagnostic":
        """Ensure declared set counts and metric contributions agree."""

        if self.relevant_count != len(self.gold_documents):
            raise ValueError("relevant_count must equal len(gold_documents)")
        gold = set(self.gold_documents)
        predicted = set(self.predicted_documents)
        expected_matched = [
            document for document in self.predicted_documents if document in gold
        ]
        expected_missed = [
            document for document in self.gold_documents if document not in predicted
        ]
        expected_false_positives = [
            document for document in self.predicted_documents if document not in gold
        ]
        if self.matched_documents != expected_matched:
            raise ValueError(
                "matched_documents must be predicted gold documents in prediction order"
            )
        if self.missed_gold_documents != expected_missed:
            raise ValueError(
                "missed_gold_documents must be unretrieved gold documents in gold order"
            )
        if self.false_positive_documents != expected_false_positives:
            raise ValueError(
                "false_positive_documents must be non-gold predictions "
                "in prediction order"
            )
        if self.relevant_retrieved_count != len(self.matched_documents):
            raise ValueError(
                "relevant_retrieved_count must equal len(matched_documents)"
            )
        if self.predicted_count != len(self.predicted_documents):
            raise ValueError("predicted_count must equal len(predicted_documents)")

        expected_recall = self.relevant_retrieved_count / self.relevant_count
        expected_precision = (
            self.relevant_retrieved_count / self.predicted_count
            if self.predicted_count
            else 0.0
        )
        if not math.isclose(
            self.recall,
            expected_recall,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "recall must equal relevant_retrieved_count/relevant_count"
            )
        if not math.isclose(
            self.precision,
            expected_precision,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "precision must equal relevant_retrieved_count/predicted_count"
            )
        return self


EvaluationMode = Literal["official_set"]


class LegalIREvaluationReport(LegalIRModel):
    """Deterministic aggregate report plus per-question diagnostics."""

    schema_version: Literal["legal-ir-evaluation-v2"] = "legal-ir-evaluation-v2"
    evaluation_mode: EvaluationMode
    metric_priority: tuple[Literal["recall"], Literal["precision"]] = (
        "recall",
        "precision",
    )
    dataset_fingerprint: Fingerprint
    aggregate: LegalIRAggregate
    per_query: List[LegalIRQueryDiagnostic] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_report_totals(self) -> "LegalIREvaluationReport":
        """Reject reports whose declared aggregate was not derived from diagnostics."""

        if self.aggregate.sample_count != len(self.per_query):
            raise ValueError("aggregate sample_count must match per_query")
        ids = [item.id for item in self.per_query]
        _reject_duplicate_values(ids, label="per_query")
        expected_multi_gold_count = sum(
            len(item.gold_documents) > 1 for item in self.per_query
        )
        if self.aggregate.multi_gold_sample_count != expected_multi_gold_count:
            raise ValueError("aggregate multi_gold_sample_count must match per_query")
        expected_recall = sum(item.recall for item in self.per_query) / len(
            self.per_query
        )
        expected_precision = sum(item.precision for item in self.per_query) / len(
            self.per_query
        )
        if not math.isclose(
            self.aggregate.recall,
            expected_recall,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("aggregate Recall must equal the per-query mean")
        if not math.isclose(
            self.aggregate.precision,
            expected_precision,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("aggregate Precision must equal the per-query mean")
        return self


def _reject_non_standard_json_constant(value: str) -> NoReturn:
    """Reject JSON extensions such as NaN and Infinity."""

    raise ValueError(f"non-standard JSON constant {value!r} is not permitted")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object while detecting duplicate keys hidden by normal decoders."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_strict_json(path: str | Path) -> tuple[Path, object]:
    """Read one UTF-8 JSON document without accepting duplicate keys or NaN."""

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {resolved}")
    try:
        encoded = resolved.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"JSON file must be UTF-8: {resolved}") from exc
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_standard_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {resolved}: {exc.msg}") from exc
    except ValueError as exc:
        raise ValueError(f"invalid JSON in {resolved}: {exc}") from exc
    return resolved, payload


def load_warmup(path: str | Path) -> List[WarmupSample]:
    """Load, strictly validate, and normalize a DSC2026 warm-up JSON file.

    Question IDs and document IDs are never coerced.  ``raw_question`` retains
    the exact source string, while ``question`` is suitable for retrieval.
    """

    resolved, payload = _read_strict_json(path)

    try:
        dataset = WarmupDataset.model_validate(payload, strict=True)
    except ValidationError as exc:
        raise ValueError(f"invalid warm-up schema in {resolved}: {exc}") from exc

    return [
        WarmupSample(
            id=question_id,
            raw_question=record.question,
            question=normalize_legal_ir_query(record.question),
            gold_documents=list(record.answer),
        )
        for question_id, record in dataset.root.items()
    ]


def _validate_question_manifest_record(
    record: object,
    *,
    allowed_fields: set[str],
    require_question: bool,
    location: str,
) -> dict[str, Any]:
    """Validate one phase-question record while deriving coverage only."""

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
        if not isinstance(question, str):
            raise TypeError(f"{location} question must be a string")
        _validate_raw_question(question)
    if "answer" in record:
        answer = record["answer"]
        if not isinstance(answer, list):
            raise TypeError(f"{location} answer must be an array of document IDs")
        if not answer:
            raise ValueError(f"{location} answer must not be empty")
        validated_answers: list[str] = []
        for index, document_id in enumerate(answer):
            if not isinstance(document_id, str):
                raise TypeError(
                    f"{location} answer[{index}] must be a string document ID"
                )
            validated_answers.append(_validate_identifier(document_id))
        _reject_duplicate_values(validated_answers, label=f"{location} answer")
    return record


def _validate_manifest_identifier(value: object, *, location: str) -> str:
    """Validate one opaque question ID without coercion or normalization."""

    if not isinstance(value, str):
        raise TypeError(f"{location} must be a string")
    try:
        return _validate_identifier(value)
    except ValueError as exc:
        raise ValueError(f"invalid {location}: {exc}") from exc


def load_legal_ir_question_ids(path: str | Path) -> List[str]:
    """Load exact coverage from Warm-up, phase input, or an ID manifest.

    Accepted roots are the organizer mapping ``id -> {question, answer?}``, an
    array of opaque string IDs, or an array of objects containing ``id`` and
    optional ``question``/``answer`` metadata.  Gold answers, when present,
    are validated but never returned or converted into predictions.
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


def _validated_ranked_documents(documents: Sequence[str]) -> List[str]:
    """Strictly validate a raw ranking supplied to a metric function."""

    if isinstance(documents, (str, bytes)):
        raise TypeError("documents must be a sequence of string IDs")
    validated: List[str] = []
    for document in documents:
        if not isinstance(document, str):
            raise TypeError("documents must contain only string IDs")
        validated.append(_validate_identifier(document))
    _reject_duplicate_values(validated, label="documents")
    return validated


def _validated_gold_documents(documents: Sequence[str]) -> List[str]:
    """Strictly validate a non-empty collection of gold document IDs."""

    validated = _validated_ranked_documents(documents)
    if not validated:
        raise ValueError("gold_documents must not be empty")
    return validated


def _first_relevant_rank(
    ranked_documents: Sequence[str],
    gold_documents: Sequence[str],
) -> tuple[int | None, str | None]:
    """Return the one-based rank and ID of the first relevant document."""

    gold = set(gold_documents)
    for rank, document in enumerate(ranked_documents, start=1):
        if document in gold:
            return rank, document
    return None, None


def _legal_ir_set_counts(
    documents: Sequence[str],
    gold_documents: Sequence[str],
) -> tuple[int, int, int]:
    """Return validated relevant, predicted, and intersection cardinalities."""

    prediction = _validated_ranked_documents(documents)
    gold = _validated_gold_documents(gold_documents)
    return len(gold), len(prediction), len(set(gold).intersection(prediction))


def _official_scored_documents(documents: Sequence[str]) -> List[str]:
    """Return the ranking eligible for the BTC scorer's per-query metric.

    The organizer scorer assigns zero Recall and Precision when an answer is
    empty or contains more than five documents.  Internal retrieval rankings
    remain unconstrained; this guard is applied only by official evaluation.
    """

    validated = _validated_ranked_documents(documents)
    if 1 <= len(validated) <= OFFICIAL_MAX_DOCUMENTS:
        return validated
    return []


def legal_ir_recall(
    documents: Sequence[str],
    gold_documents: Sequence[str],
) -> float:
    """Compute official per-query set Recall."""

    relevant_count, _, relevant_retrieved_count = _legal_ir_set_counts(
        _official_scored_documents(documents),
        gold_documents,
    )
    return relevant_retrieved_count / relevant_count


def legal_ir_precision(
    documents: Sequence[str],
    gold_documents: Sequence[str],
) -> float:
    """Compute official per-query set Precision, returning zero for no predictions."""

    _, predicted_count, relevant_retrieved_count = _legal_ir_set_counts(
        _official_scored_documents(documents),
        gold_documents,
    )
    return relevant_retrieved_count / predicted_count if predicted_count > 0 else 0.0


def legal_ir_reciprocal_rank(
    documents: Sequence[str],
    gold_document: str,
) -> float:
    """Compute the retired single-gold rank diagnostic for legacy callers."""

    ranking = _validated_ranked_documents(documents)
    if not isinstance(gold_document, str):
        raise TypeError("gold_document must be a string ID")
    gold = _validate_identifier(gold_document)
    rank, _ = _first_relevant_rank(ranking, [gold])
    return 1.0 / rank if rank is not None else 0.0


def legal_ir_recall_at_3(
    documents: Sequence[str],
    gold_document: str,
) -> float:
    """Return the retired single-gold Top-3 diagnostic for legacy callers."""

    ranking = _validated_ranked_documents(documents)
    if not isinstance(gold_document, str):
        raise TypeError("gold_document must be a string ID")
    gold = _validate_identifier(gold_document)
    return float(gold in ranking[:3])


def warmup_any_gold_reciprocal_rank(
    documents: Sequence[str],
    gold_documents: Sequence[str],
) -> float:
    """Compute the retired any-gold reciprocal-rank diagnostic."""

    ranking = _validated_ranked_documents(documents)
    gold = _validated_gold_documents(gold_documents)
    rank, _ = _first_relevant_rank(ranking, gold)
    return 1.0 / rank if rank is not None else 0.0


def warmup_any_gold_recall_at_3(
    documents: Sequence[str],
    gold_documents: Sequence[str],
) -> float:
    """Compute the retired any-gold Top-3 hit diagnostic."""

    ranking = _validated_ranked_documents(documents)
    gold = set(_validated_gold_documents(gold_documents))
    return float(bool(gold.intersection(ranking[:3])))


def _index_predictions(
    predictions: Sequence[LegalIRPrediction],
) -> dict[str, LegalIRPrediction]:
    """Index typed predictions while retaining clear duplicate diagnostics."""

    result: dict[str, LegalIRPrediction] = {}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, LegalIRPrediction):
            raise TypeError(f"predictions[{index}] must be a LegalIRPrediction")
        # Lists inside a Pydantic model remain mutable even when assignment is
        # validated. Recheck at the evaluation boundary so a caller cannot
        # append a duplicate/invalid ID after constructing the model.
        _validated_ranked_documents(prediction.documents)
        if prediction.id in result:
            raise ValueError(f"duplicate question ID in predictions: {prediction.id}")
        result[prediction.id] = prediction
    return result


def _validate_prediction_coverage(
    reference_ids: Sequence[str],
    prediction_by_id: dict[str, LegalIRPrediction],
) -> None:
    """Require exactly one prediction for every reference question."""

    expected = set(reference_ids)
    actual = set(prediction_by_id)
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


def _fingerprint(mode: EvaluationMode, references: Sequence[dict[str, Any]]) -> str:
    """Hash the exact ordered reference payload using canonical UTF-8 JSON."""

    serialized = json.dumps(
        {
            "schema_version": "legal-ir-evaluation-v2",
            "evaluation_mode": mode,
            "references": references,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _build_diagnostic(
    question_id: str,
    gold_documents: Sequence[str],
    prediction: LegalIRPrediction,
) -> LegalIRQueryDiagnostic:
    """Calculate one auditable set diagnostic from an aligned prediction."""

    validated_gold = _validated_gold_documents(gold_documents)
    # Do not truncate an invalid official answer.  BTC scores all-zero for
    # empty and >5-document answers; top-five truncation belongs solely to the
    # final-submission builder, never to the scorer-parity evaluator.
    validated_prediction = _official_scored_documents(prediction.documents)
    gold_set = set(validated_gold)
    matched = [document for document in validated_prediction if document in gold_set]
    predicted_set = set(validated_prediction)
    missed_gold = [
        document for document in validated_gold if document not in predicted_set
    ]
    false_positives = [
        document for document in validated_prediction if document not in gold_set
    ]
    relevant_count = len(validated_gold)
    predicted_count = len(validated_prediction)
    relevant_retrieved_count = len(matched)
    return LegalIRQueryDiagnostic(
        id=question_id,
        gold_documents=validated_gold,
        predicted_documents=validated_prediction,
        matched_documents=matched,
        missed_gold_documents=missed_gold,
        false_positive_documents=false_positives,
        relevant_count=relevant_count,
        predicted_count=predicted_count,
        relevant_retrieved_count=relevant_retrieved_count,
        recall=relevant_retrieved_count / relevant_count,
        precision=(
            relevant_retrieved_count / predicted_count if predicted_count else 0.0
        ),
    )


def _build_report(
    *,
    mode: EvaluationMode,
    reference_payload: Sequence[dict[str, Any]],
    diagnostics: List[LegalIRQueryDiagnostic],
) -> LegalIREvaluationReport:
    """Aggregate diagnostics and attach a stable reference fingerprint."""

    count = len(diagnostics)
    return LegalIREvaluationReport(
        evaluation_mode=mode,
        dataset_fingerprint=_fingerprint(mode, reference_payload),
        aggregate=LegalIRAggregate(
            sample_count=count,
            multi_gold_sample_count=sum(
                len(item.gold_documents) > 1 for item in diagnostics
            ),
            recall=sum(item.recall for item in diagnostics) / count,
            precision=sum(item.precision for item in diagnostics) / count,
        ),
        per_query=diagnostics,
    )


def evaluate_legal_ir(
    references: Sequence[LegalIRReference],
    predictions: Sequence[LegalIRPrediction],
) -> LegalIREvaluationReport:
    """Evaluate predictions using official macro set Recall and Precision."""

    if not references:
        raise ValueError("references must not be empty")
    typed_references: List[LegalIRReference] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, LegalIRReference):
            raise TypeError(f"references[{index}] must be a LegalIRReference")
        _validated_gold_documents(reference.gold_documents)
        typed_references.append(reference)
    _validate_unique_sample_ids(typed_references, label="references")

    prediction_by_id = _index_predictions(predictions)
    reference_ids = [reference.id for reference in typed_references]
    _validate_prediction_coverage(reference_ids, prediction_by_id)
    diagnostics = [
        _build_diagnostic(
            reference.id,
            reference.gold_documents,
            prediction_by_id[reference.id],
        )
        for reference in typed_references
    ]
    return _build_report(
        mode="official_set",
        reference_payload=[
            reference.model_dump(mode="json") for reference in typed_references
        ],
        diagnostics=diagnostics,
    )


def evaluate_warmup(
    samples: Sequence[WarmupSample],
    predictions: Sequence[LegalIRPrediction],
) -> LegalIREvaluationReport:
    """Evaluate a typed Warm-up dataset with official set metrics."""

    if not samples:
        raise ValueError("samples must not be empty")
    typed_samples: List[WarmupSample] = []
    seen: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, WarmupSample):
            raise TypeError(f"samples[{index}] must be a WarmupSample")
        if sample.id in seen:
            raise ValueError(f"duplicate question ID in warm-up samples: {sample.id}")
        _validated_gold_documents(sample.gold_documents)
        seen.add(sample.id)
        typed_samples.append(sample)

    prediction_by_id = _index_predictions(predictions)
    sample_ids = [sample.id for sample in typed_samples]
    _validate_prediction_coverage(sample_ids, prediction_by_id)
    diagnostics = [
        _build_diagnostic(
            sample.id,
            sample.gold_documents,
            prediction_by_id[sample.id],
        )
        for sample in typed_samples
    ]
    return _build_report(
        mode="official_set",
        reference_payload=[sample.model_dump(mode="json") for sample in typed_samples],
        diagnostics=diagnostics,
    )


def evaluate_warmup_any_gold(
    samples: Sequence[WarmupSample],
    predictions: Sequence[LegalIRPrediction],
) -> LegalIREvaluationReport:
    """Call :func:`evaluate_warmup` for backward import compatibility."""

    return evaluate_warmup(samples, predictions)


__all__ = [
    "LegalIRAggregate",
    "LegalIREvaluationReport",
    "LegalIRPrediction",
    "LegalIRPredictionSet",
    "LegalIRQueryDiagnostic",
    "LegalIRReference",
    "LegalIRReferenceSet",
    "WarmupDataset",
    "WarmupSample",
    "evaluate_legal_ir",
    "evaluate_warmup",
    "evaluate_warmup_any_gold",
    "OFFICIAL_MAX_DOCUMENTS",
    "legal_ir_precision",
    "legal_ir_prediction_from_hits",
    "legal_ir_recall",
    "legal_ir_recall_at_3",
    "legal_ir_reciprocal_rank",
    "load_legal_ir_question_ids",
    "load_warmup",
    "normalize_legal_ir_query",
    "normalize_legal_ir_matching_question",
    "warmup_any_gold_recall_at_3",
    "warmup_any_gold_reciprocal_rank",
]
