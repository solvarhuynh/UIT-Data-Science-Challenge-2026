"""Pydantic contracts for benchmark inputs and deterministic reports."""

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from udsc2026.contracts.retrieval import RetrievalHit

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256Fingerprint = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
Difficulty = Literal["easy", "medium", "hard"]


class EvaluationModel(BaseModel):
    """Strict base contract shared only by evaluation-owned schemas."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BenchmarkSample(EvaluationModel):
    """One reference question supplied by TV4 or adapted from an official test set."""

    question_id: NonEmptyString
    question: NonEmptyString
    answer: NonEmptyString
    gold_chunk_ids: List[NonEmptyString] = Field(min_length=1)
    gold_citations: List[NonEmptyString] = Field(default_factory=list)
    law_name: Optional[NonEmptyString] = None
    article: Optional[NonEmptyString] = None
    difficulty: Difficulty
    question_type: Optional[NonEmptyString] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("gold_chunk_ids", "gold_citations")
    @classmethod
    def reject_duplicate_values(cls, values: List[str]) -> List[str]:
        """Reject ambiguous gold labels instead of silently changing the dataset."""

        if len(values) != len(set(values)):
            raise ValueError("values must be unique")
        return values


class PredictionSample(EvaluationModel):
    """Store retrieved candidates and optional QA output for one question."""

    question_id: NonEmptyString
    hits: List[RetrievalHit] = Field(default_factory=list)
    answer: Optional[str] = None
    latency_ms: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("answer")
    @classmethod
    def normalize_optional_answer(cls, value: Optional[str]) -> Optional[str]:
        """Trim an optional answer while preserving an intentional empty value."""

        return value.strip() if value is not None else None

    @field_validator("hits")
    @classmethod
    def validate_hit_ids(cls, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        """Reject ambiguous IDs or ranks at the evaluation boundary."""

        chunk_ids = [hit.chunk_id.strip() for hit in hits]
        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError("hit chunk_id values must not be blank")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("hit chunk_id values must be unique")
        for expected_rank, hit in enumerate(hits, start=1):
            if hit.rank is not None and hit.rank != expected_rank:
                raise ValueError(
                    "explicit hit ranks must match their one-based list positions"
                )
        return hits


class RetrievalMetrics(EvaluationModel):
    """Aggregate ranking quality for a retrieval run."""

    mrr: float = Field(ge=0, le=1, allow_inf_nan=False)
    recall_at_k: Dict[int, float]

    @field_validator("recall_at_k")
    @classmethod
    def validate_recall_values(cls, values: Dict[int, float]) -> Dict[int, float]:
        """Require positive cutoffs and finite recall values in the unit interval."""

        if not values:
            raise ValueError("recall_at_k must not be empty")
        for cutoff, score in values.items():
            if isinstance(cutoff, bool) or cutoff <= 0:
                raise ValueError("recall cutoffs must be positive integers")
            if not 0 <= score <= 1:
                raise ValueError("recall values must be between 0 and 1")
        return dict(sorted(values.items()))


class QAMetrics(EvaluationModel):
    """Aggregate answer quality for predictions that contain generated answers."""

    sample_count: int = Field(ge=1)
    rouge_l: float = Field(ge=0, le=1, allow_inf_nan=False)


class LatencyStats(EvaluationModel):
    """Latency distribution in milliseconds for a complete evaluation run."""

    count: int = Field(ge=1)
    total_ms: float = Field(ge=0, allow_inf_nan=False)
    min_ms: float = Field(ge=0, allow_inf_nan=False)
    max_ms: float = Field(ge=0, allow_inf_nan=False)
    mean_ms: float = Field(ge=0, allow_inf_nan=False)
    median_ms: float = Field(ge=0, allow_inf_nan=False)
    p95_ms: float = Field(ge=0, allow_inf_nan=False)
    p99_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_distribution(self) -> "LatencyStats":
        """Catch internally inconsistent reports loaded from external JSON."""

        if self.min_ms > self.max_ms:
            raise ValueError("min_ms cannot exceed max_ms")
        bounded = (self.mean_ms, self.median_ms, self.p95_ms, self.p99_ms)
        if any(value < self.min_ms or value > self.max_ms for value in bounded):
            raise ValueError("latency aggregates must lie between min_ms and max_ms")
        return self


class EvaluationReport(EvaluationModel):
    """Complete deterministic report for one ordered set of predictions."""

    report_type: Literal["evaluation"] = "evaluation"
    name: NonEmptyString = "evaluation"
    sample_count: int = Field(ge=1)
    dataset_fingerprint: Sha256Fingerprint
    k_values: List[int] = Field(min_length=1)
    retrieval: RetrievalMetrics
    qa: Optional[QAMetrics] = None
    latency: Optional[LatencyStats] = None

    @field_validator("k_values")
    @classmethod
    def normalize_k_values(cls, values: List[int]) -> List[int]:
        """Reject invalid cutoffs and store one canonical sorted representation."""

        if any(isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("k_values must contain positive integers")
        if len(values) != len(set(values)):
            raise ValueError("k_values must be unique")
        return sorted(values)

    @model_validator(mode="after")
    def match_recall_cutoffs(self) -> "EvaluationReport":
        """Ensure report metadata and metric keys cannot disagree."""

        if self.k_values != sorted(self.retrieval.recall_at_k):
            raise ValueError("k_values must match recall_at_k cutoffs")
        if self.qa is not None and self.qa.sample_count != self.sample_count:
            raise ValueError("QA metrics must cover every evaluated sample")
        if self.latency is not None and self.latency.count != self.sample_count:
            raise ValueError("latency metrics must cover every evaluated sample")
        return self


class EvaluationDelta(EvaluationModel):
    """Signed metric changes calculated as after minus before."""

    mrr: float = Field(ge=-1, le=1, allow_inf_nan=False)
    recall_at_k: Dict[int, float]
    rouge_l: Optional[float] = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    mean_latency_ms: Optional[float] = Field(default=None, allow_inf_nan=False)
    p95_latency_ms: Optional[float] = Field(default=None, allow_inf_nan=False)

    @field_validator("recall_at_k")
    @classmethod
    def validate_recall_deltas(cls, values: Dict[int, float]) -> Dict[int, float]:
        """Validate and canonicalize per-cutoff recall deltas."""

        if not values:
            raise ValueError("recall_at_k deltas must not be empty")
        if any(not -1 <= value <= 1 for value in values.values()):
            raise ValueError("recall deltas must be between -1 and 1")
        return dict(sorted(values.items()))

    @classmethod
    def between(
        cls,
        before: EvaluationReport,
        after: EvaluationReport,
    ) -> "EvaluationDelta":
        """Calculate signed after-minus-before deltas for comparable reports."""

        return cls(
            mrr=after.retrieval.mrr - before.retrieval.mrr,
            recall_at_k={
                cutoff: (
                    after.retrieval.recall_at_k[cutoff]
                    - before.retrieval.recall_at_k[cutoff]
                )
                for cutoff in before.k_values
            },
            rouge_l=(
                (after.qa.rouge_l - before.qa.rouge_l)
                if after.qa is not None and before.qa is not None
                else None
            ),
            mean_latency_ms=(
                (after.latency.mean_ms - before.latency.mean_ms)
                if after.latency is not None and before.latency is not None
                else None
            ),
            p95_latency_ms=(
                (after.latency.p95_ms - before.latency.p95_ms)
                if after.latency is not None and before.latency is not None
                else None
            ),
        )


class EvaluationComparison(EvaluationModel):
    """Before/after evaluation pair plus signed deltas for regression checks."""

    report_type: Literal["comparison"] = "comparison"
    before: EvaluationReport
    after: EvaluationReport
    delta: EvaluationDelta

    @model_validator(mode="after")
    def validate_comparable_reports(self) -> "EvaluationComparison":
        """Disallow comparisons over different samples or retrieval cutoffs."""

        if self.before.sample_count != self.after.sample_count:
            raise ValueError("before and after reports must have equal sample counts")
        if self.before.dataset_fingerprint != self.after.dataset_fingerprint:
            raise ValueError(
                "before and after reports must use the same benchmark fingerprint"
            )
        if self.before.k_values != self.after.k_values:
            raise ValueError("before and after reports must use equal k_values")
        if (self.before.qa is None) != (self.after.qa is None):
            raise ValueError(
                "before and after reports must either contain or omit QA metrics"
            )
        if (self.before.latency is None) != (self.after.latency is None):
            raise ValueError(
                "before and after reports must either contain or omit latency metrics"
            )
        if sorted(self.delta.recall_at_k) != self.before.k_values:
            raise ValueError("delta recall cutoffs must match compared reports")
        if self.delta != EvaluationDelta.between(self.before, self.after):
            raise ValueError("delta must equal after minus before for every metric")
        return self
