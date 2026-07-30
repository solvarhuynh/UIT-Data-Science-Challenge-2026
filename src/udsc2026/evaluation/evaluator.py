"""Evaluation orchestration over shared retrieval contracts."""

import hashlib
import json
from typing import List, Optional, Sequence, TypeVar

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.metrics import (
    aggregate_latencies,
    mean_recall_at_k,
    mean_reciprocal_rank,
    mean_rouge_l,
)
from udsc2026.evaluation.models import (
    BenchmarkSample,
    EvaluationComparison,
    EvaluationDelta,
    EvaluationReport,
    PredictionSample,
    QAMetrics,
    RetrievalMetrics,
)

SampleT = TypeVar("SampleT", BenchmarkSample, PredictionSample)


def _fingerprint(payload: object) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible inputs."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _normalize_k_values(k_values: Sequence[int]) -> List[int]:
    """Return canonical cutoffs after strict validation."""

    if not k_values:
        raise ValueError("k_values must not be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in k_values
    ):
        raise ValueError("k_values must contain positive integers")
    if len(k_values) != len(set(k_values)):
        raise ValueError("k_values must be unique")
    return sorted(k_values)


def _validate_optional_metric_input(
    values: Optional[Sequence[object]],
    *,
    sample_count: int,
    name: str,
) -> None:
    """Require an optional metric to cover either all samples or none."""

    if values is not None and len(values) != sample_count:
        raise ValueError(f"{name} must contain exactly {sample_count} values")


def evaluate_retrieval(
    predictions: List[List[RetrievalHit]],
    gold_chunk_ids: List[List[str]],
    k_values: List[int],
    *,
    predicted_answers: Optional[List[str]] = None,
    reference_answers: Optional[List[str]] = None,
    latencies_ms: Optional[List[float]] = None,
    name: str = "evaluation",
    dataset_fingerprint: Optional[str] = None,
) -> EvaluationReport:
    """Evaluate ranked hits and optional answer/latency outputs.

    The first three arguments intentionally match the TV5 team contract. Extra
    metrics are keyword-only so retrieval callers do not need QA or a live model.
    """

    if not predictions:
        raise ValueError("at least one prediction sample is required")
    sample_count = len(predictions)
    if len(gold_chunk_ids) != sample_count:
        raise ValueError("predictions and gold_chunk_ids must have equal length")
    cutoffs = _normalize_k_values(k_values)
    for sample_index, hits in enumerate(predictions):
        if not isinstance(hits, list):
            raise TypeError(f"predictions[{sample_index}] must be a list")
        chunk_ids: list[str] = []
        for hit_index, hit in enumerate(hits):
            if not isinstance(hit, RetrievalHit):
                raise TypeError(
                    f"predictions[{sample_index}][{hit_index}] must be a RetrievalHit"
                )
            chunk_ids.append(hit.chunk_id)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                f"predictions[{sample_index}] contains duplicate chunk_id values"
            )
    predicted_ids = [[hit.chunk_id for hit in hits] for hits in predictions]

    qa_inputs_present = predicted_answers is not None or reference_answers is not None
    if qa_inputs_present and (predicted_answers is None or reference_answers is None):
        raise ValueError(
            "predicted_answers and reference_answers must be provided together"
        )
    _validate_optional_metric_input(
        predicted_answers, sample_count=sample_count, name="predicted_answers"
    )
    _validate_optional_metric_input(
        reference_answers, sample_count=sample_count, name="reference_answers"
    )
    _validate_optional_metric_input(
        latencies_ms, sample_count=sample_count, name="latencies_ms"
    )

    qa_metrics = None
    if predicted_answers is not None and reference_answers is not None:
        qa_metrics = QAMetrics(
            sample_count=sample_count,
            rouge_l=mean_rouge_l(predicted_answers, reference_answers),
        )

    return EvaluationReport(
        name=name,
        sample_count=sample_count,
        dataset_fingerprint=(
            dataset_fingerprint
            if dataset_fingerprint is not None
            else _fingerprint({"gold_chunk_ids": gold_chunk_ids})
        ),
        k_values=cutoffs,
        retrieval=RetrievalMetrics(
            mrr=mean_reciprocal_rank(predicted_ids, gold_chunk_ids),
            recall_at_k={
                cutoff: mean_recall_at_k(predicted_ids, gold_chunk_ids, cutoff)
                for cutoff in cutoffs
            },
        ),
        qa=qa_metrics,
        latency=aggregate_latencies(latencies_ms) if latencies_ms is not None else None,
    )


def evaluate_predictions(
    benchmark: Sequence[BenchmarkSample],
    predictions: Sequence[PredictionSample],
    k_values: Sequence[int],
    *,
    name: str = "evaluation",
) -> EvaluationReport:
    """Align predictions by question ID and evaluate a complete benchmark run."""

    if not benchmark:
        raise ValueError("benchmark must not be empty")
    benchmark_by_id = _index_unique(benchmark, "benchmark")
    prediction_by_id = _index_unique(predictions, "predictions")
    missing = sorted(set(benchmark_by_id).difference(prediction_by_id))
    extra = sorted(set(prediction_by_id).difference(benchmark_by_id))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing IDs: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected IDs: {', '.join(extra)}")
        raise ValueError(
            "prediction IDs do not match benchmark (" + "; ".join(details) + ")"
        )

    ordered_predictions = [prediction_by_id[sample.question_id] for sample in benchmark]
    answers = [prediction.answer for prediction in ordered_predictions]
    if any(answer is None for answer in answers) and not all(
        answer is None for answer in answers
    ):
        raise ValueError("answers must be supplied for every prediction or none")
    latencies = [prediction.latency_ms for prediction in ordered_predictions]
    if any(latency is None for latency in latencies) and not all(
        latency is None for latency in latencies
    ):
        raise ValueError("latency_ms must be supplied for every prediction or none")

    return evaluate_retrieval(
        predictions=[prediction.hits for prediction in ordered_predictions],
        gold_chunk_ids=[sample.gold_chunk_ids for sample in benchmark],
        k_values=list(k_values),
        predicted_answers=(
            [answer for answer in answers if answer is not None]
            if answers and answers[0] is not None
            else None
        ),
        reference_answers=(
            [sample.answer for sample in benchmark]
            if answers and answers[0] is not None
            else None
        ),
        latencies_ms=(
            [latency for latency in latencies if latency is not None]
            if latencies and latencies[0] is not None
            else None
        ),
        name=name,
        dataset_fingerprint=_fingerprint(
            [sample.model_dump(mode="json") for sample in benchmark]
        ),
    )


def _index_unique(
    samples: Sequence[SampleT],
    label: str,
) -> dict[str, SampleT]:
    """Index typed samples while rejecting duplicate IDs."""

    indexed: dict[str, SampleT] = {}
    for sample in samples:
        if sample.question_id in indexed:
            raise ValueError(f"duplicate question_id in {label}: {sample.question_id}")
        indexed[sample.question_id] = sample
    return indexed


def compare_reports(
    before: EvaluationReport,
    after: EvaluationReport,
) -> EvaluationComparison:
    """Build a signed after-minus-before comparison for equal benchmark runs."""

    if before.sample_count != after.sample_count:
        raise ValueError("reports must have equal sample counts")
    if before.dataset_fingerprint != after.dataset_fingerprint:
        raise ValueError("reports must use the same benchmark fingerprint")
    if before.k_values != after.k_values:
        raise ValueError("reports must use equal k_values")
    if (before.qa is None) != (after.qa is None):
        raise ValueError("both reports must either contain or omit QA metrics")
    if (before.latency is None) != (after.latency is None):
        raise ValueError("both reports must either contain or omit latency metrics")

    return EvaluationComparison(
        before=before,
        after=after,
        delta=EvaluationDelta.between(before, after),
    )


def validate_rerank_candidate_pools(
    before: Sequence[PredictionSample],
    after: Sequence[PredictionSample],
) -> None:
    """Validate candidate provenance and ordering for a rerank comparison."""
    before_by_id = _index_unique(before, "before predictions")
    after_by_id = _index_unique(after, "after predictions")
    if set(before_by_id) != set(after_by_id):
        missing = sorted(set(before_by_id).difference(after_by_id))
        extra = sorted(set(after_by_id).difference(before_by_id))
        raise ValueError(
            "rerank prediction IDs do not match baseline "
            f"(missing={missing}, unexpected={extra})"
        )

    preserved_fields = {
        "chunk_id",
        "doc_id",
        "text",
        "score",
        "source",
        "law_name",
        "article",
        "clause",
        "metadata",
        "dense_score",
        "sparse_score",
        "hybrid_score",
    }
    for question_id, baseline_sample in before_by_id.items():
        reranked_sample = after_by_id[question_id]
        if baseline_sample.hits and not reranked_sample.hits:
            raise ValueError(
                f"rerank output {question_id!r} cannot be empty when the "
                "baseline candidate pool is non-empty"
            )
        baseline_hits = {hit.chunk_id: hit for hit in baseline_sample.hits}
        baseline_positions = {
            hit.chunk_id: position for position, hit in enumerate(baseline_sample.hits)
        }
        previous_score: float | None = None
        previous_baseline_position: int | None = None
        for expected_rank, hit in enumerate(reranked_sample.hits, start=1):
            baseline_hit = baseline_hits.get(hit.chunk_id)
            if baseline_hit is None:
                raise ValueError(
                    f"rerank output {question_id!r} introduced candidate "
                    f"{hit.chunk_id!r} that is absent from the baseline pool"
                )
            before_values = baseline_hit.model_dump(include=preserved_fields)
            after_values = hit.model_dump(include=preserved_fields)
            if before_values != after_values:
                raise ValueError(
                    f"rerank output {question_id!r} changed preserved fields "
                    f"for candidate {hit.chunk_id!r}"
                )
            if hit.rerank_score is None or hit.final_score is None:
                raise ValueError(
                    f"rerank output {question_id!r} must provide rerank_score "
                    f"and final_score for candidate {hit.chunk_id!r}"
                )
            if hit.rank != expected_rank:
                raise ValueError(
                    f"rerank output {question_id!r} has rank {hit.rank!r} for "
                    f"candidate {hit.chunk_id!r}; expected {expected_rank}"
                )
            if hit.final_score != hit.rerank_score:
                raise ValueError(
                    f"rerank output {question_id!r} has inconsistent final_score "
                    f"for candidate {hit.chunk_id!r}"
                )
            if previous_score is not None and hit.rerank_score > previous_score:
                raise ValueError(
                    f"rerank output {question_id!r} is not sorted by descending "
                    "rerank_score"
                )
            baseline_position = baseline_positions[hit.chunk_id]
            if (
                previous_score is not None
                and hit.rerank_score == previous_score
                and previous_baseline_position is not None
                and baseline_position <= previous_baseline_position
            ):
                raise ValueError(
                    f"rerank output {question_id!r} must preserve baseline order "
                    "for candidates with equal rerank_score"
                )
            previous_score = hit.rerank_score
            previous_baseline_position = baseline_position
