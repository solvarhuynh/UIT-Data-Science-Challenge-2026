"""Reusable early document aggregation for LegalIR candidate rankings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.legal_ir import LegalIRPrediction
from udsc2026.evaluation.models import PredictionSample

AggregationMethod = Literal[
    "first_occurrence",
    "max",
    "max_plus_0.10_second",
    "max_plus_0.20_second",
    "top2_mean",
    "rrf_cap2",
    "rrf_cap3",
]

AGGREGATION_METHODS: tuple[AggregationMethod, ...] = (
    "first_occurrence",
    "max",
    "max_plus_0.10_second",
    "max_plus_0.20_second",
    "top2_mean",
    "rrf_cap2",
    "rrf_cap3",
)
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class AggregatedDocument:
    """One document collapsed from an unchanged ranked chunk candidate pool."""

    doc_id: str
    first_chunk_rank: int
    max_dense_score: float
    second_dense_score: float | None
    top_evidence_chunk_ids: tuple[str, ...]
    source_ranks: tuple[int, ...]
    number_of_supporting_chunks: int
    aggregation_method: AggregationMethod
    aggregation_score: float

    def model_dump(self) -> dict[str, object]:
        """Return a JSON-ready representation without a Pydantic dependency."""

        return asdict(self)


@dataclass(frozen=True)
class _ScoredHit:
    hit: RetrievalHit
    rank: int
    dense_score: float


def _dense_score(hit: RetrievalHit) -> float:
    """Return the best available dense-compatible score from one hit."""

    for value in (hit.dense_score, hit.final_score, hit.score):
        if value is not None:
            return float(value)
    return 0.0


def _aggregation_score(
    method: AggregationMethod,
    scored_hits: Sequence[_ScoredHit],
    *,
    rrf_k: int,
) -> float:
    scores = sorted((item.dense_score for item in scored_hits), reverse=True)
    if method == "first_occurrence":
        return -float(scored_hits[0].rank)
    if method == "max":
        return scores[0]
    if method == "max_plus_0.10_second":
        return scores[0] + 0.10 * (scores[1] if len(scores) > 1 else 0.0)
    if method == "max_plus_0.20_second":
        return scores[0] + 0.20 * (scores[1] if len(scores) > 1 else 0.0)
    if method == "top2_mean":
        return sum(scores[:2]) / min(len(scores), 2)
    if method == "rrf_cap2":
        cap = 2
    elif method == "rrf_cap3":
        cap = 3
    else:
        raise ValueError(f"unsupported aggregation method: {method}")
    return sum(1.0 / (rrf_k + item.rank) for item in scored_hits[:cap])


def aggregate_document_candidates(
    prediction: PredictionSample,
    *,
    method: AggregationMethod = "first_occurrence",
    evidence_limit: int = 3,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[AggregatedDocument]:
    """Collapse chunks by document while retaining evidence and source ranks.

    The input candidate hits are never reordered or modified. Documents are
    only ranked in the returned list according to the selected method.
    """

    if method not in AGGREGATION_METHODS:
        raise ValueError(f"unsupported aggregation method: {method}")
    if isinstance(evidence_limit, bool) or evidence_limit <= 0:
        raise ValueError("evidence_limit must be a positive integer")
    if isinstance(rrf_k, bool) or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer")

    grouped: dict[str, list[_ScoredHit]] = defaultdict(list)
    for index, hit in enumerate(prediction.hits, start=1):
        rank = hit.rank if hit.rank is not None else index
        grouped[hit.doc_id].append(
            _ScoredHit(hit=hit, rank=rank, dense_score=_dense_score(hit))
        )

    documents: list[AggregatedDocument] = []
    for doc_id, scored_hits in grouped.items():
        by_rank = sorted(scored_hits, key=lambda item: item.rank)
        by_score = sorted(
            scored_hits,
            key=lambda item: (-item.dense_score, item.rank, item.hit.chunk_id),
        )
        documents.append(
            AggregatedDocument(
                doc_id=doc_id,
                first_chunk_rank=by_rank[0].rank,
                max_dense_score=by_score[0].dense_score,
                second_dense_score=(
                    by_score[1].dense_score if len(by_score) > 1 else None
                ),
                top_evidence_chunk_ids=tuple(
                    item.hit.chunk_id for item in by_score[:evidence_limit]
                ),
                source_ranks=tuple(item.rank for item in by_rank),
                number_of_supporting_chunks=len(scored_hits),
                aggregation_method=method,
                aggregation_score=_aggregation_score(
                    method,
                    by_rank,
                    rrf_k=rrf_k,
                ),
            )
        )
    return sorted(
        documents,
        key=lambda item: (
            -item.aggregation_score,
            item.first_chunk_rank,
            item.doc_id,
        ),
    )


def aggregate_to_legal_ir_prediction(
    prediction: PredictionSample,
    *,
    method: AggregationMethod = "first_occurrence",
    max_documents: int = 5,
    evidence_limit: int = 3,
    rrf_k: int = DEFAULT_RRF_K,
) -> LegalIRPrediction:
    """Convert an early-collapsed candidate pool into an official document list."""

    if isinstance(max_documents, bool) or max_documents <= 0:
        raise ValueError("max_documents must be a positive integer")
    documents = aggregate_document_candidates(
        prediction,
        method=method,
        evidence_limit=evidence_limit,
        rrf_k=rrf_k,
    )
    return LegalIRPrediction(
        id=prediction.question_id,
        documents=[item.doc_id for item in documents[:max_documents]],
    )


__all__ = [
    "AGGREGATION_METHODS",
    "DEFAULT_RRF_K",
    "AggregationMethod",
    "AggregatedDocument",
    "aggregate_document_candidates",
    "aggregate_to_legal_ir_prediction",
]
