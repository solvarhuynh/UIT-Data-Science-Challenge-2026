"""Reusable early document aggregation for LegalIR candidate rankings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.retrieval_metrics import LegalIRPrediction
from udsc2026.evaluation.models import PredictionSample

AggregationMethod = Literal[
    "first_occurrence",
    "max",
    "max_plus_0.10_second",
    "max_plus_0.20_second",
    "top2_mean",
    "rrf_cap2",
    "rrf_cap3",
    "rrf_cap2_dense_signal",
]
ScoreField = Literal[
    "score",
    "dense_score",
    "sparse_score",
    "hybrid_score",
    "rerank_score",
    "final_score",
]

AGGREGATION_METHODS: tuple[AggregationMethod, ...] = (
    "first_occurrence",
    "max",
    "max_plus_0.10_second",
    "max_plus_0.20_second",
    "top2_mean",
    "rrf_cap2",
    "rrf_cap3",
    "rrf_cap2_dense_signal",
)
DEFAULT_RRF_K = 60
DEFAULT_DENSE_RRF_K = 30
DEFAULT_DENSE_SIGNAL_WEIGHT = 0.30


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
    first_dense_rank: int | None = None

    def model_dump(self) -> dict[str, object]:
        """Return a JSON-ready representation without a Pydantic dependency."""

        return asdict(self)


@dataclass(frozen=True)
class _ScoredHit:
    hit: RetrievalHit
    rank: int
    selected_score: float | None
    dense_score: float | None
    dense_rank: int | None


def _score(hit: RetrievalHit, score_field: ScoreField) -> float | None:
    """Read one score explicitly; never silently substitute another score."""

    value = getattr(hit, score_field)
    return None if value is None else float(value)


def _aggregation_score(
    method: AggregationMethod,
    scored_hits: Sequence[_ScoredHit],
    *,
    rrf_k: int,
    dense_rrf_k: int,
    dense_signal_weight: float,
) -> float:
    eligible = [item for item in scored_hits if item.selected_score is not None]
    scores = sorted(
        (item.selected_score for item in eligible if item.selected_score is not None),
        reverse=True,
    )
    if method == "first_occurrence":
        return -float(scored_hits[0].rank)
    if not eligible:
        return 0.0
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
    elif method == "rrf_cap2_dense_signal":
        cap = 2
    else:
        raise ValueError(f"unsupported aggregation method: {method}")
    score = sum(1.0 / (rrf_k + item.rank) for item in eligible[:cap])
    if method == "rrf_cap2_dense_signal":
        first_dense_rank = min(
            (item.dense_rank for item in eligible if item.dense_rank is not None),
            default=None,
        )
        if first_dense_rank is not None:
            score += dense_signal_weight / (dense_rrf_k + first_dense_rank)
    return score


def aggregate_document_candidates(
    prediction: PredictionSample,
    *,
    method: AggregationMethod = "first_occurrence",
    evidence_limit: int = 3,
    rrf_k: int = DEFAULT_RRF_K,
    dense_rrf_k: int = DEFAULT_DENSE_RRF_K,
    dense_signal_weight: float = DEFAULT_DENSE_SIGNAL_WEIGHT,
    score_field: ScoreField = "dense_score",
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
    if isinstance(dense_rrf_k, bool) or dense_rrf_k <= 0:
        raise ValueError("dense_rrf_k must be a positive integer")
    if dense_signal_weight < 0:
        raise ValueError("dense_signal_weight must be non-negative")

    grouped: dict[str, list[_ScoredHit]] = defaultdict(list)
    dense_ranked = sorted(
        (
            hit
            for hit in prediction.hits
            if hit.dense_score is not None
        ),
        key=lambda hit: (
            -float(hit.dense_score),
            hit.rank if hit.rank is not None else len(prediction.hits) + 1,
            hit.chunk_id,
        ),
    )
    dense_rank_by_doc: dict[str, int] = {}
    for rank, hit in enumerate(dense_ranked, start=1):
        dense_rank_by_doc.setdefault(hit.doc_id, rank)

    for index, hit in enumerate(prediction.hits, start=1):
        rank = hit.rank if hit.rank is not None else index
        grouped[hit.doc_id].append(
            _ScoredHit(
                hit=hit,
                rank=rank,
                selected_score=_score(hit, score_field),
                dense_score=(
                    None if hit.dense_score is None else float(hit.dense_score)
                ),
                dense_rank=dense_rank_by_doc.get(hit.doc_id),
            )
        )

    documents: list[AggregatedDocument] = []
    for doc_id, scored_hits in grouped.items():
        by_rank = sorted(scored_hits, key=lambda item: item.rank)
        by_score = sorted(
            scored_hits,
            key=lambda item: (
                item.selected_score is None,
                -(
                    item.selected_score
                    if item.selected_score is not None
                    else 0.0
                ),
                item.rank,
                item.hit.chunk_id,
            ),
        )
        if method != "first_occurrence" and not any(
            item.selected_score is not None for item in scored_hits
        ):
            continue
        dense_rank = dense_rank_by_doc.get(doc_id)
        documents.append(
            AggregatedDocument(
                doc_id=doc_id,
                first_chunk_rank=by_rank[0].rank,
                max_dense_score=by_score[0].dense_score or 0.0,
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
                    dense_rrf_k=dense_rrf_k,
                    dense_signal_weight=dense_signal_weight,
                ),
                first_dense_rank=dense_rank,
            )
        )
    return sorted(
        documents,
        key=lambda item: (
            -item.aggregation_score,
            item.first_dense_rank if item.first_dense_rank is not None else 10**9,
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
    dense_rrf_k: int = DEFAULT_DENSE_RRF_K,
    dense_signal_weight: float = DEFAULT_DENSE_SIGNAL_WEIGHT,
    score_field: ScoreField = "dense_score",
) -> LegalIRPrediction:
    """Convert an early-collapsed candidate pool into an official document list."""

    if isinstance(max_documents, bool) or max_documents <= 0:
        raise ValueError("max_documents must be a positive integer")
    documents = aggregate_document_candidates(
        prediction,
        method=method,
        evidence_limit=evidence_limit,
        rrf_k=rrf_k,
        dense_rrf_k=dense_rrf_k,
        dense_signal_weight=dense_signal_weight,
        score_field=score_field,
    )
    return LegalIRPrediction(
        id=prediction.question_id,
        documents=[item.doc_id for item in documents[:max_documents]],
    )


__all__ = [
    "AGGREGATION_METHODS",
    "DEFAULT_RRF_K",
    "AggregationMethod",
    "ScoreField",
    "AggregatedDocument",
    "DEFAULT_DENSE_RRF_K",
    "DEFAULT_DENSE_SIGNAL_WEIGHT",
    "aggregate_document_candidates",
    "aggregate_to_legal_ir_prediction",
]
