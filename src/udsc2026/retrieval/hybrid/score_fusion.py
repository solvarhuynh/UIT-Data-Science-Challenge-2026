"""Score normalization and citation-preserving hybrid fusion."""

import math
from typing import Any

from udsc2026.contracts.retrieval import RetrievalHit

_NORMALIZABLE_SCORE_FIELDS = {"dense_score", "sparse_score"}


def normalize_scores(hits: list[RetrievalHit], score_field: str) -> list[RetrievalHit]:
    """Return copies with min-max scores in [0, 1].

    Missing scores remain ``None``. Equal positive evidence maps to 1.0; an
    all-zero/non-positive list maps to 0.0 so absent evidence cannot become a
    maximum hybrid score.
    """
    if score_field not in _NORMALIZABLE_SCORE_FIELDS:
        raise ValueError("score_field must be either 'dense_score' or 'sparse_score'")
    _validate_hits(hits, "hits")
    values = [
        getattr(hit, score_field)
        for hit in hits
        if getattr(hit, score_field) is not None
    ]
    if not values:
        return [hit.model_copy() for hit in hits]
    minimum, maximum = min(values), max(values)
    updates = []
    for hit in hits:
        value = getattr(hit, score_field)
        normalized = (
            None
            if value is None
            else (
                (1.0 if maximum > 0 else 0.0)
                if maximum == minimum
                else (value - minimum) / (maximum - minimum)
            )
        )
        updates.append(hit.model_copy(update={score_field: normalized}))
    return updates


def fuse_scores(
    dense_hits: list[RetrievalHit],
    sparse_hits: list[RetrievalHit],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[RetrievalHit]:
    """Merge dense/sparse candidates, preserve citations, and rank by hybrid score."""
    _validate_hits(dense_hits, "dense_hits")
    _validate_hits(sparse_hits, "sparse_hits")
    _validate_weights(dense_weight, sparse_weight)
    dense = {hit.chunk_id: hit for hit in normalize_scores(dense_hits, "dense_score")}
    sparse = {
        hit.chunk_id: hit for hit in normalize_scores(sparse_hits, "sparse_score")
    }
    ordered_chunk_ids = list(dense)
    ordered_chunk_ids.extend(chunk_id for chunk_id in sparse if chunk_id not in dense)
    candidate_order = {
        chunk_id: index for index, chunk_id in enumerate(ordered_chunk_ids)
    }
    merged: list[RetrievalHit] = []
    for chunk_id in ordered_chunk_ids:
        dense_hit, sparse_hit = dense.get(chunk_id), sparse.get(chunk_id)
        primary = dense_hit or sparse_hit
        if primary is None:  # pragma: no cover - ordered IDs come from these mappings
            raise RuntimeError("hybrid candidate disappeared during fusion")
        secondary = sparse_hit if dense_hit else None
        if dense_hit is not None and sparse_hit is not None:
            _validate_matching_candidate(dense_hit, sparse_hit)
        values: dict[str, Any] = {}
        for field in ("doc_id", "text", "source", "law_name", "article", "clause"):
            values[field] = getattr(primary, field) or (
                getattr(secondary, field) if secondary else None
            )
        metadata = dict(getattr(secondary, "metadata", {}) or {})
        metadata.update(getattr(primary, "metadata", {}) or {})
        dense_score = dense_hit.dense_score if dense_hit else None
        sparse_score = sparse_hit.sparse_score if sparse_hit else None
        hybrid_score = dense_weight * (dense_score or 0.0) + sparse_weight * (
            sparse_score or 0.0
        )
        merged.append(
            RetrievalHit(
                chunk_id=chunk_id,
                **values,
                metadata=metadata,
                dense_score=dense_score,
                sparse_score=sparse_score,
                hybrid_score=hybrid_score,
                final_score=hybrid_score,
            )
        )
    merged.sort(
        key=lambda hit: (
            -(hit.final_score if hit.final_score is not None else 0.0),
            candidate_order[hit.chunk_id],
        )
    )
    return [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(merged, 1)]


def _validate_hits(hits: list[RetrievalHit], name: str) -> None:
    if not isinstance(hits, list):
        raise TypeError(f"{name} must be a list of RetrievalHit objects")
    chunk_ids: list[str] = []
    for index, hit in enumerate(hits):
        if not isinstance(hit, RetrievalHit):
            raise TypeError(f"{name}[{index}] must be a RetrievalHit")
        chunk_ids.append(hit.chunk_id)
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(f"{name} must not contain duplicate chunk_id values")


def _validate_weights(dense_weight: float, sparse_weight: float) -> None:
    for name, value in (
        ("dense_weight", dense_weight),
        ("sparse_weight", sparse_weight),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite non-negative number")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
    if not math.isclose(
        float(dense_weight) + float(sparse_weight),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("dense_weight and sparse_weight must sum to 1.0")


def _validate_matching_candidate(
    dense_hit: RetrievalHit,
    sparse_hit: RetrievalHit,
) -> None:
    for field in ("doc_id", "text", "source", "law_name", "article", "clause"):
        dense_value = getattr(dense_hit, field)
        sparse_value = getattr(sparse_hit, field)
        if (
            dense_value is not None
            and sparse_value is not None
            and dense_value != sparse_value
        ):
            raise ValueError(
                f"candidate {dense_hit.chunk_id!r} has conflicting {field} values"
            )
    overlapping_metadata = dense_hit.metadata.keys() & sparse_hit.metadata.keys()
    for key in sorted(overlapping_metadata):
        if dense_hit.metadata[key] != sparse_hit.metadata[key]:
            raise ValueError(
                f"candidate {dense_hit.chunk_id!r} has conflicting metadata "
                f"value for {key!r}"
            )
