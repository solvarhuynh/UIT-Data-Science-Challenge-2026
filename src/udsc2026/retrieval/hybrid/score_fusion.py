"""Score normalization and citation-preserving hybrid fusion."""

from typing import Any

from udsc2026.contracts.retrieval import RetrievalHit


def normalize_scores(hits: list[RetrievalHit], score_field: str) -> list[RetrievalHit]:
    """Return copies with min-max scores in [0, 1].

    Missing scores remain ``None``.  If all present scores are equal, each
    present score is conventionally normalized to 1.0 because it is equally
    relevant within that candidate list.
    """
    values = [getattr(hit, score_field) for hit in hits if getattr(hit, score_field) is not None]
    if not values:
        return [hit.model_copy() for hit in hits]
    minimum, maximum = min(values), max(values)
    updates = []
    for hit in hits:
        value = getattr(hit, score_field)
        normalized = None if value is None else (1.0 if maximum == minimum else (value - minimum) / (maximum - minimum))
        updates.append(hit.model_copy(update={score_field: normalized}))
    return updates


def fuse_scores(
    dense_hits: list[RetrievalHit],
    sparse_hits: list[RetrievalHit],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[RetrievalHit]:
    """Merge dense/sparse candidates, preserve citations, and rank by hybrid score."""
    dense = {hit.chunk_id: hit for hit in normalize_scores(dense_hits, "dense_score")}
    sparse = {hit.chunk_id: hit for hit in normalize_scores(sparse_hits, "sparse_score")}
    merged: list[RetrievalHit] = []
    for chunk_id in dense.keys() | sparse.keys():
        dense_hit, sparse_hit = dense.get(chunk_id), sparse.get(chunk_id)
        primary = dense_hit or sparse_hit
        secondary = sparse_hit if dense_hit else None
        values: dict[str, Any] = {}
        for field in ("doc_id", "text", "source", "law_name", "article", "clause"):
            values[field] = getattr(primary, field) or (getattr(secondary, field) if secondary else None)
        metadata = dict(getattr(secondary, "metadata", {}) or {})
        metadata.update(getattr(primary, "metadata", {}) or {})
        dense_score = dense_hit.dense_score if dense_hit else None
        sparse_score = sparse_hit.sparse_score if sparse_hit else None
        hybrid_score = dense_weight * (dense_score or 0.0) + sparse_weight * (sparse_score or 0.0)
        merged.append(RetrievalHit(chunk_id=chunk_id, **values, metadata=metadata,
                                   dense_score=dense_score, sparse_score=sparse_score,
                                   hybrid_score=hybrid_score, final_score=hybrid_score))
    merged.sort(key=lambda hit: hit.final_score or 0.0, reverse=True)
    return [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(merged, 1)]
