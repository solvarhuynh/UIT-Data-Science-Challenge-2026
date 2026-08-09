"""Shared interface and payload mapping for vector database adapters."""

import math
import re
from abc import ABC, abstractmethod
from numbers import Real
from typing import Any, cast

from udsc2026.contracts import LegalChunk
from udsc2026.contracts.retrieval import RetrievalHit

_COLLECTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def validate_collection_name(name: str) -> str:
    """Return a safe single path/service segment for a vector collection."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("collection_name must be a non-empty string")
    normalized = name.strip()
    if _COLLECTION_NAME_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "collection_name must start with an ASCII letter/digit and contain "
            "only letters, digits, '.', '_' or '-' (maximum 128 characters)"
        )
    return normalized


def validate_vector_size(vector_size: int) -> int:
    """Validate a collection vector dimension."""

    if (
        isinstance(vector_size, bool)
        or not isinstance(vector_size, int)
        or vector_size <= 0
    ):
        raise ValueError("vector_size must be a positive integer")
    return vector_size


def validate_top_k(top_k: int) -> int:
    """Validate a vector search limit."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def validate_query_vector(query_vector: list[float]) -> list[float]:
    """Reject empty, non-scalar, boolean, or non-finite query vectors."""

    if not isinstance(query_vector, list) or not query_vector:
        raise ValueError("query_vector must be a non-empty list")
    for index, value in enumerate(cast(list[object], query_vector)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"query_vector[{index}] must be a finite real number")
    return query_vector


def chunk_payload(chunk: LegalChunk) -> dict[str, Any]:
    """Map a chunk to the payload shared by Qdrant and FAISS side stores."""
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "law_name": chunk.law_name,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "article": chunk.article,
        "clause": chunk.clause,
        "point": chunk.point,
        "effective_date": chunk.effective_date,
        "source": chunk.source,
        "metadata": chunk.metadata,
    }


def payload_to_hit(payload: dict[str, Any], score: float) -> RetrievalHit:
    """Convert a stored payload to the common retrieval result contract."""
    metadata = dict(payload.get("metadata", {}))
    for field in (
        "law_name",
        "article",
        "clause",
        "source",
        "chapter",
        "section",
        "effective_date",
        "point",
        "parent_id",
    ):
        if payload.get(field) is not None:
            metadata.setdefault(field, payload[field])
    return RetrievalHit(
        chunk_id=payload["chunk_id"],
        parent_id=payload.get("parent_id"),
        doc_id=payload["doc_id"],
        text=payload["text"],
        score=float(score),
        dense_score=float(score),
        source=payload.get("source"),
        law_name=payload.get("law_name"),
        article=payload.get("article"),
        clause=payload.get("clause"),
        metadata=metadata,
    )


class VectorDBAdapter(ABC):
    """Backend-neutral vector storage contract."""

    @abstractmethod
    def create_collection(
        self, name: str, vector_size: int, distance: str = "cosine"
    ) -> None:
        """Create or validate a named vector collection."""
        ...

    @abstractmethod
    def upsert(self, chunks: list[LegalChunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks and their embeddings."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, str | int | list[str]] | None = None,
    ) -> list[RetrievalHit]:
        """Return nearest vector hits that match optional filters."""
        ...

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Delete a named vector collection."""
        ...
