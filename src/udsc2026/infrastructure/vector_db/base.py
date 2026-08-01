"""Shared interface and payload mapping for vector database adapters."""

from abc import ABC, abstractmethod
from typing import Any

from udsc2026.contracts import LegalChunk
from udsc2026.contracts.retrieval import RetrievalHit


def chunk_payload(chunk: LegalChunk) -> dict[str, Any]:
    """Map a chunk to the payload shared by Qdrant and FAISS side stores."""
    return {
        "chunk_id": chunk.chunk_id, "parent_id": chunk.parent_id,
        "doc_id": chunk.doc_id, "text": chunk.text,
        "parent_text": chunk.parent_text,
        "law_name": chunk.law_name, "chapter": chunk.chapter, "section": chunk.section,
        "article": chunk.article, "clause": chunk.clause, "point": chunk.point,
        "effective_date": chunk.effective_date, "source": chunk.source,
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
        "parent_text",
    ):
        if payload.get(field) is not None:
            metadata.setdefault(field, payload[field])
    return RetrievalHit(
        chunk_id=payload["chunk_id"], doc_id=payload["doc_id"], text=payload["text"],
        score=float(score), dense_score=float(score), source=payload.get("source"),
        law_name=payload.get("law_name"), article=payload.get("article"),
        clause=payload.get("clause"), metadata=metadata,
    )


class VectorDBAdapter(ABC):
    """Backend-neutral vector storage contract."""

    @abstractmethod
    def create_collection(self, name: str, vector_size: int, distance: str = "cosine") -> None: ...

    @abstractmethod
    def upsert(self, chunks: list[LegalChunk], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int,
               filters: dict[str, str | int | list[str]] | None = None) -> list[RetrievalHit]: ...

    @abstractmethod
    def delete_collection(self, name: str) -> None: ...
