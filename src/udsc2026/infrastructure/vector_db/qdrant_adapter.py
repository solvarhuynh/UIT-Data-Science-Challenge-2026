"""Qdrant implementation of the vector database contract."""

import uuid

from qdrant_client import QdrantClient, models

from udsc2026.contracts import LegalChunk
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.vector_db.base import (
    VectorDBAdapter,
    chunk_payload,
    payload_to_hit,
    validate_collection_name,
    validate_query_vector,
    validate_top_k,
    validate_vector_size,
)

_TOP_LEVEL_PAYLOAD_FIELDS = {
    "article",
    "chapter",
    "chunk_id",
    "clause",
    "doc_id",
    "effective_date",
    "law_name",
    "parent_id",
    "parent_text",
    "point",
    "section",
    "source",
    "text",
}


def _filter_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("filter keys must be non-empty strings")
    normalized = key.strip()
    if normalized in _TOP_LEVEL_PAYLOAD_FIELDS or normalized.startswith("metadata."):
        return normalized
    return f"metadata.{normalized}"


class QdrantAdapter(VectorDBAdapter):
    """Store and search legal chunk vectors in Qdrant."""

    def __init__(
        self, url: str, collection_name: str, api_key: str | None = None
    ) -> None:
        """Initialize a client for the configured Qdrant collection."""
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = validate_collection_name(collection_name)
        self._vector_size: int | None = None

    def create_collection(
        self, name: str, vector_size: int, distance: str = "cosine"
    ) -> None:
        """Create or validate a Qdrant collection."""
        validated_name = validate_collection_name(name)
        validate_vector_size(vector_size)
        distances = {
            "cosine": models.Distance.COSINE,
            "dot": models.Distance.DOT,
            "euclid": models.Distance.EUCLID,
        }
        if distance not in distances:
            raise ValueError(f"Unsupported vector distance: {distance}")
        expected_distance = distances[distance]
        if self.client.collection_exists(collection_name=validated_name):
            collection = self.client.get_collection(collection_name=validated_name)
            vectors = collection.config.params.vectors
            if isinstance(vectors, dict):
                raise ValueError(
                    "Qdrant collection uses named vectors, which this adapter "
                    "does not support"
                )
            if (
                getattr(vectors, "size", None) != vector_size
                or getattr(vectors, "distance", None) != expected_distance
            ):
                raise ValueError(
                    "Qdrant collection already exists with incompatible "
                    "vector size or distance"
                )
            self.collection_name = validated_name
            self._vector_size = vector_size
            return
        self.client.create_collection(
            collection_name=validated_name,
            vectors_config=models.VectorParams(
                size=vector_size, distance=expected_distance
            ),
        )
        self.collection_name = validated_name
        self._vector_size = vector_size

    def upsert(self, chunks: list[LegalChunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks in the active Qdrant collection."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        chunk_ids: list[str] = []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not isinstance(chunk, LegalChunk):
                raise TypeError(f"chunks[{index}] must be a LegalChunk")
            if not chunk.chunk_id.strip():
                raise ValueError(f"chunks[{index}].chunk_id must not be blank")
            validate_query_vector(embedding)
            if self._vector_size is not None and len(embedding) != self._vector_size:
                raise ValueError(
                    f"embeddings[{index}] must have dimension {self._vector_size}"
                )
            chunk_ids.append(chunk.chunk_id)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("upsert chunks must have unique chunk_id values")
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector=embedding,
                payload=chunk_payload(chunk),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, str | int | list[str]] | None = None,
    ) -> list[RetrievalHit]:
        """Return nearest Qdrant hits that match optional filters."""
        validate_query_vector(query_vector)
        validate_top_k(top_k)
        if self._vector_size is not None and len(query_vector) != self._vector_size:
            raise ValueError(
                f"query vector must have dimension {self._vector_size}, "
                f"got {len(query_vector)}"
            )
        query_filter = None
        if filters:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=_filter_key(key),
                        match=(
                            models.MatchAny(any=value)
                            if isinstance(value, list)
                            else models.MatchValue(value=value)
                        ),
                    )
                    for key, value in filters.items()
                ]
            )
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return [
            payload_to_hit(result.payload or {}, result.score) for result in results
        ]

    def delete_collection(self, name: str) -> None:
        """Delete a Qdrant collection."""
        validated_name = validate_collection_name(name)
        self.client.delete_collection(collection_name=validated_name)
        if validated_name == self.collection_name:
            self._vector_size = None
