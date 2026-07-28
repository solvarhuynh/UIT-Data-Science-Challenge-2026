"""Qdrant implementation of the vector database contract."""

import uuid

from qdrant_client import QdrantClient, models

from udsc2026.contracts import LegalChunk
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.vector_db.base import VectorDBAdapter, chunk_payload, payload_to_hit


class QdrantAdapter(VectorDBAdapter):
    """Store and search legal chunk vectors in Qdrant."""

    def __init__(self, url: str, collection_name: str, api_key: str | None = None) -> None:
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def create_collection(self, name: str, vector_size: int, distance: str = "cosine") -> None:
        distances = {"cosine": models.Distance.COSINE, "dot": models.Distance.DOT, "euclid": models.Distance.EUCLID}
        if distance not in distances:
            raise ValueError(f"Unsupported vector distance: {distance}")
        self.client.recreate_collection(collection_name=name, vectors_config=models.VectorParams(size=vector_size, distance=distances[distance]))
        self.collection_name = name

    def upsert(self, chunks: list[LegalChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)), vector=embedding, payload=chunk_payload(chunk)) for chunk, embedding in zip(chunks, embeddings)]
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int, filters: dict[str, str | int | list[str]] | None = None) -> list[RetrievalHit]:
        query_filter = None
        if filters:
            query_filter = models.Filter(must=[models.FieldCondition(key=key, match=models.MatchAny(any=value) if isinstance(value, list) else models.MatchValue(value=value)) for key, value in filters.items()])
        results = self.client.search(collection_name=self.collection_name, query_vector=query_vector, limit=top_k, query_filter=query_filter)
        return [payload_to_hit(result.payload or {}, result.score) for result in results]

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(collection_name=name)
