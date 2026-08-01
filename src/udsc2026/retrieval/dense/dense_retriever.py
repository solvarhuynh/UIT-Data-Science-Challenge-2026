"""Dense retrieval orchestration using injected embedding and vector-store clients."""

from functools import lru_cache
import logging
import time
from typing import TYPE_CHECKING

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.config import load_config
from udsc2026.infrastructure.vector_db.base import VectorDBAdapter

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from udsc2026.infrastructure.embedding.bkai_client import EmbeddingClient


class DenseRetriever:
    """Encode a query and delegate vector search without fusion or reranking logic."""

    def __init__(self, embedding_client: "EmbeddingClient", vector_db: VectorDBAdapter) -> None:
        self.embedding_client = embedding_client
        self.vector_db = vector_db

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str | int | list[str]] | None = None,
    ) -> list[RetrievalHit]:
        """Return ranked dense hits from the configured vector database."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        started = time.perf_counter()
        vector = self.embedding_client.embed_query(query)
        hits = self.vector_db.search(vector, top_k, filters)
        ranked_hits = []
        for rank, hit in enumerate(hits, start=1):
            score = hit.dense_score if hit.dense_score is not None else hit.score
            ranked_hits.append(hit.model_copy(update={"dense_score": score, "rank": rank}))
        LOGGER.info("dense search latency_ms=%.2f", (time.perf_counter() - started) * 1000)
        return ranked_hits


@lru_cache(maxsize=1)
def _default_retriever() -> DenseRetriever:
    """Build the default retriever once, on the first module-level search call."""
    config = load_config()
    embedding_config = config.get("embedding", {})
    from udsc2026.infrastructure.embedding.bkai_client import EmbeddingClient

    embedding_client = EmbeddingClient(
        model_path=embedding_config["model_path"],
        device=embedding_config.get("device", "cpu"),
        batch_size=embedding_config.get("batch_size", 32),
        max_length=embedding_config.get("max_length", 256),
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
    )
    from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter

    return DenseRetriever(embedding_client, get_vector_db_adapter(config))


def search(
    query: str,
    top_k: int,
    filters: dict[str, str | int | list[str]] | None = None,
) -> list[RetrievalHit]:
    """Search with the lazily initialized default embedding/vector-store clients."""
    return _default_retriever().search(query, top_k, filters)
