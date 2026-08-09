"""Dense retrieval orchestration using injected embedding and vector-store clients."""

import logging
import time
from functools import lru_cache
from typing import TYPE_CHECKING

from udsc2026.config import load_project_config
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.vector_db.base import VectorDBAdapter

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from udsc2026.infrastructure.embedding.client import EmbeddingClient


class DenseRetriever:
    """Encode a query and delegate vector search without fusion or reranking logic."""

    def __init__(
        self, embedding_client: "EmbeddingClient", vector_db: VectorDBAdapter
    ) -> None:
        """Initialize dense retrieval with embedding and vector-store clients."""
        self.embedding_client = embedding_client
        self.vector_db = vector_db

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str | int | list[str]] | None = None,
    ) -> list[RetrievalHit]:
        """Return ranked dense hits from the configured vector database."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        started = time.perf_counter()
        vector = self.embedding_client.embed_query(query)
        hits = self.vector_db.search(vector, top_k, filters)
        if len(hits) > top_k:
            raise RuntimeError(
                f"vector database returned {len(hits)} hits for top_k={top_k}"
            )
        ranked_hits = []
        for rank, hit in enumerate(hits, start=1):
            score = hit.dense_score if hit.dense_score is not None else hit.score
            ranked_hits.append(
                hit.model_copy(update={"dense_score": score, "rank": rank})
            )
        LOGGER.info(
            "dense search latency_ms=%.2f", (time.perf_counter() - started) * 1000
        )

        return ranked_hits


@lru_cache(maxsize=1)
def _default_retriever() -> DenseRetriever:
    """Build the default retriever once, on the first module-level search call."""
    config = load_project_config()
    from udsc2026.infrastructure.embedding.config import load_embedding_config

    embedding_config = load_embedding_config()
    from udsc2026.infrastructure.embedding.client import EmbeddingClient

    embedding_client = EmbeddingClient(
        model_path=embedding_config.get(
            "model_path", embedding_config["embedder_model_path"]
        ),
        device=embedding_config.get("device", "cpu"),
        batch_size=embedding_config.get("batch_size", 32),
        max_length=embedding_config.get("max_length", 256),
        normalize_embeddings=embedding_config.get("normalize_embeddings", True),
        output_dimension=embedding_config.get("output_dimension"),
        window_long_texts=embedding_config.get("window_long_texts", False),
        window_overlap_tokens=embedding_config.get("window_overlap_tokens", 32),
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
