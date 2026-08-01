"""Hybrid retrieval orchestration for dense and sparse retrievers."""

from functools import lru_cache
import logging
import time
from typing import Any
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.config import load_config
from udsc2026.retrieval.dense.dense_retriever import DenseRetriever
from udsc2026.retrieval.hybrid.score_fusion import fuse_scores
from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever

LOGGER = logging.getLogger(__name__)


def _config() -> dict[str, Any]:
    return load_config()


class HybridRetriever:
    """Fuse dense and BM25 candidates without reranking."""

    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: BM25Retriever,
                 dense_weight: float | None = None, sparse_weight: float | None = None,
                 candidate_k: int | None = None, min_score: float | None = None) -> None:
        settings = _config().get("hybrid", {})
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = settings.get("dense_weight", 0.5) if dense_weight is None else dense_weight
        self.sparse_weight = settings.get("sparse_weight", 0.5) if sparse_weight is None else sparse_weight
        self.candidate_k = settings.get("candidate_k", 50) if candidate_k is None else candidate_k
        self.min_score = settings.get("min_score", 0.0) if min_score is None else min_score

    def search(self, query: str, top_k: int,
               filters: dict[str, str | int | list[str]] | None = None) -> list[RetrievalHit]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        started = time.perf_counter()
        dense_hits = self.dense_retriever.search(query, self.candidate_k, filters)
        sparse_hits = self.sparse_retriever.search(query, self.candidate_k, filters)
        fused = fuse_scores(dense_hits, sparse_hits, self.dense_weight, self.sparse_weight)
        result = [hit for hit in fused if (hit.final_score or 0.0) >= self.min_score][:top_k]
        LOGGER.info("hybrid search latency_ms=%.2f", (time.perf_counter() - started) * 1000)
        return result


@lru_cache(maxsize=1)
def _default_retriever() -> HybridRetriever:
    config = _config()
    from udsc2026.infrastructure.embedding.bkai_client import EmbeddingClient
    from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter

    embedding = config.get("embedding", {})
    client = EmbeddingClient(
        model_path=embedding["model_path"],
        device=embedding.get("device", "cpu"),
        batch_size=embedding.get("batch_size", 32),
        max_length=embedding.get("max_length", 256),
        normalize_embeddings=embedding.get("normalize_embeddings", True),
    )
    dense = DenseRetriever(client, get_vector_db_adapter(config))
    sparse = BM25Retriever(config.get("sparse", {}).get("bm25_index_path"))
    sparse.load()
    return HybridRetriever(dense, sparse)


def search(query: str, top_k: int,
           filters: dict[str, str | int | list[str]] | None = None) -> list[RetrievalHit]:
    """Search using lazily initialized default dense and sparse retrievers."""
    return _default_retriever().search(query, top_k, filters)
