"""Hybrid retrieval orchestration for dense and sparse retrievers."""

import logging
import time
from functools import lru_cache
from typing import Any

from udsc2026.config import load_project_config
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.retrieval.dense.dense_retriever import DenseRetriever
from udsc2026.retrieval.hybrid.config import HybridSettings, load_hybrid_settings
from udsc2026.retrieval.hybrid.score_fusion import fuse_scores
from udsc2026.retrieval.reranking.pipeline import SearchRetriever
from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever

LOGGER = logging.getLogger(__name__)


def _config() -> dict[str, Any]:
    return load_project_config()


class HybridRetriever:
    """Fuse dense and BM25 candidates without reranking."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: BM25Retriever,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        candidate_k: int | None = None,
        min_score: float | None = None,
    ) -> None:
        """Initialize weighted dense and sparse retrieval."""
        base_settings = (
            load_hybrid_settings()
            if any(
                value is None
                for value in (
                    dense_weight,
                    sparse_weight,
                    candidate_k,
                    min_score,
                )
            )
            else HybridSettings()
        )
        settings = HybridSettings.model_validate(
            {
                **base_settings.model_dump(),
                **({"dense_weight": dense_weight} if dense_weight is not None else {}),
                **(
                    {"sparse_weight": sparse_weight}
                    if sparse_weight is not None
                    else {}
                ),
                **({"candidate_k": candidate_k} if candidate_k is not None else {}),
                **({"min_score": min_score} if min_score is not None else {}),
            }
        )
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.dense_weight = settings.dense_weight
        self.sparse_weight = settings.sparse_weight
        self.candidate_k = settings.candidate_k
        self.min_score = settings.min_score

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str | int | list[str]] | None = None,
    ) -> list[RetrievalHit]:
        """Fuse dense and sparse candidates into ranked hybrid hits."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        started = time.perf_counter()
        dense_hits = self.dense_retriever.search(query, self.candidate_k, filters)
        sparse_hits = self.sparse_retriever.search(query, self.candidate_k, filters)
        fused = fuse_scores(
            dense_hits, sparse_hits, self.dense_weight, self.sparse_weight
        )
        result = [hit for hit in fused if (hit.final_score or 0.0) >= self.min_score][
            :top_k
        ]
        if len(result) > top_k:
            raise RuntimeError(
                f"hybrid retriever returned {len(result)} hits for top_k={top_k}"
            )
        LOGGER.info(
            "hybrid search latency_ms=%.2f", (time.perf_counter() - started) * 1000
        )
        return result


@lru_cache(maxsize=1)
def _default_retriever() -> SearchRetriever:
    config = _config()
    from udsc2026.infrastructure.embedding.bkai_client import EmbeddingClient
    from udsc2026.infrastructure.embedding.config import load_embedding_config
    from udsc2026.infrastructure.reranker.config import load_reranker_settings
    from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter
    from udsc2026.retrieval.reranking import CrossEncoderReranker, RerankedRetriever

    embedding = load_embedding_config()
    hybrid_settings = load_hybrid_settings()
    client = EmbeddingClient(
        model_path=embedding["embedder_model_path"],
        device=embedding.get("device", "cpu"),
        batch_size=embedding.get("batch_size", 32),
        max_length=embedding.get("max_length", 256),
        normalize_embeddings=embedding.get("normalize_embeddings", True),
    )
    dense = DenseRetriever(client, get_vector_db_adapter(config))
    sparse = BM25Retriever(hybrid_settings.bm25_index_path)
    sparse.load()
    hybrid = HybridRetriever(
        dense,
        sparse,
        dense_weight=hybrid_settings.dense_weight,
        sparse_weight=hybrid_settings.sparse_weight,
        candidate_k=hybrid_settings.candidate_k,
        min_score=hybrid_settings.min_score,
    )
    reranker_settings = load_reranker_settings()
    if not reranker_settings.enabled:
        return hybrid
    if reranker_settings.top_n > hybrid.candidate_k:
        raise ValueError("reranker.top_n must not exceed hybrid.candidate_k")
    reranker = CrossEncoderReranker(reranker_settings.create_client())
    return RerankedRetriever(
        hybrid,
        reranker,
        candidate_k=hybrid.candidate_k,
        top_n=reranker_settings.top_n,
    )


def search(
    query: str, top_k: int, filters: dict[str, str | int | list[str]] | None = None
) -> list[RetrievalHit]:
    """Search using lazily initialized default dense and sparse retrievers."""
    return _default_retriever().search(query, top_k, filters)
