"""Production orchestration from hybrid retrieval to cross-encoder reranking."""

from __future__ import annotations

from typing import Protocol

from udsc2026.contracts.retrieval import RetrievalHit

RetrievalFilters = dict[str, str | int | list[str]]


class SearchRetriever(Protocol):
    """Structural search interface shared by hybrid and reranked retrieval."""

    def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        """Return ranked retrieval hits."""


class CandidateReranker(Protocol):
    """Structural reranker interface used by the production pipeline."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        """Rerank a candidate pool and return at most ``top_n`` hits."""


class RerankedRetriever:
    """Fetch a broad hybrid pool, then return a cross-encoder-ranked subset."""

    def __init__(
        self,
        hybrid_retriever: SearchRetriever,
        reranker: CandidateReranker,
        *,
        candidate_k: int,
        top_n: int,
    ) -> None:
        """Initialize candidate retrieval and final reranking limits."""
        _validate_positive_integer(candidate_k, "candidate_k")
        _validate_positive_integer(top_n, "top_n")
        if top_n > candidate_k:
            raise ValueError("top_n must not exceed candidate_k")
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.top_n = top_n

    def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        """Search hybrid candidates and rerank only the requested final count."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        _validate_positive_integer(top_k, "top_k")
        candidates = self.hybrid_retriever.search(
            query,
            self.candidate_k,
            filters,
        )
        result = self.reranker.rerank(
            query,
            candidates,
            min(top_k, self.top_n),
        )
        if len(result) > top_k:
            raise RuntimeError(
                f"reranker returned {len(result)} hits for top_k={top_k}"
            )
        return result


def _validate_positive_integer(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
