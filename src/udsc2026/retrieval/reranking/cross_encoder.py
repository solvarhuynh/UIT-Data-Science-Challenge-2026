"""Citation-preserving reranking of shared ``RetrievalHit`` contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.infrastructure.reranker import RerankerClient


class CrossEncoderReranker:
    """Reorder retrieval candidates using an injected relevance scorer.

    The cross-encoder score becomes both ``rerank_score`` and ``final_score``.
    Dense, sparse, hybrid and legacy ``score`` values remain untouched.
    Equal reranker scores retain candidate input order for deterministic output.
    """

    def __init__(self, client: RerankerClient) -> None:
        """Initialize reranking with a compatible scoring client."""
        if not isinstance(client, RerankerClient):
            raise TypeError(
                "client must implement RerankerClient.score(query, documents)"
            )
        self._client = client

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        """Return the highest-scoring copied candidates with one-based ranks."""

        _validate_query(query)
        _validate_top_n(top_n)
        _validate_candidates(candidates)
        if not candidates:
            return []

        scores = self._client.score(
            query,
            tuple(candidate.text for candidate in candidates),
        )
        validated_scores = _validate_scores(scores, len(candidates))

        indexed_candidates = list(enumerate(zip(candidates, validated_scores)))
        indexed_candidates.sort(key=lambda item: (-item[1][1], item[0]))

        output: list[RetrievalHit] = []
        for rank, (_, (candidate, score)) in enumerate(
            indexed_candidates[:top_n],
            start=1,
        ):
            output.append(
                candidate.model_copy(
                    deep=True,
                    update={
                        "rerank_score": score,
                        "final_score": score,
                        "rank": rank,
                    },
                )
            )
        return output


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")


def _validate_top_n(top_n: int) -> None:
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")


def _validate_candidates(candidates: list[RetrievalHit]) -> None:
    if not isinstance(candidates, list):
        raise TypeError("candidates must be a list of RetrievalHit objects")
    seen_chunk_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, RetrievalHit):
            raise TypeError(f"candidates[{index}] must be a RetrievalHit object")
        if not isinstance(candidate.text, str) or not candidate.text.strip():
            raise ValueError(f"candidates[{index}].text must be a non-empty string")
        if candidate.chunk_id in seen_chunk_ids:
            raise ValueError(
                f"candidates contains duplicate chunk_id: {candidate.chunk_id}"
            )
        seen_chunk_ids.add(candidate.chunk_id)


def _validate_scores(
    scores: Sequence[object],
    expected_count: int,
) -> list[float]:
    if isinstance(scores, (str, bytes)) or not isinstance(scores, Sequence):
        raise TypeError("reranker client must return a sequence of scores")
    if len(scores) != expected_count:
        raise ValueError(
            "reranker client returned "
            f"{len(scores)} score(s) for {expected_count} candidate(s)"
        )

    validated: list[float] = []
    for index, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError(f"reranker score at index {index} must be a real number")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError(f"reranker score at index {index} must be finite")
        validated.append(numeric_score)
    return validated
