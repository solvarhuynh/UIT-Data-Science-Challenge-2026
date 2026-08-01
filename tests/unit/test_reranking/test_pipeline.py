"""Tests for the production hybrid-to-reranker orchestration."""

from __future__ import annotations

from typing import Any

import pytest

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.retrieval.reranking import RerankedRetriever


def _hit(chunk_id: str) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=f"Legal text {chunk_id}",
        hybrid_score=0.5,
    )


class FakeHybridRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, object]] = []

    def search(
        self,
        query: str,
        top_k: int,
        filters: object = None,
    ) -> list[RetrievalHit]:
        self.calls.append((query, top_k, filters))
        return list(self.hits[:top_k])


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        self.calls.append((query, [hit.chunk_id for hit in candidates], top_n))
        return [
            hit.model_copy(
                update={
                    "rerank_score": float(index),
                    "final_score": float(index),
                    "rank": rank,
                }
            )
            for rank, (index, hit) in enumerate(
                reversed(list(enumerate(candidates))),
                start=1,
            )
        ][:top_n]


def test_fetches_candidate_pool_and_honors_requested_final_count() -> None:
    hybrid = FakeHybridRetriever([_hit("a"), _hit("b"), _hit("c")])
    reranker = FakeReranker()
    pipeline = RerankedRetriever(
        hybrid,
        reranker,
        candidate_k=3,
        top_n=2,
    )
    filters = {"law_name": "Bộ luật Lao động"}

    result = pipeline.search("chấm dứt hợp đồng", 1, filters)

    assert [hit.chunk_id for hit in result] == ["c"]
    assert [hit.rank for hit in result] == [1]
    assert hybrid.calls == [("chấm dứt hợp đồng", 3, filters)]
    assert reranker.calls == [
        ("chấm dứt hợp đồng", ["a", "b", "c"], 1),
    ]


def test_top_n_caps_large_caller_request() -> None:
    hybrid = FakeHybridRetriever([_hit("a"), _hit("b"), _hit("c")])
    reranker = FakeReranker()
    pipeline = RerankedRetriever(
        hybrid,
        reranker,
        candidate_k=3,
        top_n=2,
    )

    result = pipeline.search("valid query", 99)

    assert len(result) == 2
    assert reranker.calls[0][2] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate_k": 0, "top_n": 1},
        {"candidate_k": 2, "top_n": 3},
        {"candidate_k": True, "top_n": 1},
        {"candidate_k": 2, "top_n": 0},
    ],
)
def test_rejects_invalid_configuration(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RerankedRetriever(
            FakeHybridRetriever([]),
            FakeReranker(),
            **kwargs,
        )


@pytest.mark.parametrize(
    ("query", "top_k"),
    [
        ("", 1),
        (" ", 1),
        ("valid", 0),
        ("valid", True),
    ],
)
def test_rejects_invalid_search_inputs(query: str, top_k: Any) -> None:
    pipeline = RerankedRetriever(
        FakeHybridRetriever([]),
        FakeReranker(),
        candidate_k=2,
        top_n=1,
    )

    with pytest.raises(ValueError):
        pipeline.search(query, top_k)
