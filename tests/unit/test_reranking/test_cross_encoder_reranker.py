"""Unit tests for citation-preserving cross-encoder reranking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.retrieval.reranking import CrossEncoderReranker


class FakeRerankerClient:
    """Deterministic injectable scorer with no model or network dependency."""

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.calls: list[tuple[str, tuple[str, ...], int | None]] = []

    def score(
        self,
        query: str,
        documents: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[float]:
        self.calls.append((query, tuple(documents), batch_size))
        return list(self.scores)


@pytest.fixture
def candidates() -> list[RetrievalHit]:
    return [
        RetrievalHit(
            chunk_id="chunk-1",
            doc_id="doc-1",
            text="Điều 35 quy định về quyền đơn phương chấm dứt hợp đồng.",
            score=0.81,
            source="bo-luat-lao-dong.pdf",
            law_name="Bộ luật Lao động 2019",
            article="Điều 35",
            clause="Khoản 1",
            metadata={
                "point": "a",
                "parent_id": "article-35",
                "citation": {"page": 21},
            },
            dense_score=0.82,
            sparse_score=0.72,
            hybrid_score=0.77,
            final_score=0.77,
            rank=1,
        ),
        RetrievalHit(
            chunk_id="chunk-2",
            doc_id="doc-1",
            text="Điều 40 quy định hậu quả chấm dứt hợp đồng trái pháp luật.",
            score=0.75,
            source="bo-luat-lao-dong.pdf",
            law_name="Bộ luật Lao động 2019",
            article="Điều 40",
            metadata={"parent_id": "article-40", "citation": {"page": 24}},
            dense_score=0.61,
            sparse_score=0.89,
            hybrid_score=0.75,
            final_score=0.75,
            rank=2,
        ),
        RetrievalHit(
            chunk_id="chunk-3",
            doc_id="doc-2",
            text="Nghị định hướng dẫn thời hạn báo trước.",
            score=0.63,
            source="nghi-dinh.pdf",
            law_name="Nghị định hướng dẫn",
            article="Điều 7",
            clause="Khoản 2",
            metadata={"point": "b", "parent_id": "nd-article-7"},
            dense_score=0.70,
            sparse_score=0.51,
            hybrid_score=0.605,
            final_score=0.605,
            rank=3,
        ),
    ]


@pytest.mark.unit
def test_reranks_and_limits_results(candidates: list[RetrievalHit]) -> None:
    client = FakeRerankerClient([0.2, 0.95, 0.6])
    reranker = CrossEncoderReranker(client)

    result = reranker.rerank("Chấm dứt hợp đồng trái luật?", candidates, 2)

    assert [hit.chunk_id for hit in result] == ["chunk-2", "chunk-3"]
    assert [hit.rerank_score for hit in result] == [0.95, 0.6]
    assert [hit.final_score for hit in result] == [0.95, 0.6]
    assert [hit.rank for hit in result] == [1, 2]
    assert client.calls == [
        (
            "Chấm dứt hợp đồng trái luật?",
            tuple(hit.text for hit in candidates),
            None,
        )
    ]


@pytest.mark.unit
def test_preserves_scores_citations_and_nested_metadata(
    candidates: list[RetrievalHit],
) -> None:
    original_dump = [candidate.model_dump() for candidate in candidates]
    reranker = CrossEncoderReranker(FakeRerankerClient([0.9, 0.8, 0.7]))

    result = reranker.rerank("Khi nào được chấm dứt hợp đồng?", candidates, 3)

    for original, reranked in zip(candidates, result):
        assert reranked is not original
        assert reranked.metadata is not original.metadata
        assert reranked.metadata == original.metadata
        assert reranked.chunk_id == original.chunk_id
        assert reranked.doc_id == original.doc_id
        assert reranked.source == original.source
        assert reranked.law_name == original.law_name
        assert reranked.article == original.article
        assert reranked.clause == original.clause
        assert reranked.score == original.score
        assert reranked.dense_score == original.dense_score
        assert reranked.sparse_score == original.sparse_score
        assert reranked.hybrid_score == original.hybrid_score

    result[0].metadata["citation"]["page"] = 999
    assert candidates[0].metadata["citation"]["page"] == 21
    assert [candidate.model_dump() for candidate in candidates] == original_dump


@pytest.mark.unit
def test_equal_scores_keep_input_order(candidates: list[RetrievalHit]) -> None:
    reranker = CrossEncoderReranker(FakeRerankerClient([0.5, 0.5, 0.5]))

    first = reranker.rerank("câu hỏi", candidates, 3)
    second = reranker.rerank("câu hỏi", candidates, 3)

    expected_ids = ["chunk-1", "chunk-2", "chunk-3"]
    assert [hit.chunk_id for hit in first] == expected_ids
    assert [hit.chunk_id for hit in second] == expected_ids


@pytest.mark.unit
def test_top_n_larger_than_candidates_returns_all(
    candidates: list[RetrievalHit],
) -> None:
    reranker = CrossEncoderReranker(FakeRerankerClient([0.1, 0.2, 0.3]))

    result = reranker.rerank("câu hỏi", candidates, 99)

    assert len(result) == 3
    assert [hit.rank for hit in result] == [1, 2, 3]


@pytest.mark.unit
def test_empty_candidates_skip_client() -> None:
    client = FakeRerankerClient([])
    reranker = CrossEncoderReranker(client)

    assert reranker.rerank("câu hỏi hợp lệ", [], 5) == []
    assert client.calls == []


@pytest.mark.unit
@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_rejects_empty_query(
    query: str,
    candidates: list[RetrievalHit],
) -> None:
    reranker = CrossEncoderReranker(FakeRerankerClient([1.0] * 3))

    with pytest.raises(ValueError, match="query"):
        reranker.rerank(query, candidates, 3)


@pytest.mark.unit
@pytest.mark.parametrize("top_n", [0, -1, True, 1.5, "2"])
def test_rejects_invalid_top_n(
    top_n: Any,
    candidates: list[RetrievalHit],
) -> None:
    reranker = CrossEncoderReranker(FakeRerankerClient([1.0] * 3))

    with pytest.raises(ValueError, match="top_n"):
        reranker.rerank("câu hỏi", candidates, top_n)


@pytest.mark.unit
def test_rejects_non_list_candidates(
    candidates: list[RetrievalHit],
) -> None:
    reranker = CrossEncoderReranker(FakeRerankerClient([1.0] * 3))

    with pytest.raises(TypeError, match="list"):
        reranker.rerank("câu hỏi", tuple(candidates), 3)  # type: ignore[arg-type]


@pytest.mark.unit
def test_rejects_invalid_candidate_item() -> None:
    reranker = CrossEncoderReranker(FakeRerankerClient([1.0]))

    with pytest.raises(TypeError, match=r"candidates\[0\]"):
        reranker.rerank("câu hỏi", ["not a hit"], 1)  # type: ignore[list-item]


@pytest.mark.unit
def test_rejects_blank_candidate_text() -> None:
    blank_hit = RetrievalHit.model_construct(
        chunk_id="chunk",
        doc_id="doc",
        text=" ",
        metadata={},
    )
    reranker = CrossEncoderReranker(FakeRerankerClient([1.0]))

    with pytest.raises(ValueError, match=r"candidates\[0\]\.text"):
        reranker.rerank("câu hỏi", [blank_hit], 1)


@pytest.mark.unit
def test_rejects_duplicate_candidate_chunk_ids(
    candidates: list[RetrievalHit],
) -> None:
    reranker = CrossEncoderReranker(FakeRerankerClient([1.0, 0.5]))

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        reranker.rerank(
            "câu hỏi",
            [candidates[0], candidates[0].model_copy()],
            2,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("scores", "exception", "message"),
    [
        ([0.1], ValueError, "1 score"),
        ([0.1, 0.2, float("nan")], ValueError, "finite"),
        ([0.1, 0.2, True], TypeError, "real number"),
    ],
)
def test_rejects_invalid_client_scores(
    scores: list[Any],
    exception: type[Exception],
    message: str,
    candidates: list[RetrievalHit],
) -> None:
    reranker = CrossEncoderReranker(FakeRerankerClient(scores))

    with pytest.raises(exception, match=message):
        reranker.rerank("câu hỏi", candidates, 3)


@pytest.mark.unit
def test_rejects_client_without_score_method() -> None:
    with pytest.raises(TypeError, match="RerankerClient"):
        CrossEncoderReranker(object())  # type: ignore[arg-type]
