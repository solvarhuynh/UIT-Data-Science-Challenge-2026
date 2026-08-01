"""Determinism and boundary tests for dense and hybrid retrieval."""

from typing import Any

import pytest

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.retrieval.dense.dense_retriever import DenseRetriever
from udsc2026.retrieval.hybrid.hybrid_retriever import HybridRetriever
from udsc2026.retrieval.hybrid.score_fusion import fuse_scores, normalize_scores


def _hit(
    chunk_id: str,
    *,
    dense_score: float | None = None,
    sparse_score: float | None = None,
    text: str | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text or f"Text {chunk_id}",
        dense_score=dense_score,
        sparse_score=sparse_score,
        metadata={"citation": chunk_id},
    )


class _FakeEmbedding:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1, 0.2]


class _FakeVectorDB:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[list[float], int, object]] = []

    def search(
        self,
        vector: list[float],
        top_k: int,
        filters: object,
    ) -> list[RetrievalHit]:
        self.calls.append((vector, top_k, filters))
        return self.hits[:top_k]


class _FakeRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, object]] = []

    def search(
        self,
        query: str,
        top_k: int,
        filters: object,
    ) -> list[RetrievalHit]:
        self.calls.append((query, top_k, filters))
        return list(self.hits)


def test_dense_retriever_ranks_copied_hits() -> None:
    embedding = _FakeEmbedding()
    vector_db = _FakeVectorDB([_hit("a", dense_score=0.9), _hit("b")])
    retriever = DenseRetriever(embedding, vector_db)  # type: ignore[arg-type]

    result = retriever.search("quyền lao động", 2, {"article": "Điều 35"})

    assert [hit.rank for hit in result] == [1, 2]
    assert result[0].dense_score == 0.9
    assert result[1].dense_score is None
    assert result[0] is not vector_db.hits[0]
    assert embedding.queries == ["quyền lao động"]
    assert vector_db.calls == [
        ([0.1, 0.2], 2, {"article": "Điều 35"}),
    ]


@pytest.mark.parametrize(("query", "top_k"), [("", 1), ("valid", 0), ("valid", True)])
def test_dense_retriever_rejects_invalid_inputs(query: str, top_k: Any) -> None:
    retriever = DenseRetriever(  # type: ignore[arg-type]
        _FakeEmbedding(),
        _FakeVectorDB([]),
    )
    with pytest.raises(ValueError):
        retriever.search(query, top_k)


def test_fusion_is_deterministic_for_equal_scores_and_preserves_metadata() -> None:
    dense = [_hit("dense-first", dense_score=5), _hit("both", dense_score=5)]
    sparse = [_hit("both", sparse_score=7), _hit("sparse-only", sparse_score=7)]

    first = fuse_scores(dense, sparse)
    second = fuse_scores(dense, sparse)

    assert [hit.chunk_id for hit in first] == [
        "both",
        "dense-first",
        "sparse-only",
    ]
    assert [hit.model_dump() for hit in second] == [hit.model_dump() for hit in first]
    assert [hit.rank for hit in first] == [1, 2, 3]
    assert first[0].metadata == {"citation": "both"}


def test_fusion_rejects_duplicates_conflicts_and_invalid_weights() -> None:
    duplicate = _hit("same", dense_score=1)
    with pytest.raises(ValueError, match="duplicate"):
        fuse_scores([duplicate, duplicate.model_copy()], [])

    dense = _hit("same", dense_score=1, text="dense")
    sparse = RetrievalHit(
        chunk_id="same",
        doc_id=dense.doc_id,
        text="sparse",
        sparse_score=1,
    )
    with pytest.raises(ValueError, match="conflicting text"):
        fuse_scores([dense], [sparse])

    dense_metadata = _hit("metadata", dense_score=1)
    sparse_metadata = _hit("metadata", sparse_score=1).model_copy(
        update={"metadata": {"citation": "different"}}
    )
    with pytest.raises(ValueError, match="conflicting metadata"):
        fuse_scores([dense_metadata], [sparse_metadata])

    with pytest.raises(ValueError, match="sum to 1"):
        fuse_scores([], [], dense_weight=0.8, sparse_weight=0.8)
    with pytest.raises(ValueError, match="finite"):
        fuse_scores([], [], dense_weight=float("nan"), sparse_weight=0.5)


def test_normalize_scores_handles_equal_and_missing_values() -> None:
    normalized = normalize_scores(
        [
            _hit("a", dense_score=3),
            _hit("b", dense_score=3),
            _hit("c", dense_score=None),
        ],
        "dense_score",
    )

    assert [hit.dense_score for hit in normalized] == [1.0, 1.0, None]
    zero_evidence = normalize_scores(
        [_hit("zero-a", sparse_score=0), _hit("zero-b", sparse_score=0)],
        "sparse_score",
    )
    assert [hit.sparse_score for hit in zero_evidence] == [0.0, 0.0]
    with pytest.raises(ValueError, match="score_field"):
        normalize_scores([], "final_score")


def test_hybrid_retriever_uses_candidate_pool_filters_and_threshold() -> None:
    dense = _FakeRetriever([_hit("a", dense_score=2), _hit("b", dense_score=1)])
    sparse = _FakeRetriever([_hit("b", sparse_score=2), _hit("a", sparse_score=1)])
    retriever = HybridRetriever(  # type: ignore[arg-type]
        dense,
        sparse,
        dense_weight=0.5,
        sparse_weight=0.5,
        candidate_k=7,
        min_score=0.5,
    )

    result = retriever.search("hợp đồng", 3, {"law_name": "Bộ luật Lao động"})

    assert [hit.chunk_id for hit in result] == ["a", "b"]
    assert dense.calls == [
        ("hợp đồng", 7, {"law_name": "Bộ luật Lao động"}),
    ]
    assert sparse.calls == dense.calls


def test_fully_injected_hybrid_does_not_require_project_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "UDSC2026_CONFIG_PATH",
        "definitely-missing-config.yaml",
    )

    retriever = HybridRetriever(  # type: ignore[arg-type]
        _FakeRetriever([]),
        _FakeRetriever([]),
        dense_weight=0.5,
        sparse_weight=0.5,
        candidate_k=5,
        min_score=0.0,
    )

    assert retriever.search("query", top_k=1) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate_k": 0},
        {"candidate_k": True},
        {"min_score": float("inf")},
        {"min_score": True},
    ],
)
def test_hybrid_retriever_rejects_invalid_configuration(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(  # type: ignore[arg-type]
            _FakeRetriever([]),
            _FakeRetriever([]),
            dense_weight=0.5,
            sparse_weight=0.5,
            **kwargs,
        )
