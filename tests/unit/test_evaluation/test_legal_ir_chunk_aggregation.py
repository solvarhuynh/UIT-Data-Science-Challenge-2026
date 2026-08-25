"""Regression tests for raw-chunk document aggregation."""

from __future__ import annotations

import pytest

from udsc2026.evaluation.legal_ir_chunk_aggregation import (
    aggregate_chunks,
    top_document_ids,
)


def _hit(
    doc: str, chunk: str, rank: int, dense: float, bge: float
) -> dict[str, object]:
    return {
        "doc_id": doc,
        "chunk_id": chunk,
        "dense_rank": rank,
        "dense_score": dense,
        "bge_score": bge,
        "article": f"article-{chunk}",
    }


def test_multiple_chunks_support_one_document_and_keep_provenance() -> None:
    ranking = aggregate_chunks(
        [
            _hit("single", "s1", 1, 0.9, 0.80),
            _hit("multi", "m1", 2, 0.8, 0.75),
            _hit("multi", "m2", 3, 0.7, 0.74),
        ],
        method="bge_logsumexp_top2",
    )

    assert ranking[0].doc_id == "multi"
    assert ranking[0].used_chunk_ids == ("m1", "m2")
    assert [item.chunk_id for item in ranking[0].supporting_chunks] == ["m1", "m2"]
    assert ranking[0].supporting_chunks[0].provenance["article"] == "article-m1"


def test_rank_cap_limits_each_document_contribution() -> None:
    hits = [
        _hit("a", "a1", 1, 0.9, 0.9),
        _hit("a", "a2", 2, 0.8, 0.8),
        _hit("b", "b1", 3, 0.7, 0.7),
    ]

    cap1 = aggregate_chunks(hits, method="rank_cap1_k0")
    cap2 = aggregate_chunks(hits, method="rank_cap2_k0")

    assert cap1[0].used_chunk_ids == ("a1",)
    assert cap2[0].used_chunk_ids == ("a1", "a2")
    assert cap2[0].score > cap1[0].score


def test_ties_are_stable_across_input_order() -> None:
    hits = [
        _hit("b", "b1", 1, 0.5, 0.5),
        _hit("a", "a1", 1, 0.5, 0.5),
    ]

    forward = top_document_ids(hits, method="bge_max")
    reverse = top_document_ids(reversed(hits), method="bge_max")

    assert forward == reverse == ["a", "b"]


def test_top5_contains_five_distinct_documents() -> None:
    hits = [
        _hit(f"d{index}", f"c{index}", index, 1 / index, 1 / index)
        for index in range(1, 8)
    ]
    hits.insert(1, _hit("d1", "c1b", 8, 0.01, 2.0))

    result = top_document_ids(hits, method="bge_max", limit=5)

    assert len(result) == len(set(result)) == 5


def test_dense_aggregation_does_not_require_bge_scores() -> None:
    hits = [
        {"doc_id": "a", "chunk_id": "a1", "rank": 1, "dense_score": 0.5},
        {"doc_id": "b", "chunk_id": "b1", "rank": 2, "dense_score": 0.4},
    ]

    assert top_document_ids(hits, method="dense_mean_top3") == ["a", "b"]
    with pytest.raises(ValueError, match="no BGE score"):
        top_document_ids(hits, method="bge_max")
