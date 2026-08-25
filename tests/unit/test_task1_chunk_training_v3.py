"""Tests for inference-matched Task1 fine-tuning V3 samples."""

from __future__ import annotations

from udsc2026.training.task1_chunk_training_v3 import RankBands, build_training_record


def _hit(doc: str, rank: int, law: str = "") -> dict[str, object]:
    return {
        "doc_id": doc,
        "chunk_id": f"{doc}-{rank}",
        "text": f"raw chunk {doc} {rank}",
        "rank": rank,
        "score": 1 / rank,
        "law_name": law,
    }


def test_builder_uses_retrieved_gold_chunks_and_balances_rank_bands() -> None:
    hits = [
        _hit("gold", 1, "law-a"),
        _hit("hard", 2),
        _hit("medium", 30),
        _hit("same-law", 31, "law-a"),
        _hit("easy", 120),
    ]

    record = build_training_record(
        query_id="q",
        question="question",
        gold_documents=["gold"],
        raw_hits=hits,
        bands=RankBands(20, 100),
        negatives_per_band=1,
    )

    assert record is not None
    assert record["positive_chunks"][0]["chunk_id"] == "gold-1"
    assert set(record["negative_chunks"]) == {"hard", "medium", "easy", "same_law"}
    assert all(len(values) <= 1 for values in record["negative_chunks"].values())
    assert record["negative_chunks"]["same_law"][0]["doc_id"] == "same-law"


def test_builder_skips_query_without_retrieved_positive() -> None:
    assert (
        build_training_record(
            query_id="q",
            question="question",
            gold_documents=["missing"],
            raw_hits=[_hit("negative", 1)],
        )
        is None
    )
