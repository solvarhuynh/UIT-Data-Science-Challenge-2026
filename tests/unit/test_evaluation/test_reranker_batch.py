"""Tests for TV5 batch reranking and prediction artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation import (
    BenchmarkSample,
    PredictionSample,
    load_predictions,
    rerank_prediction_samples,
    write_predictions,
    write_run_manifest,
)
from udsc2026.retrieval.reranking import CrossEncoderReranker


class FixedClient:
    """Return scores derived from document text for deterministic tests."""

    def score(
        self,
        query: str,
        documents: tuple[str, ...],
        *,
        batch_size: int | None = None,
    ) -> list[float]:
        del query, batch_size
        return [float(document.rsplit(" ", 1)[-1]) for document in documents]


def _benchmark(question_id: str, gold: str) -> BenchmarkSample:
    return BenchmarkSample(
        question_id=question_id,
        question=f"question {question_id}",
        answer="reference answer",
        gold_chunk_ids=[gold],
        difficulty="easy",
    )


def _prediction(
    question_id: str,
    chunks: list[tuple[str, float]],
    *,
    latency_ms: float | None,
) -> PredictionSample:
    return PredictionSample(
        question_id=question_id,
        hits=[
            RetrievalHit(
                chunk_id=chunk_id,
                doc_id="law",
                text=f"document {score}",
                hybrid_score=0.9 - index / 10,
            )
            for index, (chunk_id, score) in enumerate(chunks)
        ],
        answer="existing answer",
        latency_ms=latency_ms,
    )


def test_batch_aligns_by_id_truncates_and_preserves_provenance() -> None:
    benchmark = [_benchmark("q1", "gold-1"), _benchmark("q2", "gold-2")]
    predictions = [
        _prediction("q2", [("gold-2", 0.2), ("other-2", 0.8)], latency_ms=None),
        _prediction(
            "q1",
            [("other-1", 0.1), ("gold-1", 0.9), ("discarded", 1.0)],
            latency_ms=5.0,
        ),
    ]
    ticks = iter([1.0, 1.01, 2.0, 2.02])

    result = rerank_prediction_samples(
        benchmark,
        predictions,
        CrossEncoderReranker(FixedClient()),
        candidate_k=2,
        top_n=2,
        clock=lambda: next(ticks),
    )

    assert [sample.question_id for sample in result.before] == ["q1", "q2"]
    assert [hit.chunk_id for hit in result.before[0].hits] == ["other-1", "gold-1"]
    assert [hit.chunk_id for hit in result.after[0].hits] == ["gold-1", "other-1"]
    assert result.after[0].hits[0].hybrid_score == 0.8
    assert result.after[0].hits[0].rerank_score == 0.9
    assert result.after[0].hits[0].final_score == 0.9
    assert result.after[0].hits[0].rank == 1
    assert result.after[0].answer == "existing answer"
    assert result.after[0].latency_ms == pytest.approx(15.0)
    assert result.after[1].latency_ms is None
    assert result.rerank_latencies_ms == pytest.approx([10.0, 20.0])
    assert result.candidate_pair_count == 4


@pytest.mark.parametrize(
    ("candidate_k", "top_n", "message"),
    [(0, 1, "candidate_k"), (2, 0, "top_n"), (1, 2, "must not exceed")],
)
def test_batch_rejects_invalid_limits(
    candidate_k: int,
    top_n: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rerank_prediction_samples(
            [_benchmark("q1", "gold")],
            [_prediction("q1", [("gold", 1.0)], latency_ms=None)],
            CrossEncoderReranker(FixedClient()),
            candidate_k=candidate_k,
            top_n=top_n,
        )


def test_batch_requires_exact_question_coverage() -> None:
    with pytest.raises(ValueError, match="missing IDs: q2"):
        rerank_prediction_samples(
            [_benchmark("q1", "gold-1"), _benchmark("q2", "gold-2")],
            [_prediction("q1", [("gold-1", 1.0)], latency_ms=None)],
            CrossEncoderReranker(FixedClient()),
            candidate_k=1,
            top_n=1,
        )


def test_prediction_and_manifest_writers_are_loadable(tmp_path: Path) -> None:
    samples = [_prediction("q1", [("gold", 1.0)], latency_ms=3.0)]
    json_path = write_predictions(samples, tmp_path / "predictions.json")
    jsonl_path = write_predictions(samples, tmp_path / "predictions.jsonl")
    manifest_path = write_run_manifest(
        {"schema_version": 1, "model": "test"},
        tmp_path / "manifest.json",
    )

    assert load_predictions(json_path) == samples
    assert load_predictions(jsonl_path) == samples
    assert json.loads(manifest_path.read_text("utf-8"))["schema_version"] == 1
