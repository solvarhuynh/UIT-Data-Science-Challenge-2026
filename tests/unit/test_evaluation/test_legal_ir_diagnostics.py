"""Tests for deep candidate diagnostics kept separate from official scoring."""

from __future__ import annotations

import pytest

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.legal_ir import LegalIRReference
from udsc2026.evaluation.legal_ir_diagnostics import (
    document_ids_from_hits,
    evaluate_legal_ir_document_diagnostics,
)
from udsc2026.evaluation.models import PredictionSample


def _prediction(question_id: str, document_ids: list[str]) -> PredictionSample:
    return PredictionSample(
        question_id=question_id,
        hits=[
            RetrievalHit(
                chunk_id=f"{question_id}-chunk-{index}",
                doc_id=document_id,
                text=f"text {index}",
                rank=index,
            )
            for index, document_id in enumerate(document_ids, start=1)
        ],
    )


def test_document_diagnostics_report_official_and_deep_candidate_metrics() -> None:
    references = [
        LegalIRReference(id="q1", gold_documents=["gold-1"]),
        LegalIRReference(id="q2", gold_documents=["gold-2", "gold-2b"]),
    ]
    predictions = [
        _prediction("q1", ["a", "b", "c", "d", "e", "gold-1"]),
        _prediction("q2", ["gold-2", "gold-2b", "x"]),
    ]

    report = evaluate_legal_ir_document_diagnostics(references, predictions)

    # q1's gold appears in the deep pool but is outside the final top five;
    # candidate metrics retain that signal while the official result does not.
    assert report["official_recall"] == pytest.approx(0.5)
    assert report["official_precision"] == pytest.approx(1 / 3)
    assert report["candidate_doc_recall_at_5"] == pytest.approx(0.5)
    assert report["candidate_doc_recall_at_10"] == 1.0
    assert report["mean_unique_docs_at_chunk_depth_5"] == pytest.approx(4.0)
    assert report["gold_absent_from_candidate_count"] == 0
    assert report["gold_present_but_final_dropped_count"] == 1
    assert report["multi_gold_recall"] == 1.0
    assert set(report["candidate_doc_recall_at_k"]) >= {
        "5",
        "10",
        "20",
        "50",
        "100",
        "200",
    }


def test_document_diagnostics_can_score_shallow_final_ranking_against_deep_pool() -> (
    None
):
    references = [LegalIRReference(id="q", gold_documents=["gold"])]
    candidates = [_prediction("q", ["a", "gold"])]
    final = [_prediction("q", ["gold"])]

    report = evaluate_legal_ir_document_diagnostics(
        references,
        final,
        candidate_predictions=candidates,
    )

    assert report["official_recall"] == 1.0
    assert report["candidate_doc_recall_at_5"] == 1.0


def test_document_diagnostics_reject_coverage_mismatch() -> None:
    with pytest.raises(ValueError, match="missing IDs: q"):
        evaluate_legal_ir_document_diagnostics(
            [LegalIRReference(id="q", gold_documents=["gold"])],
            [],
        )


def test_document_collapse_keeps_best_unique_document_order() -> None:
    prediction = _prediction("q", ["d1", "d1", "d2", "d3"])
    assert document_ids_from_hits(prediction, chunk_depth=3) == ["d1", "d2"]
    assert document_ids_from_hits(prediction, max_documents=2) == ["d1", "d2"]
