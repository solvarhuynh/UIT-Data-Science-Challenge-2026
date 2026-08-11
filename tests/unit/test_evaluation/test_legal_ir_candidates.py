"""Synthetic tests for LegalIR candidate coverage and document collapse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.evaluation.analyze_legal_ir_candidates import build_parser, run

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.evaluation.legal_ir_candidates import (
    AGGREGATION_METHODS,
    aggregate_document_candidates,
)
from udsc2026.evaluation.models import PredictionSample


def _prediction(question_id: str, documents: list[str]) -> PredictionSample:
    return PredictionSample(
        question_id=question_id,
        hits=[
            RetrievalHit(
                chunk_id=f"{question_id}-{index}",
                doc_id=document,
                text=f"evidence {index}",
                dense_score=1.0 - index / 100,
                score=1.0 - index / 100,
                rank=index,
            )
            for index, document in enumerate(documents, start=1)
        ],
    )


def test_aggregation_preserves_raw_hits_and_exposes_all_requested_methods() -> None:
    prediction = PredictionSample(
        question_id="q",
        hits=[
            RetrievalHit(
                chunk_id="d1-1",
                doc_id="d1",
                text="one",
                dense_score=0.80,
                rank=1,
            ),
            RetrievalHit(
                chunk_id="d2-1",
                doc_id="d2",
                text="two",
                dense_score=0.79,
                rank=2,
            ),
            RetrievalHit(
                chunk_id="d1-2",
                doc_id="d1",
                text="one more",
                dense_score=0.70,
                rank=3,
            ),
            RetrievalHit(
                chunk_id="d3-1",
                doc_id="d3",
                text="three",
                dense_score=0.78,
                rank=4,
            ),
            RetrievalHit(
                chunk_id="d3-2",
                doc_id="d3",
                text="three more",
                dense_score=0.77,
                rank=5,
            ),
        ],
    )
    original_chunks = [hit.chunk_id for hit in prediction.hits]

    results = {
        method: aggregate_document_candidates(prediction, method=method)
        for method in AGGREGATION_METHODS
    }

    assert [item.doc_id for item in results["first_occurrence"]] == ["d1", "d2", "d3"]
    assert [item.doc_id for item in results["max_plus_0.10_second"]] == [
        "d1",
        "d3",
        "d2",
    ]
    assert [item.doc_id for item in results["top2_mean"]] == ["d2", "d3", "d1"]
    first = results["max"][0]
    assert first.doc_id == "d1"
    assert first.first_chunk_rank == 1
    assert first.max_dense_score == pytest.approx(0.80)
    assert first.second_dense_score == pytest.approx(0.70)
    assert first.top_evidence_chunk_ids == ("d1-1", "d1-2")
    assert first.source_ranks == (1, 3)
    assert first.number_of_supporting_chunks == 2
    assert [hit.chunk_id for hit in prediction.hits] == original_chunks


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_candidate_analyzer_writes_coverage_and_all_failure_buckets(
    tmp_path: Path,
) -> None:
    gold = tmp_path / "train.json"
    gold.write_text(
        json.dumps(
            {
                "q1": {"question": "Q1?", "answer": ["g1"]},
                "q2": {"question": "Q2?", "answer": ["g2"]},
                "q3": {"question": "Q3?", "answer": ["g3"]},
                "q4": {"question": "Q4?", "answer": ["g4a", "g4b"]},
                "q5": {"question": "Q5?", "answer": ["g5"]},
            }
        ),
        encoding="utf-8",
    )
    predictions = [
        _prediction("q1", ["g1", "a", "b", "c", "d", "e"]),
        _prediction("q2", ["a", "b", "c", "d", "e", "g2"]),
        _prediction("q3", ["a", "b", "c", "d", "e", "f"]),
        _prediction("q4", ["g4a", "a", "b", "c", "d", "e"]),
        _prediction("q5", ["g5", "a", "b", "c", "d", "e"]),
    ]
    prediction_path = tmp_path / "cached_dense200_predictions.jsonl"
    _write_jsonl(
        prediction_path,
        [item.model_dump(mode="json", exclude_none=True) for item in predictions],
    )
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text('{"cached":true}\n', encoding="utf-8")
    manifest = tmp_path / "cached_dense200_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "candidate_k": 6,
                "question_count": 5,
                "output": str(prediction_path),
                "benchmark": str(benchmark),
                "benchmark_sha256": hashlib.sha256(benchmark.read_bytes()).hexdigest(),
                "model_id": "cached-test-model",
                "model_path": "models/cached-test-model",
                "index_manifest": {"corpus_hash": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    strict_folds = tmp_path / "folds.json"
    strict_folds.write_text(
        json.dumps(
            {
                "schema_version": "legal-ir-strict-cv-v2",
                "folds": [
                    {
                        "fold": index,
                        "validation_ids": [f"q{index + 1}"],
                        "training_ids": [],
                    }
                    for index in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "p2_candidate"
    args = build_parser().parse_args(
        [
            "--gold",
            str(gold),
            "--predictions",
            str(prediction_path),
            "--manifest",
            str(manifest),
            "--strict-folds",
            str(strict_folds),
            "--k",
            "5",
            "6",
            "--output-dir",
            str(output_dir),
        ]
    )

    written = run(args)

    assert [path.name for path in written] == [
        "summary.json",
        "summary.md",
        "failure_cases.jsonl",
    ]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["manifest_validation"]["status"] == "passed"
    candidate_recall = summary["candidate_metrics"]["candidate_doc_recall_at_k"]
    assert candidate_recall["6"] == pytest.approx(0.7)
    assert summary["failure_buckets"] == {
        "candidate_present_final_miss": 1,
        "multi_gold_partial": 1,
        "retrieval_miss": 1,
        "success": 2,
    }
    records = [
        json.loads(line)
        for line in (output_dir / "failure_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_id = {record["id"]: record for record in records}
    assert by_id["q2"]["gold_first_chunk_rank"]["g2"] == 6
    assert by_id["q2"]["gold_first_document_rank"]["g2"] == 6
    assert by_id["q2"]["bucket"] == "candidate_present_final_miss"
    assert by_id["q4"]["bucket"] == "multi_gold_partial"
    assert by_id["q3"]["bucket"] == "retrieval_miss"
