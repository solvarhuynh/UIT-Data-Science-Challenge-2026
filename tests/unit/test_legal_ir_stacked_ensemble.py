"""Focused contracts for the leakage-safe LegalIR stacked ensemble."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "submission" / "build_legal_ir_stacked_ensemble.py"
SPEC = importlib.util.spec_from_file_location("legal_ir_stacked_ensemble", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _FixedModel:
    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = np.asarray(probabilities, dtype=np.float64)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        assert len(features) == len(self._probabilities)
        return np.column_stack((1.0 - self._probabilities, self._probabilities))


def _question(question_id: str, text: str, *documents: str):
    return MODULE.Question(question_id, text, documents)


def test_score_loader_streams_and_aggregates_max_per_document(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    rows = [
        {
            "question_id": "q1",
            "hits": [
                {"doc_id": "d1", "dense_score": 0.2, "rerank_score": 0.7},
                {"doc_id": "d1", "dense_score": 0.5, "rerank_score": 0.6},
                {"doc_id": "d2", "dense_score": 0.1, "rerank_score": 0.9},
            ],
        }
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    scores = MODULE.load_document_scores(
        path, expected_ids=["q1"], allowed_document_ids={"d1", "d2"}
    )

    assert scores == {"q1": {"d1": (0.5, 0.7), "d2": (0.1, 0.9)}}


def test_score_loader_rejects_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"question_id":"q1","hits":['
        '{"doc_id":"d1","dense_score":NaN,"rerank_score":0.5}]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-standard JSON constant"):
        MODULE.load_document_scores(
            path, expected_ids=["q1"], allowed_document_ids={"d1"}
        )


def test_semantic_neighbors_exclude_complete_validation_group() -> None:
    training = {
        "q1": _question("q1", "Cùng một câu?", "leaked"),
        "q1-copy": _question("q1-copy", "cùng một câu", "also-leaked"),
        "q2": _question("q2", "Câu khác", "safe"),
    }
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    excluded = {MODULE.normalize_legal_ir_matching_question("Cùng một câu")}

    rankings = MODULE.semantic_document_rankings(
        queries={"q1": training["q1"]},
        query_embeddings=embeddings,
        query_embedding_ids=["q1", "q1-copy", "q2"],
        training=training,
        training_embeddings=embeddings,
        training_embedding_ids=["q1", "q1-copy", "q2"],
        excluded_normalized=excluded,
    )

    assert rankings["q1"].documents == ("safe",)


def test_rank_blend_uses_within_query_ranks_and_returns_five_unique_documents() -> None:
    documents = (("a", "b", "c", "d", "e", "f"),)
    features = np.zeros((6, 79), dtype=np.float32)
    batch = MODULE.FeatureBatch(
        features=features,
        labels=np.zeros(6, dtype=np.uint8),
        query_ids=("q",),
        query_documents=documents,
        query_slices=((0, 6),),
        fallback_scores=np.zeros(6, dtype=np.float32),
    )
    base = _FixedModel([0.1, 0.9, 0.8, 0.7, 0.6, 0.5])
    scored = _FixedModel([0.99, 0.1, 0.2, 0.3, 0.4, 0.5])

    predictions = MODULE.rank_blended_feature_batch(
        base_model=base,
        scored_model=scored,
        base_batch=batch,
        scored_batch=batch,
        base_weight=0.75,
    )

    assert predictions == [{"id": "q", "documents": ["b", "c", "d", "e", "f"]}]
    assert len(set(predictions[0]["documents"])) == 5


def test_exact_overlay_uses_external_labels_without_duplicate_documents() -> None:
    target = {"public": _question("public", "Mức phạt là bao nhiêu?")}
    labels = {"warm": _question("warm", "mức phạt là bao nhiêu", "gold")}
    predictions: list[dict[str, object]] = [
        {"id": "public", "documents": ["a", "b", "gold", "c", "d"]}
    ]

    count = MODULE.apply_exact_overlay(predictions, target, [labels])

    assert count == 1
    assert predictions[0]["documents"] == ["gold", "a", "b", "c", "d"]


def test_document_score_mask_does_not_mutate_original_batch() -> None:
    features = np.ones((2, 79), dtype=np.float32)
    batch = MODULE.FeatureBatch(
        features=features,
        labels=np.asarray([0, 1], dtype=np.uint8),
        query_ids=("q",),
        query_documents=(("a", "b"),),
        query_slices=((0, 2),),
        fallback_scores=np.zeros(2, dtype=np.float32),
    )

    masked = MODULE.without_document_score_features(batch)

    assert np.all(batch.features == 1.0)
    assert np.all(masked.features[:, MODULE.DOCUMENT_SCORE_FEATURE_SLICE] == 0.0)
