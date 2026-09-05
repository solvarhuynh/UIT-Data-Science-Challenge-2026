"""Contracts for the label-free Task 2 Qwen/extractive ensemble."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path("scripts/submission/build_legal_qa_qwen_extractive_ensemble.py")


def _load() -> object:
    spec = importlib.util.spec_from_file_location("task2_ensemble_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ensemble_prefixes_evidence_and_preserves_exact_overlay() -> None:
    module = _load()
    qwen = [
        {"id": "q1", "answer": "generated one"},
        {"id": "q2", "answer": "known exact"},
    ]
    extractive = [
        {"id": "q1", "answer": "a b c d"},
        {"id": "q2", "answer": "must not be prepended"},
    ]

    output, counts = module.build_ensemble(
        qwen,
        extractive,
        evidence_words=3,
        exact_overlay_ids={"q2"},
    )

    assert output == [
        {"id": "q1", "answer": "a b c\ngenerated one"},
        {"id": "q2", "answer": "known exact"},
    ]
    assert counts == {
        "question_count": 2,
        "exact_overlays_preserved": 1,
        "extractive_prefixes_truncated": 1,
        "evidence_words": 3,
    }


def test_ensemble_rejects_misaligned_predictions() -> None:
    module = _load()
    with pytest.raises(ValueError, match="IDs/order differ"):
        module.build_ensemble(
            [{"id": "q1", "answer": "a"}],
            [{"id": "q2", "answer": "b"}],
            evidence_words=3,
        )
