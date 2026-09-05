"""Tests for P15 continuation/preference preparation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "prepare_task2_p15_adaptation.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_task2_p15", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_keeps_gate_disjoint_and_builds_preferences() -> None:
    script = _load()
    questions = {
        str(index): {"question": f"Question {index}?", "answer": f"Answer {index}"}
        for index in range(20)
    }
    labels = [
        {
            "question_id": str(index),
            "parent_text": f"Context {index}",
            "reference_overlap": 0.8,
            "candidate_rank": 1,
            "doc_id": "d",
            "parent_id": str(index),
        }
        for index in range(20)
    ]
    baseline = [
        {"id": str(index), "answer": f"Rejected {index}"} for index in range(20)
    ]

    outputs, report = script.prepare(
        questions,
        list(questions),
        labels,
        baseline,
        contexts_per_question=1,
        dev_fraction=0.25,
        seed=2026,
    )

    train_ids = {row["id"] for row in outputs["adaptation_train"]}
    dev_ids = {row["id"] for row in outputs["adaptation_dev"]}
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids | dev_ids == set(questions)
    assert report["train_dev_overlap"] == 0
    assert outputs["preference_train"][0]["chosen"].startswith("Answer")
    assert outputs["preference_train"][0]["rejected"].startswith("Rejected")
