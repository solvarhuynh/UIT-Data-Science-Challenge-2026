"""Model-free tests for P15 global profile selection/application."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluation" / "select_task2_p15_profile.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("select_task2_p15_profile", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_profile_is_reference_free_and_preserves_order() -> None:
    script = _load()
    questions = {
        "q2": {"question": "Second?", "answer": "must not be read"},
        "q1": {"question": "First?", "answer": "must not be read"},
    }

    rows = script.apply_profile(
        questions,
        {"q1": "Qwen one", "q2": "Qwen two"},
        {"q1": "Evidence one", "q2": "Evidence two"},
        "qwen",
    )

    assert rows == [
        {"id": "q2", "answer": "Qwen two"},
        {"id": "q1", "answer": "Qwen one"},
    ]
