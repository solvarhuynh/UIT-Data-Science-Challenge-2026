"""Tests for P15 source-only synthetic rankings."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "submission"
    / "build_legal_qa_rankings_from_extractive.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p15_extractive_rankings", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rankings_preserve_requested_order_and_text() -> None:
    script = _load()
    rows = script.build_rankings(
        {"q1": {}, "q2": {}},
        [
            {"id": "q1", "answer": "Evidence one"},
            {"id": "q2", "answer": "Evidence two"},
        ],
        ["q2", "q1"],
    )

    assert [row["question_id"] for row in rows] == ["q2", "q1"]
    assert rows[0]["parents"][0]["parent_text"] == "Evidence two"
    assert rows[0]["parents"][0]["rank"] == 1


def test_submission_prefix_recovery_is_bounded(tmp_path: Path) -> None:
    script = _load()
    archive = tmp_path / "submission.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "submission.json",
            json.dumps({"q": {"answer": "one two three four"}}),
        )

    rows = script._load_submission_prefixes(archive, 3)

    assert rows == [{"id": "q", "answer": "one two three"}]
