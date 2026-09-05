"""Tests for the verified P18 preference bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/cloud/package_task2_p18_modal.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("package_p18_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, ids: list[str]) -> None:
    rows = [
        {
            "id": question_id,
            "question": "Câu hỏi",
            "contexts": [{"text": "Căn cứ"}],
            "chosen": "Đáp án đúng",
            "rejected": "Đáp án kém",
        }
        for question_id in ids
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_package_writes_verified_preference_sets(tmp_path: Path) -> None:
    script = _load()
    train = tmp_path / "train.jsonl"
    all_data = tmp_path / "all.jsonl"
    _write(train, ["q1", "q2"])
    _write(all_data, ["q1", "q2", "q3"])
    output = tmp_path / "p18.zip"

    archive, manifest_path = script.package(
        argparse.Namespace(
            preference_train=train,
            preference_all=all_data,
            output=output,
        )
    )

    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {
            "preferences/train.jsonl",
            "preferences/all.jsonl",
        }
        assert bundle.testzip() is None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "task2-p18-modal-input-v1"
    assert [row["records"] for row in manifest["members"]] == [2, 3]
    assert manifest["train_ids_are_strict_subset"] is True
