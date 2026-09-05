"""Tests for compact, deterministic P16 Modal input packaging."""

from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/cloud/package_task2_p16_modal.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("package_task2_p16", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compact_row_preserves_union_of_rank_axes() -> None:
    script = _load()
    row = {
        "question_id": "q1",
        "parents": [
            {"rank": 1, "candidate_rank": 9, "parent_text": "rank"},
            {"rank": 9, "candidate_rank": 1, "parent_text": "candidate"},
            {"rank": 9, "candidate_rank": 9, "parent_text": "drop"},
        ],
    }

    compact = script.compact_row(row, rank_limit=2, candidate_limit=2)

    assert [parent["parent_text"] for parent in compact["parents"]] == [
        "rank",
        "candidate",
    ]


def test_package_writes_verified_member_contract(
    tmp_path: Path, monkeypatch: object
) -> None:
    script = _load()
    local_run = tmp_path / "run"
    local_run.mkdir()
    monkeypatch.setattr(script, "P14_RUN", local_run)  # type: ignore[attr-defined]
    rankings = tmp_path / "heldout.jsonl"
    ranking_row = {
        "question_id": "q1",
        "parents": [{"rank": 1, "candidate_rank": 1, "parent_text": "law"}],
    }
    rankings.write_text(json.dumps(ranking_row) + "\n", encoding="utf-8")
    ids = tmp_path / "ids.json"
    ids.write_text('["q1"]', encoding="utf-8")
    predictions = '[{"id":"q1","answer":"answer"}]'
    (local_run / "strict_predictions.json").write_text(
        predictions, encoding="utf-8"
    )
    (local_run / "extractive_predictions.json").write_text(
        predictions, encoding="utf-8"
    )
    p14_result = tmp_path / "p14.zip"
    with zipfile.ZipFile(p14_result, "w") as bundle:
        bundle.writestr(
            f"{script.P14_ZIP_RUN}/public_rankings.jsonl",
            json.dumps(ranking_row) + "\n",
        )
        bundle.writestr(
            f"{script.P14_ZIP_RUN}/public_qwen_predictions.json",
            predictions,
        )
        bundle.writestr(
            f"{script.P14_ZIP_RUN}/public_extractive_predictions.json", predictions
        )
    output = tmp_path / "p16.zip"
    args = argparse.Namespace(
        heldout_rankings=rankings,
        p14_result=p14_result,
        dev_ids=ids,
        heldout_ids=ids,
        rank_limit=2,
        candidate_limit=2,
        output=output,
    )

    archive, manifest = script.package(args)

    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {
            "rankings/heldout_rankings.jsonl",
            "rankings/public_rankings.jsonl",
            "training/dev_ids.json",
            "training/heldout_ids.json",
            "predictions/heldout_qwen.json",
            "predictions/heldout_extractive.json",
            "predictions/public_qwen.json",
            "predictions/public_extractive.json",
        }
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "task2-p16-modal-input-v1"
    assert payload["ranking_compaction"]["heldout"]["questions"] == 1
