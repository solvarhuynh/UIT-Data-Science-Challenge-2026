"""Tests for the verified P17 Modal input bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/cloud/package_task2_p17_modal.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("package_p17_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_writes_exact_verified_members(tmp_path: Path) -> None:
    script = _load()
    bank = tmp_path / "bank.jsonl"
    bank.write_text('{"id":"q1","profile":"p","answer":"a"}\n')
    ids = tmp_path / "ids.json"
    ids.write_text('["q1"]')
    baseline = tmp_path / "baseline.json"
    baseline.write_text('[{"id":"q1","answer":"a"}]')
    nltk = tmp_path / "nltk/corpora"
    nltk.mkdir(parents=True)
    (nltk / "wordnet.zip").write_bytes(b"wordnet")
    (nltk / "omw-1.4.zip").write_bytes(b"omw")
    p14 = tmp_path / "p14.zip"
    with zipfile.ZipFile(p14, "w") as bundle:
        bundle.writestr(
            f"{script.P14_ZIP_RUN}/public_qwen_predictions.json",
            '[{"id":"q1","answer":"q"}]',
        )
        bundle.writestr(
            f"{script.P14_ZIP_RUN}/public_extractive_predictions.json",
            '[{"id":"q1","answer":"e"}]',
        )
    output = tmp_path / "p17.zip"

    archive, manifest_path = script.package(
        argparse.Namespace(
            candidate_bank=bank,
            question_ids=ids,
            p14_result=p14,
            nltk_data=tmp_path / "nltk",
            baseline_predictions=baseline,
            output=output,
        )
    )

    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {
            "heldout/candidate_bank.jsonl",
            "heldout/question_ids.json",
            "heldout/baseline_predictions.json",
            "nltk_data/corpora/wordnet.zip",
            "nltk_data/corpora/omw-1.4.zip",
            "public/qwen.json",
            "public/extractive.json",
        }
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "task2-p17-modal-input-v1"
    assert len(manifest["members"]) == 7
