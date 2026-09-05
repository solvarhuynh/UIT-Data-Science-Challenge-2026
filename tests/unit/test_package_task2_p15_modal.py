"""Tests for deterministic P15 Modal overlay packaging."""

from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "cloud" / "package_task2_p15_modal.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("package_task2_p15", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_contains_only_declared_members(tmp_path: Path) -> None:
    script = _load()
    adapter = tmp_path / "adapter"
    adaptation = tmp_path / "adaptation"
    nltk_data = tmp_path / "nltk" / "corpora"
    adapter.mkdir()
    adaptation.mkdir()
    nltk_data.mkdir(parents=True)
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_manifest.json",
    ):
        (adapter / name).write_text(name, encoding="utf-8")
    for name in (
        "adaptation_train.jsonl",
        "adaptation_all.jsonl",
        "adaptation_dev_ids.json",
        "baseline_dev_predictions.json",
        "adaptation_manifest.json",
        "dev_rankings.jsonl",
        "public_rankings.jsonl",
        "dev_extractive_predictions.json",
        "public_extractive_predictions.json",
    ):
        (adaptation / name).write_text(name, encoding="utf-8")
    for name in ("wordnet.zip", "omw-1.4.zip"):
        (nltk_data / name).write_text(name, encoding="utf-8")
    selector = tmp_path / "selector.joblib"
    selector.write_bytes(b"selector")
    output = tmp_path / "bundle.zip"
    args = argparse.Namespace(
        initial_adapter=adapter,
        adaptation_dir=adaptation,
        nltk_data=nltk_data.parent,
        selector=selector,
        output=output,
    )

    archive, manifest = script.package(args)

    with zipfile.ZipFile(archive) as bundle:
        assert "initial_adapter/adapter_model.safetensors" in bundle.namelist()
        assert "training/adaptation_train.jsonl" in bundle.namelist()
        assert "nltk_data/corpora/wordnet.zip" in bundle.namelist()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["archive_bytes"] == archive.stat().st_size
    assert len(payload["archive_sha256"]) == 64
