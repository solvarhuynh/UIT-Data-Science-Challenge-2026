"""Tests for the weight-free Task1 Kaggle source bundle."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "package_kaggle_task1.py"
SPEC = importlib.util.spec_from_file_location("package_kaggle_task1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_bundle_excludes_runtime_data_and_weights(tmp_path: Path) -> None:
    output = tmp_path / "task1.zip"
    names = MODULE.create_bundle(ROOT, output)

    assert "task1_bundle_manifest.json" in names
    with zipfile.ZipFile(output) as archive:
        members = archive.namelist()
    assert all(not name.startswith("data/") for name in members)
    assert all(not name.startswith("models/") for name in members)
    assert all(not name.startswith("artifacts/") for name in members)
    assert "docs/members/tv2/tv2_task1_kaggle.md" in members
    assert "scripts/training/finetune_task1_bge_reranker.py" in members


def test_help_parser_has_output_option() -> None:
    args = MODULE.build_parser().parse_args(["--output", "bundle.zip"])
    assert args.output == Path("bundle.zip")
