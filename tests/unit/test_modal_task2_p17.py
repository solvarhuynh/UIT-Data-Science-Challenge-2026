"""Model-free contracts for the Modal P17 cross-encoder run."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/cloud/modal_task2_p17.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("modal_p17_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p17_is_h100_only_and_strictly_gated() -> None:
    script = _load()

    assert script.MODAL_GPU_PRIORITY == ["H100"]
    assert script.FOLDS == 5
    assert script.MINIMUM_OFFICIAL_METEOR >= 0.585
    assert script.MINIMUM_FAST_METEOR >= 0.58


def test_remote_code_root_precedes_core_import() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    path_setup = source.index('Path("/root/udsc2026")')
    core_import = source.index("from scripts.cloud import modal_task2_p15 as core")
    assert path_setup < core_import
    assert ".parents[2]" not in source[:core_import]


def test_train_command_is_listwise_frozen_and_bounded(tmp_path: Path) -> None:
    script = _load()
    pipeline = SimpleNamespace(_python=lambda *arguments: ["python", *arguments])

    command = script._train_command(
        pipeline,
        train_data=tmp_path / "train.jsonl",
        eval_data=tmp_path / "eval.jsonl",
        base=tmp_path / "base",
        output=tmp_path / "output",
        seed=2026,
    )

    assert command[command.index("--objective") + 1] == "listwise"
    assert command[command.index("--target-temperature") + 1] == "0.05"
    assert command[command.index("--freeze-layers") + 1] == "8"
    assert command[command.index("--batch-size") + 1] == "4"
    assert "--fit-all" not in command
