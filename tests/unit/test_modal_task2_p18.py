"""Model-free contracts for the gated Modal P18 run."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/cloud/modal_task2_p18.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("modal_p18_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p18_is_h100_only_and_strictly_gated() -> None:
    script = _load()

    assert script.MODAL_GPU_PRIORITY == ["H100"]
    assert script.MINIMUM_METEOR_GAIN >= 0.01
    assert script.MINIMUM_DEV_METEOR >= 0.60
    assert script.MINIMUM_ROUGE_GAIN >= -0.015
    assert script.DPO_CONFIG["epochs"] == 1


def test_remote_code_root_precedes_core_import() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    path_setup = source.index('Path("/root/udsc2026")')
    core_import = source.index("from scripts.cloud import modal_task2_p15 as core")
    assert path_setup < core_import
    assert ".parents[2]" not in source[:core_import]


def test_training_command_uses_length_normalized_dpo_contract(tmp_path: Path) -> None:
    script = _load()
    pipeline = SimpleNamespace(_python=lambda *arguments: ["python", *arguments])

    command = script._training_command(
        pipeline,
        preferences=tmp_path / "preferences.jsonl",
        base_model=tmp_path / "base",
        initial_adapter=tmp_path / "initial",
        output=tmp_path / "output",
    )

    assert command[1] == "scripts/training/train_task2_p18_dpo.py"
    assert command[command.index("--learning-rate") + 1] == "5e-06"
    assert command[command.index("--beta") + 1] == "0.2"
    assert command[command.index("--sft-weight") + 1] == "0.1"
    assert command[command.index("--dtype") + 1] == "bfloat16"


def test_profile_selection_targets_meteor_without_rouge_filter(tmp_path: Path) -> None:
    script = _load()
    pipeline = SimpleNamespace(_python=lambda *arguments: ["python", *arguments])

    command = script._profile_command(
        pipeline,
        p15_inputs=tmp_path / "p15",
        dev_qwen=tmp_path / "dev.json",
        output=tmp_path / "profile",
    )

    assert command[command.index("--meteor-weight") + 1] == "1"
    assert command[command.index("--rouge-weight") + 1] == "0"
    assert command[command.index("--maximum-rouge-drop") + 1] == "1"
    assert "--selection-only" in command
