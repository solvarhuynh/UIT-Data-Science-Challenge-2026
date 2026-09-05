"""Model-free contract tests for the Modal P16 RAG sweep."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/cloud/modal_task2_p16.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("modal_task2_p16_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_experiments_are_bounded_and_h100_only() -> None:
    script = _load()

    assert script.MODAL_GPU_PRIORITY == ["H100"]
    assert {row["name"] for row in script.DEV_EXPERIMENTS} == {
        "base_top2_fusion",
        "base_top3_fusion",
        "adapter_top3_fusion",
        "adapter_top3_ce",
    }
    assert all(row["top_parents"] <= 3 for row in script.DEV_EXPERIMENTS)
    assert all(row["max_context_tokens"] <= 1800 for row in script.DEV_EXPERIMENTS)


def test_remote_code_root_is_added_before_shared_core_import() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    path_setup = source.index('Path("/root/udsc2026")')
    core_import = source.index("from scripts.cloud import modal_task2_p15 as core")
    assert path_setup < core_import
    assert ".parents[2]" not in source[:core_import]


def test_generation_command_pins_offline_base_and_ranking_contract(
    tmp_path: Path,
) -> None:
    script = _load()
    pipeline = SimpleNamespace(_python=lambda *arguments: ["python", *arguments])
    config = dict(script.DEV_EXPERIMENTS[2])

    command = script._generation_command(
        pipeline,
        questions="questions.json",
        rankings=tmp_path / "rankings.jsonl",
        model=tmp_path / "adapter",
        base_model=tmp_path / "base",
        output=tmp_path / "predictions.json",
        diagnostics=tmp_path / "diagnostics.json",
        config=config,
        question_ids=tmp_path / "ids.json",
    )

    assert command[command.index("--base-model-dir") + 1] == str(tmp_path / "base")
    assert command[command.index("--top-parents") + 1] == "3"
    assert command[command.index("--ce-weight") + 1] == "0.3"
    assert command[command.index("--retrieval-weight") + 1] == "0.7"
    assert command[command.index("--question-ids") + 1] == str(tmp_path / "ids.json")
    assert "--known-answers" not in command


def test_profile_command_uses_meteor_only_without_public_leakage(
    tmp_path: Path,
) -> None:
    script = _load()
    pipeline = SimpleNamespace(_python=lambda *arguments: ["python", *arguments])

    command = script._profile_command(
        pipeline,
        question_ids=tmp_path / "ids.json",
        qwen=tmp_path / "qwen.json",
        extractive=tmp_path / "extractive.json",
        output=tmp_path / "profile",
    )

    assert command[command.index("--meteor-weight") + 1] == "1"
    assert command[command.index("--rouge-weight") + 1] == "0"
    assert "--selection-only" in command
    assert "--public-qwen" not in command
