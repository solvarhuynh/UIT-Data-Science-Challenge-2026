"""Model-free tests for the provider-independent Task 2 Kaggle runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts/cloud/kaggle_task2_p14.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("kaggle_task2_p14", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_to_guarded_full_run() -> None:
    script = _load_script()

    args = script.build_parser().parse_args([])

    assert args.stage == "all"
    assert args.epochs == 2
    assert args.single_gpu is False


def test_two_gpu_qwen_command_enables_data_parallel(tmp_path, monkeypatch) -> None:
    script = _load_script()
    captured: list[list[str]] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        captured.append(list(command))
        return 0

    monkeypatch.setattr(script, "_run", fake_run)
    adapter = script._run_qwen(
        tmp_path,
        tmp_path / "output",
        tmp_path / "qwen",
        gpu_count=2,
        single_gpu=False,
        epochs=2,
        dry_run=True,
    )

    assert adapter == tmp_path / "output/qwen_lora"
    assert "--data-parallel" in captured[0]
    assert captured[0][captured[0].index("--batch-size") + 1] == "2"
    assert captured[0][captured[0].index("--dtype") + 1] == "float16"
    assert captured[0][captured[0].index("--max-length") + 1] == "2048"


def test_explicit_archive_is_resolved(tmp_path) -> None:
    script = _load_script()
    archive = tmp_path / script.ARCHIVE_NAME
    archive.write_bytes(b"archive")

    assert script._find_archive(archive) == archive.resolve()
