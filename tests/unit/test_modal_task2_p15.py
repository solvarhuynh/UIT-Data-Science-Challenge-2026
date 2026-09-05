"""Model-free contract tests for the Modal P15 runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "cloud" / "modal_task2_p15.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("modal_task2_p15_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_input_requires_matching_size_hash_and_schema(tmp_path: Path) -> None:
    script = _load()
    archive = tmp_path / "input.zip"
    archive.write_bytes(b"verified")
    manifest = archive.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "task2-p15-modal-input-v1",
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": hashlib.sha256(b"verified").hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    resolved, resolved_manifest, payload = script._validated_local_input(
        archive, "task2-p15-modal-input-v1"
    )

    assert resolved == archive.resolve()
    assert resolved_manifest == manifest.resolve()
    assert payload["archive_bytes"] == len(b"verified")


def test_commands_pin_continuation_and_selection_contracts(tmp_path: Path) -> None:
    script = _load()
    pipeline = SimpleNamespace(
        _python=lambda *arguments: ["python", *arguments],
    )
    train = script._training_command(
        pipeline,
        train_data=tmp_path / "train.jsonl",
        base_model=tmp_path / "base",
        initial_adapter=tmp_path / "initial",
        output=tmp_path / "output",
        batch_size=4,
        gradient_accumulation=4,
    )
    generate = script._generation_command(
        pipeline,
        questions="questions.json",
        rankings=tmp_path / "rankings.jsonl",
        model=tmp_path / "adapter",
        base_model=tmp_path / "base",
        output=tmp_path / "predictions.json",
        diagnostics=tmp_path / "diagnostics.json",
        batch_size=16,
        max_new_tokens=16,
    )
    selection = script._profile_command(
        pipeline,
        extracted=tmp_path,
        dev_qwen=tmp_path / "dev.json",
        public_qwen=None,
        output=tmp_path / "profile",
    )

    assert "--initial-adapter" in train
    assert train[train.index("--learning-rate") + 1] == "1e-5"
    assert train[train.index("--batch-size") + 1] == "4"
    assert train[train.index("--gradient-accumulation") + 1] == "4"
    assert generate[generate.index("--base-model-dir") + 1] == str(
        tmp_path / "base"
    )
    assert generate[generate.index("--max-new-tokens") + 1] == "16"
    assert "--selection-only" in selection
    assert "--public-qwen" not in selection
