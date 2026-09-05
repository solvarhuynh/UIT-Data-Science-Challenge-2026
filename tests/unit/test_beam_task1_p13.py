from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
import zipfile
from pathlib import Path
from typing import Any


class _FakeImage:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.packages: list[str] = []

    def add_python_packages(self, packages: list[str]) -> _FakeImage:
        self.packages.extend(packages)
        return self


class _FakeVolume:
    def __init__(self, *, name: str, mount_path: str) -> None:
        self.name = name
        self.mount_path = mount_path


def _fake_function(**kwargs: Any) -> Any:
    def decorate(handler: Any) -> Any:
        handler._beam_test_kwargs = kwargs
        return handler

    return decorate


def _load_module(monkeypatch: Any) -> Any:
    monkeypatch.delenv("UDSC_BEAM_GPU", raising=False)
    fake_beam = types.ModuleType("beam")
    fake_beam.Image = _FakeImage
    fake_beam.Volume = _FakeVolume
    fake_beam.function = _fake_function
    monkeypatch.setitem(sys.modules, "beam", fake_beam)
    path = Path("scripts/cloud/beam_task1_p13.py")
    spec = importlib.util.spec_from_file_location("beam_task1_p13_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_windows_cli_module() -> Any:
    path = Path("scripts/cloud/beam_windows_cli.py")
    spec = importlib.util.spec_from_file_location("beam_windows_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_result_validator_module() -> Any:
    path = Path("scripts/cloud/validate_beam_result.py")
    spec = importlib.util.spec_from_file_location("validate_beam_result_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_beam_configuration_targets_one_rtx4090_by_default(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)

    assert module.VOLUME_NAME == "udsc-task1-p13"
    assert module.VOLUME_MOUNT_PATH == "/mnt/task1"
    assert module.BEAM_MODULE_NAME == "scripts.cloud.beam_task1_p13"
    assert module.BEAM_GPU_TYPE == "RTX4090"
    assert module.BEAM_MAX_RETRIES == 2
    assert module.smoke._beam_test_kwargs["gpu"] == "RTX4090"
    assert module.smoke._beam_test_kwargs["env"] == {"UDSC_BEAM_GPU": "RTX4090"}
    assert module.full._beam_test_kwargs["gpu"] == "RTX4090"
    assert module.full._beam_test_kwargs["env"] == {"UDSC_BEAM_GPU": "RTX4090"}
    assert module.input_probe._beam_test_kwargs["cpu"] == 2
    assert module.input_probe._beam_test_kwargs["memory"] == "4Gi"
    assert "gpu" not in module.input_probe._beam_test_kwargs
    assert module.CANDIDATE_DEPTH == 200
    assert module.MAX_TRAINING_PAIRS == 24_000
    assert module.BEAM_IMAGE.kwargs["base_image"].endswith(
        "pytorch:2.10.0-cuda12.8-cudnn9-runtime"
    )
    assert "transformers==5.0.0" in module.BEAM_IMAGE.packages


def test_mounted_input_probe_requires_exact_size_and_hash(
    monkeypatch: Any, tmp_path: Path
) -> None:
    module = _load_module(monkeypatch)
    archive = tmp_path / "archive.tar.zst"
    archive.write_bytes(b"verified archive")
    monkeypatch.setattr(module, "INPUT_ARCHIVE", archive)
    monkeypatch.setattr(module, "INPUT_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(module, "INPUT_ARCHIVE_SHA256", "expected")
    monkeypatch.setattr(module, "_sha256", lambda _path: "expected")

    ready = module._run_input_probe()
    assert ready["status"] == "INPUT_READY"
    assert ready["size"] == archive.stat().st_size
    assert ready["sha256"] == "expected"

    monkeypatch.setattr(module, "INPUT_ARCHIVE_BYTES", archive.stat().st_size + 1)
    invalid = module._run_input_probe()
    assert invalid["status"] == "INPUT_INVALID"
    assert invalid["sha256"] is None


def test_beam_sdk_ignore_file_is_a_strict_source_allowlist() -> None:
    beta9_ignore = Path(".beta9ignore").read_text(encoding="utf-8").splitlines()
    beam_ignore = Path(".beamignore").read_text(encoding="utf-8").splitlines()

    assert beta9_ignore == beam_ignore
    assert beta9_ignore[0] == "/*"
    for required in (
        "!/.beta9ignore",
        "!/.beamignore",
        "!/pyproject.toml",
        "!/configs/**",
        "!/scripts/**",
        "!/src/**",
    ):
        assert required in beta9_ignore


def test_smoke_command_exercises_full_memory_profile(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)

    command = module._training_command(smoke=True)

    assert command[0] == sys.executable
    assert command[1:4] == [
        "-u",
        "-m",
        "scripts.training.finetune_task1_bge_reranker",
    ]
    assert command[command.index("--folds-to-run") + 1] == "0"
    assert command[command.index("--max-training-pairs") + 1] == "800"
    assert command[command.index("--max-validation-queries") + 1] == "20"
    assert command[command.index("--batch-size") + 1] == "8"
    assert command[command.index("--inference-batch-size") + 1] == "32"
    assert "--diagnostic-only" in command
    assert "--resume" in command


def test_beam_checkpoint_accepts_both_transformers_weight_formats(
    monkeypatch: Any, tmp_path: Path
) -> None:
    module = _load_module(monkeypatch)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    assert module._checkpoint_weight(checkpoint) is None

    pytorch_weight = checkpoint / "pytorch_model.bin"
    pytorch_weight.write_bytes(b"weights")
    assert module._checkpoint_weight(checkpoint) == pytorch_weight

    safetensors_weight = checkpoint / "model.safetensors"
    safetensors_weight.write_bytes(b"safe weights")
    assert module._checkpoint_weight(checkpoint) == safetensors_weight


def test_gpu_manifest_returns_transport_safe_builtin_versions(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)

    class TorchVersion(str):
        pass

    class _Cuda:
        amp = object()
        version = TorchVersion("13.0")

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_index: int) -> str:
            return "NVIDIA GeForce RTX 4090"

        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            return (8, 9)

        @staticmethod
        def get_arch_list() -> list[str]:
            return ["sm_89"]

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def get_device_properties(_index: int) -> Any:
            return types.SimpleNamespace(total_memory=24_000_000_000)

    class _Probe:
        def __matmul__(self, _other: Any) -> _Probe:
            return self

    fake_torch = types.SimpleNamespace(
        __version__=TorchVersion("2.13.0+cu130"),
        version=types.SimpleNamespace(cuda=TorchVersion("13.0")),
        cuda=_Cuda(),
        float16=object(),
        randn=lambda *_args, **_kwargs: _Probe(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    manifest = module._gpu_manifest()

    assert manifest["torch_version"] == "2.13.0+cu130"
    assert type(manifest["torch_version"]) is str
    assert manifest["cuda_version"] == "13.0"
    assert type(manifest["cuda_version"]) is str


def test_full_command_is_five_fold_and_resume_safe(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)

    command = module._training_command(smoke=False)
    folds_at = command.index("--folds-to-run")

    assert command[folds_at + 1 : folds_at + 6] == ["0", "1", "2", "3", "4"]
    assert command[command.index("--max-training-pairs") + 1] == "24000"
    assert "--max-validation-queries" not in command
    assert "--diagnostic-only" not in command
    assert "--resume" in command


def test_public_stage_uses_same_candidate_contract(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)

    commands = module._public_commands(
        full_root=Path("/mnt/task1/full"), public_root=Path("/mnt/task1/public")
    )
    inference = commands[0]

    assert len(commands) == 3
    assert inference[inference.index("--candidate-depth") + 1] == "200"
    assert inference[inference.index("--evidence-limit") + 1] == "1"
    assert inference[inference.index("--batch-size") + 1] == "32"
    assert commands[-1][1].endswith("validate_legal_ir_submission.py")


def test_remote_failure_is_persisted_and_returned(
    monkeypatch: Any, tmp_path: Path
) -> None:
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "RESULT_ROOT", tmp_path)

    def fail() -> dict[str, Any]:
        raise RuntimeError("diagnostic failure")

    result = module._run_guarded("smoke", fail)

    assert result["status"] == "SMOKE_FAILED"
    assert result["exception_type"] == "RuntimeError"
    assert "diagnostic failure" in result["traceback"]
    failure_path = tmp_path / "smoke_failure.json"
    assert failure_path.is_file()
    assert json.loads(failure_path.read_text(encoding="utf-8"))["failure_path"] == str(
        failure_path
    )


def test_run_contract_versions_resume_roots(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    versions_a = {"torch": "2.10.0", "transformers": "5.0.0"}
    versions_b = {"torch": "2.10.1", "transformers": "5.0.0"}

    smoke_a = module._run_contract_sha("smoke", "source-a", versions_a)
    smoke_b = module._run_contract_sha("smoke", "source-b", versions_a)
    full_a = module._run_contract_sha("full", "source-a", versions_a)
    changed_runtime = module._run_contract_sha("smoke", "source-a", versions_b)

    assert smoke_a != smoke_b
    assert smoke_a != full_a
    assert smoke_a != changed_runtime
    assert module._versioned_root(Path("base"), smoke_a) == Path("base") / smoke_a[:24]


def test_result_archive_accepts_workspace_and_results_siblings(
    monkeypatch: Any, tmp_path: Path
) -> None:
    module = _load_module(monkeypatch)
    volume_root = tmp_path / "mnt" / "task1"
    result_root = volume_root / "results"
    full_root = volume_root / "workspace" / "artifacts" / "full" / "contract"
    public_root = volume_root / "workspace" / "artifacts" / "public" / "contract"
    monkeypatch.setattr(module, "VOLUME_ROOT", volume_root)
    monkeypatch.setattr(module, "RESULT_ROOT", result_root)

    required_files = {
        full_root / "final_decision.json": "{}\n",
        full_root / "comparison.json": "{}\n",
        full_root / "comparison.md": "comparison\n",
        full_root / "training_manifest.json": "{}\n",
        full_root / "oof_predictions.jsonl": "{}\n",
        public_root / "predictions.json": "[]\n",
        public_root / "report.json": "{}\n",
        public_root / "submission.zip": "submission",
    }
    required_files.update(
        {full_root / f"fold_{fold}" / "metrics.json": "{}\n" for fold in range(5)}
    )
    for path, content in required_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    smoke_sha = "a" * 64
    smoke_marker = result_root / "smoke" / f"{smoke_sha}.json"
    smoke_marker.parent.mkdir(parents=True, exist_ok=True)
    smoke_marker.write_text("{}\n", encoding="utf-8")

    archive = module._build_result_archive(
        {"status": "PUBLIC_CANDIDATE_READY", "smoke_run_contract_sha256": smoke_sha},
        full_root=full_root,
        public_root=public_root,
    )

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
    assert "results/full_run_summary.json" in names
    assert f"results/smoke/{smoke_sha}.json" in names
    assert "workspace/artifacts/full/contract/final_decision.json" in names
    assert "workspace/artifacts/public/contract/submission.zip" in names


def _write_valid_beam_result(
    path: Path,
    *,
    include_submission: bool = True,
    source_sha_override: str | None = None,
) -> dict[str, Any]:
    validator = _load_result_validator_module()
    source_sha = source_sha_override or validator.compute_source_tree_sha256(
        validator.PROJECT_ROOT
    )
    smoke_sha = "2" * 64
    full_sha = "3" * 64
    full_root = validator.FULL_ROOT_BASE / full_sha[:24]
    public_root = validator.PUBLIC_ROOT_BASE / full_sha[:24]
    smoke_root = validator.SMOKE_ROOT_BASE / smoke_sha[:24]
    full_relative = full_root.relative_to(validator.VOLUME_ROOT).as_posix()
    public_relative = public_root.relative_to(validator.VOLUME_ROOT).as_posix()
    decision = {
        "status": "PROMOTE_CANDIDATE",
        "promotable": True,
        "complete_oof": True,
    }
    summary = {
        "schema_version": "task1-p13-beam-full-v1",
        "status": "PUBLIC_CANDIDATE_READY",
        "archive_sha256": validator.EXPECTED_ARCHIVE_SHA256,
        "source_tree_sha256": source_sha,
        "smoke_run_contract_sha256": smoke_sha,
        "run_contract_sha256": full_sha,
        "full_root": str(full_root),
        "public_root": str(public_root),
        "download_path": "beam://udsc-task1-p13/results/task1_p13_beam_result.zip",
        "dependency_versions": {"torch": "2.10.0"},
        "promotable": True,
        "decision": decision,
    }
    smoke = {
        "status": "SMOKE_PASS",
        "archive_sha256": validator.EXPECTED_ARCHIVE_SHA256,
        "source_tree_sha256": source_sha,
        "run_contract_sha256": smoke_sha,
        "smoke_root": str(smoke_root),
    }
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as submission:
        submission.writestr("submission.json", json.dumps({"q1": ["doc1"]}))
    entries: dict[str, bytes | str] = {
        "results/full_run_summary.json": json.dumps(summary),
        f"results/smoke/{smoke_sha}.json": json.dumps(smoke),
        f"{full_relative}/final_decision.json": json.dumps(decision),
        f"{full_relative}/comparison.json": "{}",
        f"{full_relative}/comparison.md": "comparison",
        f"{full_relative}/training_manifest.json": "{}",
        f"{full_relative}/oof_predictions.jsonl": "{}\n",
        f"{public_relative}/predictions.json": json.dumps(
            [{"id": "q1", "documents": ["doc1"]}]
        ),
        f"{public_relative}/report.json": "{}",
    }
    if include_submission:
        entries[f"{public_relative}/submission.zip"] = nested.getvalue()
    entries.update(
        {
            f"{full_relative}/fold_{fold}/metrics.json": json.dumps(
                {"status": "COMPLETE", "fold": fold, "validation_query_count": 1}
            )
            for fold in range(5)
        }
    )
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in entries.items():
            bundle.writestr(name, content)
    return summary


def test_windows_wrapper_uses_idempotent_volume_and_multipart_upload() -> None:
    wrapper = Path("scripts/cloud/run_task1_p13_beam.ps1").read_text(encoding="utf-8")

    assert "Invoke-Beam volume create $VolumeName" in wrapper
    assert '"beam://$RemoteArchivePath"' in wrapper
    assert "Invoke-NativeInteractive" in wrapper
    assert "--no-multipart" not in wrapper
    assert '"scripts\\cloud\\beam_windows_cli.py"' in wrapper
    assert '$env:MULTIPART_REQUEST_TIMEOUT = "300"' in wrapper
    assert '$env:MULTIPART_MAX_WORKERS = "8"' in wrapper
    assert wrapper.count("Ensure-BeamInputArchive") == 4
    assert "Get-BeamMountedArchiveProbe" in wrapper
    assert "Test-BeamMountedArchiveProbe" in wrapper
    assert "Invoke-Beam rm $RemoteArchivePath" in wrapper
    assert "mounted archive verification failed" in wrapper


def test_windows_wrapper_invokes_functions_through_beam_sdk_python() -> None:
    wrapper = Path("scripts/cloud/run_task1_p13_beam.ps1").read_text(encoding="utf-8")

    assert "Invoke-BeamFunction smoke" in wrapper
    assert "Invoke-BeamFunction full" in wrapper
    assert '"Check" {' in wrapper
    assert "Invoke-Beam run" not in wrapper


def test_windows_wrapper_can_fetch_logs_without_beam_on_path() -> None:
    wrapper = Path("scripts/cloud/run_task1_p13_beam.ps1").read_text(encoding="utf-8")

    assert '"Logs" {' in wrapper
    assert '"logs", "--task-id", $TaskId' in wrapper
    assert "function Invoke-NativeLogged" in wrapper
    assert '$ErrorActionPreference = "Continue"' in wrapper
    assert '$env:PYTHONIOENCODING = "utf-8"' in wrapper
    assert '"beam_task_${TaskId}.log"' in wrapper
    assert '"beam_${Name}_client.log"' in wrapper


def test_windows_download_uses_posix_remote_paths() -> None:
    module = _load_windows_cli_module()
    remote = types.SimpleNamespace(
        scheme="beam",
        volume_name="udsc-task1-p13",
        volume_path="results/task1_p13_beam_result.zip",
    )

    assert module._posix_remote_path(remote) == (
        "udsc-task1-p13/results/task1_p13_beam_result.zip"
    )
    remote.volume_path = r"results\nested\result.zip"
    assert module._posix_remote_path(remote) == (
        "udsc-task1-p13/results/nested/result.zip"
    )
    remote.volume_path = ""
    assert module._posix_remote_path(remote) == "udsc-task1-p13/"
    assert "SECRET" not in module._redact_download_error(
        "GET https://storage.test/result?X-Amz-Signature=SECRET&token=SECRET"
    )

    wrapper = Path("scripts/cloud/run_task1_p13_beam.ps1").read_text(encoding="utf-8")
    assert '"scripts\\cloud\\beam_windows_cli.py"' in wrapper
    download_block = wrapper.split('    "Download" {', maxsplit=1)[1]
    assert "Invoke-Beam cp" not in download_block
    assert '"beam://$VolumeName/results/task1_p13_beam_result.zip"' in download_block
    assert "$partialResult" in download_block
    assert '"scripts\\cloud\\validate_beam_result.py"' in download_block
    assert "[System.IO.File]::Replace" in download_block
    assert "Move-Item -LiteralPath $partialResult" not in download_block


def test_downloaded_result_must_be_complete_before_publish(tmp_path: Path) -> None:
    module = _load_result_validator_module()
    archive = tmp_path / "result.zip"
    summary = _write_valid_beam_result(archive)

    assert module.validate_result_archive(archive) == summary


def test_downloaded_result_rejects_missing_submission(tmp_path: Path) -> None:
    module = _load_result_validator_module()
    archive = tmp_path / "result.zip"
    _write_valid_beam_result(archive, include_submission=False)

    try:
        module.validate_result_archive(archive)
    except ValueError as exc:
        assert "missing required artifacts" in str(exc)
    else:
        raise AssertionError("incomplete promoted Beam result was accepted")


def test_downloaded_result_rejects_stale_source_contract(tmp_path: Path) -> None:
    module = _load_result_validator_module()
    archive = tmp_path / "stale-result.zip"
    _write_valid_beam_result(archive, source_sha_override="f" * 64)

    try:
        module.validate_result_archive(archive)
    except ValueError as exc:
        assert "does not match the current local code" in str(exc)
    else:
        raise AssertionError("stale Beam source contract was accepted")
