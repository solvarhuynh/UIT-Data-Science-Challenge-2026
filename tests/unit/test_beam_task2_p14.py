"""Model-free contracts for the Beam Task 2 P14 runner and packager."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any


class FakeImage:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.packages: list[str] = []

    def add_python_packages(self, packages: list[str]) -> FakeImage:
        self.packages.extend(packages)
        return self


class FakeVolume:
    def __init__(self, *, name: str, mount_path: str) -> None:
        self.name = name
        self.mount_path = mount_path


class FakeTaskPolicy:
    def __init__(
        self,
        *,
        max_retries: int = 0,
        timeout: int = 0,
        ttl: int = 0,
    ) -> None:
        self.max_retries = max_retries
        self.timeout = timeout
        self.ttl = ttl


def fake_function(**kwargs: Any) -> Any:
    def decorate(handler: Any) -> Any:
        handler._beam_test_kwargs = kwargs
        return handler

    return decorate


def load_runner(monkeypatch: Any) -> Any:
    monkeypatch.delenv("UDSC_BEAM_GPU", raising=False)
    fake_beta9 = types.ModuleType("beta9")
    fake_beta9.TaskPolicy = FakeTaskPolicy
    fake_beam = types.ModuleType("beam")
    fake_beam.Image = FakeImage
    fake_beam.Volume = FakeVolume
    fake_beam.function = fake_function
    monkeypatch.setitem(sys.modules, "beta9", fake_beta9)
    monkeypatch.setitem(sys.modules, "beam", fake_beam)
    path = Path("scripts/cloud/beam_task2_p14.py")
    spec = importlib.util.spec_from_file_location("beam_task2_p14_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_packager() -> Any:
    path = Path("scripts/cloud/package_task2_p14_beam.py")
    spec = importlib.util.spec_from_file_location("task2_packager_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_parent_preflight() -> Any:
    path = Path("scripts/cloud/task2_p14_gpu_preflight.py")
    spec = importlib.util.spec_from_file_location("task2_parent_preflight_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_beam_task2_uses_cached_image_volume_and_no_auto_retry(
    monkeypatch: Any,
) -> None:
    module = load_runner(monkeypatch)

    assert module.VOLUME_NAME == "udsc-task2-p14"
    assert module.VOLUME_MOUNT == "/mnt/task2"
    assert module.GPU_TYPES == ("RTX5090", "RTX4090", "A10G")
    assert module.GPU_REQUEST == ["RTX5090", "RTX4090", "A10G"]
    assert module.smoke._beam_test_kwargs["gpu"] == [
        "RTX5090",
        "RTX4090",
        "A10G",
    ]
    assert module.smoke._beam_test_kwargs["cpu"] == 8
    assert module.smoke._beam_test_kwargs["memory"] == "64Gi"
    assert module.smoke._beam_test_kwargs["allow_marketplace"] is False
    assert module.smoke._beam_test_kwargs["retries"] == 0
    assert module.smoke._beam_test_kwargs["timeout"] == 30 * 60
    assert module.smoke._beam_test_kwargs["headless"] is True
    smoke_policy = module.smoke._beam_test_kwargs["task_policy"]
    assert smoke_policy.timeout == 30 * 60
    assert smoke_policy.ttl == 45 * 60
    assert module.full._beam_test_kwargs["timeout"] == 8 * 60 * 60
    assert module.full._beam_test_kwargs["headless"] is True
    full_policy = module.full._beam_test_kwargs["task_policy"]
    assert full_policy.timeout == 8 * 60 * 60
    assert full_policy.ttl == 12 * 60 * 60
    assert module.full._beam_test_kwargs["retries"] == 0
    assert module.model_prepare._beam_test_kwargs["cpu"] == 4
    assert "gpu" not in module.model_prepare._beam_test_kwargs
    assert module.BEAM_IMAGE.kwargs["base_image"].endswith(
        "pytorch:2.10.0-cuda12.8-cudnn9-runtime"
    )
    assert "transformers==5.0.0" in module.BEAM_IMAGE.packages
    assert "peft>=0.17,<1.0" in module.BEAM_IMAGE.packages
    assert module.full._beam_test_kwargs["name"] == "udsc-task2-p14-full"
    assert module.full._beam_test_kwargs["env"] == {
        "UDSC_BEAM_GPU": "RTX5090,RTX4090,A10G"
    }
    assert module.RUN_CONTRACT_VERSION == "p14-fast-v4"
    assert module.GPU_CAPABILITY_MINIMUMS["H100"] == (9, 0)
    assert module.QWEN_REVISION == ("b270ee9f3c8ea72cea9ff0f82ece6b40bb8b67a8")


def test_allocation_nonce_creates_a_fresh_function_name(monkeypatch: Any) -> None:
    monkeypatch.setenv("UDSC_BEAM_ALLOCATION_NONCE", "Retry-ABC_123")
    module = load_runner(monkeypatch)

    assert module.ALLOCATION_NONCE == "retryabc123"
    assert module.full._beam_test_kwargs["name"] == ("udsc-task2-p14-full-retryabc123")
    assert (
        module.full._beam_test_kwargs["env"]["UDSC_BEAM_ALLOCATION_NONCE"]
        == "retryabc123"
    )


def test_pool_name_is_forwarded_to_remote_environment(monkeypatch: Any) -> None:
    monkeypatch.setenv("UDSC_BEAM_POOL", "udsc-h100")
    module = load_runner(monkeypatch)

    assert module.full._beam_test_kwargs["pool"] == "udsc-h100"
    assert module.full._beam_test_kwargs["env"]["UDSC_BEAM_POOL"] == "udsc-h100"


def test_input_probe_uses_uploaded_sidecar_hash(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    archive = tmp_path / "input.tar.zst"
    manifest = tmp_path / "input.tar.manifest.json"
    archive.write_bytes(b"verified")
    observed = module._sha256(archive)
    manifest.write_text(
        json.dumps(
            {"archive_bytes": archive.stat().st_size, "archive_sha256": observed}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "INPUT_ARCHIVE", archive)
    monkeypatch.setattr(module, "INPUT_MANIFEST", manifest)

    ready = module._input_state(verify_hash=True)
    assert ready["status"] == "INPUT_READY"
    assert ready["sha256"] == observed

    archive.write_bytes(b"partial")
    assert module._input_state(verify_hash=True)["status"] == "INPUT_INVALID"


def test_run_contract_changes_with_source_hash(monkeypatch: Any) -> None:
    module = load_runner(monkeypatch)

    first = module._run_contract("archive", "source-a")
    second = module._run_contract("archive", "source-b")

    assert first != second
    assert first == module._run_contract("archive", "source-a")


def test_source_tree_hash_ignores_python_cache_files(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts" / "runner.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    before = module._source_tree_sha256(tmp_path)
    cache = tmp_path / "scripts" / "__pycache__"
    cache.mkdir()
    (cache / "runner.pyc").write_bytes(b"volatile")

    assert module._source_tree_sha256(tmp_path) == before


def test_source_snapshot_never_copies_or_hashes_fresh_volume(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    project = tmp_path / "code"
    module_file = project / "scripts" / "cloud" / "beam_task2_p14.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("# stable source\n", encoding="utf-8")
    (project / "src").mkdir()
    (project / "configs").mkdir()
    (project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    volume_workspace = tmp_path / "volume" / "workspace"
    volume_workspace.mkdir(parents=True)
    monkeypatch.setattr(module, "__file__", str(module_file))
    monkeypatch.setattr(module, "WORKSPACE", volume_workspace)
    original_hash = module._source_tree_sha256
    observed_roots: list[Path] = []

    def guarded_hash(root: Path) -> str:
        observed_roots.append(root)
        if root == volume_workspace:
            raise AssertionError("fresh Beam Volume must not be hashed")
        return original_hash(root)

    monkeypatch.setattr(module, "_source_tree_sha256", guarded_hash)

    source_hash = module._overlay_source()

    assert observed_roots == [project]
    assert source_hash == original_hash(project)
    assert list(volume_workspace.iterdir()) == []


def test_subprocess_uses_local_code_but_volume_working_data(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    project = tmp_path / "code"
    module_file = project / "scripts" / "cloud" / "beam_task2_p14.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("# source\n", encoding="utf-8")
    (project / "src").mkdir()
    workspace = tmp_path / "volume" / "workspace"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(module, "__file__", str(module_file))
    monkeypatch.setattr(module, "WORKSPACE", workspace)
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(workspace), str(workspace / "src"), "external")),
    )

    command = module._python("scripts/training/train.py", "--flag")
    environment = module._environment()

    assert Path(command[1]) == project / "scripts" / "training" / "train.py"
    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(project),
        str(project / "src"),
    ]


def test_subprocess_executes_code_locally_and_reads_data_from_workspace(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    project = tmp_path / "code"
    module_file = project / "scripts" / "cloud" / "beam_task2_p14.py"
    job = project / "scripts" / "training" / "job.py"
    package = project / "src" / "sample_package"
    module_file.parent.mkdir(parents=True)
    job.parent.mkdir(parents=True)
    package.mkdir(parents=True)
    module_file.write_text("# source\n", encoding="utf-8")
    (package / "__init__.py").write_text("VALUE = 'local-code'\n", encoding="utf-8")
    job.write_text(
        "from pathlib import Path\n"
        "from sample_package import VALUE\n"
        "assert VALUE == 'local-code'\n"
        "assert Path('input.txt').read_text() == 'volume-data'\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "volume" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "input.txt").write_text("volume-data", encoding="utf-8")
    stale_package = workspace / "src" / "sample_package"
    stale_package.mkdir(parents=True)
    (stale_package / "__init__.py").write_text(
        "VALUE = 'volume-code'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "__file__", str(module_file))
    monkeypatch.setattr(module, "WORKSPACE", workspace)
    monkeypatch.setenv("PYTHONPATH", "")

    assert module._run(module._python("scripts/training/job.py")) == 0


def test_pinned_qwen_cache_skips_network_when_manifest_matches(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"cached")
    manifest = model_dir / ".beam_model_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": module.QWEN_REPO,
                "revision": module.QWEN_REVISION,
                "model_bytes": len(b"cached"),
                "config_sha256": module._sha256(model_dir / "config.json"),
                "tensor_count": 1,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "QWEN_DIR", model_dir)
    monkeypatch.setattr(module, "QWEN_CACHE_MANIFEST", manifest)

    module._download_qwen()


def test_gpu_manifest_returns_only_portable_version_strings(monkeypatch: Any) -> None:
    module = load_runner(monkeypatch)

    class TorchVersion(str):
        pass

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            assert index == 0
            return (8, 9)

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "Fake RTX 4090"

        @staticmethod
        def get_device_properties(index: int) -> Any:
            assert index == 0
            return types.SimpleNamespace(total_memory=24_000_000_000)

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            assert index == 0
            return (20_000_000_000, 24_000_000_000)

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def memory_allocated(index: int) -> int:
            assert index == 0
            return 0

        @staticmethod
        def memory_reserved(index: int) -> int:
            assert index == 0
            return 0

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = TorchVersion("2.10.0+cu128")
    fake_torch.version = types.SimpleNamespace(cuda=TorchVersion("12.8"))
    fake_torch.cuda = FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: types.SimpleNamespace(
            returncode=0,
            stdout="321, python3, 4096\n",
            stderr="",
        ),
    )

    manifest = module._gpu_manifest()

    assert type(manifest["torch_version"]) is str
    assert type(manifest["cuda_version"]) is str
    assert manifest["torch_version"] == "2.10.0+cu128"
    assert manifest["cuda_version"] == "12.8"
    assert manifest["free_vram_bytes"] == 20_000_000_000
    assert manifest["vram_bytes"] == 24_000_000_000
    assert manifest["actual_type"] == "RTX4090"
    assert manifest["requested_types"] == ["RTX5090", "RTX4090", "A10G"]
    assert manifest["nvidia_smi_compute_processes"] == ["321, python3, 4096"]
    assert module.GPU_CAPABILITY_MINIMUMS["A10G"] == (8, 6)


def test_vram_preflight_rejects_an_already_occupied_beam_gpu(
    monkeypatch: Any,
) -> None:
    module = load_runner(monkeypatch)

    try:
        module._require_free_vram(
            {"free_vram_bytes": 24 * 1024**2, "vram_bytes": 24 * 1024**3}
        )
    except RuntimeError as exc:
        assert "already occupied" in str(exc)
        assert "changing batch size cannot fix" in str(exc)
    else:
        raise AssertionError("occupied GPU was accepted")


def test_h100_preflight_requires_at_least_70_gib_free(monkeypatch: Any) -> None:
    module = load_runner(monkeypatch)
    gpu = {
        "actual_type": "H100",
        "free_vram_bytes": 69 * 1024**3,
        "vram_bytes": 80 * 1024**3,
    }

    try:
        module._require_free_vram(gpu)
    except RuntimeError as exc:
        assert "70.00 GiB" in str(exc)
    else:
        raise AssertionError("partly occupied H100 was accepted")

    gpu["free_vram_bytes"] = 71 * 1024**3
    module._require_free_vram(gpu)


def test_performance_profile_scales_batches_without_changing_effective_batch(
    monkeypatch: Any,
) -> None:
    module = load_runner(monkeypatch)

    h100 = module._performance_profile({"vram_bytes": 80 * 1024**3})
    rtx5090 = module._performance_profile({"vram_bytes": 32 * 1024**3})
    safe = module._performance_profile({"vram_bytes": 24 * 1024**3})

    for profile in (h100, rtx5090, safe):
        assert profile["qwen_train_batch"] * profile["qwen_gradient_accumulation"] == 16
        assert (
            profile["parent_train_batch"] * profile["parent_gradient_accumulation"]
            == 16
        )


def test_incomplete_qwen_epoch_checkpoint_is_preserved_for_resume(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    output = tmp_path / "qwen"
    checkpoint = output / "checkpoint-epoch-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "training_state.pt").write_bytes(b"state")

    module._ensure_clean_incomplete(
        output,
        output / "training_manifest.json",
        preserve_epoch_resume=True,
    )

    assert checkpoint.is_dir()


def test_incomplete_parent_latest_checkpoint_is_preserved_for_resume(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    output = tmp_path / "parent"
    checkpoint = output / "checkpoint-latest"
    checkpoint.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "config.json",
        "tokenizer_config.json",
        "training_metrics.json",
        "training_state.pt",
    ):
        (checkpoint / name).write_bytes(b"complete")

    module._ensure_clean_incomplete(
        output,
        output / "training_manifest.json",
        preserve_epoch_resume=True,
    )

    assert checkpoint.is_dir()


def test_prediction_completion_requires_exact_order_and_diagnostics(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    questions = tmp_path / "questions.json"
    ids = tmp_path / "ids.json"
    predictions = tmp_path / "predictions.json"
    diagnostics = tmp_path / "diagnostics.json"
    questions.write_text('{"q1": {}, "q2": {}}', encoding="utf-8")
    ids.write_text('["q2", "q1"]', encoding="utf-8")
    predictions.write_text('[{"id": "q2", "answer": "a"}]', encoding="utf-8")
    diagnostics.write_text('[{"id": "q2"}]', encoding="utf-8")

    assert not module._prediction_outputs_complete(
        predictions,
        diagnostics,
        questions=questions,
        question_ids=ids,
    )
    predictions.write_text(
        '[{"id": "q2", "answer": "a"}, {"id": "q1", "answer": "b"}]',
        encoding="utf-8",
    )
    diagnostics.write_text('[{"id": "q2"}, {"id": "q1"}]', encoding="utf-8")
    assert module._prediction_outputs_complete(
        predictions,
        diagnostics,
        questions=questions,
        question_ids=ids,
    )


def test_windows_wrapper_has_one_command_run_and_fast_serverless_profile() -> None:
    wrapper = Path("scripts/cloud/run_task2_p14_beam.ps1").read_text(encoding="utf-8")

    assert '"FastServerless"' in wrapper
    assert '@("RTX5090", "RTX4090", "A10G")' in wrapper
    assert "$env:UDSC_BEAM_GPU = $attemptGpuRequest" in wrapper
    assert "($attempt - 1) % $attemptPlan.Count" in wrapper
    assert "Get-GpuAttemptPlan" in wrapper
    assert "return @($GpuType)" in wrapper
    assert "gpu_attempt_plan" in wrapper
    assert "rotating $attemptGpuRequest -> $nextGpuRequest" in wrapper
    assert "$nextGpuRequest -ne $attemptGpuRequest" in wrapper
    assert "before switching Beam GPU pools" in wrapper
    assert '"Run" {' in wrapper
    assert "Invoke-GpuStage smoke" in wrapper
    assert "Invoke-GpuStage full" in wrapper
    assert "Receive-BeamResult" in wrapper
    assert "Assert-NoActiveTask2Job" in wrapper
    assert 'GpuType -eq "H100"' in wrapper
    assert '$ExpectedRunContractVersion = "p14-fast-v4"' in wrapper
    assert "$summary.archive_sha256 -ne $expected.archive_sha256" in wrapper
    assert "$summary.contract_version -ne $expected.contract_version" in wrapper
    assert "$summary.source_sha256 -ne $expected.source_sha256" in wrapper
    assert "Write-BeamLaunchManifest" in wrapper
    assert '"Stop" {' in wrapper
    assert "Invoke-Beam task stop $TaskId" in wrapper
    assert "Test-UnknownDeviceAllocationFailure" in wrapper
    assert "Test-RetryableGpuAllocationFailureContent" in wrapper
    assert "Get-LocallyConfirmedTerminalTaskIds" in wrapper
    assert "Get-LocallyConfirmedTerminalTaskMap" in wrapper
    assert "Get-CanonicalBeamTaskId" in wrapper
    assert "ConvertFrom-BeamTaskListJson" in wrapper
    assert "task-list flattening self-test failed" in wrapper
    assert "$terminalTaskMap.ContainsKey($candidateId)" in wrapper
    assert "terminal-task classifier self-test failed" in wrapper
    assert "terminal-task classifier accepted non-terminal text" in wrapper
    assert "allocation-error classifier self-test failed" in wrapper
    assert "FastServerless rotation self-test failed" in wrapper
    assert "allocation classifier accepted an application failure" in wrapper


def test_windows_wrapper_rotates_complete_serverless_priority_lists() -> None:
    wrapper = Path("scripts/cloud/run_task2_p14_beam.ps1").read_text(encoding="utf-8")
    gpu_plan = wrapper.split("function Get-GpuAttemptPlan", 1)[1].split(
        "function Invoke-GpuStage", 1
    )[0]

    assert '$FastServerlessGpuOrder = @("RTX5090", "RTX4090", "A10G")' in (wrapper)
    assert "$rotations = @()" in gpu_plan
    assert "$offset -lt $FastServerlessGpuOrder.Count" in gpu_plan
    assert "$index -lt $FastServerlessGpuOrder.Count" in gpu_plan
    assert "($offset + $index) % $FastServerlessGpuOrder.Count" in gpu_plan
    assert '$rotations += ,($ordered -join ",")' in gpu_plan
    assert (
        '"RTX5090,RTX4090,A10G|" +\n'
        '                "RTX4090,A10G,RTX5090|" +\n'
        '                "A10G,RTX5090,RTX4090"' in wrapper
    )


def test_windows_wrapper_task_query_is_deep_and_keeps_stderr_out_of_json() -> None:
    wrapper = Path("scripts/cloud/run_task2_p14_beam.ps1").read_text(encoding="utf-8")
    task_query = wrapper.split("function Get-Task2BeamJobState", 1)[1].split(
        "function Assert-NoActiveTask2Job", 1
    )[0]

    assert "--limit 1000" in task_query
    assert '--filter "status=running,pending,retry"' in task_query
    assert "$stdout = @(" in task_query
    assert "2> $stderrPath" in task_query
    assert "2>&1" not in task_query
    assert "ConvertFrom-BeamTaskListJson -Json ($stdout -join" in task_query
    assert 'Write-Warning ("Beam task-list warning: " + $stderr.Trim())' in (task_query)


def test_windows_wrapper_logs_and_stop_require_one_exact_task_uuid() -> None:
    wrapper = Path("scripts/cloud/run_task2_p14_beam.ps1").read_text(encoding="utf-8")
    exact_id = wrapper.split("function Resolve-ExactBeamTaskId", 1)[1].split(
        "function Get-LocallyConfirmedTerminalTaskMap", 1
    )[0]
    logs_stage = wrapper.split('"Logs" {', 1)[1].split('"Stop" {', 1)[0]
    stop_stage = wrapper.split('"Stop" {', 1)[1].split('"Download" {', 1)[0]

    assert "$trimmed = ([string]$Value).Trim()" in exact_id
    assert "$trimmed.ToLowerInvariant() -ne $canonical" in exact_id
    assert "Task ID must be one exact UUID copied from Beam" in exact_id
    assert "$TaskId = Resolve-ExactBeamTaskId -Value $TaskId" in logs_stage
    assert "$TaskId = Resolve-ExactBeamTaskId -Value $TaskId" in stop_stage
    assert '@("logs", "--task-id", $TaskId' in logs_stage
    assert "Invoke-Beam task stop $TaskId" in stop_stage


def test_windows_wrapper_capacity_stage_is_read_only() -> None:
    wrapper = Path("scripts/cloud/run_task2_p14_beam.ps1").read_text(encoding="utf-8")
    capacity_stage = wrapper.split('"Capacity" {', 1)[1].split('"Guard" {', 1)[0]

    assert '[ValidateSet("Check", "Capacity", "Guard"' in wrapper
    assert "Invoke-Beam machine list --no-offers" in capacity_stage
    assert "machine reserve" not in capacity_stage
    assert "machine release" not in capacity_stage


def test_paid_h100_wrapper_caps_spend_and_always_releases_pool() -> None:
    wrapper = Path("scripts/cloud/run_task2_p14_h100.ps1").read_text(encoding="utf-8")

    assert '[string]$Ttl = "4h"' in wrapper
    assert "[double]$MaxSpendUsd = 16.0" in wrapper
    assert "machine reserve" in wrapper
    assert "--gpu H100" in wrapper
    assert "--max-spend $MaxSpendText" in wrapper
    assert "-Stage Run" in wrapper
    assert "-GpuType H100" in wrapper
    assert "-PoolName $PoolName" in wrapper
    assert "finally" in wrapper
    assert "machine release --pool $PoolName --yes" in wrapper
    assert "Serverless-only credits may not cover" in wrapper


def test_full_accepts_a_different_requested_gpu_and_repreflights_actual_gpu(
    monkeypatch: Any,
) -> None:
    module = load_runner(monkeypatch)
    full_source = inspect.getsource(module._run_full)

    assert 'get("requested_types")' not in full_source
    assert 'get("actual_type")' in full_source
    assert "_training_preflight(performance)" in full_source


def test_public_candidate_resumes_checkpoints_without_training_again(
    monkeypatch: Any,
) -> None:
    module = load_runner(monkeypatch)
    source = inspect.getsource(module._run_public_candidate)

    assert "_resumable_full_run()" in source
    assert "public_qwen_predictions.json" in source
    assert "build_legal_qa_qwen_extractive_ensemble.py" not in source
    assert '"train"' not in source


def test_run_enforces_subprocess_timeout(monkeypatch: Any, tmp_path: Path) -> None:
    module = load_runner(monkeypatch)
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)

    def time_out(command: list[str], **kwargs: Any) -> None:
        assert kwargs["timeout"] == 17
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(module.subprocess, "run", time_out)

    try:
        module._run(["python", "stuck.py"], timeout_seconds=17)
    except RuntimeError as exc:
        assert "exceeded 17 seconds" in str(exc)
        assert "stuck.py" in str(exc)
    else:
        raise AssertionError("stuck subprocess was not rejected")


def test_reranker_preflight_has_a_fail_fast_timeout(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> int:
        observed["command"] = command
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_localize_directory",
        lambda source, label: tmp_path / label,
    )
    module._reranker_cuda_preflight(
        {"parent_train_batch": 4, "parent_gradient_accumulation": 4}
    )

    assert (
        Path(observed["command"][1])
        .as_posix()
        .endswith("scripts/cloud/task2_p14_gpu_preflight.py")
    )
    assert "--train-data" in observed["command"]
    assert "--groups-per-batch" in observed["command"]
    assert (
        observed["command"][observed["command"].index("--groups-per-batch") + 1] == "4"
    )
    assert (
        observed["command"][observed["command"].index("--gradient-accumulation") + 1]
        == "4"
    )
    assert observed["command"][-2:] == ["--freeze-layers", "6"]
    assert observed["timeout_seconds"] == 300


def test_qwen_preflight_has_a_fail_fast_timeout(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> int:
        observed["command"] = command
        observed.update(kwargs)
        return 0

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(
        module,
        "_localize_directory",
        lambda source, label: tmp_path / label,
    )
    monkeypatch.setattr(module, "LOCAL_RUNTIME_ROOT", tmp_path / "runtime")
    module._qwen_training_preflight(
        {"qwen_train_batch": 2, "qwen_gradient_accumulation": 8}
    )

    assert (
        Path(observed["command"][1])
        .as_posix()
        .endswith("scripts/training/finetune_task2_qwen_lora.py")
    )
    assert "--smoke-only" in observed["command"]
    assert observed["timeout_seconds"] == 900


def test_smoke_invalidates_an_old_pass_before_remote_work(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    result_root = tmp_path / "results"
    result_root.mkdir()
    smoke_path = result_root / "smoke.json"
    smoke_path.write_text(
        json.dumps({"status": "SMOKE_PASS", "source_sha256": "old"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "RESULT_ROOT", result_root)

    def fake_extract() -> dict[str, Any]:
        invalidated = json.loads(smoke_path.read_text(encoding="utf-8"))
        assert invalidated["status"] == "SMOKE_RUNNING"
        return {"expected_sha256": "archive"}

    monkeypatch.setattr(module, "_extract_workspace", fake_extract)
    monkeypatch.setattr(module, "_overlay_source", lambda: "source")
    monkeypatch.setattr(module, "_download_qwen", lambda: None)
    monkeypatch.setattr(
        module,
        "_gpu_manifest",
        lambda: {
            "actual_type": "RTX4090",
            "free_vram_bytes": 23 * 1024**3,
            "vram_bytes": 24 * 1024**3,
        },
    )
    monkeypatch.setattr(module, "_require_free_vram", lambda gpu: None)
    monkeypatch.setattr(module, "_training_preflight", lambda performance: None)

    payload = module._run_smoke()

    assert payload["status"] == "SMOKE_PASS"
    assert json.loads(smoke_path.read_text(encoding="utf-8"))["status"] == (
        "SMOKE_PASS"
    )


def test_smoke_imports_every_full_entrypoint_from_local_code(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    project = tmp_path / "code"
    module_file = project / "scripts" / "cloud" / "beam_task2_p14.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("# runner\n", encoding="utf-8")
    for relative in module.FULL_SOURCE_SCRIPTS:
        script = project / relative
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("# entrypoint\n", encoding="utf-8")
    workspace = tmp_path / "volume" / "workspace"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(module, "__file__", str(module_file))
    monkeypatch.setattr(module, "WORKSPACE", workspace)
    observed: list[tuple[list[str], int]] = []

    def fake_run(command: list[str], *, timeout_seconds: int) -> int:
        observed.append((command, timeout_seconds))
        return 0

    monkeypatch.setattr(module, "_run", fake_run)

    module._full_source_import_preflight()

    assert len(observed) == len(module.FULL_SOURCE_SCRIPTS)
    for command, timeout_seconds in observed:
        assert Path(command[1]).is_relative_to(project)
        assert not Path(command[1]).is_relative_to(workspace)
        assert command[-1] == "--help"
        assert timeout_seconds == 60


def test_parent_preflight_derives_real_worst_group_sizes(tmp_path: Path) -> None:
    module = load_parent_preflight()
    training = tmp_path / "parent.jsonl"
    training.write_text(
        "\n".join(
            json.dumps({"question_id": question_id})
            for question_id in ("q1", "q1", "q2", "q2", "q2")
        )
        + "\n",
        encoding="utf-8",
    )

    assert module._largest_group_sizes(training, 2) == [3, 2]


def test_parent_preflight_rejects_invalid_question_id(tmp_path: Path) -> None:
    module = load_parent_preflight()
    training = tmp_path / "parent.jsonl"
    training.write_text('{"question_id": ""}\n', encoding="utf-8")

    try:
        module._largest_group_sizes(training, 1)
    except ValueError as exc:
        assert "invalid question_id" in str(exc)
    else:
        raise AssertionError("invalid listwise data were accepted")


def test_full_localizes_models_and_does_not_run_a_disposable_cuda_probe(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    module = load_runner(monkeypatch)
    source = tmp_path / "volume-model"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    runtime = tmp_path / "runtime"
    stale = runtime / "parent_base"
    stale.mkdir(parents=True)
    (stale / "stale.txt").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(module, "LOCAL_RUNTIME_ROOT", runtime)

    localized = module._localize_directory(source, "parent_base")

    assert localized == runtime / "parent_base"
    assert (localized / "model.safetensors").read_bytes() == b"weights"
    assert not (localized / "stale.txt").exists()
    full_source = inspect.getsource(module._run_full)
    assert "_reranker_cuda_preflight()" not in full_source
    assert "_localize_directory" in full_source


def test_packager_omits_optimizer_state_and_local_qwen() -> None:
    module = load_packager()

    assert "training_state.pt" in module.EXCLUDED_NAMES
    assert all(
        "models/qwen3-legal" not in path.as_posix() for path in module.INCLUDE_PATHS
    )
    assert Path("artifacts/task2/training/task2_qwen_sft_train.jsonl") in (
        module.INCLUDE_PATHS
    )
    assert Path("data/processed_v3/parents") in module.INCLUDE_PATHS
