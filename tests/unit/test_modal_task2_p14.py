"""Model-free contracts for the Modal Task 2 P14 runner."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

SCRIPT = Path("scripts/cloud/modal_task2_p14.py")


class FakeImage:
    @classmethod
    def from_registry(cls, *_: Any, **__: Any) -> FakeImage:
        return cls()

    @classmethod
    def debian_slim(cls, *_: Any, **__: Any) -> FakeImage:
        return cls()

    def entrypoint(self, *_: Any, **__: Any) -> FakeImage:
        return self

    def apt_install(self, *_: Any, **__: Any) -> FakeImage:
        return self

    def pip_install(self, *_: Any, **__: Any) -> FakeImage:
        return self

    def add_local_dir(self, *_: Any, **__: Any) -> FakeImage:
        return self

    def add_local_file(self, *_: Any, **__: Any) -> FakeImage:
        return self


class FakeVolume:
    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> FakeVolume:
        instance = cls()
        instance.name = name
        instance.kwargs = kwargs
        return instance


class FakeApp:
    def __init__(self, name: str) -> None:
        self.name = name

    def function(self, **kwargs: Any) -> Any:
        def decorate(handler: Any) -> Any:
            handler._modal_test_kwargs = kwargs
            return handler

        return decorate

    def local_entrypoint(self, **_: Any) -> Any:
        return lambda handler: handler


def _load(monkeypatch: Any, *, with_modal: bool) -> Any:
    if with_modal:
        fake = types.ModuleType("modal")
        fake.App = FakeApp
        fake.Image = FakeImage
        fake.Volume = FakeVolume
        monkeypatch.setitem(sys.modules, "modal", fake)
    else:
        monkeypatch.setitem(sys.modules, "modal", None)
    spec = importlib.util.spec_from_file_location("modal_task2_p14_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modal_resources_use_fast_gpu_fallback_and_persistent_volume(
    monkeypatch: Any,
) -> None:
    module = _load(monkeypatch, with_modal=True)

    assert module.app.name == "udsc-task2-p14-modal"
    assert module.task2_volume.name == "udsc-task2-p14-modal"
    assert module.task2_volume.kwargs == {"create_if_missing": True}
    assert module.MODAL_GPU_PRIORITY == ["H100", "A100-80GB", "L40S"]
    assert module.TORCH_VERSION == "2.10.0"
    assert module.TORCH_INDEX_URL.endswith("/cu128")
    assert module.smoke._modal_test_kwargs["gpu"] == module.MODAL_GPU_PRIORITY
    assert module.smoke._modal_test_kwargs["retries"] == 0
    assert module.smoke._modal_test_kwargs["timeout"] == 30 * 60
    assert module.full._modal_test_kwargs["gpu"] == module.MODAL_GPU_PRIORITY
    assert module.full._modal_test_kwargs["timeout"] == 8 * 60 * 60
    assert module.full._modal_test_kwargs["volumes"] == {
        "/mnt/task2": module.task2_volume
    }
    assert module.run_all._modal_test_kwargs["gpu"] == module.MODAL_GPU_PRIORITY
    assert module.run_all._modal_test_kwargs["timeout"] == 10 * 60 * 60
    assert module.public_candidate._modal_test_kwargs["gpu"] == (
        module.MODAL_GPU_PRIORITY
    )
    assert module.public_candidate._modal_test_kwargs["timeout"] == 4 * 60 * 60


def test_project_root_resolution_supports_relocated_modal_entrypoint(
    monkeypatch: Any, tmp_path: Path
) -> None:
    module = _load(monkeypatch, with_modal=False)
    container_root = tmp_path / "root"
    remote_code_root = container_root / "udsc2026"
    (remote_code_root / "scripts/cloud").mkdir(parents=True)
    (remote_code_root / "src").mkdir()
    (remote_code_root / "pyproject.toml").write_text("[project]\nname='test'\n")
    (remote_code_root / "scripts/cloud/beam_task2_p14.py").write_text(
        "PROVIDER_NAME = 'modal'\n"
    )

    observed = module._resolve_project_root(
        container_root / "modal_task2_p14.py",
        remote_code_root=remote_code_root,
        cwd=container_root,
    )

    assert observed == remote_code_root.resolve()


def test_relocated_modal_entrypoint_imports_from_root_layout(
    monkeypatch: Any, tmp_path: Path
) -> None:
    container_root = tmp_path / "root"
    remote_code_root = container_root / "udsc2026"
    (remote_code_root / "scripts/cloud").mkdir(parents=True)
    (remote_code_root / "src").mkdir()
    (remote_code_root / "pyproject.toml").write_text("[project]\nname='test'\n")
    (remote_code_root / "scripts/cloud/beam_task2_p14.py").write_text(
        "PROVIDER_NAME = 'modal'\n"
    )
    relocated_entrypoint = container_root / "modal_task2_p14.py"
    relocated_source = SCRIPT.read_text(encoding="utf-8").replace(
        'REMOTE_CODE_ROOT = Path("/root/udsc2026")',
        f"REMOTE_CODE_ROOT = Path({str(remote_code_root)!r})",
        1,
    )
    relocated_entrypoint.write_text(relocated_source, encoding="utf-8")
    monkeypatch.setitem(sys.modules, "modal", None)

    spec = importlib.util.spec_from_file_location(
        "relocated_modal_task2_p14_test", relocated_entrypoint
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.PROJECT_ROOT == remote_code_root.resolve()


def test_local_input_verifies_size_and_sha256(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load(monkeypatch, with_modal=False)
    archive = tmp_path / "task2_p14_beam_input.tar.zst"
    archive.write_bytes(b"verified archive")
    manifest = archive.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": module._sha256(archive),
            }
        ),
        encoding="utf-8",
    )

    observed_archive, observed_manifest, payload = module._local_input(archive)

    assert observed_archive == archive.resolve()
    assert observed_manifest == manifest.resolve()
    assert payload["archive_bytes"] == archive.stat().st_size
    archive.write_bytes(b"tampered")
    try:
        module._local_input(archive)
    except ValueError as exc:
        assert "size mismatch" in str(exc) or "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("tampered Modal input must be rejected")


def test_provider_stubs_load_shared_core_without_beam_sdk(monkeypatch: Any) -> None:
    module = _load(monkeypatch, with_modal=False)
    module._install_shared_core_stubs()

    assert sys.modules["beam"].Image is module._ProviderImageStub
    assert sys.modules["beta9"].TaskPolicy is module._ProviderTaskPolicyStub


def test_download_validator_requires_modal_summary(
    monkeypatch: Any, tmp_path: Path
) -> None:
    module = _load(monkeypatch, with_modal=False)
    result = tmp_path / "task2_p14_modal_result.zip"
    summary = {
        "schema_version": "task2-p14-modal-full-v1",
        "provider": "modal",
        "status": "PUBLIC_CANDIDATE_READY",
    }
    with zipfile.ZipFile(result, "w") as bundle:
        bundle.writestr("results/full_run_summary.json", json.dumps(summary))

    assert module._validate_download(result) == summary


def test_extract_submission_validates_nested_zip(
    monkeypatch: Any, tmp_path: Path
) -> None:
    module = _load(monkeypatch, with_modal=False)
    nested_buffer = io.BytesIO()
    with zipfile.ZipFile(nested_buffer, "w") as nested:
        nested.writestr("submission.json", '{"q1":{"answer":"ok"}}')
    result = tmp_path / "result.zip"
    with zipfile.ZipFile(result, "w") as bundle:
        bundle.writestr("results/runs/abc/submission.zip", nested_buffer.getvalue())

    output = module._extract_submission(result, tmp_path)

    assert output == tmp_path / "submission.zip"
    with zipfile.ZipFile(output) as submission:
        assert submission.namelist() == ["submission.json"]


def test_wrapper_exposes_guarded_lifecycle() -> None:
    wrapper = Path("scripts/cloud/run_task2_p14_modal.ps1").read_text(
        encoding="utf-8-sig"
    )

    stages = (
        "Setup",
        "Check",
        "Prepare",
        "Smoke",
        "Full",
        "Run",
        "Public",
        "Status",
        "Logs",
        "Download",
        "Stop",
    )
    for stage in stages:
        assert f'"{stage}"' in wrapper
    assert "task2_p14_modal_result.zip" in wrapper
    assert "Invoke-NativeChecked" in wrapper
