"""Run the guarded Task 2 P14 pipeline on Modal.

The training implementation remains shared with the already tested Beam runner;
this module only replaces the unreliable Beam allocation/client layer with
Modal functions and a persistent Modal Volume.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import sys
import types
import zipfile
from pathlib import Path
from typing import Any, Callable

try:
    import modal
except ImportError:  # Local unit tests and ``--help`` remain model-free.
    modal = None  # type: ignore[assignment]

APP_NAME = "udsc-task2-p14-modal"
VOLUME_NAME = "udsc-task2-p14-modal"
VOLUME_MOUNT = "/mnt/task2"
REMOTE_CODE_ROOT = Path("/root/udsc2026")


def _is_project_root(candidate: Path) -> bool:
    """Return whether *candidate* contains the Task 2 runtime source tree."""

    return (
        (candidate / "pyproject.toml").is_file()
        and (candidate / "scripts/cloud/beam_task2_p14.py").is_file()
        and (candidate / "src").is_dir()
    )


def _resolve_project_root(
    module_file: str | Path | None = None,
    *,
    remote_code_root: str | Path = REMOTE_CODE_ROOT,
    cwd: str | Path | None = None,
) -> Path:
    """Locate the repository without assuming a fixed entrypoint depth.

    Modal relocates this entrypoint to ``/root/modal_task2_p14.py`` when a
    function starts.  The source tree baked into the image remains at
    ``/root/udsc2026``.  Walking for project markers works in both that layout
    and the normal local ``scripts/cloud`` layout.
    """

    module_path = Path(module_file or __file__).expanduser().resolve()
    working_dir = Path(cwd or Path.cwd()).expanduser().resolve()
    preferred_remote = Path(remote_code_root).expanduser().resolve()
    candidates: list[Path] = []
    # Modal places the relocated entrypoint beside the baked source directory:
    # /root/modal_task2_p14.py and /root/udsc2026. Prefer that exact layout so
    # an unrelated working directory cannot win the marker search.
    if module_path.parent == preferred_remote.parent:
        candidates.append(preferred_remote)
    candidates.extend(
        [
            module_path.parent,
            *module_path.parents,
            working_dir,
            *working_dir.parents,
            preferred_remote,
        ]
    )
    checked: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(key)
        if _is_project_root(candidate):
            return candidate
    raise RuntimeError(
        "Cannot locate the Task 2 project root. Checked: " + ", ".join(checked)
    )


PROJECT_ROOT = _resolve_project_root()
DEFAULT_ARCHIVE = PROJECT_ROOT / "artifacts/task2/task2_p14_beam_input.tar.zst"
DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "artifacts/task2/modal_download"
REMOTE_ARCHIVE = "input/task2_p14_beam_input.tar.zst"
REMOTE_MANIFEST = "input/task2_p14_beam_input.tar.manifest.json"
REMOTE_RESULT = "results/task2_p14_modal_result.zip"
REMOTE_SUMMARY = "results/full_run_summary.json"

# Modal tries the list in order and uses the first GPU type with capacity.
# All three support bfloat16 and have enough VRAM for the validated profiles.
MODAL_GPU_PRIORITY = ["H100", "A100-80GB", "L40S"]
PIPELINE_GPU_PROFILE = "H100,A100,L40S"
TORCH_VERSION = "2.10.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
PYTHON_PACKAGES = (
    "accelerate>=1.1,<2.0",
    "huggingface-hub>=1.3,<2.0",
    "nltk>=3.8,<4.0",
    "numpy==1.26.4",
    "peft>=0.17,<1.0",
    "pydantic>=2.8,<3.0",
    "pyyaml>=6.0",
    "rapidfuzz>=3.9,<4.0",
    "rouge-score>=0.1,<1.0",
    "safetensors>=0.5,<1.0",
    "scikit-learn>=1.5,<2.0",
    "sentencepiece>=0.2,<1.0",
    "tqdm>=4.66,<5.0",
    "transformers==5.0.0",
    "zstandard==0.23.0",
)
ACCEPTED_STATUSES = {
    "input": {"INPUT_READY"},
    "model": {"MODEL_READY"},
    "smoke": {"SMOKE_PASS"},
    "full": {"PUBLIC_CANDIDATE_READY", "HELDOUT_REJECTED"},
    "public": {"PUBLIC_CANDIDATE_READY"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_input(archive: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    resolved = Path(archive).expanduser().resolve()
    manifest_path = resolved.with_suffix(".manifest.json")
    if not resolved.is_file():
        raise FileNotFoundError(f"Task 2 input archive is missing: {resolved}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Task 2 input manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise ValueError("Task 2 input manifest must be a JSON object")
    expected_size = int(manifest.get("archive_bytes", -1))
    expected_hash = str(manifest.get("archive_sha256", "")).lower()
    if expected_size != resolved.stat().st_size:
        raise ValueError(
            f"Task 2 archive size mismatch: expected {expected_size}, "
            f"observed {resolved.stat().st_size}"
        )
    observed_hash = _sha256(resolved)
    if expected_hash != observed_hash:
        raise ValueError(
            f"Task 2 archive SHA-256 mismatch: expected {expected_hash}, "
            f"observed {observed_hash}"
        )
    return resolved, manifest_path, manifest


class _ProviderImageStub:
    """Minimal Beam image stand-in used only while loading the shared core."""

    def __init__(self, **_: Any) -> None:
        pass

    def add_python_packages(self, _: list[str]) -> _ProviderImageStub:
        return self


class _ProviderVolumeStub:
    def __init__(self, **_: Any) -> None:
        pass


class _ProviderTaskPolicyStub:
    def __init__(self, **_: Any) -> None:
        pass


class _ProviderHandlerStub:
    def __init__(self, handler: Callable[..., Any]) -> None:
        self._handler = handler
        self.parent = types.SimpleNamespace(handler=None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._handler(*args, **kwargs)

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self._handler(*args, **kwargs)


def _provider_function_stub(**_: Any) -> Callable[[Callable[..., Any]], Any]:
    def decorate(handler: Callable[..., Any]) -> _ProviderHandlerStub:
        return _ProviderHandlerStub(handler)

    return decorate


def _install_shared_core_stubs() -> None:
    """Let Modal import the provider-neutral core without installing Beam."""

    fake_beam = types.ModuleType("beam")
    fake_beam.Image = _ProviderImageStub
    fake_beam.Volume = _ProviderVolumeStub
    fake_beam.function = _provider_function_stub
    fake_beta9 = types.ModuleType("beta9")
    fake_beta9.TaskPolicy = _ProviderTaskPolicyStub
    sys.modules["beam"] = fake_beam
    sys.modules["beta9"] = fake_beta9


def _load_pipeline() -> Any:
    os.environ["UDSC_TASK2_PROVIDER"] = "modal"
    os.environ["UDSC_BEAM_GPU"] = PIPELINE_GPU_PROFILE
    code_root = PROJECT_ROOT
    if not _is_project_root(code_root):
        raise RuntimeError(f"Task 2 runtime source tree is incomplete: {code_root}")
    for root in (code_root, code_root / "src"):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    _install_shared_core_stubs()
    pipeline = importlib.import_module("scripts.cloud.beam_task2_p14")
    if pipeline.PROVIDER_NAME != "modal":
        raise RuntimeError("Task 2 shared core was imported with the wrong provider")
    return pipeline


def _assert_status(stage: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Modal {stage} returned no JSON result")
    status = str(payload.get("status", ""))
    if status not in ACCEPTED_STATUSES[stage]:
        raise RuntimeError(
            f"Modal {stage} failed with status {status or '<missing>'}: "
            + json.dumps(payload, ensure_ascii=False)
        )
    return payload


def _read_volume_json(volume: Any, remote_path: str) -> dict[str, Any] | None:
    try:
        content = b"".join(volume.read_file(remote_path))
    except FileNotFoundError:
        return None
    if not content:
        return None
    payload = json.loads(content.decode("utf-8-sig"))
    return payload if isinstance(payload, dict) else None


def _validate_download(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Modal result is missing or empty: {path}")
    with zipfile.ZipFile(path) as bundle:
        corrupt = bundle.testzip()
        if corrupt is not None:
            raise ValueError(f"Modal result contains corrupt member: {corrupt}")
        summaries = [
            name for name in bundle.namelist() if name.endswith("full_run_summary.json")
        ]
        if len(summaries) != 1:
            raise ValueError("Modal result must contain exactly one full summary")
        summary = json.loads(bundle.read(summaries[0]).decode("utf-8-sig"))
    if not isinstance(summary, dict) or summary.get("provider") != "modal":
        raise ValueError("Modal result summary has an invalid provider contract")
    if summary.get("status") not in ACCEPTED_STATUSES["full"]:
        raise ValueError("Modal result summary has an invalid completion status")
    return summary


def _extract_submission(result: Path, destination: Path) -> Path:
    """Extract and validate the sole nested official submission ZIP."""

    with zipfile.ZipFile(result) as bundle:
        candidates = [
            name
            for name in bundle.namelist()
            if name == "submission.zip" or name.endswith("/submission.zip")
        ]
        if len(candidates) != 1:
            raise ValueError("Modal result must contain exactly one submission.zip")
        content = bundle.read(candidates[0])
    with zipfile.ZipFile(io.BytesIO(content)) as submission:
        if submission.testzip() is not None:
            raise ValueError("nested Modal submission ZIP is corrupt")
        if submission.namelist() != ["submission.json"]:
            raise ValueError(
                "nested Modal submission must contain only submission.json"
            )
    output = destination / "submission.zip"
    temporary = output.with_name(f".{output.name}.partial")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_bytes(content)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


if modal is not None:
    app = modal.App(APP_NAME)
    task2_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    pipeline_image = (
        # Use Modal's own base instead of relying on a third-party registry
        # image.  The CUDA 12.8 wheel contains the runtime libraries; Modal
        # supplies the host NVIDIA driver for the allocated GPU.
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libgomp1")
        .pip_install(
            f"torch=={TORCH_VERSION}",
            index_url=TORCH_INDEX_URL,
        )
        .pip_install(*PYTHON_PACKAGES)
    )
    for _relative in ("scripts", "src", "configs"):
        _source = PROJECT_ROOT / _relative
        if _source.is_dir():
            pipeline_image = pipeline_image.add_local_dir(
                str(_source),
                remote_path=(REMOTE_CODE_ROOT / _relative).as_posix(),
                copy=True,
            )
    pipeline_image = pipeline_image.add_local_file(
        str(PROJECT_ROOT / "pyproject.toml"),
        remote_path=(REMOTE_CODE_ROOT / "pyproject.toml").as_posix(),
        copy=True,
    )

    def _commit_result(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        try:
            return operation()
        finally:
            task2_volume.commit()

    @app.function(
        image=pipeline_image,
        cpu=2,
        memory=4096,
        timeout=15 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def input_probe() -> dict[str, Any]:
        task2_volume.reload()
        pipeline = _load_pipeline()
        payload = pipeline._input_state(verify_hash=True)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload

    @app.function(
        image=pipeline_image,
        cpu=4,
        memory=16384,
        timeout=60 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def model_prepare() -> dict[str, Any]:
        task2_volume.reload()
        pipeline = _load_pipeline()

        def operation() -> dict[str, Any]:
            with pipeline._exclusive_run():
                pipeline._download_qwen()
                return {
                    "schema_version": "task2-p14-modal-model-v1",
                    "provider": "modal",
                    "status": "MODEL_READY",
                    "repo_id": pipeline.QWEN_REPO,
                    "revision": pipeline.QWEN_REVISION,
                    "manifest": pipeline._read_json(pipeline.QWEN_CACHE_MANIFEST),
                }

        return _commit_result(lambda: pipeline._guarded("model", operation))

    @app.function(
        image=pipeline_image,
        gpu=MODAL_GPU_PRIORITY,
        cpu=8,
        memory=65536,
        timeout=30 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def smoke() -> dict[str, Any]:
        task2_volume.reload()
        pipeline = _load_pipeline()

        def operation() -> dict[str, Any]:
            print("MODAL_REMOTE_STAGE_START=smoke", flush=True)
            with pipeline._exclusive_run():
                return pipeline._run_smoke()

        return _commit_result(lambda: pipeline._guarded("smoke", operation))

    @app.function(
        image=pipeline_image,
        gpu=MODAL_GPU_PRIORITY,
        cpu=8,
        memory=65536,
        timeout=8 * 60 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def full() -> dict[str, Any]:
        task2_volume.reload()
        pipeline = _load_pipeline()

        def operation() -> dict[str, Any]:
            print("MODAL_REMOTE_STAGE_START=full", flush=True)
            with pipeline._exclusive_run():
                return pipeline._run_full()

        return _commit_result(lambda: pipeline._guarded("full", operation))

    @app.function(
        image=pipeline_image,
        gpu=MODAL_GPU_PRIORITY,
        cpu=8,
        memory=65536,
        timeout=10 * 60 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def run_all() -> dict[str, Any]:
        """Run Smoke and Full on one allocation so no second GPU can stall."""

        task2_volume.reload()
        pipeline = _load_pipeline()

        def operation() -> dict[str, Any]:
            with pipeline._exclusive_run():
                print("MODAL_REMOTE_STAGE_START=smoke", flush=True)
                smoke_result = pipeline._guarded("smoke", pipeline._run_smoke)
                if smoke_result.get("status") != "SMOKE_PASS":
                    return smoke_result
                print("MODAL_REMOTE_STAGE_START=full", flush=True)
                return pipeline._run_full()

        return _commit_result(lambda: pipeline._guarded("full", operation))

    @app.function(
        image=pipeline_image,
        gpu=MODAL_GPU_PRIORITY,
        cpu=8,
        memory=65536,
        timeout=4 * 60 * 60,
        retries=0,
        volumes={VOLUME_MOUNT: task2_volume},
    )
    def public_candidate() -> dict[str, Any]:
        """Resume trained artifacts and generate only the public candidate."""

        task2_volume.reload()
        pipeline = _load_pipeline()

        def operation() -> dict[str, Any]:
            print("MODAL_REMOTE_STAGE_START=public", flush=True)
            with pipeline._exclusive_run():
                return pipeline._run_public_candidate()

        return _commit_result(lambda: pipeline._guarded("public", operation))

    def _upload_input(archive: str | Path) -> dict[str, Any]:
        local_archive, local_manifest, manifest = _local_input(archive)
        expected_hash = str(manifest["archive_sha256"])
        current = input_probe.remote()
        if (
            isinstance(current, dict)
            and current.get("status") == "INPUT_READY"
            and current.get("expected_sha256") == expected_hash
        ):
            print("MODAL_INPUT_ALREADY_READY", flush=True)
            return current
        archive_gb = local_archive.stat().st_size / 1e9
        print(
            f"Uploading verified Task 2 archive ({archive_gb:.2f} GB)",
            flush=True,
        )
        with task2_volume.batch_upload(force=True) as batch:
            batch.put_file(str(local_archive), f"/{REMOTE_ARCHIVE}")
            batch.put_file(str(local_manifest), f"/{REMOTE_MANIFEST}")
        verified = input_probe.remote()
        return _assert_status("input", verified)

    def _download_result(download_dir: str | Path) -> tuple[Path, dict[str, Any]]:
        destination = Path(download_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / Path(REMOTE_RESULT).name
        temporary = output.with_name(f".{output.name}.partial")
        temporary.unlink(missing_ok=True)
        try:
            with temporary.open("wb") as stream:
                for chunk in task2_volume.read_file(REMOTE_RESULT):
                    stream.write(chunk)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        summary = _validate_download(output)
        print(f"MODAL_RESULT_DOWNLOADED={output}", flush=True)
        if summary.get("status") == "PUBLIC_CANDIDATE_READY":
            submission = _extract_submission(output, destination)
            print(f"MODAL_SUBMISSION_READY={submission}", flush=True)
        print(
            "MODAL_RESULT_SUMMARY=" + json.dumps(summary, ensure_ascii=False),
            flush=True,
        )
        return output, summary

    @app.local_entrypoint()
    def main(
        stage: str = "run",
        archive: str = str(DEFAULT_ARCHIVE),
        download_dir: str = str(DEFAULT_DOWNLOAD_DIR),
    ) -> None:
        selected = stage.strip().lower()
        if selected == "check":
            _local_input(archive)
            print(
                json.dumps(
                    {
                        "status": "LOCAL_CHECK_PASS",
                        "app": APP_NAME,
                        "volume": VOLUME_NAME,
                        "gpu_priority": MODAL_GPU_PRIORITY,
                        "archive": str(Path(archive).resolve()),
                    },
                    indent=2,
                )
            )
            return
        if selected == "status":
            payload = _read_volume_json(task2_volume, REMOTE_SUMMARY)
            status_payload = payload or {"status": "NO_FULL_RESULT"}
            print(
                "MODAL_STATUS_JSON="
                + json.dumps(status_payload, ensure_ascii=False)
            )
            return
        if selected == "download":
            _download_result(download_dir)
            return
        if selected not in {"prepare", "smoke", "full", "run", "public"}:
            raise ValueError(
                "stage must be one of check, prepare, smoke, full, run, public, "
                "status, download"
            )
        if selected in {"prepare", "smoke", "run", "public"}:
            _upload_input(archive)
            model = _assert_status("model", model_prepare.remote())
            print("MODAL MODEL COMPLETE: " + str(model["status"]), flush=True)
        if selected == "smoke":
            smoke_result = _assert_status("smoke", smoke.remote())
            print("MODAL SMOKE COMPLETE: " + str(smoke_result["status"]), flush=True)
        if selected == "full":
            full_result = _assert_status("full", full.remote())
            print("MODAL FULL COMPLETE: " + str(full_result["status"]), flush=True)
            _download_result(download_dir)
        if selected == "run":
            full_result = _assert_status("full", run_all.remote())
            print("MODAL FULL COMPLETE: " + str(full_result["status"]), flush=True)
            _download_result(download_dir)
        if selected == "public":
            public_result = _assert_status("public", public_candidate.remote())
            print(
                "MODAL PUBLIC COMPLETE: " + str(public_result["status"]),
                flush=True,
            )
            _download_result(download_dir)

else:
    app = None
    task2_volume = None


if __name__ == "__main__" and modal is None:
    raise SystemExit(
        "Modal SDK is missing. Install it with: python -m pip install -U modal"
    )
