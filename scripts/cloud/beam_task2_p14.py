"""Run the guarded Task 2 P14 pipeline on one persistent Beam GPU Volume."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from beam import Image, Volume, function
from beta9 import TaskPolicy

VOLUME_NAME = "udsc-task2-p14"
VOLUME_MOUNT = "/mnt/task2"
VOLUME_ROOT = Path(VOLUME_MOUNT)
PROVIDER_NAME = os.getenv("UDSC_TASK2_PROVIDER", "beam").strip().lower()
if PROVIDER_NAME not in {"beam", "modal"}:
    raise ValueError("UDSC_TASK2_PROVIDER must be either 'beam' or 'modal'")
PROVIDER_DISPLAY = PROVIDER_NAME.capitalize()
INPUT_ARCHIVE = VOLUME_ROOT / "input/task2_p14_beam_input.tar.zst"
INPUT_MANIFEST = VOLUME_ROOT / "input/task2_p14_beam_input.tar.manifest.json"
WORKSPACE = VOLUME_ROOT / "workspace"
MODEL_ROOT = VOLUME_ROOT / "models"
RESULT_ROOT = VOLUME_ROOT / "results"
LOCAL_RUNTIME_ROOT = Path("/tmp/udsc-task2-p14-runtime")
MODULE_NAME = "scripts.cloud.beam_task2_p14"
RUN_CONTRACT_VERSION = "p14-fast-v4"
SOURCE_DIRECTORIES = ("scripts", "src", "configs")
SOURCE_FILES = ("pyproject.toml",)
GPU_PROFILE = os.getenv("UDSC_BEAM_GPU", "RTX5090,RTX4090,A10G").strip().upper()
GPU_TYPES = tuple(
    dict.fromkeys(value.strip() for value in GPU_PROFILE.split(",") if value.strip())
)
GPU_REQUEST: str | list[str] = GPU_TYPES[0] if len(GPU_TYPES) == 1 else list(GPU_TYPES)
POOL_NAME = os.getenv("UDSC_BEAM_POOL", "").strip() or None
ALLOCATION_NONCE = "".join(
    character
    for character in os.getenv("UDSC_BEAM_ALLOCATION_NONCE", "").lower()
    if character.isalnum()
)[:12]
RUNTIME_ENV = {"UDSC_BEAM_GPU": GPU_PROFILE}
if POOL_NAME is not None:
    RUNTIME_ENV["UDSC_BEAM_POOL"] = POOL_NAME
if ALLOCATION_NONCE:
    RUNTIME_ENV["UDSC_BEAM_ALLOCATION_NONCE"] = ALLOCATION_NONCE
GPU_CAPABILITY_MINIMUMS = {
    "A10G": (8, 6),
    # The two entries below are also used by the provider-neutral training
    # core when Modal supplies an A100 80 GB or L40S fallback.  Beam's Windows
    # wrapper continues to expose only the Beam GPU names it actually accepts.
    "A100": (8, 0),
    "L40S": (8, 9),
    "RTX4090": (8, 9),
    "RTX5090": (12, 0),
    "H100": (9, 0),
}
GPU_NAME_MARKERS = {
    "A10G": ("A10G",),
    "A100": ("A100",),
    "L40S": ("L40S",),
    "RTX4090": ("RTX 4090", "RTX4090"),
    "RTX5090": ("RTX 5090", "RTX5090"),
    "H100": ("H100",),
}
if not GPU_TYPES or any(value not in GPU_CAPABILITY_MINIMUMS for value in GPU_TYPES):
    raise ValueError(
        "UDSC_BEAM_GPU must be a comma-separated priority list drawn from "
        f"{sorted(GPU_CAPABILITY_MINIMUMS)}"
    )

QWEN_REPO = "thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2"
QWEN_REVISION = "b270ee9f3c8ea72cea9ff0f82ece6b40bb8b67a8"
QWEN_DIR = MODEL_ROOT / "qwen3-legal"
QWEN_CACHE_MANIFEST = QWEN_DIR / ".beam_model_manifest.json"
PARENT_TRAIN_DATA_SHA256 = (
    "44acfda30b42ebd009e80c816ebfa26121fd01e91430e1c6837f200b7e6a5bec"
)
PARENT_EVAL_DATA_SHA256 = (
    "22fd5b07e1a32ade741aff78398be66bfb4b645ecb9db42e74e016e3fc9e9cc8"
)
MINIMUM_FREE_VRAM_BYTES = 16 * 1024**3
MINIMUM_FREE_VRAM_RATIO = 0.80
PROMOTION_METEOR = float(os.getenv("UDSC_TASK2_MINIMUM_METEOR", "0.57"))
if not 0.0 <= PROMOTION_METEOR <= 1.0:
    raise ValueError("UDSC_TASK2_MINIMUM_METEOR must be in [0, 1]")
FAST_PUBLIC_PROMOTION_METEOR = 0.56
FAST_PUBLIC_EVIDENCE_WORDS = 352
PUBLIC_PARENT_SCORE_TIMEOUT_SECONDS = 20 * 60
RERANKER_PREFLIGHT_TIMEOUT_SECONDS = 300
QWEN_PREFLIGHT_TIMEOUT_SECONDS = 900
SMOKE_TIMEOUT_SECONDS = 30 * 60
SMOKE_TTL_SECONDS = 45 * 60
FULL_TIMEOUT_SECONDS = 8 * 60 * 60
FULL_TTL_SECONDS = 12 * 60 * 60
FULL_SOURCE_SCRIPTS = (
    "scripts/training/train_legal_qa_parent_crossencoder.py",
    "scripts/training/finetune_task2_qwen_lora.py",
    "scripts/submission/build_legal_qa_parent_scores.py",
    "scripts/submission/write_legal_qa_submission.py",
    "scripts/submission/validate_legal_qa_submission.py",
)
BASE_IMAGE = "docker.io/pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
PYTHON_PACKAGES = [
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
]
BEAM_IMAGE = Image(base_image=BASE_IMAGE).add_python_packages(PYTHON_PACKAGES)


def _function_name(stage: str) -> str:
    base = f"udsc-task2-p14-{stage}"
    return f"{base}-{ALLOCATION_NONCE}" if ALLOCATION_NONCE else base


def _volume() -> Volume:
    return Volume(name=VOLUME_NAME, mount_path=VOLUME_MOUNT)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _input_state(*, verify_hash: bool) -> dict[str, Any]:
    if not INPUT_MANIFEST.is_file():
        return {"status": "INPUT_INVALID", "reason": "manifest_missing"}
    manifest = _read_json(INPUT_MANIFEST)
    expected_size = int(manifest.get("archive_bytes", -1))
    expected_hash = str(manifest.get("archive_sha256", ""))
    is_file = INPUT_ARCHIVE.is_file()
    size = INPUT_ARCHIVE.stat().st_size if is_file else 0
    observed_hash = None
    if verify_hash and is_file and size == expected_size:
        observed_hash = _sha256(INPUT_ARCHIVE)
    ready = (
        is_file
        and size == expected_size
        and (not verify_hash or observed_hash == expected_hash)
    )
    return {
        "schema_version": f"task2-p14-{PROVIDER_NAME}-input-probe-v1",
        "status": "INPUT_READY" if ready else "INPUT_INVALID",
        "archive": str(INPUT_ARCHIVE),
        "manifest": str(INPUT_MANIFEST),
        "is_file": is_file,
        "size": size,
        "expected_size": expected_size,
        "sha256": observed_hash,
        "expected_sha256": expected_hash,
    }


def _extract_workspace() -> dict[str, Any]:
    state = _input_state(verify_hash=True)
    if state["status"] != "INPUT_READY":
        raise FileNotFoundError(
            f"{PROVIDER_DISPLAY} Task 2 input is not ready: {state}"
        )
    archive_hash = str(state["expected_sha256"])
    marker = WORKSPACE / ".task2_p14_input.json"
    if marker.is_file() and _read_json(marker).get("archive_sha256") == archive_hash:
        return state
    staging = VOLUME_ROOT / ".workspace_extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    import zstandard

    print(
        f"extracting {INPUT_ARCHIVE} ({INPUT_ARCHIVE.stat().st_size / 1e9:.2f} GB)",
        flush=True,
    )
    with INPUT_ARCHIVE.open("rb") as source:
        with zstandard.ZstdDecompressor().stream_reader(source) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                archive.extractall(staging, filter="data")
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    os.replace(staging, WORKSPACE)
    _write_json(marker, {"archive_sha256": archive_hash})
    print("finished extracting Task 2 archive", flush=True)
    return state


def _source_tree_sha256(project_root: Path) -> str:
    """Hash exactly the source files synchronized by the Beam ignore contract."""

    digest = hashlib.sha256()
    hashed: list[Path] = []
    for relative in SOURCE_FILES:
        source = project_root / relative
        if source.is_file():
            hashed.append(source)
    for relative in SOURCE_DIRECTORIES:
        root = project_root / relative
        if root.is_dir():
            hashed.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
    for path in sorted(
        hashed,
        key=lambda item: item.relative_to(project_root).as_posix(),
    ):
        digest.update(path.relative_to(project_root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _code_root() -> Path:
    """Return Beam's immutable, container-local synchronized source root."""

    return Path(__file__).resolve().parents[2]


def _overlay_source() -> str:
    """Hash local source without ever copying/executing code on Beam Volume.

    Beam Volume is appropriate for persistent data and checkpoints, but its
    metadata may be briefly inconsistent after writes. All Python source and
    imports must stay on the immutable container-local ``/mnt/code`` snapshot.
    """

    return _source_tree_sha256(_code_root())


def _run_contract(archive_sha256: str, source_sha256: str) -> str:
    material = f"{archive_sha256}:{RUN_CONTRACT_VERSION}:{source_sha256}"
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def _download_qwen() -> None:
    required = (QWEN_DIR / "config.json", QWEN_DIR / "model.safetensors")
    cached = _read_json(QWEN_CACHE_MANIFEST) if QWEN_CACHE_MANIFEST.is_file() else {}
    if (
        cached.get("repo_id") == QWEN_REPO
        and cached.get("revision") == QWEN_REVISION
        and all(path.is_file() and path.stat().st_size > 0 for path in required)
        and cached.get("model_bytes") == (QWEN_DIR / "model.safetensors").stat().st_size
        and cached.get("config_sha256") == _sha256(QWEN_DIR / "config.json")
        and int(cached.get("tensor_count", 0)) > 0
    ):
        print(f"Qwen cache ready: {QWEN_DIR}", flush=True)
        return
    from huggingface_hub import snapshot_download

    QWEN_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"verifying/downloading {QWEN_REPO}@{QWEN_REVISION} into "
        f"{PROVIDER_DISPLAY} Volume",
        flush=True,
    )
    snapshot_download(
        repo_id=QWEN_REPO,
        revision=QWEN_REVISION,
        local_dir=QWEN_DIR,
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise RuntimeError("Qwen download completed without required model files")
    from safetensors import safe_open

    try:
        with safe_open(QWEN_DIR / "model.safetensors", framework="pt") as weights:
            tensor_count = len(weights.keys())
    except Exception as exc:
        QWEN_CACHE_MANIFEST.unlink(missing_ok=True)
        (QWEN_DIR / "model.safetensors").unlink(missing_ok=True)
        raise RuntimeError(
            "Qwen safetensors cache is corrupt; the bad weight file was removed. "
            "Retry Prepare/Run to download it again."
        ) from exc
    if tensor_count < 1:
        raise RuntimeError("Qwen safetensors validation returned no tensors")
    _write_json(
        QWEN_CACHE_MANIFEST,
        {
            "schema_version": 1,
            "repo_id": QWEN_REPO,
            "revision": QWEN_REVISION,
            "model_bytes": (QWEN_DIR / "model.safetensors").stat().st_size,
            "tensor_count": tensor_count,
            "config_sha256": _sha256(QWEN_DIR / "config.json"),
            "completed_at_utc": _utc_now(),
        },
    )


def _environment() -> dict[str, str]:
    code_root = _code_root()
    roots = [str(code_root), str(code_root / "src")]
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(roots),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TQDM_DISABLE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def _run(
    command: Sequence[str],
    *,
    accepted: set[int] | None = None,
    timeout_seconds: int | None = None,
) -> int:
    print("RUN:\n" + " ".join(command), flush=True)
    try:
        completed = subprocess.run(
            list(command),
            cwd=WORKSPACE,
            env=_environment(),
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        print(
            f"PROCESS TIMEOUT: no completion after {timeout_seconds} seconds",
            flush=True,
        )
        raise RuntimeError(
            f"Beam subprocess exceeded {timeout_seconds} seconds: " + " ".join(command)
        ) from exc
    print(f"PROCESS EXIT: {completed.returncode}", flush=True)
    allowed = {0} if accepted is None else accepted
    if completed.returncode not in allowed:
        raise subprocess.CalledProcessError(completed.returncode, list(command))
    return completed.returncode


def _python(*arguments: str) -> list[str]:
    resolved = list(arguments)
    if resolved:
        script = Path(resolved[0])
        if not script.is_absolute() and script.parts[:1] == ("scripts",):
            resolved[0] = str(_code_root() / script)
    return [sys.executable, *resolved]


def _gpu_manifest() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Beam allocated no usable CUDA GPU")
    device_name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    actual_type = next(
        (
            requested
            for requested in GPU_TYPES
            if any(
                marker in device_name.upper() for marker in GPU_NAME_MARKERS[requested]
            )
        ),
        None,
    )
    if actual_type is None:
        raise RuntimeError(
            f"Beam allocated {device_name!r}, which does not match requested "
            f"GPU priority list {list(GPU_TYPES)}"
        )
    minimum = GPU_CAPABILITY_MINIMUMS[actual_type]
    if capability < minimum:
        raise RuntimeError(
            f"GPU capability {capability} is below {minimum} for {actual_type}"
        )
    torch.cuda.empty_cache()
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    diagnostic = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return {
        "name": device_name,
        "actual_type": actual_type,
        "requested_types": list(GPU_TYPES),
        "pool": POOL_NAME,
        "allocation_nonce": ALLOCATION_NONCE or None,
        "capability": list(capability),
        # torch.__version__ is a torch.torch_version.TorchVersion subclass.
        # Returning it directly makes Beam's Windows client import torch while
        # unpickling the otherwise primitive result dictionary.  The Beam CLI
        # environment deliberately has no GPU stack, so keep the RPC payload
        # JSON/cloudpickle-portable just like the Task 1 runner does.
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "vram_bytes": int(total_vram),
        "free_vram_bytes": int(free_vram),
        "current_pid": os.getpid(),
        "current_process_allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "current_process_reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "nvidia_smi_exit_code": diagnostic.returncode,
        "nvidia_smi_compute_processes": [
            line.strip() for line in diagnostic.stdout.splitlines() if line.strip()
        ],
        "nvidia_smi_stderr": diagnostic.stderr.strip(),
    }


def _require_free_vram(gpu: dict[str, Any]) -> None:
    print("GPU_PREFLIGHT_JSON=" + json.dumps(gpu, ensure_ascii=True), flush=True)
    free_vram = int(gpu["free_vram_bytes"])
    total_vram = int(gpu["vram_bytes"])
    required_vram = max(
        MINIMUM_FREE_VRAM_BYTES,
        int(total_vram * MINIMUM_FREE_VRAM_RATIO),
    )
    if gpu.get("actual_type") == "H100":
        required_vram = max(required_vram, 70 * 1024**3)
    gpu["required_free_vram_bytes"] = required_vram
    if free_vram < required_vram:
        raise RuntimeError(
            f"{PROVIDER_DISPLAY} GPU is already occupied before Task 2 starts: "
            f"only {free_vram / 1024**3:.2f} GiB of "
            f"{total_vram / 1024**3:.2f} GiB is free; "
            f"at least {required_vram / 1024**3:.2f} GiB is required. "
            f"Retry on another {PROVIDER_DISPLAY} GPU node; changing batch size "
            "cannot fix this host."
        )


def _performance_profile(gpu: dict[str, Any]) -> dict[str, int | str]:
    """Keep the effective batch constant while using larger GPUs efficiently."""

    total_gib = int(gpu["vram_bytes"]) / 1024**3
    if total_gib >= 70:
        return {
            "name": "h100-80gb",
            "parent_train_batch": 8,
            "parent_gradient_accumulation": 2,
            "parent_eval_batch": 16,
            "parent_score_batch": 128,
            "qwen_train_batch": 8,
            "qwen_gradient_accumulation": 2,
            "generation_batch": 16,
        }
    if total_gib >= 30:
        return {
            "name": "rtx5090-32gb",
            "parent_train_batch": 4,
            "parent_gradient_accumulation": 4,
            "parent_eval_batch": 8,
            "parent_score_batch": 48,
            "qwen_train_batch": 2,
            "qwen_gradient_accumulation": 8,
            "generation_batch": 8,
        }
    return {
        "name": "24gb-safe",
        "parent_train_batch": 2,
        "parent_gradient_accumulation": 8,
        "parent_eval_batch": 2,
        "parent_score_batch": 32,
        "qwen_train_batch": 1,
        "qwen_gradient_accumulation": 16,
        "generation_batch": 4,
    }


@contextmanager
def _exclusive_run() -> Iterator[None]:
    """Prevent duplicate headless jobs from writing the shared Volume."""

    fcntl: Any = importlib.import_module("fcntl")
    VOLUME_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = VOLUME_ROOT / ".task2_p14.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another Task 2 {PROVIDER_DISPLAY} job is already running; "
                "stop the duplicate "
                "or wait for it to finish"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _localize_directory(source: Path, label: str) -> Path:
    """Copy model files off the Beam Volume before safetensors mmap/CUDA use."""

    if not source.is_dir():
        raise FileNotFoundError(f"runtime model directory is missing: {source}")
    target = LOCAL_RUNTIME_ROOT / label
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"localizing model directory {source} -> {target}", flush=True)
    shutil.copytree(source, target)
    print(f"finished localizing {label}", flush=True)
    return target


def _reranker_cuda_preflight(performance: dict[str, int | str]) -> None:
    """Exercise the worst listwise microbatch, including AdamW state allocation."""

    local_parent_base = _localize_directory(
        WORKSPACE
        / "artifacts/task2/models"
        / "dek21-parent-crossencoder-hn16-v2/checkpoint-best",
        "parent_base_smoke",
    )
    _run(
        _python(
            "scripts/cloud/task2_p14_gpu_preflight.py",
            "--model-dir",
            str(local_parent_base),
            "--device",
            "cuda",
            "--max-length",
            "256",
            "--train-data",
            "artifacts/task2/training/parent_ce_hn16_train.jsonl",
            "--groups-per-batch",
            str(performance["parent_train_batch"]),
            "--gradient-accumulation",
            str(performance["parent_gradient_accumulation"]),
            "--freeze-layers",
            "6",
        ),
        timeout_seconds=RERANKER_PREFLIGHT_TIMEOUT_SECONDS,
    )


def _qwen_training_preflight(performance: dict[str, int | str]) -> None:
    """Run the exact worst-shape LoRA backward and optimizer allocation."""

    local_qwen_base = _localize_directory(QWEN_DIR, "qwen3-legal-smoke")
    smoke_output = LOCAL_RUNTIME_ROOT / "qwen_training_smoke"
    if smoke_output.exists():
        shutil.rmtree(smoke_output)
    _run(
        _python(
            "scripts/training/finetune_task2_qwen_lora.py",
            "train",
            "--train-data",
            "artifacts/task2/training/task2_qwen_sft_train.jsonl",
            "--model-dir",
            str(local_qwen_base),
            "--output-dir",
            str(smoke_output),
            "--epochs",
            "2",
            "--batch-size",
            str(performance["qwen_train_batch"]),
            "--gradient-accumulation",
            str(performance["qwen_gradient_accumulation"]),
            "--learning-rate",
            "2e-4",
            "--max-length",
            "3072",
            "--max-context-tokens",
            "1800",
            "--max-answer-tokens",
            "1024",
            "--lora-rank",
            "32",
            "--lora-alpha",
            "64",
            "--dtype",
            "bfloat16",
            "--device",
            "cuda",
            "--smoke-only",
        ),
        timeout_seconds=QWEN_PREFLIGHT_TIMEOUT_SECONDS,
    )


def _full_source_import_preflight() -> None:
    """Import every Full-stage entry point from immutable local source."""

    for relative in FULL_SOURCE_SCRIPTS:
        command = _python(relative, "--help")
        script = Path(command[1])
        if not script.is_file() or WORKSPACE in script.parents:
            raise RuntimeError(f"Full-stage source is not container-local: {script}")
        _run(command, timeout_seconds=60)
    print("FULL_SOURCE_IMPORT_PREFLIGHT_PASS", flush=True)


def _training_preflight(performance: dict[str, int | str]) -> None:
    _full_source_import_preflight()
    _reranker_cuda_preflight(performance)
    _qwen_training_preflight(performance)


def _run_smoke() -> dict[str, Any]:
    started = time.monotonic()
    smoke_result = RESULT_ROOT / "smoke.json"
    # Replace any earlier pass before doing work.  A worker loss or failed
    # retry must never leave Full able to consume a stale SMOKE_PASS from the
    # same source/input/GPU profile.
    smoke_result.unlink(missing_ok=True)
    _write_json(
        smoke_result,
        {
            "schema_version": f"task2-p14-{PROVIDER_NAME}-smoke-v1",
            "status": "SMOKE_RUNNING",
            "started_at_utc": _utc_now(),
        },
    )
    state = _extract_workspace()
    source_hash = _overlay_source()
    _download_qwen()
    gpu = _gpu_manifest()
    _require_free_vram(gpu)
    performance = _performance_profile(gpu)
    _training_preflight(performance)
    payload = {
        "schema_version": f"task2-p14-{PROVIDER_NAME}-smoke-v1",
        "provider": PROVIDER_NAME,
        "status": "SMOKE_PASS",
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "archive_sha256": state["expected_sha256"],
        "source_sha256": source_hash,
        "gpu": gpu,
        "performance_profile": performance,
        "qwen_repo": QWEN_REPO,
        "qwen_revision": QWEN_REVISION,
    }
    # _write_json commits through os.replace, so Full can observe either the
    # invalidated state above or this complete success, never a partial JSON.
    _write_json(smoke_result, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def _ensure_clean_incomplete(
    directory: Path,
    completion: Path,
    *,
    preserve_epoch_resume: bool = False,
) -> None:
    if completion.is_file():
        return
    qwen_resume = any(
        (checkpoint / "adapter_model.safetensors").is_file()
        and (checkpoint / "training_state.pt").is_file()
        for checkpoint in directory.glob("checkpoint-epoch-*")
    )
    parent_latest = directory / "checkpoint-latest"
    parent_resume = (
        (parent_latest / "training_state.pt").is_file()
        and (parent_latest / "config.json").is_file()
        and (parent_latest / "tokenizer_config.json").is_file()
        and (parent_latest / "training_metrics.json").is_file()
        and any(
            (parent_latest / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        )
    )
    if preserve_epoch_resume and (qwen_resume or parent_resume):
        print(f"preserving resumable partial output: {directory}", flush=True)
        return
    if directory.exists():
        shutil.rmtree(directory)


def _prediction_outputs_complete(
    predictions: Path,
    diagnostics: Path,
    *,
    questions: Path,
    question_ids: Path | None = None,
) -> bool:
    if not predictions.is_file() or not diagnostics.is_file():
        return False
    try:
        question_payload = json.loads(questions.read_text(encoding="utf-8-sig"))
        expected = (
            json.loads(question_ids.read_text(encoding="utf-8-sig"))
            if question_ids is not None
            else list(question_payload)
        )
        prediction_payload = json.loads(predictions.read_text(encoding="utf-8"))
        diagnostic_payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(expected, list):
        return False
    expected_ids = [str(value) for value in expected]
    prediction_ids = (
        [str(row.get("id", "")) for row in prediction_payload if isinstance(row, dict)]
        if isinstance(prediction_payload, list)
        else []
    )
    diagnostic_ids = (
        [str(row.get("id", "")) for row in diagnostic_payload if isinstance(row, dict)]
        if isinstance(diagnostic_payload, list)
        else []
    )
    return (
        len(expected_ids) == len(set(expected_ids))
        and prediction_ids == expected_ids
        and diagnostic_ids == expected_ids
    )


def _score_parents(
    checkpoint: Path,
    output: Path,
    *,
    public: bool,
    batch_size: int,
) -> None:
    questions = (
        "data/raw/btc/LegalQA/public-official.json"
        if public
        else "data/raw/btc/LegalQA/train.json"
    )
    candidates = (
        ["artifacts/task2/public-bge-reranker-all/predictions_after.jsonl"]
        if public
        else [
            "artifacts/task2/train500-bge-reranker-all/predictions_after.jsonl",
            "artifacts/task2/train500b-bge-reranker-all/predictions_after.jsonl",
        ]
    )
    command = _python(
        "scripts/training/train_legal_qa_parent_crossencoder.py",
        "score",
        "--questions",
        questions,
    )
    for candidate in candidates:
        command.extend(("--candidates", candidate))
    command.extend(
        (
            "--parents-dir",
            "data/processed_v3/parents",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--candidate-k",
            "50",
            "--batch-size",
            str(batch_size),
            "--max-length",
            "256",
            "--device",
            "cuda",
            "--amp",
        )
    )
    _run(
        command,
        timeout_seconds=(PUBLIC_PARENT_SCORE_TIMEOUT_SECONDS if public else None),
    )


def _build_result_archive(summary: dict[str, Any], run_root: Path) -> Path:
    archive = RESULT_ROOT / f"task2_p14_{PROVIDER_NAME}_result.zip"
    summary["download_path"] = (
        f"{PROVIDER_NAME}://{VOLUME_NAME}/results/{archive.name}"
    )
    _write_json(RESULT_ROOT / "full_run_summary.json", summary)
    temporary = archive.with_name(f".{archive.name}.partial")
    temporary.unlink(missing_ok=True)
    files: list[Path] = [RESULT_ROOT / "full_run_summary.json"]
    for relative in (
        "strict_predictions.json",
        "strict_diagnostics.json",
        "strict_metrics.json",
        "strict_ensemble_predictions.json",
        "strict_ensemble_predictions.manifest.json",
        "strict_ensemble_metrics.json",
        "extractive_predictions.json",
        "extractive_metrics.json",
        "public_rankings.jsonl",
        "public_extractive_predictions.json",
        "public_qwen_predictions.json",
        "public_qwen_diagnostics.json",
        "public_predictions.json",
        "public_predictions.manifest.json",
        "submission.zip",
    ):
        path = run_root / relative
        if path.is_file():
            files.append(path)
    adapter = run_root / "qwen_lora"
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_manifest.json",
    ):
        path = adapter / name
        if path.is_file():
            files.append(path)
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for path in files:
            bundle.write(path, arcname=path.relative_to(VOLUME_ROOT).as_posix())
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError(
                f"Task 2 {PROVIDER_DISPLAY} result archive is corrupt"
            )
    return archive


def _run_full() -> dict[str, Any]:
    started = time.monotonic()
    state = _extract_workspace()
    source_hash = _overlay_source()
    _download_qwen()
    gpu = _gpu_manifest()
    _require_free_vram(gpu)
    performance = _performance_profile(gpu)
    print(
        "PERFORMANCE_PROFILE_JSON=" + json.dumps(performance, ensure_ascii=True),
        flush=True,
    )
    smoke = RESULT_ROOT / "smoke.json"
    if not smoke.is_file():
        raise RuntimeError(
            f"run and pass Task 2 {PROVIDER_DISPLAY} Smoke before Full"
        )
    smoke_payload = _read_json(smoke)
    if (
        smoke_payload.get("status") != "SMOKE_PASS"
        or smoke_payload.get("archive_sha256") != state["expected_sha256"]
        or smoke_payload.get("source_sha256") != source_hash
    ):
        raise RuntimeError(
            f"Task 2 {PROVIDER_DISPLAY} Smoke is stale; run Smoke again before Full"
        )
    if (
        smoke_payload.get("gpu", {}).get("actual_type") != gpu.get("actual_type")
        or smoke_payload.get("performance_profile") != performance
    ):
        print(
            "Full received a different GPU profile than Smoke; running the "
            "exact training preflight again on this GPU.",
            flush=True,
        )
        _training_preflight(performance)
    contract = _run_contract(state["expected_sha256"], source_hash)
    run_root = RESULT_ROOT / "runs" / contract
    run_root.mkdir(parents=True, exist_ok=True)

    listwise = run_root / "parent_listwise"
    _ensure_clean_incomplete(
        listwise,
        listwise / "training_manifest.json",
        preserve_epoch_resume=True,
    )
    if not (listwise / "training_manifest.json").is_file():
        local_parent_base = _localize_directory(
            WORKSPACE
            / "artifacts/task2/models"
            / "dek21-parent-crossencoder-hn16-v2/checkpoint-best",
            "parent_base",
        )
        _run(
            _python(
                "scripts/training/train_legal_qa_parent_crossencoder.py",
                "train",
                "--train-data",
                "artifacts/task2/training/parent_ce_hn16_train.jsonl",
                "--eval-data",
                "artifacts/task2/training/parent_ce_hn16_eval.jsonl",
                "--preverified-input-archive-sha256",
                str(state["expected_sha256"]),
                "--preverified-train-data-sha256",
                PARENT_TRAIN_DATA_SHA256,
                "--preverified-eval-data-sha256",
                PARENT_EVAL_DATA_SHA256,
                "--model-dir",
                str(local_parent_base),
                "--output-dir",
                str(listwise),
                "--objective",
                "listwise",
                "--target-temperature",
                "0.10",
                "--epochs",
                "3",
                "--learning-rate",
                "1e-5",
                "--batch-size",
                str(performance["parent_train_batch"]),
                "--eval-batch-size",
                str(performance["parent_eval_batch"]),
                "--gradient-accumulation",
                str(performance["parent_gradient_accumulation"]),
                "--max-length",
                "256",
                "--device",
                "cuda",
                "--amp",
                "--resume",
            )
        )

    local_listwise_checkpoint = _localize_directory(
        listwise / "checkpoint-best",
        "parent_listwise_checkpoint",
    )
    strict_rankings = run_root / "strict_rankings.jsonl"
    if not strict_rankings.is_file():
        _score_parents(
            local_listwise_checkpoint,
            strict_rankings,
            public=False,
            batch_size=int(performance["parent_score_batch"]),
        )
    extractive = run_root / "extractive_predictions.json"
    if not extractive.is_file():
        _run(
            _python(
                "scripts/submission/build_legal_qa_parent_scores.py",
                "--scores",
                str(strict_rankings),
                "--questions",
                "data/raw/btc/LegalQA/train.json",
                "--question-ids",
                "artifacts/task2/training/parent_ce_eval_ids.json",
                "--output",
                str(extractive),
                "--top-parents",
                "2",
                "--max-parent-tokens",
                "512",
                "--max-total-tokens",
                "1024",
                "--ce-weight",
                "0.30",
                "--retrieval-weight",
                "0.70",
                "--rrf-k",
                "0",
            )
        )
    extractive_metrics = run_root / "extractive_metrics.json"
    if not extractive_metrics.is_file():
        _run(
            _python(
                "scripts/training/finetune_task2_qwen_lora.py",
                "evaluate",
                "--questions",
                "data/raw/btc/LegalQA/train.json",
                "--question-ids",
                "artifacts/task2/training/parent_ce_eval_ids.json",
                "--predictions",
                str(extractive),
                "--output",
                str(extractive_metrics),
                "--minimum-meteor",
                "0",
            )
        )

    qwen_lora = run_root / "qwen_lora"
    _ensure_clean_incomplete(
        qwen_lora,
        qwen_lora / "training_manifest.json",
        preserve_epoch_resume=True,
    )
    local_qwen_base = _localize_directory(QWEN_DIR, "qwen3-legal")
    if not (qwen_lora / "training_manifest.json").is_file():
        _run(
            _python(
                "scripts/training/finetune_task2_qwen_lora.py",
                "train",
                "--train-data",
                "artifacts/task2/training/task2_qwen_sft_train.jsonl",
                "--model-dir",
                str(local_qwen_base),
                "--output-dir",
                str(qwen_lora),
                "--epochs",
                "2",
                "--batch-size",
                str(performance["qwen_train_batch"]),
                "--gradient-accumulation",
                str(performance["qwen_gradient_accumulation"]),
                "--learning-rate",
                "2e-4",
                "--max-length",
                "3072",
                "--max-context-tokens",
                "1800",
                "--max-answer-tokens",
                "1024",
                "--lora-rank",
                "32",
                "--lora-alpha",
                "64",
                "--dtype",
                "bfloat16",
                "--device",
                "cuda",
                "--resume",
            )
        )

    persistent_adapter_config = qwen_lora / "adapter_config.json"
    if persistent_adapter_config.is_file():
        adapter_payload = _read_json(persistent_adapter_config)
        adapter_payload["base_model_name_or_path"] = QWEN_REPO
        adapter_payload["revision"] = QWEN_REVISION
        _write_json(persistent_adapter_config, adapter_payload)

    local_qwen_lora = _localize_directory(qwen_lora, "qwen_lora")
    adapter_config = local_qwen_lora / "adapter_config.json"
    if adapter_config.is_file():
        adapter_payload = _read_json(adapter_config)
        adapter_payload["base_model_name_or_path"] = str(local_qwen_base)
        adapter_payload["revision"] = None
        _write_json(adapter_config, adapter_payload)

    strict_predictions = run_root / "strict_predictions.json"
    strict_diagnostics = run_root / "strict_diagnostics.json"
    if not _prediction_outputs_complete(
        strict_predictions,
        strict_diagnostics,
        questions=WORKSPACE / "data/raw/btc/LegalQA/train.json",
        question_ids=(WORKSPACE / "artifacts/task2/training/parent_ce_eval_ids.json"),
    ):
        _run(
            _python(
                "scripts/training/finetune_task2_qwen_lora.py",
                "generate",
                "--questions",
                "data/raw/btc/LegalQA/train.json",
                "--question-ids",
                "artifacts/task2/training/parent_ce_eval_ids.json",
                "--rankings",
                str(strict_rankings),
                "--model-dir",
                str(local_qwen_lora),
                "--output",
                str(strict_predictions),
                "--diagnostics",
                str(strict_diagnostics),
                "--batch-size",
                str(performance["generation_batch"]),
                "--top-parents",
                "2",
                "--max-parent-words",
                "512",
                "--max-context-tokens",
                "1800",
                "--max-new-tokens",
                "1024",
                "--ce-weight",
                "0.30",
                "--retrieval-weight",
                "0.70",
                "--rrf-k",
                "0",
                "--dtype",
                "bfloat16",
                "--device",
                "cuda",
                "--resume",
            )
        )
    strict_metrics = run_root / "strict_metrics.json"
    if not strict_metrics.is_file():
        _run(
            _python(
                "scripts/training/finetune_task2_qwen_lora.py",
                "evaluate",
                "--questions",
                "data/raw/btc/LegalQA/train.json",
                "--question-ids",
                "artifacts/task2/training/parent_ce_eval_ids.json",
                "--predictions",
                str(strict_predictions),
                "--output",
                str(strict_metrics),
                "--minimum-meteor",
                str(PROMOTION_METEOR),
            ),
            accepted={0, 1},
        )
    heldout = _read_json(strict_metrics)
    promoted = heldout.get("status") == "PROMOTED"

    if promoted:
        public_rankings = run_root / "public_rankings.jsonl"
        if not public_rankings.is_file():
            _score_parents(
                local_listwise_checkpoint,
                public_rankings,
                public=True,
                batch_size=int(performance["parent_score_batch"]),
            )
        public_predictions = run_root / "public_predictions.json"
        public_diagnostics = run_root / "public_diagnostics.json"
        if not _prediction_outputs_complete(
            public_predictions,
            public_diagnostics,
            questions=WORKSPACE / "data/raw/btc/LegalQA/public-official.json",
        ):
            _run(
                _python(
                    "scripts/training/finetune_task2_qwen_lora.py",
                    "generate",
                    "--questions",
                    "data/raw/btc/LegalQA/public-official.json",
                    "--rankings",
                    str(public_rankings),
                    "--model-dir",
                    str(local_qwen_lora),
                    "--output",
                    str(public_predictions),
                    "--diagnostics",
                    str(public_diagnostics),
                    "--known-answers",
                    "data/raw/btc/LegalQA/train.json",
                    "--batch-size",
                    str(performance["generation_batch"]),
                    "--top-parents",
                    "2",
                    "--max-parent-words",
                    "512",
                    "--max-context-tokens",
                    "1800",
                    "--max-new-tokens",
                    "1024",
                    "--ce-weight",
                    "0.30",
                    "--retrieval-weight",
                    "0.70",
                    "--rrf-k",
                    "0",
                    "--dtype",
                    "bfloat16",
                    "--device",
                    "cuda",
                    "--resume",
                )
            )
        submission = run_root / "submission.zip"
        if not submission.is_file():
            _run(
                _python(
                    "scripts/submission/write_legal_qa_submission.py",
                    "--input",
                    str(public_predictions),
                    "--questions",
                    "data/raw/btc/LegalQA/public-official.json",
                    "--empty-answer-policy",
                    "error",
                    "--output",
                    str(submission),
                )
            )
            _run(
                _python(
                    "scripts/submission/validate_legal_qa_submission.py",
                    "--input",
                    str(submission),
                    "--questions",
                    "data/raw/btc/LegalQA/public-official.json",
                )
            )

    summary = {
        "schema_version": f"task2-p14-{PROVIDER_NAME}-full-v1",
        "provider": PROVIDER_NAME,
        "status": "PUBLIC_CANDIDATE_READY" if promoted else "HELDOUT_REJECTED",
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "archive_sha256": state["expected_sha256"],
        "source_sha256": source_hash,
        "contract": contract,
        "contract_version": RUN_CONTRACT_VERSION,
        "gpu": gpu,
        "performance_profile": performance,
        "promotion_meteor": PROMOTION_METEOR,
        "qwen_repo": QWEN_REPO,
        "qwen_revision": QWEN_REVISION,
        "generation_profile": {
            "top_parents": 2,
            "max_parent_words": 512,
            "max_context_tokens": 1800,
            "max_new_tokens": 1024,
            "ce_weight": 0.30,
            "retrieval_weight": 0.70,
            "rrf_k": 0,
            "selection_basis": (
                "best stable configuration on both strict held-out halves"
            ),
        },
        "heldout": heldout,
        "run_root": str(run_root),
        "leaderboard_note": (
            "Held-out METEOR is a selection gate; only Codabench establishes "
            "the public leaderboard score."
        ),
    }
    _build_result_archive(summary, run_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _resumable_full_run() -> tuple[Path, dict[str, Any]]:
    """Return the last completed full run without tying it to new source hash."""

    summary_path = RESULT_ROOT / "full_run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"No completed Task 2 run was found at {summary_path}"
        )
    previous = _read_json(summary_path)
    contract = str(previous.get("contract", ""))
    if len(contract) != 24 or any(
        value not in "0123456789abcdef" for value in contract
    ):
        raise ValueError("The completed Task 2 summary has an invalid contract ID")
    run_root = RESULT_ROOT / "runs" / contract
    required = (
        run_root / "parent_listwise/checkpoint-best/model.safetensors",
        run_root / "qwen_lora/adapter_config.json",
        run_root / "qwen_lora/adapter_model.safetensors",
        run_root / "strict_predictions.json",
        run_root / "strict_diagnostics.json",
        run_root / "extractive_predictions.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The completed Task 2 run is not resumable; missing: "
            + ", ".join(missing)
        )
    return run_root, previous


def _build_ensemble_predictions(
    *,
    qwen: Path,
    extractive: Path,
    diagnostics: Path,
    output: Path,
) -> None:
    _run(
        _python(
            "scripts/submission/build_legal_qa_qwen_extractive_ensemble.py",
            "--qwen",
            str(qwen),
            "--extractive",
            str(extractive),
            "--diagnostics",
            str(diagnostics),
            "--evidence-words",
            str(FAST_PUBLIC_EVIDENCE_WORDS),
            "--output",
            str(output),
        )
    )


def _run_public_candidate() -> dict[str, Any]:
    """Resume trained checkpoints and produce a guarded public submission."""

    started = time.monotonic()
    state = _extract_workspace()
    source_hash = _overlay_source()
    _download_qwen()
    gpu = _gpu_manifest()
    _require_free_vram(gpu)
    performance = _performance_profile(gpu)
    print(
        "PERFORMANCE_PROFILE_JSON=" + json.dumps(performance, ensure_ascii=True),
        flush=True,
    )
    run_root, previous = _resumable_full_run()
    print(f"RESUMING_COMPLETED_RUN={run_root}", flush=True)

    strict_ensemble = run_root / "strict_ensemble_predictions.json"
    if not strict_ensemble.is_file():
        _build_ensemble_predictions(
            qwen=run_root / "strict_predictions.json",
            extractive=run_root / "extractive_predictions.json",
            diagnostics=run_root / "strict_diagnostics.json",
            output=strict_ensemble,
        )
    strict_ensemble_metrics = run_root / "strict_ensemble_metrics.json"
    if not strict_ensemble_metrics.is_file():
        _run(
            _python(
                "scripts/training/finetune_task2_qwen_lora.py",
                "evaluate",
                "--questions",
                "data/raw/btc/LegalQA/train.json",
                "--question-ids",
                "artifacts/task2/training/parent_ce_eval_ids.json",
                "--predictions",
                str(strict_ensemble),
                "--output",
                str(strict_ensemble_metrics),
                "--minimum-meteor",
                str(FAST_PUBLIC_PROMOTION_METEOR),
            ),
            accepted={0, 1},
        )
    heldout = _read_json(strict_ensemble_metrics)
    if heldout.get("status") != "PROMOTED":
        raise RuntimeError(
            "The fixed label-free ensemble did not pass its 0.56 held-out gate"
        )

    public_rankings = run_root / "public_rankings.jsonl"
    if not public_rankings.is_file():
        local_listwise_checkpoint = _localize_directory(
            run_root / "parent_listwise/checkpoint-best",
            "parent_listwise_checkpoint_public",
        )
        _score_parents(
            local_listwise_checkpoint,
            public_rankings,
            public=True,
            batch_size=int(performance["parent_score_batch"]),
        )
    public_extractive = run_root / "public_extractive_predictions.json"
    if not public_extractive.is_file():
        _run(
            _python(
                "scripts/submission/build_legal_qa_parent_scores.py",
                "--scores",
                str(public_rankings),
                "--questions",
                "data/raw/btc/LegalQA/public-official.json",
                "--output",
                str(public_extractive),
                "--top-parents",
                "2",
                "--max-parent-tokens",
                "512",
                "--max-total-tokens",
                "1024",
                "--ce-weight",
                "0.30",
                "--retrieval-weight",
                "0.70",
                "--rrf-k",
                "0",
            )
        )

    public_qwen = run_root / "public_qwen_predictions.json"
    public_qwen_diagnostics = run_root / "public_qwen_diagnostics.json"
    if not _prediction_outputs_complete(
        public_qwen,
        public_qwen_diagnostics,
        questions=WORKSPACE / "data/raw/btc/LegalQA/public-official.json",
    ):
        local_qwen_base = _localize_directory(QWEN_DIR, "qwen3-legal-public")
        local_qwen_lora = _localize_directory(
            run_root / "qwen_lora", "qwen_lora_public"
        )
        local_adapter_config = local_qwen_lora / "adapter_config.json"
        adapter_payload = _read_json(local_adapter_config)
        adapter_payload["base_model_name_or_path"] = str(local_qwen_base)
        adapter_payload["revision"] = None
        _write_json(local_adapter_config, adapter_payload)
        _run(
            _python(
                "scripts/training/finetune_task2_qwen_lora.py",
                "generate",
                "--questions",
                "data/raw/btc/LegalQA/public-official.json",
                "--rankings",
                str(public_rankings),
                "--model-dir",
                str(local_qwen_lora),
                "--output",
                str(public_qwen),
                "--diagnostics",
                str(public_qwen_diagnostics),
                "--known-answers",
                "data/raw/btc/LegalQA/train.json",
                "--batch-size",
                str(performance["generation_batch"]),
                "--top-parents",
                "2",
                "--max-parent-words",
                "512",
                "--max-context-tokens",
                "1800",
                "--max-new-tokens",
                "1024",
                "--ce-weight",
                "0.30",
                "--retrieval-weight",
                "0.70",
                "--rrf-k",
                "0",
                "--dtype",
                "bfloat16",
                "--device",
                "cuda",
                "--resume",
            )
        )
    public_predictions = run_root / "public_predictions.json"
    if not public_predictions.is_file():
        _build_ensemble_predictions(
            qwen=public_qwen,
            extractive=public_extractive,
            diagnostics=public_qwen_diagnostics,
            output=public_predictions,
        )
    submission = run_root / "submission.zip"
    submission.unlink(missing_ok=True)
    _run(
        _python(
            "scripts/submission/write_legal_qa_submission.py",
            "--input",
            str(public_predictions),
            "--questions",
            "data/raw/btc/LegalQA/public-official.json",
            "--empty-answer-policy",
            "error",
            "--output",
            str(submission),
        )
    )
    _run(
        _python(
            "scripts/submission/validate_legal_qa_submission.py",
            "--input",
            str(submission),
            "--questions",
            "data/raw/btc/LegalQA/public-official.json",
        )
    )

    summary = {
        "schema_version": f"task2-p14-{PROVIDER_NAME}-public-v1",
        "provider": PROVIDER_NAME,
        "status": "PUBLIC_CANDIDATE_READY",
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "archive_sha256": state["expected_sha256"],
        "source_sha256": source_hash,
        "contract": run_root.name,
        "contract_version": RUN_CONTRACT_VERSION,
        "gpu": gpu,
        "performance_profile": performance,
        "heldout": heldout,
        "base_heldout": previous.get("heldout"),
        "ensemble": {
            "evidence_words": FAST_PUBLIC_EVIDENCE_WORDS,
            "exact_known_answers_preserved": True,
            "label_free_public_transform": True,
            "minimum_meteor": FAST_PUBLIC_PROMOTION_METEOR,
        },
        "run_root": str(run_root),
        "submission": str(submission),
        "leaderboard_note": (
            "The 0.56 strict held-out gate and prior public overlay uplift make "
            "0.57-0.58 plausible, but only Codabench establishes the score."
        ),
    }
    _build_result_archive(summary, run_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _guarded(stage: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return operation()
    except Exception as exc:
        payload = {
            "schema_version": f"task2-p14-{PROVIDER_NAME}-failure-v1",
            "provider": PROVIDER_NAME,
            "status": f"{stage.upper()}_FAILED",
            "failed_at_utc": _utc_now(),
            "stage": stage,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            _write_json(RESULT_ROOT / f"{stage}_failure.json", payload)
        except Exception as persistence_error:
            payload["persistence_error"] = str(persistence_error)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload


@function(  # type: ignore[misc]
    name="udsc-task2-p14-input",
    cpu=2,
    memory="4Gi",
    image=BEAM_IMAGE,
    timeout=900,
    retries=0,
    volumes=[_volume()],
    headless=True,
)
def input_probe() -> dict[str, Any]:
    payload = _input_state(verify_hash=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


@function(  # type: ignore[misc]
    name="udsc-task2-p14-model-cache",
    cpu=4,
    memory="16Gi",
    image=BEAM_IMAGE,
    timeout=1800,
    retries=0,
    volumes=[_volume()],
    headless=True,
)
def model_prepare() -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        with _exclusive_run():
            _download_qwen()
            payload = {
                "schema_version": f"task2-p14-{PROVIDER_NAME}-model-v1",
                "provider": PROVIDER_NAME,
                "status": "MODEL_READY",
                "repo_id": QWEN_REPO,
                "revision": QWEN_REVISION,
                "manifest": _read_json(QWEN_CACHE_MANIFEST),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            return payload

    return _guarded("model", operation)


@function(  # type: ignore[misc]
    name=_function_name("smoke"),
    cpu=8,
    memory="64Gi",
    gpu=GPU_REQUEST,
    gpu_count=1,
    image=BEAM_IMAGE,
    timeout=SMOKE_TIMEOUT_SECONDS,
    retries=0,
    task_policy=TaskPolicy(
        max_retries=0,
        timeout=SMOKE_TIMEOUT_SECONDS,
        ttl=SMOKE_TTL_SECONDS,
    ),
    env=RUNTIME_ENV,
    volumes=[_volume()],
    pool=POOL_NAME,
    allow_marketplace=False,
    headless=True,
)
def smoke() -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        print("REMOTE_STAGE_START=smoke", flush=True)
        with _exclusive_run():
            return _run_smoke()

    return _guarded("smoke", operation)


@function(  # type: ignore[misc]
    name=_function_name("full"),
    cpu=8,
    memory="64Gi",
    gpu=GPU_REQUEST,
    gpu_count=1,
    image=BEAM_IMAGE,
    timeout=FULL_TIMEOUT_SECONDS,
    retries=0,
    task_policy=TaskPolicy(
        max_retries=0,
        timeout=FULL_TIMEOUT_SECONDS,
        ttl=FULL_TTL_SECONDS,
    ),
    env=RUNTIME_ENV,
    volumes=[_volume()],
    pool=POOL_NAME,
    allow_marketplace=False,
    headless=True,
)
def full() -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        print("REMOTE_STAGE_START=full", flush=True)
        with _exclusive_run():
            return _run_full()

    return _guarded("full", operation)


def _pin_windows_handlers() -> None:
    handlers = (
        ("input_probe", input_probe),
        ("model_prepare", model_prepare),
        ("smoke", smoke),
        ("full", full),
    )
    for name, handler in handlers:
        parent = getattr(handler, "parent", None)
        if parent is not None:
            parent.handler = f"{MODULE_NAME}:{name}"


_pin_windows_handlers()


def _invoke(stage: str) -> int:
    handlers = {
        "input": input_probe,
        "model": model_prepare,
        "smoke": smoke,
        "full": full,
    }
    if stage == "check":
        payload = {
            "status": "LOCAL_CHECK_PASS",
            "source_sha256": _source_tree_sha256(_code_root()),
            "gpu_priority": list(GPU_TYPES),
            "pool": POOL_NAME,
            "volume": VOLUME_NAME,
            "handlers": {
                name: getattr(handler.parent, "handler", None)
                for name, handler in handlers.items()
            },
        }
        print(json.dumps(payload, indent=2), flush=True)
        return 0
    if stage == "source":
        source_hash = _source_tree_sha256(_code_root())
        print(f"LOCAL_SOURCE_SHA256={source_hash}", flush=True)
        return 0
    handler = handlers.get(stage)
    if handler is None:
        print(
            "usage: beam_task2_p14.py {check|source|input|model|smoke|full}",
            file=sys.stderr,
        )
        return 2
    result = handler.remote()
    if not isinstance(result, dict):
        print("Beam returned no completed Task 2 result", file=sys.stderr)
        return 1
    status = result.get("status")
    if stage == "input":
        print("BEAM_INPUT_PROBE_JSON=" + json.dumps(result), flush=True)
        return 0 if status == "INPUT_READY" else 3
    accepted = {
        "model": {"MODEL_READY"},
        "smoke": {"SMOKE_PASS"},
        "full": {"PUBLIC_CANDIDATE_READY", "HELDOUT_REJECTED"},
    }[stage]
    if status not in accepted:
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(f"BEAM {stage.upper()} COMPLETE: {status}", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = _invoke(sys.argv[1] if len(sys.argv) == 2 else "")
    except BaseException as exc:
        print(f"Beam Task 2 invocation failed: {exc}", file=sys.stderr, flush=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(exit_code)
