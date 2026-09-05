"""Run the guarded Task 1 P13 pipeline on one Beam RTX GPU.

The large, ignored competition artifacts are uploaded once to a Beam Volume.
This module is synced as source code only.  Both functions are deliberately
idempotent: smoke is cached after a real CUDA/checkpoint round trip, while the
full job asks the P13 runner to resume any completed folds in the volume.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from beam import Image, Volume, function

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cloud.beam_source_contract import (  # noqa: E402
    compute_source_tree_sha256,
)

VOLUME_NAME = "udsc-task1-p13"
VOLUME_MOUNT_PATH = "/mnt/task1"
VOLUME_ROOT = Path(VOLUME_MOUNT_PATH)
BEAM_MODULE_NAME = "scripts.cloud.beam_task1_p13"
SUPPORTED_BEAM_CLIENT_VERSION = "0.2.207"
SUPPORTED_BETA9_VERSION = "0.1.265"
BEAM_GPU_TYPE = os.getenv("UDSC_BEAM_GPU", "RTX4090").strip().upper()
GPU_CAPABILITY_MINIMUMS = {"RTX4090": (8, 9), "RTX5090": (12, 0)}
if BEAM_GPU_TYPE not in GPU_CAPABILITY_MINIMUMS:
    raise ValueError(
        f"UDSC_BEAM_GPU must be one of {sorted(GPU_CAPABILITY_MINIMUMS)}, "
        f"got {BEAM_GPU_TYPE!r}"
    )
BEAM_MAX_RETRIES = 2
INPUT_ARCHIVE = VOLUME_ROOT / "input" / "task1_p13_kaggle_upload.tar.zst"
INPUT_ARCHIVE_SHA256 = (
    "21dda96ecab20686ef81b85d1f6813d8c6fa0d80f892e43c61604a3f90469921"
)
INPUT_ARCHIVE_BYTES = 1_763_448_793
INPUT_PROBE_PREFIX = "BEAM_INPUT_PROBE_JSON="
WORKSPACE = VOLUME_ROOT / "workspace"
RESULT_ROOT = VOLUME_ROOT / "results"
SMOKE_ROOT_BASE = WORKSPACE / "artifacts/task1/models/bge_reranker_finetune/beam_smoke"
FULL_ROOT_BASE = (
    WORKSPACE / "artifacts/task1/models/bge_reranker_finetune/beam_full_oof"
)
PUBLIC_ROOT_BASE = WORKSPACE / "artifacts/task1/beam_finetuned_oof_public"
TARGET_RECALL = 0.9591
CANDIDATE_DEPTH = 200
MAX_TRAINING_PAIRS = 24_000
TRAIN_BATCH_SIZE = 8
INFERENCE_BATCH_SIZE = 32
MAX_LENGTH = 512

BEAM_BASE_IMAGE = "docker.io/pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime"
BEAM_PYTHON_PACKAGES = (
    "accelerate>=1.1,<2.0",
    "numpy==1.26.4",
    "pydantic>=2.8,<3.0",
    "pyyaml>=6.0",
    "sentence-transformers==5.4.1",
    "tqdm>=4.66,<5.0",
    "transformers==5.0.0",
    "zstandard==0.23.0",
)
BEAM_IMAGE = Image(base_image=BEAM_BASE_IMAGE).add_python_packages(
    list(BEAM_PYTHON_PACKAGES)
)


def _beam_volume() -> Volume:
    # Keep a literal POSIX path. Path('/mnt/task1') stringifies to
    # '\\mnt\\task1' while the Beam decorator is inspected on Windows.
    return Volume(name=VOLUME_NAME, mount_path=VOLUME_MOUNT_PATH)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected an object in {path}")
    return payload


def _extract_tar_zst(archive: Path, destination: Path) -> None:
    import zstandard

    destination.mkdir(parents=True, exist_ok=True)
    print(
        f"extracting {archive} ({archive.stat().st_size / 1e9:.2f} GB)",
        flush=True,
    )
    with archive.open("rb") as source:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(source) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as bundle:
                bundle.extractall(destination, filter="data")
    print(f"finished extracting {archive.name}", flush=True)


def _prepare_workspace() -> None:
    if not INPUT_ARCHIVE.is_file():
        raise FileNotFoundError(
            f"Beam input archive is missing: {INPUT_ARCHIVE}. "
            "Upload it with scripts/cloud/run_task1_p13_beam.ps1 -Stage Prepare."
        )
    observed_hash = _sha256(INPUT_ARCHIVE)
    if observed_hash != INPUT_ARCHIVE_SHA256:
        raise ValueError(
            "Beam input archive SHA-256 mismatch: "
            f"expected {INPUT_ARCHIVE_SHA256}, got {observed_hash}"
        )

    marker = WORKSPACE / ".beam_bundle_ready.json"
    if marker.is_file():
        payload = _read_json(marker)
        if payload.get("archive_sha256") != INPUT_ARCHIVE_SHA256:
            raise ValueError("existing Beam workspace came from a different archive")
        return
    if WORKSPACE.exists():
        raise RuntimeError(
            f"refusing to overwrite incomplete workspace {WORKSPACE}; inspect the "
            "volume before removing it"
        )

    staging = VOLUME_ROOT / ".workspace_extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _extract_tar_zst(INPUT_ARCHIVE, staging)
        nested = staging / "artifacts/task1/task1_p13_gpu_bundle.tar.zst"
        if not nested.is_file():
            raise FileNotFoundError(f"nested P13 bundle missing: {nested}")
        _extract_tar_zst(nested, staging)
        required = (
            staging / "pyproject.toml",
            staging / "models/reranker/model.safetensors",
            staging / "data/raw/btc/LegalIR/train.json",
            staging / "data/raw/btc/LegalIR/public-official.json",
            staging / "artifacts/task1/evaluation/strict_cv_v2/folds.json",
            staging / "artifacts/task1/training/full_dense500_candidates.jsonl",
            staging / "artifacts/task1/training/public_dense500_candidates.jsonl",
            staging / "artifacts/task1/training/negatives/fold_4.jsonl",
            staging / "artifacts/task1/corpus_document_ids.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Beam bundle is incomplete: {missing}")
        _write_json_atomic(
            staging / ".beam_bundle_ready.json",
            {
                "schema_version": "task1-p13-beam-bundle-v1",
                "prepared_at_utc": _utc_now(),
                "archive_sha256": INPUT_ARCHIVE_SHA256,
            },
        )
        os.replace(staging, WORKSPACE)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _overlay_synced_code() -> str:
    sync_root = PROJECT_ROOT
    source_sha = compute_source_tree_sha256(sync_root)

    for relative in ("configs", "scripts", "src"):
        source = sync_root / relative
        if not source.is_dir():
            raise FileNotFoundError(f"synced source directory missing: {source}")
        target = WORKSPACE / relative
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copy2(sync_root / "pyproject.toml", WORKSPACE / "pyproject.toml")
    return source_sha


def _runtime_dependency_versions() -> dict[str, str]:
    distributions = (
        "accelerate",
        "huggingface-hub",
        "numpy",
        "pydantic",
        "PyYAML",
        "safetensors",
        "scikit-learn",
        "sentence-transformers",
        "tokenizers",
        "torch",
        "tqdm",
        "transformers",
        "zstandard",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required Beam image dependency is missing: {distribution}"
            ) from exc
    return versions


def _run_contract_sha(
    stage: str, source_sha: str, dependency_versions: dict[str, str]
) -> str:
    if stage not in {"smoke", "full"}:
        raise ValueError(f"unsupported Beam stage: {stage}")
    payload = {
        "schema_version": "task1-p13-beam-run-contract-v1",
        "stage": stage,
        "archive_sha256": INPUT_ARCHIVE_SHA256,
        "source_tree_sha256": source_sha,
        "image": {
            "base": BEAM_BASE_IMAGE,
            "python_packages": list(BEAM_PYTHON_PACKAGES),
            "resolved_versions": dependency_versions,
        },
        "beam_runtime": {
            "gpu_type": BEAM_GPU_TYPE,
            "gpu_count": 1,
            "cpu_millicores": 8_000,
            "memory_mb": 65_536,
            "max_retries": BEAM_MAX_RETRIES,
        },
        "training": {
            "ablation": "semi-hard-plus-hard",
            "base_model": "models/reranker",
            "candidate_depth": CANDIDATE_DEPTH,
            "device": "cuda",
            "evidence_limit": 1,
            "learning_rate": 2e-5,
            "epochs": 2,
            "batch_size": TRAIN_BATCH_SIZE,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "gradient_accumulation": 4,
            "max_length": MAX_LENGTH,
            "warmup_ratio": 0.1,
            "fp16": True,
            "seed": 2026,
            "same_law_boost": 2,
            "selection_policy": "fixed_epochs",
            "early_stopping_patience": 0,
            "resume": True,
            "folds": [0] if stage == "smoke" else [0, 1, 2, 3, 4],
            "max_training_pairs": 800 if stage == "smoke" else MAX_TRAINING_PAIRS,
            "max_validation_queries": 20 if stage == "smoke" else 0,
            "diagnostic_only": stage == "smoke",
        },
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _versioned_root(base: Path, contract_sha: str) -> Path:
    return base / contract_sha[:24]


def _checkpoint_weight(checkpoint: Path) -> Path | None:
    """Return either supported Transformers checkpoint weight artifact."""

    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = checkpoint / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _artifact_inventory(root: Path, *, limit: int = 40) -> list[str]:
    """Describe bounded remote output state in guarded failure payloads."""

    if not root.exists():
        return ["<output root does not exist>"]
    records: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            records.append(
                f"{path.relative_to(root).as_posix()} ({path.stat().st_size} B)"
            )
            if len(records) >= limit:
                records.append("<inventory truncated>")
                break
    return records or ["<output root contains no files>"]


def _runtime_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_path = [str(WORKSPACE), str(WORKSPACE / "src")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "OMP_NUM_THREADS": "8",
        }
    )
    return environment


def _run(command: Sequence[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    completed = subprocess.run(
        list(command),
        cwd=WORKSPACE,
        env=_runtime_environment(),
        check=False,
    )
    print(f"PROCESS EXIT: {completed.returncode}", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, list(command))


def _gpu_manifest() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Beam task does not have a CUDA GPU")
    name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    architectures = list(torch.cuda.get_arch_list())
    expected_model = BEAM_GPU_TYPE.removeprefix("RTX")
    if expected_model not in name.casefold().replace(" ", ""):
        raise RuntimeError(f"expected {BEAM_GPU_TYPE}, received {name!r}")
    minimum_capability = GPU_CAPABILITY_MINIMUMS[BEAM_GPU_TYPE]
    if capability < minimum_capability:
        raise RuntimeError(
            f"{BEAM_GPU_TYPE} capability check failed: expected at least "
            f"{minimum_capability}, got {capability}"
        )
    probe = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
    result = probe @ probe
    torch.cuda.synchronize()
    del probe, result
    torch.cuda.empty_cache()
    payload = {
        "name": name,
        "requested_type": BEAM_GPU_TYPE,
        "capability": list(capability),
        # torch.__version__ is a torch.torch_version.TorchVersion subclass.
        # Returning that object through Beam makes the Windows client import
        # torch while deserializing the result, even though the Beam CLI's
        # lightweight local environment intentionally has no torch installed.
        # Keep the remote result transport-only by converting both values to
        # plain built-in strings.
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "supported_architectures": architectures,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


@contextmanager
def _exclusive_run() -> Iterator[None]:
    # Imported lazily because Beam's Windows CLI imports this module locally to
    # inspect the decorated handlers, while the function itself runs on Linux.
    fcntl: Any = importlib.import_module("fcntl")

    VOLUME_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = VOLUME_ROOT / ".task1_p13.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Task 1 Beam job is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _training_command(*, smoke: bool, output: Path | None = None) -> list[str]:
    if output is None:
        # Unit tests and local inspection use a deterministic placeholder; a
        # remote run always supplies its contract-keyed output directory.
        output = SMOKE_ROOT_BASE if smoke else FULL_ROOT_BASE
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.training.finetune_task1_bge_reranker",
        "--train",
        "data/raw/btc/LegalIR/train.json",
        "--folds",
        "artifacts/task1/evaluation/strict_cv_v2/folds.json",
        "--negatives-dir",
        "artifacts/task1/training/negatives",
        "--candidates",
        "artifacts/task1/training/full_dense500_candidates.jsonl",
        "--base-model",
        "models/reranker",
        "--output-dir",
        str(output),
        "--ablation",
        "semi-hard-plus-hard",
        "--candidate-depth",
        str(CANDIDATE_DEPTH),
        "--evidence-limit",
        "1",
        "--learning-rate",
        "2e-5",
        "--epochs",
        "2",
        "--batch-size",
        str(TRAIN_BATCH_SIZE),
        "--inference-batch-size",
        str(INFERENCE_BATCH_SIZE),
        "--gradient-accumulation",
        "4",
        "--max-length",
        str(MAX_LENGTH),
        "--warmup-ratio",
        "0.1",
        "--device",
        "cuda",
        "--resume",
    ]
    if smoke:
        command.extend(
            [
                "--folds-to-run",
                "0",
                "--max-training-pairs",
                "800",
                "--max-validation-queries",
                "20",
                "--diagnostic-only",
            ]
        )
    else:
        command.extend(
            [
                "--folds-to-run",
                "0",
                "1",
                "2",
                "3",
                "4",
                "--max-training-pairs",
                str(MAX_TRAINING_PAIRS),
            ]
        )
    return command


def _smoke_passed(*, source_sha: str, contract_sha: str) -> dict[str, Any] | None:
    marker = RESULT_ROOT / "smoke" / f"{contract_sha}.json"
    if not marker.is_file():
        return None
    try:
        payload = _read_json(marker)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        payload.get("status") != "SMOKE_PASS"
        or payload.get("source_tree_sha256") != source_sha
        or payload.get("run_contract_sha256") != contract_sha
        or payload.get("archive_sha256") != INPUT_ARCHIVE_SHA256
    ):
        return None
    smoke_root_raw = payload.get("smoke_root")
    if not isinstance(smoke_root_raw, str):
        return None
    smoke_root = Path(smoke_root_raw)
    expected_root = _versioned_root(SMOKE_ROOT_BASE, contract_sha)
    if smoke_root != expected_root:
        return None
    metrics_path = smoke_root / "fold_0/metrics.json"
    checkpoint_dir = smoke_root / "fold_0/checkpoint"
    checkpoint = _checkpoint_weight(checkpoint_dir)
    checkpoint_metadata = smoke_root / "fold_0/checkpoint/p13_checkpoint_metadata.json"
    if (
        not metrics_path.is_file()
        or checkpoint is None
        or not checkpoint_metadata.is_file()
    ):
        return None
    try:
        metrics = _read_json(metrics_path)
        metadata = _read_json(checkpoint_metadata)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        metrics.get("status") != "COMPLETE"
        or metrics.get("validation_query_count") != 20
        or metrics.get("checkpoint") != str(smoke_root / "fold_0/checkpoint")
        or payload.get("fold_0") != metrics
        or metadata.get("schema_version") != "task1-p13-checkpoint-v1"
        or metadata.get("fold") != 0
    ):
        return None
    return payload


def _quarantine_incomplete_smoke(smoke_root: Path) -> None:
    if not smoke_root.exists():
        return
    quarantine = smoke_root.with_name(f"{smoke_root.name}.invalid-{time.time_ns()}")
    os.replace(smoke_root, quarantine)
    print(f"moved incomplete smoke output to {quarantine}", flush=True)


def _run_smoke() -> dict[str, Any]:
    with _exclusive_run():
        started = time.monotonic()
        _prepare_workspace()
        source_sha = _overlay_synced_code()
        dependency_versions = _runtime_dependency_versions()
        contract_sha = _run_contract_sha("smoke", source_sha, dependency_versions)
        smoke_root = _versioned_root(SMOKE_ROOT_BASE, contract_sha)
        gpu = _gpu_manifest()
        cached = _smoke_passed(source_sha=source_sha, contract_sha=contract_sha)
        if cached is not None:
            print("reusing validated Beam smoke result for this contract", flush=True)
            return cached
        _quarantine_incomplete_smoke(smoke_root)
        _run(_training_command(smoke=True, output=smoke_root))
        metrics_path = smoke_root / "fold_0/metrics.json"
        checkpoint_dir = smoke_root / "fold_0/checkpoint"
        checkpoint = _checkpoint_weight(checkpoint_dir)
        if not metrics_path.is_file() or checkpoint is None:
            raise RuntimeError(
                "Beam smoke did not produce a complete metrics/checkpoint pair. "
                f"metrics_exists={metrics_path.is_file()}, "
                f"checkpoint_weight={str(checkpoint) if checkpoint else None}, "
                f"artifact_inventory={_artifact_inventory(smoke_root)}"
            )
        metrics = _read_json(metrics_path)
        if (
            metrics.get("status") != "COMPLETE"
            or metrics.get("validation_query_count") != 20
        ):
            raise RuntimeError(f"Beam smoke metrics are incomplete: {metrics}")
        payload = {
            "schema_version": "task1-p13-beam-smoke-v1",
            "status": "SMOKE_PASS",
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "archive_sha256": INPUT_ARCHIVE_SHA256,
            "source_tree_sha256": source_sha,
            "run_contract_sha256": contract_sha,
            "dependency_versions": dependency_versions,
            "smoke_root": str(smoke_root),
            "gpu": gpu,
            "configuration": {
                "candidate_depth": CANDIDATE_DEPTH,
                "batch_size": TRAIN_BATCH_SIZE,
                "inference_batch_size": INFERENCE_BATCH_SIZE,
                "max_length": MAX_LENGTH,
            },
            "fold_0": metrics,
        }
        _write_json_atomic(RESULT_ROOT / "smoke" / f"{contract_sha}.json", payload)
        print(f"BEAM {BEAM_GPU_TYPE} SMOKE PASS", flush=True)
        return payload


def _public_commands(*, full_root: Path, public_root: Path) -> list[list[str]]:
    predictions = public_root / "predictions.json"
    report = public_root / "report.json"
    submission = public_root / "submission.zip"
    return [
        [
            sys.executable,
            "scripts/submission/build_legal_ir_finetuned_fold_ensemble.py",
            "--questions",
            "data/raw/btc/LegalIR/public-official.json",
            "--overlay-labels",
            "data/raw/btc/LegalIR/train.json",
            "--candidates",
            "artifacts/task1/training/public_dense500_candidates.jsonl",
            "--checkpoint-root",
            str(full_root),
            "--candidate-depth",
            str(CANDIDATE_DEPTH),
            "--evidence-limit",
            "1",
            "--batch-size",
            str(INFERENCE_BATCH_SIZE),
            "--max-length",
            str(MAX_LENGTH),
            "--device",
            "cuda",
            "--output",
            str(predictions),
            "--report",
            str(report),
        ],
        [
            sys.executable,
            "scripts/submission/write_legal_ir_submission.py",
            "--input",
            str(predictions),
            "--questions",
            "data/raw/btc/LegalIR/public-official.json",
            "--corpus-manifest",
            "artifacts/task1/corpus_document_ids.json",
            "--output",
            str(submission),
        ],
        [
            sys.executable,
            "scripts/submission/validate_legal_ir_submission.py",
            "--input",
            str(submission),
            "--questions",
            "data/raw/btc/LegalIR/public-official.json",
            "--corpus-manifest",
            "artifacts/task1/corpus_document_ids.json",
        ],
    ]


def _build_result_archive(
    summary: dict[str, Any], *, full_root: Path, public_root: Path
) -> Path:
    _write_json_atomic(RESULT_ROOT / "full_run_summary.json", summary)
    archive = RESULT_ROOT / "task1_p13_beam_result.zip"
    temporary = archive.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    required_paths: list[Path] = [
        RESULT_ROOT / "full_run_summary.json",
        full_root / "final_decision.json",
        full_root / "comparison.json",
        full_root / "comparison.md",
        full_root / "training_manifest.json",
        full_root / "oof_predictions.jsonl",
    ]
    smoke_contract_sha = summary.get("smoke_run_contract_sha256")
    if isinstance(smoke_contract_sha, str) and smoke_contract_sha:
        required_paths.append(RESULT_ROOT / "smoke" / f"{smoke_contract_sha}.json")
    else:
        raise RuntimeError("full summary is missing its smoke run contract")
    required_paths.extend(full_root / f"fold_{fold}/metrics.json" for fold in range(5))
    if summary.get("status") == "PUBLIC_CANDIDATE_READY":
        required_paths.extend(
            (
                public_root / "predictions.json",
                public_root / "report.json",
                public_root / "submission.zip",
            )
        )
    missing = [
        str(path)
        for path in required_paths
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise RuntimeError(f"refusing to archive incomplete Beam results: {missing}")
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for path in required_paths:
            bundle.write(path, arcname=path.relative_to(VOLUME_ROOT).as_posix())
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as bundle:
        corrupt = bundle.testzip()
    if corrupt is not None:
        raise RuntimeError(f"Beam result archive is corrupt at {corrupt}")
    return archive


def _run_full() -> dict[str, Any]:
    with _exclusive_run():
        started = time.monotonic()
        _prepare_workspace()
        source_sha = _overlay_synced_code()
        dependency_versions = _runtime_dependency_versions()
        smoke_contract_sha = _run_contract_sha("smoke", source_sha, dependency_versions)
        smoke_result = _smoke_passed(
            source_sha=source_sha, contract_sha=smoke_contract_sha
        )
        if smoke_result is None:
            raise RuntimeError(
                f"run and pass the Beam {BEAM_GPU_TYPE} smoke for the current "
                "source contract "
                "before full OOF"
            )
        full_contract_sha = _run_contract_sha("full", source_sha, dependency_versions)
        full_root = _versioned_root(FULL_ROOT_BASE, full_contract_sha)
        public_root = _versioned_root(PUBLIC_ROOT_BASE, full_contract_sha)
        gpu = _gpu_manifest()
        _run(_training_command(smoke=False, output=full_root))
        decision_path = full_root / "final_decision.json"
        if not decision_path.is_file():
            raise RuntimeError("full OOF did not produce final_decision.json")
        decision = _read_json(decision_path)
        oof = decision.get("oof")
        tuned_recall = None
        if isinstance(oof, dict) and isinstance(oof.get("fine_tuned"), dict):
            tuned_recall = oof["fine_tuned"].get("official_macro_recall")
        promotable = (
            decision.get("status") == "PROMOTE_CANDIDATE"
            and decision.get("promotable") is True
            and decision.get("complete_oof") is True
        )
        if promotable:
            for command in _public_commands(
                full_root=full_root, public_root=public_root
            ):
                _run(command)
        summary = {
            "schema_version": "task1-p13-beam-full-v1",
            "status": "PUBLIC_CANDIDATE_READY" if promotable else "OOF_REJECTED",
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "archive_sha256": INPUT_ARCHIVE_SHA256,
            "source_tree_sha256": source_sha,
            "smoke_run_contract_sha256": smoke_contract_sha,
            "run_contract_sha256": full_contract_sha,
            "dependency_versions": dependency_versions,
            "full_root": str(full_root),
            "public_root": str(public_root),
            "gpu": gpu,
            "configuration": {
                "candidate_depth": CANDIDATE_DEPTH,
                "max_training_pairs_per_fold": MAX_TRAINING_PAIRS,
                "epochs": 2,
                "learning_rate": 2e-5,
                "batch_size": TRAIN_BATCH_SIZE,
                "inference_batch_size": INFERENCE_BATCH_SIZE,
                "gradient_accumulation": 4,
                "max_length": MAX_LENGTH,
            },
            "oof_tuned_recall": tuned_recall,
            "target_recall": TARGET_RECALL,
            "target_proxy_pass": (
                isinstance(tuned_recall, (int, float))
                and float(tuned_recall) >= TARGET_RECALL
            ),
            "promotable": promotable,
            "leaderboard_note": (
                "OOF is a selection proxy; only Codabench can establish a public "
                "leaderboard score above 0.9591."
            ),
            "decision": decision,
        }
        archive = _build_result_archive(
            summary, full_root=full_root, public_root=public_root
        )
        summary["download_path"] = f"beam://{VOLUME_NAME}/results/{archive.name}"
        _write_json_atomic(RESULT_ROOT / "full_run_summary.json", summary)
        # Rebuild once so the archive includes the final download path as well.
        _build_result_archive(summary, full_root=full_root, public_root=public_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary


def _run_guarded(stage: str, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Return remote failures as data so Windows does not depend on Beam logs."""

    try:
        return operation()
    except Exception as exc:
        payload: dict[str, Any] = {
            "schema_version": "task1-p13-beam-failure-v1",
            "status": f"{stage.upper()}_FAILED",
            "failed_at_utc": _utc_now(),
            "stage": stage,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure_path = RESULT_ROOT / f"{stage}_failure.json"
        payload["failure_path"] = str(failure_path)
        try:
            _write_json_atomic(failure_path, payload)
        except Exception as persistence_error:
            payload["persistence_error"] = (
                f"{type(persistence_error).__name__}: {persistence_error}"
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return payload


def _run_input_probe() -> dict[str, Any]:
    """Inspect the archive through the mount, not cached Volume metadata."""

    is_file = INPUT_ARCHIVE.is_file()
    size = INPUT_ARCHIVE.stat().st_size if is_file else 0
    observed_hash: str | None = None
    if is_file and size == INPUT_ARCHIVE_BYTES:
        observed_hash = _sha256(INPUT_ARCHIVE)
    ready = (
        is_file
        and size == INPUT_ARCHIVE_BYTES
        and observed_hash == INPUT_ARCHIVE_SHA256
    )
    payload = {
        "schema_version": "task1-p13-beam-input-probe-v1",
        "status": "INPUT_READY" if ready else "INPUT_INVALID",
        "path": str(INPUT_ARCHIVE),
        "is_file": is_file,
        "size": size,
        "expected_size": INPUT_ARCHIVE_BYTES,
        "sha256": observed_hash,
        "expected_sha256": INPUT_ARCHIVE_SHA256,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


@function(  # type: ignore[misc]
    name="udsc-task1-p13-input-probe",
    cpu=2,
    memory="4Gi",
    image=BEAM_IMAGE,
    timeout=900,
    retries=BEAM_MAX_RETRIES,
    env={"UDSC_BEAM_GPU": BEAM_GPU_TYPE},
    volumes=[_beam_volume()],
    headless=True,
)
def input_probe() -> dict[str, Any]:
    """Verify the exact mounted input size/hash without allocating a GPU."""

    return _run_guarded("input_probe", _run_input_probe)


@function(  # type: ignore[misc]
    name="udsc-task1-p13-smoke",
    cpu=8,
    memory="64Gi",
    gpu=BEAM_GPU_TYPE,
    gpu_count=1,
    image=BEAM_IMAGE,
    timeout=-1,
    retries=BEAM_MAX_RETRIES,
    env={"UDSC_BEAM_GPU": BEAM_GPU_TYPE},
    volumes=[_beam_volume()],
    headless=True,
)
def smoke() -> dict[str, Any]:
    """Validate the exact Beam GPU, image, data and checkpoint round trip."""

    return _run_guarded("smoke", _run_smoke)


@function(  # type: ignore[misc]
    name="udsc-task1-p13-full",
    cpu=8,
    memory="64Gi",
    gpu=BEAM_GPU_TYPE,
    gpu_count=1,
    image=BEAM_IMAGE,
    timeout=-1,
    retries=BEAM_MAX_RETRIES,
    env={"UDSC_BEAM_GPU": BEAM_GPU_TYPE},
    volumes=[_beam_volume()],
    headless=True,
)
def full() -> dict[str, Any]:
    """Resume five strict folds, gate OOF, and build a validated public candidate."""

    return _run_guarded("full", _run_full)


def _pin_portable_beam_handlers() -> None:
    """Bypass Beam 0.2.207's Windows-only handler path conversion bug."""

    for name, handler in (
        ("input_probe", input_probe),
        ("smoke", smoke),
        ("full", full),
    ):
        parent = getattr(handler, "parent", None)
        if parent is not None:
            # SDK 0.2.207 replaces '/' but not '\\' when deriving this value on
            # Windows. The Linux runner requires an importable dotted module.
            parent.handler = f"{BEAM_MODULE_NAME}:{name}"


_pin_portable_beam_handlers()


def _validate_local_beam_contract() -> int:
    """Validate serialized handler/resource values without allocating a GPU."""

    errors: list[str] = []
    observed: dict[str, Any] = {
        "sdk_versions": {
            "beam-client": importlib.metadata.version("beam-client"),
            "beta9": importlib.metadata.version("beta9"),
        }
    }
    expected_versions = {
        "beam-client": SUPPORTED_BEAM_CLIENT_VERSION,
        "beta9": SUPPORTED_BETA9_VERSION,
    }
    if observed["sdk_versions"] != expected_versions:
        errors.append(
            "unsupported Beam SDK versions: "
            f"expected {expected_versions}, got {observed['sdk_versions']}"
        )
    for name, handler in (("smoke", smoke), ("full", full)):
        parent = getattr(handler, "parent", None)
        if parent is None:
            errors.append(f"{name}: missing Beam parent")
            continue
        volumes = getattr(parent, "volumes", [])
        task_policy = getattr(parent, "task_policy", None)
        mount_paths = [getattr(volume, "mount_path", None) for volume in volumes]
        serialized_env = getattr(parent, "env", {})
        if isinstance(serialized_env, dict):
            remote_env = serialized_env
        else:
            remote_env = {
                key: value
                for entry in serialized_env
                if "=" in entry
                for key, value in [entry.split("=", maxsplit=1)]
            }
        observed[name] = {
            "handler": getattr(parent, "handler", None),
            "mount_paths": mount_paths,
            "gpu": getattr(parent, "gpu", None),
            "gpu_count": getattr(parent, "gpu_count", None),
            "cpu_millicores": getattr(parent, "cpu", None),
            "memory_mb": getattr(parent, "memory", None),
            "timeout_seconds": getattr(task_policy, "timeout", None),
            "max_retries": getattr(task_policy, "max_retries", None),
            "headless": getattr(parent, "headless", None),
            "remote_env": remote_env,
        }
        expected_handler = f"{BEAM_MODULE_NAME}:{name}"
        if parent.handler != expected_handler:
            errors.append(f"{name}: handler {parent.handler!r} != {expected_handler!r}")
        if mount_paths != [VOLUME_MOUNT_PATH]:
            errors.append(f"{name}: volume mount paths are {mount_paths!r}")
        if parent.gpu != BEAM_GPU_TYPE or parent.gpu_count != 1:
            errors.append(
                f"{name}: expected one {BEAM_GPU_TYPE}, "
                f"got {parent.gpu!r} x{parent.gpu_count}"
            )
        if remote_env.get("UDSC_BEAM_GPU") != BEAM_GPU_TYPE:
            errors.append(
                f"{name}: remote UDSC_BEAM_GPU is "
                f"{remote_env.get('UDSC_BEAM_GPU')!r}, expected {BEAM_GPU_TYPE!r}"
            )
        if parent.cpu != 8_000 or parent.memory != 65_536:
            errors.append(
                f"{name}: expected cpu=8000m/memory=65536MiB, "
                f"got {parent.cpu!r}/{parent.memory!r}"
            )
        if (
            task_policy is None
            or task_policy.timeout != -1
            or task_policy.max_retries != BEAM_MAX_RETRIES
            or parent.headless is not True
        ):
            errors.append(
                f"{name}: expected timeout=-1/retries={BEAM_MAX_RETRIES}/"
                "headless=true, got "
                f"{getattr(task_policy, 'timeout', None)!r}/"
                f"{getattr(task_policy, 'max_retries', None)!r}/"
                f"{getattr(parent, 'headless', None)!r}"
            )
    probe_parent = getattr(input_probe, "parent", None)
    if probe_parent is None:
        errors.append("input_probe: missing Beam parent")
    else:
        probe_volumes = getattr(probe_parent, "volumes", [])
        probe_mounts = [getattr(volume, "mount_path", None) for volume in probe_volumes]
        probe_policy = getattr(probe_parent, "task_policy", None)
        observed["input_probe"] = {
            "handler": getattr(probe_parent, "handler", None),
            "mount_paths": probe_mounts,
            "gpu_count": getattr(probe_parent, "gpu_count", None),
            "cpu_millicores": getattr(probe_parent, "cpu", None),
            "memory_mb": getattr(probe_parent, "memory", None),
            "timeout_seconds": getattr(probe_policy, "timeout", None),
            "max_retries": getattr(probe_policy, "max_retries", None),
        }
        if probe_parent.handler != f"{BEAM_MODULE_NAME}:input_probe":
            errors.append(f"input_probe: invalid handler {probe_parent.handler!r}")
        if probe_mounts != [VOLUME_MOUNT_PATH]:
            errors.append(f"input_probe: volume mount paths are {probe_mounts!r}")
        if (
            probe_parent.gpu_count != 0
            or probe_parent.cpu != 2_000
            or probe_parent.memory != 4_096
            or probe_policy is None
            or probe_policy.timeout != 900
            or probe_policy.max_retries != BEAM_MAX_RETRIES
        ):
            errors.append("input_probe: invalid CPU-only resource contract")
    payload: dict[str, Any] = {
        "status": "LOCAL_CHECK_PASS" if not errors else "LOCAL_CHECK_FAILED"
    }
    payload["beam_contract"] = observed
    payload["errors"] = errors
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if not errors else 1


def _invoke_remote_from_cli(stage: str) -> int:
    """Invoke a decorated Beam function from the Beam SDK environment."""

    if stage == "check":
        return _validate_local_beam_contract()
    handlers = {"input": input_probe, "smoke": smoke, "full": full}
    handler = handlers.get(stage)
    if handler is None:
        print(
            "usage: beam_task1_p13.py {check|input|smoke|full}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    remote = getattr(handler, "remote", None)
    if not callable(remote):
        print("Beam SDK did not expose handler.remote()", file=sys.stderr, flush=True)
        return 2
    result = remote()
    if not isinstance(result, dict):
        print(
            f"Beam {stage} did not return a completed result; inspect Beam task logs.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    status = result.get("status")
    if stage == "input":
        print(INPUT_PROBE_PREFIX + json.dumps(result, ensure_ascii=True), flush=True)
        if status == "INPUT_READY":
            return 0
        if status == "INPUT_INVALID":
            return 3
        return 1
    completed_statuses = {
        "smoke": {"SMOKE_PASS"},
        "full": {"PUBLIC_CANDIDATE_READY", "OOF_REJECTED"},
    }
    if status not in completed_statuses[stage]:
        print(
            json.dumps(result, ensure_ascii=False, indent=2),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"BEAM {stage.upper()} COMPLETE: {status}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = _invoke_remote_from_cli(sys.argv[1] if len(sys.argv) == 2 else "")
    except BaseException as exc:  # Beam CLI boundary: always return a useful exit code.
        print(f"Beam invocation failed: {exc}", file=sys.stderr, flush=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    # Beam's gRPC worker threads can keep the uv tool interpreter alive after
    # completion on Windows. The remote result has already been persisted.
    os._exit(exit_code)
