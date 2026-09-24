"""Isolated Modal L4 route for the frozen SAME-BGE FT V2 Phase 2B.

This wrapper reuses the canonical V3 trainer and V2 scorer contracts.  It only
provides Modal image/volume plumbing, immutable checkpoint verification, and a
non-promotable smoke evidence envelope.  It never changes training math,
selection, aggregation, retrieval, or RRF policy.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "task1-same-bge-ft-v2-phase2b"
DATA_VOLUME_NAME = "udsc-p13"
MODEL_VOLUME_NAME = "udsc-task1-modal"
DATA_MOUNT = Path("/workspace/p13")
MODEL_MOUNT = Path("/data")
REMOTE_ROOT = DATA_MOUNT / "runtime/private_task1/experiments/sprint48_bge_ft_v2"
REMOTE_SMOKE_GROUPS = REMOTE_ROOT / "smoke/train_groups.jsonl"
REMOTE_SMOKE_OUTPUT = REMOTE_ROOT / "smoke/gpu_checkpoint"
REMOTE_SMOKE_EVIDENCE = REMOTE_ROOT / "preflight/remote_smoke_result.json"
REMOTE_FT_MODEL = MODEL_MOUNT / "bge_ft/sprint48_bge_ft_v2/current_ft_verified"
REMOTE_INPUT_ROOT = REMOTE_ROOT / "remote_inputs/_tmp_phase2b_remote_sync"
REMOTE_CHUNKS = REMOTE_INPUT_ROOT / "processed_pv1/chunks"
REMOTE_QUESTIONS = DATA_MOUNT / "runtime/data/raw/btc/LegalIR/train.json"
REMOTE_REFERENCE = REMOTE_INPUT_ROOT / "reconstructed_pv1_bge_scores.jsonl"
REMOTE_TRAINER_PATH = "/root/task1/finetune_task1_bge_reranker_v3.py"
REMOTE_SCORER_PATH = "/root/task1/score_same_bge_ft_v2_gpu.py"
REMOTE_TRAINER = Path(REMOTE_TRAINER_PATH)
REMOTE_SCORER = Path(REMOTE_SCORER_PATH)

MODEL_ID = "BAAI/bge-reranker-v2-m3"
BASE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
FT_WEIGHT_SHA256 = "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c"
FT_CONFIG_SHA256 = "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b"
SMOKE_GROUPS_SHA256 = "e7f776b42f7bbef0fb27fcaa53b96ea83d39412ea00fbb6e4ccb050c2acce180"

SOURCE_PATH = Path(__file__).resolve()
LOCAL_ROOT = SOURCE_PATH.parents[2] if len(SOURCE_PATH.parents) >= 3 else Path("/__modal_local_source_unavailable__")
LOCAL_TRAINER = LOCAL_ROOT / "scripts/training/finetune_task1_bge_reranker_v3.py"
LOCAL_SCORER = LOCAL_ROOT / "scripts/evaluation/score_same_bge_ft_v2_gpu.py"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers==5.0.0",
    "sentence-transformers==5.4.1",
    "accelerate>=1.1",
    "safetensors",
    "numpy",
)
image = image.add_local_file(str(LOCAL_TRAINER), remote_path=REMOTE_TRAINER_PATH)
image = image.add_local_file(str(LOCAL_SCORER), remote_path=REMOTE_SCORER_PATH)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=False)
app = modal.App(APP_NAME)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_remote_checkpoint() -> dict[str, Any]:
    config_path = REMOTE_FT_MODEL / "config.json"
    weight_path = REMOTE_FT_MODEL / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise RuntimeError("REMOTE_MODEL_FILES_MISSING")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = int(config.get("num_labels", len(config.get("id2label", {})) or -1))
    if config.get("model_type") != "xlm-roberta" or labels != 1:
        raise RuntimeError("REMOTE_MODEL_NOT_SINGLE_LABEL_BGE")
    weight_sha = sha256_file(weight_path)
    config_sha = sha256_file(config_path)
    if weight_sha != FT_WEIGHT_SHA256:
        raise RuntimeError(f"REMOTE_WEIGHT_SHA256_MISMATCH:{weight_sha}")
    if config_sha != FT_CONFIG_SHA256:
        raise RuntimeError(f"REMOTE_CONFIG_SHA256_MISMATCH:{config_sha}")
    return {
        "path": str(REMOTE_FT_MODEL),
        "model_id": MODEL_ID,
        "weight_sha256": weight_sha,
        "config_sha256": config_sha,
        "verified_before_model_load": True,
    }


def verify_smoke_groups() -> dict[str, Any]:
    if not REMOTE_SMOKE_GROUPS.is_file():
        raise RuntimeError("REMOTE_SMOKE_GROUPS_MISSING")
    digest = sha256_file(REMOTE_SMOKE_GROUPS)
    if digest != SMOKE_GROUPS_SHA256:
        raise RuntimeError(f"REMOTE_SMOKE_GROUPS_SHA256_MISMATCH:{digest}")
    rows = [json.loads(line) for line in REMOTE_SMOKE_GROUPS.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = 0
    for row in rows:
        positive = row.get("positive_chunks", [])
        negative = row.get("negative_chunks", {}).get("hard", [])
        if len(positive) != 1 or len(negative) != 1:
            raise RuntimeError("REMOTE_SMOKE_GROUP_SCHEMA_MISMATCH")
        examples += 2
    if len(rows) != 2 or examples != 4:
        raise RuntimeError(f"REMOTE_SMOKE_SIZE_MISMATCH:{len(rows)}/{examples}")
    return {"path": str(REMOTE_SMOKE_GROUPS), "sha256": digest, "pairs": len(rows), "examples": examples}


FOLD_SPECS: dict[int, dict[str, int]] = {
    1: {"examples": 52512, "steps": 3282, "qdocs": 28000, "units": 83988},
    2: {"examples": 52488, "steps": 3281, "qdocs": 28000, "units": 83991},
    3: {"examples": 52668, "steps": 3292, "qdocs": 28000, "units": 83994},
    4: {"examples": 52284, "steps": 3268, "qdocs": 28000, "units": 83985},
}
FOLD_INPUT_SHAS: dict[int, dict[str, str]] = {
    1: {
        "train_groups": "8e6c6b9d9e192cd6a6c44f7d11511e0f9036c395795afcdc61287c00da5971f5",
        "score_worklist": "c735858985c7faf9ee20f4b95af6ec60cd84eca88237ad00912cf039c99b1f97",
    },
    2: {
        "train_groups": "db924d77277851170e9ed572c56f4b2c749941d80e087ec09896af76745adc26",
        "score_worklist": "67f9400191840a0487411bba1b368e2e90dc91c9cea343c2ed4c032ceef1614a",
    },
    3: {
        "train_groups": "7aa517a55495bce7bb978f89b1fb5731d13dda7e81a2d527e221de4cfedd9546",
        "score_worklist": "bce3fe2bd901e7e60e526df6532a9227a08d9475a886d28f63d08e44b5570c89",
    },
    4: {
        "train_groups": "089f5dc11b1e5b9f09a2aa6277cf8f57fb6d82b96a630a08515e15732426fedd",
        "score_worklist": "4fa39d7d9aa124133a2ace05f86d1be59a328048bfcebe38895120b6d9b99875",
    },
}
REFERENCE_SHA256 = "5e47db9d9d706660a2943656408af561504ef18f8b6cbfdb36fde99681e93bd0"


def fold_paths(fold: int) -> dict[str, Path]:
    if fold not in FOLD_SPECS:
        raise ValueError(f"invalid fold: {fold}")
    root = REMOTE_ROOT / f"fold{fold}"
    return {
        "root": root,
        "train_groups": REMOTE_INPUT_ROOT / f"fold{fold}/train_groups.jsonl",
        "score_worklist": REMOTE_INPUT_ROOT / f"fold{fold}/score_worklist.jsonl",
        "checkpoint": root / "gpu_checkpoint",
        "scores": root / "scores_v2.jsonl",
        "metrics": root / "score_metrics.json",
        "evidence": root / "execution_result.json",
    }


def require_remote_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"REMOTE_REQUIRED_FILE_MISSING:{path}")


@app.function(
    image=image,
    volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume},
    timeout=10 * 60,
    cpu=2,
    memory=4096,
    retries=0,
)
def import_probe() -> dict[str, Any]:
    """Remote non-GPU import probe; it never loads a model or reads weights."""
    import safetensors  # noqa: F401
    import sentence_transformers  # noqa: F401
    import torch
    import transformers  # noqa: F401

    load_module(REMOTE_TRAINER, "phase2b_remote_trainer_import_probe")
    load_module(REMOTE_SCORER, "phase2b_remote_scorer_import_probe")
    return {
        "status": "MODAL_RUNTIME_IMPORT_PASS",
        "gpu_requested": False,
        "model_loaded": False,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available_without_gpu_request": bool(torch.cuda.is_available()),
        "imports": ["torch", "transformers", "safetensors", "sentence_transformers", "trainer", "scorer"],
    }


@app.function(
    image=image,
    volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume},
    timeout=10 * 60,
    cpu=2,
    memory=4096,
    retries=0,
)
def remote_input_probe() -> dict[str, Any]:
    """Verify isolated remote model and smoke input before L4 model loading."""
    data_volume.reload()
    model_volume.reload()
    model = verify_remote_checkpoint()
    groups = verify_smoke_groups()
    result = {
        "status": "REMOTE_INPUT_GATE_PASS",
        "model": model,
        "smoke_groups": groups,
        "historical_model_namespace_touched": False,
        "model_loaded": False,
        "gpu_requested": False,
    }
    atomic_json(REMOTE_ROOT / "preflight/remote_input_probe.json", result)
    data_volume.commit()
    return result


@app.function(
    image=image,
    volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume},
    timeout=30 * 60,
    cpu=2,
    memory=4096,
    retries=0,
)
def remote_fold_input_probe() -> dict[str, Any]:
    """CPU-only verify all exact fold inputs before any fold GPU call."""
    data_volume.reload()
    model_volume.reload()
    model = verify_remote_checkpoint()
    require_remote_file(REMOTE_QUESTIONS)
    require_remote_file(REMOTE_REFERENCE)
    reference_sha = sha256_file(REMOTE_REFERENCE)
    if reference_sha != REFERENCE_SHA256:
        raise RuntimeError(f"REMOTE_REFERENCE_SHA256_MISMATCH:{reference_sha}")
    expected_docs: set[str] = set()
    verified_files: list[dict[str, Any]] = []
    for fold in range(1, 5):
        paths = fold_paths(fold)
        for key in ("train_groups", "score_worklist"):
            require_remote_file(paths[key])
            actual = sha256_file(paths[key])
            expected = FOLD_INPUT_SHAS[fold][key]
            if actual != expected:
                raise RuntimeError(f"REMOTE_{key.upper()}_SHA256_MISMATCH_F{fold}:{actual}")
            verified_files.append({"fold": fold, "kind": key, "sha256": actual})
        for line in paths["score_worklist"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                expected_docs.add(str(json.loads(line)["document_id"]))
    chunk_files = list(REMOTE_CHUNKS.glob("*.jsonl"))
    missing_docs = sorted(doc for doc in expected_docs if not (REMOTE_CHUNKS / f"{doc}.jsonl").is_file())
    if missing_docs:
        raise RuntimeError(f"REMOTE_CHUNKS_MISSING:{len(missing_docs)}:{missing_docs[:5]}")
    result = {
        "status": "REMOTE_FOLD_INPUT_GATE_PASS", "gpu_requested": False,
        "model_loaded": False, "model": model,
        "questions": str(REMOTE_QUESTIONS), "reference_sha256": reference_sha,
        "verified_files": verified_files, "folds": 4,
        "unique_required_documents": len(expected_docs),
        "remote_chunk_files": len(chunk_files), "missing_required_documents": 0,
    }
    atomic_json(REMOTE_ROOT / "preflight/remote_fold_input_probe.json", result)
    data_volume.commit()
    return result


def _logits(value: Any) -> Any:
    logits = value.logits
    if logits.ndim == 2 and logits.shape[-1] == 2:
        return logits[:, 1] - logits[:, 0]
    return logits.reshape(-1)


def _smoke_pairs() -> tuple[list[tuple[str, str]], list[float]]:
    rows = [json.loads(line) for line in REMOTE_SMOKE_GROUPS.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs: list[tuple[str, str]] = []
    labels: list[float] = []
    for row in rows:
        question = str(row["question"])
        positive = row["positive_chunks"][0]
        negative = row["negative_chunks"]["hard"][0]
        pairs.extend([(question, str(positive["text"])), (question, str(negative["text"]))])
        labels.extend([1.0, 0.0])
    return pairs, labels


def _post_reload_checks(checkpoint: Path, pairs: list[tuple[str, str]], labels: list[float]) -> dict[str, Any]:
    import torch
    from torch.nn import functional
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sentence_transformers import CrossEncoder

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    student = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), local_files_only=True).to("cuda")
    teacher = AutoModelForSequenceClassification.from_pretrained(str(REMOTE_FT_MODEL), local_files_only=True).to("cuda")
    student.eval()
    teacher.eval()
    encoded = tokenizer(
        [pair[0] for pair in pairs], [pair[1] for pair in pairs],
        padding=True, truncation=True, max_length=512, return_tensors="pt",
    )
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    target = torch.tensor(labels, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        student_logits = _logits(student(**encoded))
        teacher_logits = _logits(teacher(**encoded))
        bce = functional.binary_cross_entropy_with_logits(student_logits, target)
        teacher_mse = functional.mse_loss(student_logits, teacher_logits)
        total = 0.5 * bce + 0.5 * teacher_mse
    components = {"bce": float(bce.cpu()), "teacher_mse": float(teacher_mse.cpu()), "total": float(total.cpu())}
    if any(not math.isfinite(value) for value in components.values()):
        raise RuntimeError("SMOKE_POST_RELOAD_NONFINITE_LOSS_COMPONENT")
    del student, teacher, tokenizer, encoded, target
    gc.collect()
    torch.cuda.empty_cache()
    scorer = CrossEncoder(str(checkpoint), device="cuda", max_length=512, local_files_only=True)
    with torch.inference_mode():
        scores = scorer.predict(pairs, batch_size=4, show_progress_bar=False, convert_to_numpy=True)
    finite_scores = all(math.isfinite(float(value)) for value in scores)
    del scorer
    gc.collect()
    torch.cuda.empty_cache()
    if len(scores) != len(pairs) or not finite_scores:
        raise RuntimeError("SMOKE_POST_RELOAD_SCORER_NONFINITE_OR_MISALIGNED")
    return {"dynamic_teacher_logits": "PASS", "loss_components": components, "scorer_reload": "PASS", "scores": [float(value) for value in scores]}


@app.function(
    image=image,
    gpu="L4",
    volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume},
    timeout=10 * 60,
    cpu=8,
    memory=32768,
    retries=0,
)
def training_smoke() -> dict[str, Any]:
    """Run exactly the frozen two-pair/one-step smoke on one L4."""
    import torch

    data_volume.reload()
    model_volume.reload()
    model_contract = verify_remote_checkpoint()
    groups_contract = verify_smoke_groups()
    if not torch.cuda.is_available():
        raise RuntimeError("L4_ENV_GATE_FAIL_CUDA_UNAVAILABLE")
    gpu_name = torch.cuda.get_device_name(0)
    if "L4" not in gpu_name.upper():
        raise RuntimeError(f"L4_ENV_GATE_FAIL_UNEXPECTED_GPU:{gpu_name}")
    total_vram = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    evidence: dict[str, Any] = {
        "status": "SMOKE_STARTED",
        "gpu": gpu_name,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "total_vram_bytes": int(total_vram),
        "model_contract_before_load": model_contract,
        "smoke_groups_contract": groups_contract,
        "recipe": {
            "learning_rate": 5e-7, "epochs": 1, "batch_size": 4,
            "gradient_accumulation": 4, "max_length": 512,
            "freeze_layers": 18, "teacher_weight": 0.5, "seed": 2026,
        },
        "checkpoint_promotable": False,
    }
    try:
        trainer = load_module(REMOTE_TRAINER, "phase2b_remote_trainer")
        args = trainer.build_parser().parse_args([
            "--train-groups", str(REMOTE_SMOKE_GROUPS),
            "--base-model", str(REMOTE_FT_MODEL),
            "--teacher-model", str(REMOTE_FT_MODEL),
            "--output-dir", str(REMOTE_SMOKE_OUTPUT),
            "--allow-training", "--strict-oof-f1-f4", "--device", "cuda",
            "--learning-rate", "5e-7", "--epochs", "1", "--batch-size", "4",
            "--gradient-accumulation", "4", "--max-length", "512",
            "--freeze-layers", "18", "--teacher-weight", "0.5", "--seed", "2026",
        ])
        report = trainer.run(args)
        if report.get("training_example_count") != 4 or report.get("optimizer_steps") != 1 or report.get("micro_steps") != 1:
            raise RuntimeError("SMOKE_TRAINING_SIZE_OR_STEP_MISMATCH")
        checkpoint = REMOTE_SMOKE_OUTPUT / "checkpoint"
        checkpoint_weight_sha = sha256_file(checkpoint / "model.safetensors")
        checkpoint_config_sha = sha256_file(checkpoint / "config.json")
        pairs, labels = _smoke_pairs()
        post_reload = _post_reload_checks(checkpoint, pairs, labels)
        torch.cuda.synchronize()
        evidence.update({
            "status": "SMOKE_PASS",
            "training_report": report,
            "checkpoint": str(checkpoint),
            "checkpoint_weight_sha256": checkpoint_weight_sha,
            "checkpoint_config_sha256": checkpoint_config_sha,
            "post_reload": post_reload,
            "wall_seconds": time.perf_counter() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "pairs": 2,
            "examples": 4,
            "optimizer_steps": 1,
        })
    except Exception as exc:
        evidence.update({
            "status": "SMOKE_FAIL_ENGINEERING",
            "error": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.perf_counter() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        })
        atomic_json(REMOTE_SMOKE_EVIDENCE, evidence)
        data_volume.commit()
        raise
    atomic_json(REMOTE_SMOKE_EVIDENCE, evidence)
    data_volume.commit()
    return evidence


@app.function(
    image=image,
    gpu="L4",
    volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume},
    timeout=2 * 60 * 60,
    cpu=8,
    memory=32768,
    retries=0,
)
def train_fold(fold: int) -> dict[str, Any]:
    """Train one frozen outer fold; no partial-resume or recipe changes."""
    import torch

    data_volume.reload()
    model_volume.reload()
    paths = fold_paths(fold)
    spec = FOLD_SPECS[fold]
    model_contract = verify_remote_checkpoint()
    require_remote_file(paths["train_groups"])
    if paths["checkpoint"].exists():
        raise RuntimeError(f"FOLD_OUTPUT_ALREADY_EXISTS:{paths['checkpoint']}")
    if not torch.cuda.is_available():
        raise RuntimeError("L4_ENV_GATE_FAIL_CUDA_UNAVAILABLE")
    gpu_name = torch.cuda.get_device_name(0)
    if "L4" not in gpu_name.upper():
        raise RuntimeError(f"L4_ENV_GATE_FAIL_UNEXPECTED_GPU:{gpu_name}")
    started = time.perf_counter()
    trainer = load_module(REMOTE_TRAINER, f"phase2b_remote_trainer_f{fold}")
    args = trainer.build_parser().parse_args([
        "--train-groups", str(paths["train_groups"]),
        "--base-model", str(REMOTE_FT_MODEL),
        "--teacher-model", str(REMOTE_FT_MODEL),
        "--output-dir", str(paths["checkpoint"]),
        "--allow-training", "--strict-oof-f1-f4", "--device", "cuda",
        "--learning-rate", "5e-7", "--epochs", "1", "--batch-size", "4",
        "--gradient-accumulation", "4", "--max-length", "512",
        "--freeze-layers", "18", "--teacher-weight", "0.5", "--seed", "2026",
    ])
    report = trainer.run(args)
    if report.get("training_example_count") != spec["examples"] or report.get("optimizer_steps") != spec["steps"]:
        raise RuntimeError(
            f"FOLD_TRAIN_SIZE_OR_STEP_MISMATCH:{report.get('training_example_count')}/{report.get('optimizer_steps')}"
        )
    checkpoint = paths["checkpoint"] / "checkpoint"
    checkpoint_weight_sha = sha256_file(checkpoint / "model.safetensors")
    checkpoint_config_sha = sha256_file(checkpoint / "config.json")
    result = {
        "status": "FOLD_TRAIN_PASS", "fold": fold, "gpu": gpu_name,
        "cuda_version": torch.version.cuda, "torch_version": torch.__version__,
        "model_contract": model_contract, "recipe": report,
        "expected": spec, "checkpoint": str(checkpoint),
        "checkpoint_weight_sha256": checkpoint_weight_sha,
        "checkpoint_config_sha256": checkpoint_config_sha,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(paths["evidence"], result)
    data_volume.commit()
    return result


@app.function(
    image=image,
    gpu="L4",
    volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume},
    timeout=60 * 60,
    cpu=8,
    memory=32768,
    retries=0,
)
def score_fold(fold: int) -> dict[str, Any]:
    """Score one frozen fold worklist with the canonical GPU scorer."""
    import torch

    data_volume.reload()
    model_volume.reload()
    paths = fold_paths(fold)
    spec = FOLD_SPECS[fold]
    model_contract = verify_remote_checkpoint()
    for required in (paths["score_worklist"], REMOTE_QUESTIONS, REMOTE_REFERENCE):
        require_remote_file(required)
    checkpoint = paths["checkpoint"] / "checkpoint"
    for required in (checkpoint / "config.json", checkpoint / "model.safetensors"):
        require_remote_file(required)
    if not torch.cuda.is_available():
        raise RuntimeError("L4_ENV_GATE_FAIL_CUDA_UNAVAILABLE")
    gpu_name = torch.cuda.get_device_name(0)
    if "L4" not in gpu_name.upper():
        raise RuntimeError(f"L4_ENV_GATE_FAIL_UNEXPECTED_GPU:{gpu_name}")
    started = time.perf_counter()
    scorer = load_module(REMOTE_SCORER, f"phase2b_remote_scorer_f{fold}")
    original_argv = sys.argv
    try:
        sys.argv = [
            "score_same_bge_ft_v2_gpu.py",
            "--worklist", str(paths["score_worklist"]),
            "--questions", str(REMOTE_QUESTIONS),
            "--chunks-root", str(REMOTE_CHUNKS),
            "--model", str(checkpoint),
            "--reference-scores", str(REMOTE_REFERENCE),
            "--output", str(paths["scores"]),
            "--metrics", str(paths["metrics"]),
            "--batch-size", "16", "--device", "cuda",
        ]
        scorer.main()
    finally:
        sys.argv = original_argv
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    if metrics.get("qdocs") != spec["qdocs"] or metrics.get("units") != spec["units"]:
        raise RuntimeError(f"FOLD_SCORE_COVERAGE_MISMATCH:{metrics.get('qdocs')}/{metrics.get('units')}")
    result = {
        "status": "FOLD_SCORE_PASS", "fold": fold, "gpu": gpu_name,
        "model_contract": model_contract, "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(paths["evidence"], result)
    data_volume.commit()
    return result


@app.local_entrypoint()
def main(mode: str = "import-probe", fold: int = 0) -> None:
    if mode == "import-probe":
        result = import_probe.remote()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    if mode == "remote-input-probe":
        result = remote_input_probe.remote()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    if mode == "remote-fold-input-probe":
        result = remote_fold_input_probe.remote()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    if mode == "smoke":
        call = training_smoke.spawn()
        app_id = getattr(call, "_app_id", None) or getattr(app, "_app_id", None) or "UNAVAILABLE"
        print(f"app_id={app_id}", flush=True)
        print(f"function_call_id={call.object_id}", flush=True)
        result = call.get()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    if mode in {"fold-train", "fold-score"}:
        if fold not in FOLD_SPECS:
            raise SystemExit("--fold must be one of 1,2,3,4")
        call = (train_fold if mode == "fold-train" else score_fold).spawn(fold)
        app_id = getattr(call, "_app_id", None) or getattr(app, "_app_id", None) or "UNAVAILABLE"
        print(f"app_id={app_id}", flush=True)
        print(f"function_call_id={call.object_id}", flush=True)
        result = call.get()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    raise SystemExit(f"unknown --mode: {mode}")
