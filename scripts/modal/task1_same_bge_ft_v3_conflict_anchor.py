"""Isolated Modal L4 route for SAME-BGE HARD-NEGATIVE FT V3.

V3 changes only the teacher-anchor loss mode.  The runner uses a separate
namespace and reuses the exact frozen V2 train groups/worklists.  Its CPU
input probe never loads a model; GPU functions are explicit smoke/train/score
operations and are called sequentially by the local operator.
"""

from __future__ import annotations

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


APP_NAME = "task1-same-bge-ft-v3-conflict-anchor"
DATA_VOLUME_NAME = "udsc-p13"
MODEL_VOLUME_NAME = "udsc-task1-modal"
DATA_MOUNT = Path("/workspace/p13")
MODEL_MOUNT = Path("/data")
REMOTE_ROOT = DATA_MOUNT / "runtime/private_task1/experiments/sprint48_bge_ft_v3_conflict_anchor"
REMOTE_FT_MODEL = MODEL_MOUNT / "bge_ft/sprint48_bge_ft_v2/current_ft_verified"
REMOTE_V2_ROOT = DATA_MOUNT / "runtime/private_task1/experiments/sprint48_bge_ft_v2"
REMOTE_QUESTIONS = DATA_MOUNT / "runtime/data/raw/btc/LegalIR/train.json"
REMOTE_CHUNKS = REMOTE_V2_ROOT / "remote_inputs/_tmp_phase2b_remote_sync/processed_pv1/chunks"
REMOTE_REFERENCE = REMOTE_V2_ROOT / "remote_inputs/_tmp_phase2b_remote_sync/reconstructed_pv1_bge_scores.jsonl"
REMOTE_V2_SMOKE_GROUPS = REMOTE_V2_ROOT / "smoke/train_groups.jsonl"
REMOTE_TRAINER_PATH = "/root/task1/finetune_task1_bge_reranker_v3.py"
REMOTE_SCORER_PATH = "/root/task1/score_same_bge_ft_v2_gpu.py"
REMOTE_TRAINER = Path(REMOTE_TRAINER_PATH)
REMOTE_SCORER = Path(REMOTE_SCORER_PATH)

MODEL_ID = "BAAI/bge-reranker-v2-m3"
BASE_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
CURRENT_FT_WEIGHT_SHA256 = "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c"
CURRENT_FT_CONFIG_SHA256 = "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b"
V3_CONFIG_SHA256 = "8523a19389f1314e64aeac81d0432adaf3e23639bad149e28331c51ceb52a6b7"
FOLD_SPECS: dict[int, dict[str, int]] = {
    1: {"examples": 52512, "steps": 3282, "qdocs": 28000, "units": 83988},
    2: {"examples": 52488, "steps": 3281, "qdocs": 28000, "units": 83991},
    3: {"examples": 52668, "steps": 3292, "qdocs": 28000, "units": 83994},
    4: {"examples": 52284, "steps": 3268, "qdocs": 28000, "units": 83985},
}
FOLD_INPUT_SHAS: dict[int, dict[str, str]] = {
    1: {"train_groups.jsonl": "8e6c6b9d9e192cd6a6c44f7d11511e0f9036c395795afcdc61287c00da5971f5", "score_worklist.jsonl": "c735858985c7faf9ee20f4b95af6ec60cd84eca88237ad00912cf039c99b1f97"},
    2: {"train_groups.jsonl": "db924d77277851170e9ed572c56f4b2c749941d80e087ec09896af76745adc26", "score_worklist.jsonl": "67f9400191840a0487411bba1b368e2e90dc91c9cea343c2ed4c032ceef1614a"},
    3: {"train_groups.jsonl": "7aa517a55495bce7bb978f89b1fb5731d13dda7e81a2d527e221de4cfedd9546", "score_worklist.jsonl": "bce3fe2bd901e7e60e526df6532a9227a08d9475a886d28f63d08e44b5570c89"},
    4: {"train_groups.jsonl": "089f5dc11b1e5b9f09a2aa6277cf8f57fb6d82b96a630a08515e15732426fedd", "score_worklist.jsonl": "4fa39d7d9aa124133a2ace05f86d1be59a328048bfcebe38895120b6d9b99875"},
}

SOURCE_PATH = Path(__file__).resolve()
LOCAL_ROOT = SOURCE_PATH.parents[2] if len(SOURCE_PATH.parents) >= 3 else Path("/__modal_local_source_unavailable__")
LOCAL_TRAINER = LOCAL_ROOT / "scripts/training/finetune_task1_bge_reranker_v3.py"
LOCAL_SCORER = LOCAL_ROOT / "scripts/evaluation/score_same_bge_ft_v2_gpu.py"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "transformers==5.0.0", "sentence-transformers==5.4.1", "accelerate>=1.1", "safetensors", "numpy"
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
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_remote_current_ft() -> dict[str, Any]:
    config_path = REMOTE_FT_MODEL / "config.json"
    weight_path = REMOTE_FT_MODEL / "model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise RuntimeError("REMOTE_CURRENT_FT_FILES_MISSING")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    labels = int(config.get("num_labels", len(config.get("id2label", {})) or -1))
    if config.get("model_type") != "xlm-roberta" or labels != 1:
        raise RuntimeError("REMOTE_CURRENT_FT_NOT_SINGLE_LOGIT_BGE")
    weight_sha = sha256_file(weight_path)
    config_sha = sha256_file(config_path)
    if weight_sha != CURRENT_FT_WEIGHT_SHA256 or config_sha != CURRENT_FT_CONFIG_SHA256:
        raise RuntimeError(f"REMOTE_CURRENT_FT_HASH_MISMATCH:{weight_sha}/{config_sha}")
    return {"path": str(REMOTE_FT_MODEL), "model_id": MODEL_ID, "base_revision": BASE_REVISION, "weight_sha256": weight_sha, "config_sha256": config_sha}


def fold_paths(fold: int) -> dict[str, Path]:
    if fold not in FOLD_SPECS:
        raise ValueError(f"invalid fold: {fold}")
    root = REMOTE_ROOT / f"fold{fold}"
    return {"root": root, "train_groups": root / "train_groups.jsonl", "score_worklist": root / "score_worklist.jsonl", "checkpoint": root / "gpu_checkpoint/checkpoint", "training_execution": root / "training_execution.json", "scores": root / "scores_v3.jsonl", "metrics": root / "score_metrics.json", "score_execution": root / "score_execution.json"}


def require(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"REMOTE_REQUIRED_FILE_MISSING:{path}")


@app.function(image=image, volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume}, timeout=30 * 60, cpu=2, memory=4096, retries=0)
def remote_input_probe() -> dict[str, Any]:
    """CPU-only V3 input/model provenance gate; no model load."""
    data_volume.reload()
    model_volume.reload()
    model = verify_remote_current_ft()
    config_path = REMOTE_ROOT / "v3_frozen_config.json"
    require(config_path)
    if sha256_file(config_path) != V3_CONFIG_SHA256:
        raise RuntimeError("V3_CONFIG_SHA256_MISMATCH")
    verified: list[dict[str, Any]] = []
    for fold in range(1, 5):
        paths = fold_paths(fold)
        for name in ("train_groups.jsonl", "score_worklist.jsonl"):
            path = paths[name[:-6]] if name == "train_groups.jsonl" else paths["score_worklist"]
            require(path)
            actual = sha256_file(path)
            expected = FOLD_INPUT_SHAS[fold][name]
            if actual != expected:
                raise RuntimeError(f"V3_INPUT_SHA_MISMATCH_F{fold}_{name}:{actual}")
            verified.append({"fold": fold, "file": name, "sha256": actual})
    require(REMOTE_QUESTIONS)
    require(REMOTE_CHUNKS / "300362.jsonl")
    require(REMOTE_REFERENCE)
    result = {"status": "V3_REMOTE_INPUT_GATE_PASS", "gpu_requested": False, "model_loaded": False, "model": model, "v3_config_sha256": V3_CONFIG_SHA256, "verified": verified, "reference_sha256": sha256_file(REMOTE_REFERENCE)}
    atomic_json(REMOTE_ROOT / "preflight/remote_input_probe.json", result)
    data_volume.commit()
    return result


def smoke_pairs() -> tuple[list[tuple[str, str]], list[float]]:
    rows = [json.loads(line) for line in REMOTE_V2_SMOKE_GROUPS.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs: list[tuple[str, str]] = []
    labels: list[float] = []
    for row in rows:
        positive = row["positive_chunks"][0]
        negative = row["negative_chunks"]["hard"][0]
        pairs.extend([(str(row["question"]), str(positive["text"])), (str(row["question"]), str(negative["text"]))])
        labels.extend([1.0, 0.0])
    return pairs, labels


def logits(value: Any) -> Any:
    result = value.logits
    if result.ndim == 2 and result.shape[-1] == 2:
        return result[:, 1] - result[:, 0]
    return result.reshape(-1)


@app.function(image=image, gpu="L4", volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume}, timeout=20 * 60, cpu=8, memory=32768, retries=0)
def smoke() -> dict[str, Any]:
    """One L4 smoke: 2 pairs/4 examples/1 step, including mask cases."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    data_volume.reload()
    model_volume.reload()
    model_contract = verify_remote_current_ft()
    groups_path = REMOTE_V2_SMOKE_GROUPS
    require(groups_path)
    trainer = load_module(REMOTE_TRAINER, "v3_conflict_anchor_trainer_smoke")
    output_root = REMOTE_ROOT / "smoke/gpu_checkpoint"
    if output_root.exists():
        raise RuntimeError("V3_SMOKE_OUTPUT_ALREADY_EXISTS")
    args = trainer.build_parser().parse_args([
        "--train-groups", str(groups_path), "--base-model", str(REMOTE_FT_MODEL), "--teacher-model", str(REMOTE_FT_MODEL),
        "--output-dir", str(output_root), "--allow-training", "--strict-oof-f1-f4", "--device", "cuda",
        "--learning-rate", "5e-7", "--epochs", "1", "--batch-size", "4", "--gradient-accumulation", "4", "--max-length", "512", "--freeze-layers", "18", "--teacher-weight", "0.5", "--teacher-anchor-mode", "conflict-aware", "--seed", "2026",
    ])
    report = trainer.run(args)
    if report.get("training_example_count") != 4 or report.get("optimizer_steps") != 1 or report.get("micro_steps") != 1:
        raise RuntimeError(f"V3_SMOKE_SIZE_OR_STEP_MISMATCH:{report.get('training_example_count')}/{report.get('optimizer_steps')}/{report.get('micro_steps')}")
    if report.get("teacher_consistent_examples", 0) <= 0 or report.get("teacher_conflict_examples", 0) <= 0:
        raise RuntimeError(f"V3_SMOKE_MASK_CASES_MISSING:{report}")
    checkpoint = output_root / "checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    student = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), local_files_only=True).to("cuda")
    teacher = AutoModelForSequenceClassification.from_pretrained(str(REMOTE_FT_MODEL), local_files_only=True).to("cuda")
    pairs, labels = smoke_pairs()
    encoded = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded = {key: value.to("cuda") for key, value in encoded.items()}
    target = torch.tensor(labels, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        student_logits = logits(student(**encoded))
        teacher_logits = logits(teacher(**encoded))
        loss, diagnostics = trainer.compute_teacher_anchor_loss(student_logits, teacher_logits, target, 0.5, "conflict-aware")
        bce_finite = bool(torch.isfinite(torch.nn.functional.binary_cross_entropy_with_logits(student_logits, target)))
        mse_finite = bool(torch.isfinite((student_logits - teacher_logits).square()).all())
    del student, teacher, tokenizer, encoded, target
    gc.collect()
    torch.cuda.empty_cache()
    if not math.isfinite(float(loss.cpu())) or not bce_finite or not mse_finite:
        raise RuntimeError("V3_SMOKE_NONFINITE_LOSS")
    result = {"status": "V3_SMOKE_PASS", "gpu": torch.cuda.get_device_name(0), "model_contract": model_contract, "training_report": report, "mask_check": diagnostics, "bce_finite": bce_finite, "mse_finite": mse_finite, "mse_excluded_on_conflicts": True, "checkpoint": str(checkpoint), "checkpoint_weight_sha256": sha256_file(checkpoint / "model.safetensors"), "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"), "model_loaded": True, "inference_pairs": len(pairs)}
    atomic_json(REMOTE_ROOT / "smoke/smoke_result.json", result)
    data_volume.commit()
    return result


@app.function(image=image, gpu="L4", volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume}, timeout=2 * 60 * 60, cpu=8, memory=32768, retries=0)
def train_fold(fold: int) -> dict[str, Any]:
    import torch

    data_volume.reload()
    model_volume.reload()
    paths = fold_paths(fold)
    spec = FOLD_SPECS[fold]
    model_contract = verify_remote_current_ft()
    require(paths["train_groups"])
    if sha256_file(paths["train_groups"]) != FOLD_INPUT_SHAS[fold]["train_groups.jsonl"]:
        raise RuntimeError("V3_TRAIN_GROUP_SHA_MISMATCH")
    if (paths["root"] / "gpu_checkpoint").exists():
        raise RuntimeError(f"V3_FOLD_OUTPUT_ALREADY_EXISTS:{paths['root'] / 'gpu_checkpoint'}")
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("V3_L4_ENV_GATE_FAIL")
    trainer = load_module(REMOTE_TRAINER, f"v3_conflict_anchor_trainer_f{fold}")
    args = trainer.build_parser().parse_args([
        "--train-groups", str(paths["train_groups"]), "--base-model", str(REMOTE_FT_MODEL), "--teacher-model", str(REMOTE_FT_MODEL), "--output-dir", str(REMOTE_ROOT / f"fold{fold}/gpu_checkpoint"),
        "--allow-training", "--strict-oof-f1-f4", "--device", "cuda", "--learning-rate", "5e-7", "--epochs", "1", "--batch-size", "4", "--gradient-accumulation", "4", "--max-length", "512", "--freeze-layers", "18", "--teacher-weight", "0.5", "--teacher-anchor-mode", "conflict-aware", "--seed", "2026",
    ])
    report = trainer.run(args)
    if report.get("training_example_count") != spec["examples"] or report.get("optimizer_steps") != spec["steps"] or report.get("teacher_anchor_mode") != "conflict-aware":
        raise RuntimeError(f"V3_FOLD_TRAIN_CONTRACT_MISMATCH:{report}")
    checkpoint = paths["checkpoint"]
    result = {"status": "V3_FOLD_TRAIN_PASS", "fold": fold, "gpu": torch.cuda.get_device_name(0), "model_contract": model_contract, "recipe": report, "checkpoint": str(checkpoint), "checkpoint_weight_sha256": sha256_file(checkpoint / "model.safetensors"), "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"), "expected": spec}
    atomic_json(paths["training_execution"], result)
    data_volume.commit()
    return result


@app.function(image=image, gpu="L4", volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume}, timeout=90 * 60, cpu=8, memory=32768, retries=0)
def score_fold(fold: int) -> dict[str, Any]:
    import torch

    data_volume.reload()
    model_volume.reload()
    paths = fold_paths(fold)
    spec = FOLD_SPECS[fold]
    model_contract = verify_remote_current_ft()
    for required in (paths["score_worklist"], REMOTE_QUESTIONS, REMOTE_REFERENCE, paths["checkpoint"] / "config.json", paths["checkpoint"] / "model.safetensors"):
        require(required)
    if sha256_file(paths["score_worklist"]) != FOLD_INPUT_SHAS[fold]["score_worklist.jsonl"]:
        raise RuntimeError("V3_SCORE_WORKLIST_SHA_MISMATCH")
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("V3_L4_ENV_GATE_FAIL")
    scorer = load_module(REMOTE_SCORER, f"v3_conflict_anchor_scorer_f{fold}")
    original_argv = sys.argv
    try:
        sys.argv = ["score_same_bge_ft_v2_gpu.py", "--worklist", str(paths["score_worklist"]), "--questions", str(REMOTE_QUESTIONS), "--chunks-root", str(REMOTE_CHUNKS), "--model", str(paths["checkpoint"]), "--reference-scores", str(REMOTE_REFERENCE), "--output", str(paths["scores"]), "--metrics", str(paths["metrics"]), "--batch-size", "16", "--device", "cuda"]
        scorer.main()
    finally:
        sys.argv = original_argv
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    if metrics.get("qdocs") != spec["qdocs"] or metrics.get("units") != spec["units"]:
        raise RuntimeError(f"V3_SCORE_COVERAGE_MISMATCH:{metrics.get('qdocs')}/{metrics.get('units')}")
    result = {"status": "V3_FOLD_SCORE_PASS", "fold": fold, "gpu": torch.cuda.get_device_name(0), "model_contract": model_contract, "metrics": metrics}
    atomic_json(paths["score_execution"], result)
    data_volume.commit()
    return result


@app.local_entrypoint()
def main(mode: str = "remote-input-probe", fold: int = 0) -> None:
    if mode == "remote-input-probe":
        print(json.dumps(remote_input_probe.remote(), ensure_ascii=False, indent=2), flush=True)
        return
    if mode == "smoke":
        print(json.dumps(smoke.remote(), ensure_ascii=False, indent=2), flush=True)
        return
    if mode == "train":
        if fold not in FOLD_SPECS:
            raise SystemExit("--fold must be one of 1,2,3,4")
        print(json.dumps(train_fold.remote(fold), ensure_ascii=False, indent=2), flush=True)
        return
    if mode == "score":
        if fold not in FOLD_SPECS:
            raise SystemExit("--fold must be one of 1,2,3,4")
        print(json.dumps(score_fold.remote(fold), ensure_ascii=False, indent=2), flush=True)
        return
    raise SystemExit(f"unknown mode: {mode}")
