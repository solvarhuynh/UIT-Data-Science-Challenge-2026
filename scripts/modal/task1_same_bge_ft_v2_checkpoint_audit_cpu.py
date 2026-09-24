"""CPU-only forensic tensor audit of the preserved SAME-BGE F1 checkpoint.

This function reads safetensors from the existing Modal Volumes only.  It does
not import transformers, torch, or a scorer; it cannot train or perform model
inference.  The resulting small JSON manifest is written to the data Volume
and downloaded by the local entrypoint after the read-only comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

import modal


APP_NAME = "task1-same-bge-ft-v2-checkpoint-audit-cpu"
DATA_VOLUME_NAME = "udsc-p13"
MODEL_VOLUME_NAME = "udsc-task1-modal"
DATA_MOUNT = Path("/workspace/p13")
MODEL_MOUNT = Path("/data")
REMOTE_FOLD = DATA_MOUNT / "runtime/private_task1/experiments/sprint48_bge_ft_v2/fold1"
REMOTE_CHECKPOINT = REMOTE_FOLD / "gpu_checkpoint/checkpoint"
REMOTE_CURRENT_FT = MODEL_MOUNT / "bge_ft/sprint48_bge_ft_v2/current_ft_verified"
REMOTE_MANIFEST = REMOTE_FOLD / "checkpoint_tensor_diff_manifest.json"

EXPECTED_CHECKPOINT_WEIGHT_SHA = "4bbe295ef1a1901288230c030438663cffd775d0c1eb6aa49f1c4eed9a3ce601"
EXPECTED_CURRENT_WEIGHT_SHA = "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c"
EXPECTED_CONFIG_SHA = "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b"
EXPECTED_MODEL_ID = "BAAI/bge-reranker-v2-m3"
EXPECTED_MAX_LENGTH = 512
EXPECTED_FREEZE_LAYERS = 18

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy",
    "safetensors",
)
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
    temporary.replace(path)


def tensor_group(name: str) -> tuple[str, bool, int | None]:
    if "embeddings" in name:
        return "embeddings", True, None
    match = re.search(r"encoder\.layer\.(\d+)(?:\.|$)", name)
    if match:
        layer = int(match.group(1))
        return (f"encoder_layer_{layer}", layer < EXPECTED_FREEZE_LAYERS, layer)
    if any(token in name for token in ("classifier", "score", "head")):
        return "classifier_head", False, None
    return "other", False, None


def config_label_count(config: dict[str, Any]) -> int:
    return int(config.get("num_labels", len(config.get("id2label", {})) or -1))


def finite_array(array: Any, numpy: Any) -> bool:
    try:
        return bool(numpy.isfinite(array).all())
    except (TypeError, ValueError):
        return False


@app.function(
    image=image,
    volumes={str(DATA_MOUNT): data_volume, str(MODEL_MOUNT): model_volume},
    timeout=60 * 60,
    cpu=8,
    memory=32768,
    retries=0,
)
def audit_checkpoint_cpu() -> dict[str, Any]:
    """Compare the preserved F1 safetensors to CURRENT_FT on CPU only."""
    import numpy as np
    from safetensors.numpy import safe_open

    data_volume.reload()
    model_volume.reload()
    checkpoint_weight = REMOTE_CHECKPOINT / "model.safetensors"
    checkpoint_config = REMOTE_CHECKPOINT / "config.json"
    current_weight = REMOTE_CURRENT_FT / "model.safetensors"
    current_config = REMOTE_CURRENT_FT / "config.json"
    required = (checkpoint_weight, checkpoint_config, current_weight, current_config)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"CHECKPOINT_AUDIT_REQUIRED_FILES_MISSING:{missing}")

    checkpoint_sha = sha256_file(checkpoint_weight)
    current_sha = sha256_file(current_weight)
    checkpoint_config_sha = sha256_file(checkpoint_config)
    current_config_sha = sha256_file(current_config)
    if checkpoint_sha != EXPECTED_CHECKPOINT_WEIGHT_SHA:
        raise RuntimeError(f"CHECKPOINT_IDENTITY_GATE_FAIL_WEIGHT:{checkpoint_sha}")
    if checkpoint_config_sha != EXPECTED_CONFIG_SHA or current_config_sha != EXPECTED_CONFIG_SHA:
        raise RuntimeError(f"CHECKPOINT_IDENTITY_GATE_FAIL_CONFIG:{checkpoint_config_sha}/{current_config_sha}")
    if current_sha != EXPECTED_CURRENT_WEIGHT_SHA:
        raise RuntimeError(f"CURRENT_FT_IDENTITY_GATE_FAIL_WEIGHT:{current_sha}")

    checkpoint_cfg = json.loads(checkpoint_config.read_text(encoding="utf-8"))
    current_cfg = json.loads(current_config.read_text(encoding="utf-8"))
    checkpoint_labels = config_label_count(checkpoint_cfg)
    current_labels = config_label_count(current_cfg)
    config_contract = {
        "checkpoint_model_type": checkpoint_cfg.get("model_type"),
        "current_model_type": current_cfg.get("model_type"),
        "checkpoint_num_labels": checkpoint_labels,
        "current_num_labels": current_labels,
        "config_objects_equal": checkpoint_cfg == current_cfg,
        "model_family_compatible": checkpoint_cfg.get("model_type") == "xlm-roberta" and current_cfg.get("model_type") == "xlm-roberta",
        "single_logit_head": checkpoint_labels == 1 and current_labels == 1,
        "max_length_contract": EXPECTED_MAX_LENGTH,
    }

    initial_names: set[str]
    checkpoint_names: set[str]
    with safe_open(str(current_weight), framework="numpy") as current_file:
        initial_names = set(current_file.keys())
    with safe_open(str(checkpoint_weight), framework="numpy") as checkpoint_file:
        checkpoint_names = set(checkpoint_file.keys())
    missing_tensors = sorted(initial_names - checkpoint_names)
    extra_tensors = sorted(checkpoint_names - initial_names)
    if missing_tensors or extra_tensors:
        raise RuntimeError(f"CHECKPOINT_TENSOR_SET_MISMATCH:{len(missing_tensors)}/{len(extra_tensors)}")

    tensor_records: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    nonfinite = 0
    frozen_total = frozen_changed = 0
    trainable_total = trainable_changed = trainable_unchanged = 0
    changed_parameter_count = unchanged_parameter_count = 0
    for group_name in ("embeddings", "classifier_head", "other"):
        summary[group_name] = {"tensors": 0, "changed_tensors": 0, "parameters": 0, "changed_parameters": 0, "squared_delta_sum": 0.0, "max_abs_delta": 0.0}

    with safe_open(str(current_weight), framework="numpy") as current_file, safe_open(str(checkpoint_weight), framework="numpy") as checkpoint_file:
        for name in sorted(initial_names):
            initial = np.asarray(current_file.get_tensor(name))
            trained = np.asarray(checkpoint_file.get_tensor(name))
            group, frozen_expected, layer = tensor_group(name)
            if layer is not None:
                group = f"encoder_layer_{layer}"
            summary.setdefault(group, {"tensors": 0, "changed_tensors": 0, "parameters": 0, "changed_parameters": 0, "squared_delta_sum": 0.0, "max_abs_delta": 0.0})
            same_shape = initial.shape == trained.shape
            same_dtype = initial.dtype == trained.dtype
            initial_finite = finite_array(initial, np)
            trained_finite = finite_array(trained, np)
            if not initial_finite or not trained_finite:
                nonfinite += 1
            if not same_shape or not same_dtype:
                raise RuntimeError(f"CHECKPOINT_TENSOR_SCHEMA_MISMATCH:{name}:{initial.shape}/{trained.shape}:{initial.dtype}/{trained.dtype}")
            initial_hash = hashlib.sha256(np.ascontiguousarray(initial).tobytes()).hexdigest()
            trained_hash = hashlib.sha256(np.ascontiguousarray(trained).tobytes()).hexdigest()
            exact_equal = bool(np.array_equal(initial, trained))
            parameter_count = int(initial.size)
            max_abs = 0.0
            mean_abs = 0.0
            l2 = 0.0
            if not exact_equal:
                difference = np.subtract(trained.astype(np.float64), initial.astype(np.float64))
                absolute = np.abs(difference)
                max_abs = float(absolute.max())
                mean_abs = float(absolute.mean())
                l2 = float(np.sqrt(np.square(difference).sum()))
                changed_parameter_count += parameter_count
            else:
                unchanged_parameter_count += parameter_count
            summary[group]["tensors"] += 1
            summary[group]["parameters"] += parameter_count
            summary[group]["squared_delta_sum"] += l2 * l2
            summary[group]["max_abs_delta"] = max(summary[group]["max_abs_delta"], max_abs)
            if exact_equal:
                summary[group]["changed_tensors"] += 0
            else:
                summary[group]["changed_tensors"] += 1
                summary[group]["changed_parameters"] += parameter_count
            if frozen_expected:
                frozen_total += 1
                if not exact_equal:
                    frozen_changed += 1
            else:
                trainable_total += 1
                if exact_equal:
                    trainable_unchanged += 1
                else:
                    trainable_changed += 1
            tensor_records.append({
                "name": name,
                "shape": list(initial.shape),
                "dtype": str(initial.dtype),
                "group": group,
                "expected_frozen": frozen_expected,
                "initial_tensor_sha256": initial_hash,
                "checkpoint_tensor_sha256": trained_hash,
                "exact_equal": exact_equal,
                "max_abs_delta": max_abs,
                "mean_abs_delta": mean_abs,
                "l2_delta": l2,
                "finite_initial": initial_finite,
                "finite_checkpoint": trained_finite,
            })

    for values in summary.values():
        values["l2_delta"] = math.sqrt(values.pop("squared_delta_sum"))

    training_report_path = REMOTE_CHECKPOINT.parent / "training_report.json"
    preflight_path = REMOTE_CHECKPOINT.parent / "preflight.json"
    execution_path = REMOTE_FOLD / "execution_result.json"
    metrics_path = REMOTE_FOLD / "score_metrics.json"
    training_report = json.loads(training_report_path.read_text(encoding="utf-8")) if training_report_path.is_file() else None
    preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else None
    execution = json.loads(execution_path.read_text(encoding="utf-8")) if execution_path.is_file() else None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    recorded_hashes = {
        "training_report_checkpoint": (training_report or {}).get("checkpoint_weight_sha256"),
        "execution_result_checkpoint": ((execution or {}).get("metrics") or {}).get("weight_sha256"),
        "score_metrics_checkpoint": (metrics or {}).get("weight_sha256"),
    }
    manifest = {
        "status": "CHECKPOINT_TENSOR_AUDIT_PASS" if frozen_changed == 0 and trainable_changed > 0 and nonfinite == 0 else "CHECKPOINT_TENSOR_AUDIT_FAIL",
        "mode": "CPU_ONLY_REMOTE_SAFETENSORS_READ",
        "model_loaded": False,
        "inference_run": False,
        "training_run": False,
        "checkpoint_location": str(REMOTE_CHECKPOINT),
        "current_ft_location": str(REMOTE_CURRENT_FT),
        "checkpoint_weight_sha256": checkpoint_sha,
        "current_ft_weight_sha256": current_sha,
        "checkpoint_config_sha256": checkpoint_config_sha,
        "current_ft_config_sha256": current_config_sha,
        "checkpoint_identity_gate": "PASS",
        "tensor_set": {"initial_total": len(initial_names), "checkpoint_total": len(checkpoint_names), "missing": missing_tensors, "extra": extra_tensors},
        "config_contract": config_contract,
        "expected_recipe": {"freeze_embeddings": True, "freeze_encoder_layers": "0-17", "max_length": EXPECTED_MAX_LENGTH, "learning_rate": 5e-7, "epochs": 1, "teacher_weight": 0.5, "seed": 2026},
        "frozen_tensors": {"expected": frozen_total, "changed": frozen_changed, "max_abs_delta": max((r["max_abs_delta"] for r in tensor_records if r["expected_frozen"]), default=0.0)},
        "trainable_tensors": {"total": trainable_total, "changed": trainable_changed, "unchanged": trainable_unchanged},
        "parameter_counts": {"changed": changed_parameter_count, "unchanged": unchanged_parameter_count},
        "nonfinite_tensors": nonfinite,
        "unexpected_dtype_or_shape_changes": 0,
        "module_delta_summary": summary,
        "recorded_checkpoint_hashes": recorded_hashes,
        "recorded_checkpoint_hashes_all_match": all(value == EXPECTED_CHECKPOINT_WEIGHT_SHA for value in recorded_hashes.values() if value is not None),
        "training_report": training_report,
        "preflight": preflight,
        "execution_result": execution,
        "score_metrics": metrics,
        "tensors": tensor_records,
    }
    atomic_json(REMOTE_MANIFEST, manifest)
    data_volume.commit()
    return {key: value for key, value in manifest.items() if key not in {"tensors", "training_report", "preflight", "execution_result", "score_metrics"}} | {"manifest": str(REMOTE_MANIFEST), "tensor_record_count": len(tensor_records)}


@app.local_entrypoint()
def main() -> None:
    result = audit_checkpoint_cpu.remote()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
