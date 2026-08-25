"""Beam CPU launcher for the V3 residual swap-policy pipeline.

Mounts ``udsc-p13`` and runs ``run_policy_cpu.py`` directly on the volume so
it reads the on-volume ``frozen_features.jsonl`` (no 147MB/283MB download to
the local repo).  Outputs (actions, policy predictions, fold0 evaluation) land
back in ``v3_residual/`` on the volume.

Steps run in order (all CPU, all via subprocess against the volume runtime):
  1. build_actions        -> actions.jsonl           (reads frozen_features.jsonl)
  2. train_residual_policy -> fold0_v3_policy_predictions.jsonl
  3. evaluate_v3_fold0    -> fold0_evaluation.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from beam import Image, Volume, function
except ModuleNotFoundError:
    if "--preflight" not in sys.argv:
        raise

    class Image:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            pass

    class Volume:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            pass

    def function(**kwargs):  # type: ignore[no-redef]
        def decorate(fn):
            return fn

        return decorate


ROOT = Path(__file__).resolve().parents[2]
VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
LOCAL_V3 = ROOT / "scripts/beam/task1_v3_residual"
REMOTE_V3 = RUNTIME / "scripts/beam/task1_v3_residual"
PUBLIC_OUT = RUNTIME / "artifacts/task1/recovery_096/final_public_v3"


IMAGE = Image(python_version="python3.11", python_packages=["pyvi", "scikit-learn", "numpy"])


def _sync_v3_code() -> None:
    """Đồng bộ code V3 từ code-upload vào volume trước khi chạy subprocess.

    run_policy_cpu.py chạy với cwd='runtime' (volume) và gọi các sub-script bằng
    path tương đối, nên toàn bộ file .py của task1_v3_residual phải tồn tại trong
    volume. Volume được mount tại VOLUME_ROOT nên ta ghi đè trực tiếp.
    """
    remote_v3 = REMOTE_V3
    remote_v3.mkdir(parents=True, exist_ok=True)
    for src in LOCAL_V3.rglob("*.py"):
        if "__pycache__" in src.parts:
            continue
        dst = remote_v3 / src.relative_to(LOCAL_V3)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def preflight(mode: str = "train") -> dict:
    required = [
        LOCAL_V3 / "run_policy_cpu.py",
        LOCAL_V3 / "build_actions.py",
        LOCAL_V3 / "train_residual_policy.py",
        LOCAL_V3 / "evaluate_v3_fold0.py",
        LOCAL_V3 / "common.py",
    ]
    if any(not path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError("V3 CPU policy source missing:\n" + "\n".join(missing))
    source = Path(__file__).read_text(encoding="utf-8")
    decorator = source.split("def run", 1)[0]
    if "gpu=" in decorator:
        raise AssertionError("V3 CPU policy runner must not load a reranker")
    return {
        "status": "PREFLIGHT_PASS",
        "cpu": 16,
        "memory": "64Gi",
        "gpu_parameter_present": False,
        "mode": mode,
        "reads_frozen_features_from_volume": mode == "train",
        "labels_required": mode == "train",
        "folds_required": mode == "train",
        "remote_submitted": False,
    }


@function(
    name="udsc-task1-v3-policy-cpu",
    cpu=16,
    memory="64Gi",
    image=IMAGE,
    volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=1,
    headless=True,
)
def run():
    _sync_v3_code()
    runner = RUNTIME / "scripts/beam/task1_v3_residual/run_policy_cpu.py"
    policy_root = RUNTIME / "artifacts/task1/recovery_096"
    required = [
        runner,
        RUNTIME / "artifacts/task1/recovery_096/v3_residual/frozen_features.jsonl",
        RUNTIME / "data/raw/btc/LegalIR/train.json",
        policy_root / "baseline_093_oof/predictions.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("V3 CPU policy inputs missing on volume:\n" + "\n".join(missing))
    features = required[1]
    command = [
        "python", "scripts/beam/task1_v3_residual/run_policy_cpu.py",
    ]
    result = subprocess.run(command, cwd=str(RUNTIME))
    if result.returncode:
        raise RuntimeError(f"V3 CPU policy pipeline failed: {result.returncode}")
    print(
        json.dumps({"status": "V3_POLICY_CLEAN", "runner": str(runner), "features": str(features), "output_dir": str(RUNTIME / "artifacts/task1/recovery_096/v3_residual/policy")}),
        flush=True,
    )


@function(
    name="udsc-task1-v3a-public-policy-cpu",
    cpu=16,
    memory="64Gi",
    image=IMAGE,
    volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=0,
    headless=True,
)
def run_public():
    _sync_v3_code()
    runner = RUNTIME / "scripts/beam/task1_v3_residual/run_policy_cpu.py"
    required = [
        runner,
        PUBLIC_OUT / "public_frozen_features.jsonl",
        RUNTIME / "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit.joblib",
        RUNTIME / "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit_manifest.json",
        RUNTIME / "artifacts/task1/recovery_096/public_anchor_093/reproduced_093_submission.zip",
        RUNTIME / "data/raw/btc/LegalIR/public-official.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("public V3A policy inputs missing on volume:\n" + "\n".join(missing))
    command = [
        "python", "scripts/beam/task1_v3_residual/run_policy_cpu.py", "--public",
        "--features", "artifacts/task1/recovery_096/final_public_v3/public_frozen_features.jsonl",
        "--public-model", "artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit.joblib",
        "--public-baseline", "artifacts/task1/recovery_096/public_anchor_093/reproduced_093_submission.zip",
        "--public-questions", "data/raw/btc/LegalIR/public-official.json",
        "--public-output", "artifacts/task1/recovery_096/final_public_v3/public_v3a_predictions.json",
        "--public-report", "artifacts/task1/recovery_096/final_public_v3/public_v3a_policy_report.json",
    ]
    result = subprocess.run(command, cwd=str(RUNTIME))
    if result.returncode:
        raise RuntimeError(f"public V3A policy application failed: {result.returncode}")
    print(json.dumps({"status": "PUBLIC_V3A_POLICY_COMPLETE", "output": str(PUBLIC_OUT / "public_v3a_predictions.json"), "no_public_labels_used": True}), flush=True)


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        print(json.dumps(preflight("public" if "--public" in sys.argv else "train"), indent=2))
    else:
        (run_public.remote() if "--public" in sys.argv else run.remote())
