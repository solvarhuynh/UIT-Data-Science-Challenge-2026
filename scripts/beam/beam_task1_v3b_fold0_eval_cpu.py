"""Stage B Beam CPU launcher: evaluate already-frozen V3B Fold0 predictions."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    from beam import Image, Volume, function
except ModuleNotFoundError:
    if "--preflight" not in sys.argv: raise
    class Image:
        def __init__(self, **kwargs): pass
    class Volume:
        def __init__(self, **kwargs): pass
    def function(**kwargs): return lambda fn: fn


ROOT = Path(__file__).resolve().parents[2]
VOLUME_ROOT = Path("/workspace/p13"); RUNTIME = VOLUME_ROOT / "runtime"
LOCAL_V3 = ROOT / "scripts/beam/task1_v3_residual"; REMOTE_V3 = RUNTIME / "scripts/beam/task1_v3_residual"
IMAGE = Image(python_version="python3.11")


def sync_code() -> None:
    """Make evaluation source on the mounted volume exactly match local source."""
    REMOTE_V3.mkdir(parents=True, exist_ok=True)
    for source in LOCAL_V3.rglob("*.py"):
        if "__pycache__" in source.parts: continue
        target = REMOTE_V3 / source.relative_to(LOCAL_V3)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def preflight() -> dict:
    required = [LOCAL_V3 / name for name in ("evaluate_v3b_fold0.py", "evaluate_v3_fold0.py", "common.py")]
    if any(not path.is_file() for path in required): raise FileNotFoundError("V3B Fold0 evaluation source missing")
    return {"status": "PREFLIGHT_PASS", "stage": "FOLD0_EVALUATION", "policy_training_enabled": False, "gpu_parameter_present": False, "remote_submitted": False}


@function(name="udsc-task1-v3b-fold0-eval-cpu", cpu=16, memory="64Gi", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=1, headless=True)
def run():
    sync_code()
    root = RUNTIME / "artifacts/task1/recovery_096/v3_residual"; output = root / "v3b_policy"
    command = [
        "python", "scripts/beam/task1_v3_residual/evaluate_v3b_fold0.py",
        "--baseline", "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl",
        "--v3a-predictions", str(root / "policy/fold0_v3_policy_predictions.jsonl"),
        "--v3b-predictions", str(output / "fold0_v3b_policy_predictions.jsonl"),
        "--questions", "data/raw/btc/LegalIR/train.json",
        "--report", str(output / "fold0_evaluation.json"),
    ]
    if subprocess.run(command, cwd=str(RUNTIME)).returncode: raise RuntimeError(f"V3B Fold0 evaluation failed: {command}")
    print(json.dumps({"status": "V3B_FOLD0_EVALUATION_COMPLETE", "output": str(output / "fold0_evaluation.json"), "gpu_launched": False, "no_submission_created": True}), flush=True)


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2)) if "--preflight" in sys.argv else run.remote()
