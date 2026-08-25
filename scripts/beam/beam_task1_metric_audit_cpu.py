"""CPU-only, folds1--4 metric audit launcher; no policy search or Fold0 scoring."""
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
IMAGE = Image(python_version="python3.11", python_packages=["scikit-learn", "numpy"])


def sync_code() -> None:
    REMOTE_V3.mkdir(parents=True, exist_ok=True)
    for source in LOCAL_V3.rglob("*.py"):
        if "__pycache__" in source.parts: continue
        target = REMOTE_V3 / source.relative_to(LOCAL_V3)
        target.parent.mkdir(parents=True, exist_ok=True); target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def preflight() -> dict:
    required = [LOCAL_V3 / name for name in ("audit_task1_metric_integrity.py", "train_residual_policy_v3b.py", "train_residual_policy.py", "build_actions.py", "common.py")]
    if any(not path.is_file() for path in required): raise FileNotFoundError("metric audit source missing")
    return {"status": "PREFLIGHT_PASS", "stage": "METRIC_AUDIT", "fold0_gold_used": False, "full_v3b_search_enabled": False, "fold0_evaluation_enabled": False, "gpu_parameter_present": False, "remote_submitted": False}


@function(name="udsc-task1-metric-audit-cpu", cpu=16, memory="64Gi", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=1, headless=True)
def run():
    sync_code()
    root = RUNTIME / "artifacts/task1/recovery_096/v3_residual"; policy = root / "policy"; v3b = root / "v3b_policy"; output = root / "metric_audit" / "metric_integrity_report.json"
    actions = root / "actions.jsonl" if (root / "actions.jsonl").is_file() else policy / "actions.jsonl"
    command = ["python", "scripts/beam/task1_v3_residual/audit_task1_metric_integrity.py", "--actions", str(actions), "--questions", "data/raw/btc/LegalIR/train.json", "--v3a-report", str(policy / "policy_training_report.json"), "--v3b-report", str(v3b / "v3b_policy_training_report.json"), "--output", str(output)]
    if subprocess.run(command, cwd=str(RUNTIME)).returncode: raise RuntimeError(f"metric audit failed: {command}")
    print(json.dumps({"status": "METRIC_AUDIT_COMPLETE", "output": str(output), "fold0_gold_used": False, "gpu_launched": False, "no_submission_created": True}), flush=True)


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2)) if "--preflight" in sys.argv else run.remote()
