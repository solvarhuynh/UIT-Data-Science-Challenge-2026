"""CPU-only Stage1 hard-negative forensic and CV experiment launcher."""
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


ROOT = Path(__file__).resolve().parents[2]; VOLUME_ROOT = Path("/workspace/p13"); RUNTIME = VOLUME_ROOT / "runtime"
LOCAL_V3 = ROOT / "scripts/beam/task1_v3_residual"; REMOTE_V3 = RUNTIME / "scripts/beam/task1_v3_residual"
IMAGE = Image(python_version="python3.11", python_packages=["scikit-learn", "numpy"])


def sync_code() -> None:
    REMOTE_V3.mkdir(parents=True, exist_ok=True)
    for source in LOCAL_V3.rglob("*.py"):
        if "__pycache__" in source.parts: continue
        target = REMOTE_V3 / source.relative_to(LOCAL_V3); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def preflight() -> dict:
    required = [LOCAL_V3 / name for name in ("forensic_stage1_hard_errors.py", "train_stage1_hard_negative.py", "train_residual_policy_v3b.py", "common.py")]
    if any(not path.is_file() for path in required): raise FileNotFoundError("Stage1 hard-negative source missing")
    return {"status": "PREFLIGHT_PASS", "stage": "STAGE1_HARD_NEGATIVE", "reuses_actions": True, "retrieval_enabled": False, "shortlist_enabled": False, "frozen_scoring_enabled": False, "stage2_enabled": False, "fold0_evaluation_enabled": False, "gpu_parameter_present": False, "remote_submitted": False}


@function(name="udsc-task1-stage1-hard-negative-cpu", cpu=16, memory="64Gi", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=1, headless=True)
def run():
    sync_code(); root = RUNTIME / "artifacts/task1/recovery_096/v3_residual"; policy = root / "policy"; output = root / "stage1_hard_negative"; actions = root / "actions.jsonl" if (root / "actions.jsonl").is_file() else policy / "actions.jsonl"; output.mkdir(parents=True, exist_ok=True)
    commands = [["python", "scripts/beam/task1_v3_residual/forensic_stage1_hard_errors.py", "--actions", str(actions), "--v3b-report", str(root / "v3b_policy/v3b_policy_training_report.json"), "--output-dir", str(output)], ["python", "scripts/beam/task1_v3_residual/train_stage1_hard_negative.py", "--actions", str(actions), "--v3a-report", str(policy / "policy_training_report.json"), "--v3b-report", str(root / "v3b_policy/v3b_policy_training_report.json"), "--output-dir", str(output)]]
    for command in commands:
        if subprocess.run(command, cwd=str(RUNTIME)).returncode: raise RuntimeError(f"Stage1 hard-negative step failed: {command}")
    print(json.dumps({"status": "STAGE1_HARD_NEGATIVE_COMPLETE", "output_dir": str(output), "fold0_labels_read": False, "fold0_evaluated": False, "gpu_launched": False, "no_submission_created": True}), flush=True)


if __name__ == "__main__": print(json.dumps(preflight(), indent=2)) if "--preflight" in sys.argv else run.remote()
