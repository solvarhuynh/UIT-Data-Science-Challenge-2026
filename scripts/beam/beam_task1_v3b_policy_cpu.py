"""Stage A Beam CPU launcher: V3B selection and label-free Fold0 predictions."""
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
IMAGE = Image(python_version="python3.11", python_packages=["pyvi", "scikit-learn", "numpy"])


def sync_code() -> None:
    REMOTE_V3.mkdir(parents=True, exist_ok=True)
    for source in LOCAL_V3.rglob("*.py"):
        if "__pycache__" in source.parts: continue
        target = REMOTE_V3 / source.relative_to(LOCAL_V3)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def preflight() -> dict:
    required = [LOCAL_V3 / name for name in ("forensic_v3b_policy.py", "train_residual_policy_v3b.py", "common.py")]
    if any(not path.is_file() for path in required): raise FileNotFoundError("V3B CPU source missing")
    return {"status": "PREFLIGHT_PASS", "stage": "POLICY_SELECTION", "cpu": 16, "memory": "64Gi", "fold0_gold_read": False, "fold0_evaluation_enabled": False, "gpu_parameter_present": False, "reuses_actions": True, "reruns_retrieval": False, "reruns_shortlist": False, "reruns_frozen_scoring": False, "remote_submitted": False}


@function(name="udsc-task1-v3b-policy-cpu", cpu=16, memory="64Gi", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=1, headless=True)
def run():
    sync_code()
    root = RUNTIME / "artifacts/task1/recovery_096/v3_residual"; output = root / "v3b_policy"
    actions = root / "actions.jsonl" if (root / "actions.jsonl").is_file() else root / "policy/actions.jsonl"
    v3a_report = root / "policy/policy_training_report.json"
    required = [actions, v3a_report]
    if any(not path.is_file() for path in required): raise FileNotFoundError(f"V3B requires existing V3 action/policy artifacts: {required}")
    output.mkdir(parents=True, exist_ok=True)
    commands = [
        ["python", "scripts/beam/task1_v3_residual/forensic_v3b_policy.py", "--actions", str(actions), "--v3a-report", str(v3a_report), "--report", str(output / "forensic_report.json")],
        ["python", "scripts/beam/task1_v3_residual/train_residual_policy_v3b.py", "--actions", str(actions), "--output-dir", str(output), "--v3a-report", str(v3a_report)],
    ]
    for command in commands:
        if subprocess.run(command, cwd=str(RUNTIME)).returncode: raise RuntimeError(f"V3B CPU stage failed: {command}")
    print(json.dumps({"status": "V3B_CPU_SELECTION_COMPLETE", "output_dir": str(output), "fold0_gold_read": False, "fold0_evaluated": False, "gpu_launched": False, "no_submission_created": True}), flush=True)


if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2)) if "--preflight" in sys.argv else run.remote()
