"""CPU-only launcher for Delta-Recall residual feature forensic V2."""
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
VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
LOCAL_V3 = ROOT / "scripts/beam/task1_v3_residual"
REMOTE_V3 = RUNTIME / "scripts/beam/task1_v3_residual"
IMAGE = Image(python_version="python3.11", python_packages=["numpy", "scikit-learn"])


def sync_code() -> None:
    REMOTE_V3.mkdir(parents=True, exist_ok=True)
    for source in LOCAL_V3.rglob("*.py"):
        if "__pycache__" in source.parts: continue
        target = REMOTE_V3 / source.relative_to(LOCAL_V3); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def preflight() -> dict:
    required = [LOCAL_V3 / name for name in ("common.py", "train_residual_policy.py", "train_residual_policy_v3b.py", "forensic_stage1_hard_errors.py", "train_delta_recall_residual.py", "forensic_delta_recall_feature_space.py")]
    missing = [path for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing Delta feature forensic source dependencies: {[str(path) for path in missing]}")
    return {"status": "PREFLIGHT_PASS", "stage": "DELTA_RECALL_FEATURE_FORENSIC_V2", "required_source_files": [str(path) for path in required], "retrieval_enabled": False, "shortlist_enabled": False, "frozen_scoring_enabled": False, "delta_training_enabled": False, "fold0_evaluation_enabled": False, "gpu_parameter_present": False, "remote_submitted": False}


def validate_runtime_inputs(actions: Path, v3b_report: Path) -> None:
    missing = [path for path in (actions, v3b_report) if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing Delta feature forensic runtime artifacts: {[str(path) for path in missing]}")


@function(name="udsc-task1-delta-feature-forensic-cpu", cpu=16, memory="64Gi", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=0, headless=True)
def run():
    sync_code()
    root = RUNTIME / "artifacts/task1/recovery_096/v3_residual"; policy = root / "policy"; output = root / "delta_feature_forensic_v2"
    actions = root / "actions.jsonl" if (root / "actions.jsonl").is_file() else policy / "actions.jsonl"
    v3b = root / "v3b_policy/v3b_policy_training_report.json"
    validate_runtime_inputs(actions, v3b)
    output.mkdir(parents=True, exist_ok=True)
    command = ["python", "scripts/beam/task1_v3_residual/forensic_delta_recall_feature_space.py", "--actions", str(actions), "--v3b-report", str(v3b), "--output-dir", str(output)]
    if subprocess.run(command, cwd=str(RUNTIME)).returncode: raise RuntimeError(f"Delta feature forensic failed: {command}")
    print(json.dumps({"status": "DELTA_RECALL_FEATURE_FORENSIC_V2_COMPLETE", "output_dir": str(output), "fold0_labels_read": False, "fold0_evaluated": False, "gpu_launched": False, "no_submission_created": True, "descriptive_only": True}), flush=True)


if __name__ == "__main__":
    if "--preflight" in sys.argv: print(json.dumps(preflight(), indent=2))
    else: run.remote()
