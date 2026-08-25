"""CPU launcher for the metric-aligned V3 delta-Recall residual smoke."""
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
IMAGE = Image(python_version="python3.11", python_packages=["numpy", "scikit-learn", "xgboost==3.2.0"])


def sync_code() -> None:
    REMOTE_V3.mkdir(parents=True, exist_ok=True)
    for source in LOCAL_V3.rglob("*.py"):
        if "__pycache__" in source.parts: continue
        target = REMOTE_V3 / source.relative_to(LOCAL_V3); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def preflight() -> dict:
    required = [LOCAL_V3 / name for name in ("common.py", "train_residual_policy.py", "train_residual_policy_v3b.py", "forensic_stage1_hard_errors.py", "forensic_benefit_neutral_signal.py", "train_delta_recall_residual.py")]
    missing = [path for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing delta-Recall residual source dependencies: {[str(path) for path in missing]}")
    return {"status": "PREFLIGHT_PASS", "stage": "DELTA_RECALL_RESIDUAL", "required_source_files": [str(path) for path in required], "retrieval_enabled": False, "shortlist_enabled": False, "frozen_scoring_enabled": False, "fold0_evaluation_enabled": False, "gpu_parameter_present": False, "remote_submitted": False}


def validate_runtime_inputs(actions: Path, v3a_report: Path, v3b_report: Path) -> None:
    missing = [path for path in (actions, v3a_report, v3b_report) if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing delta-Recall residual runtime artifacts: {[str(path) for path in missing]}")


@function(name="udsc-task1-delta-recall-residual-cpu", cpu=16, memory="64Gi", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=0, headless=True)
def run():
    sync_code()
    root = RUNTIME / "artifacts/task1/recovery_096/v3_residual"; policy = root / "policy"; output = root / "delta_recall_residual"
    actions = root / "actions.jsonl" if (root / "actions.jsonl").is_file() else policy / "actions.jsonl"
    v3a = policy / "policy_training_report.json"; v3b = root / "v3b_policy/v3b_policy_training_report.json"
    validate_runtime_inputs(actions, v3a, v3b)
    output.mkdir(parents=True, exist_ok=True)
    forensic = ["python", "scripts/beam/task1_v3_residual/forensic_benefit_neutral_signal.py", "--actions", str(actions), "--v3b-report", str(v3b), "--output-dir", str(output)]
    if subprocess.run(forensic, cwd=str(RUNTIME)).returncode: raise RuntimeError(f"Benefit-neutral forensic failed: {forensic}")
    trainer = ["python", "scripts/beam/task1_v3_residual/train_delta_recall_residual.py", "--actions", str(actions), "--v3a-report", str(v3a), "--v3b-report", str(v3b), "--output-dir", str(output)]
    if subprocess.run(trainer, cwd=str(RUNTIME)).returncode: raise RuntimeError(f"Delta-Recall residual failed: {trainer}")
    print(json.dumps({"status": "DELTA_RECALL_RESIDUAL_COMPLETE", "output_dir": str(output), "fold0_labels_read": False, "fold0_evaluated": False, "gpu_launched": False, "no_submission_created": True}), flush=True)


if __name__ == "__main__":
    if "--preflight" in sys.argv: print(json.dumps(preflight(), indent=2))
    else: run.remote()
