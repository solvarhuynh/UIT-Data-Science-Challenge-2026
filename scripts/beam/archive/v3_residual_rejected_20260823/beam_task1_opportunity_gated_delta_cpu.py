"""CPU Beam launcher for the opportunity-gated Delta-Recall selector."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    from beam import Image, Volume, function
except ModuleNotFoundError:
    if "--preflight" not in sys.argv:
        raise

    class Image:
        def __init__(self, **kwargs):
            pass

    class Volume:
        def __init__(self, **kwargs):
            pass

    def function(**kwargs):
        return lambda fn: fn


ROOT = Path(__file__).resolve().parents[2]
VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
LOCAL_BEAM = ROOT / "scripts" / "beam"
REMOTE_BEAM = RUNTIME / "scripts/beam"
IMAGE = Image(python_version="python3.11", python_packages=["numpy", "scikit-learn", "xgboost==3.2.0"])


def sync_code() -> None:
    REMOTE_BEAM.mkdir(parents=True, exist_ok=True)
    for source in LOCAL_BEAM.rglob("*.py"):
        if "__pycache__" in source.parts:
            continue
        target = REMOTE_BEAM / source.relative_to(LOCAL_BEAM)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def preflight() -> dict:
    required = [
        LOCAL_BEAM / "task1_v3_residual" / name
        for name in (
            "common.py",
            "train_residual_policy.py",
            "train_residual_policy_v3b.py",
            "forensic_stage1_hard_errors.py",
            "train_opportunity_gated_delta_selector.py",
        )
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing opportunity-gated selector source dependencies: {[str(path) for path in missing]}")
    return {
        "status": "PREFLIGHT_PASS",
        "required_source_files": [str(path) for path in required],
        "runtime": str(RUNTIME),
        "top_k": 20,
        "retrieval_enabled": False,
        "shortlist_enabled": False,
        "frozen_scoring_enabled": False,
        "delta_v1_enabled": False,
        "forensic_v2_enabled": False,
        "fold0_evaluation_enabled": False,
        "gpu_parameter_present": False,
        "remote_submitted": False,
    }


def validate_runtime_inputs(actions: Path, v3a_report: Path, v3b_report: Path) -> None:
    missing = [path for path in (actions, v3a_report, v3b_report) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing opportunity-gated selector runtime artifacts: {[str(path) for path in missing]}")


@function(
    name="udsc-task1-opportunity-gated-delta-cpu",
    cpu=16,
    memory="64Gi",
    image=IMAGE,
    volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=0,
    headless=True,
)
def run() -> None:
    sync_code()
    root = RUNTIME / "artifacts/task1/recovery_096/v3_residual"
    policy = root / "policy"
    output = root / "opportunity_gated_delta"
    actions = root / "actions.jsonl" if (root / "actions.jsonl").is_file() else policy / "actions.jsonl"
    v3a_report = policy / "policy_training_report.json"
    v3b_report = root / "v3b_policy" / "v3b_policy_training_report.json"
    validate_runtime_inputs(actions, v3a_report, v3b_report)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "scripts/beam/task1_v3_residual/train_opportunity_gated_delta_selector.py",
        "--actions", str(actions),
        "--v3a-report", str(v3a_report),
        "--v3b-report", str(v3b_report),
        "--output-dir", str(output),
    ]
    if subprocess.run(command, cwd=str(RUNTIME)).returncode:
        raise RuntimeError(f"Opportunity-gated Delta-Recall selector failed: {command}")
    print(json.dumps({"status": "OPPORTUNITY_GATED_DELTA_SELECTOR_COMPLETE", "output_dir": str(output), "fold0_labels_read": False, "fold0_evaluated": False, "gpu_launched": False, "no_submission_created": True}), flush=True)


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        print(json.dumps(preflight(), indent=2))
    else:
        run.remote()
