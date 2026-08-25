"""CPU launcher for the model-free incoming-document relevance signal forensic."""
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
LOCAL_V3 = ROOT / "scripts" / "beam" / "task1_v3_residual"; REMOTE_V3 = RUNTIME / "scripts" / "beam" / "task1_v3_residual"
IMAGE = Image(python_version="python3.11")


def sync_code() -> None:
    REMOTE_V3.mkdir(parents=True, exist_ok=True)
    for name in ("common.py", "forensic_incoming_relevance_signal.py"):
        source = LOCAL_V3 / name
        (REMOTE_V3 / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def preflight() -> dict:
    required = [LOCAL_V3 / "common.py", LOCAL_V3 / "forensic_incoming_relevance_signal.py"]
    missing = [path for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing incoming relevance forensic source dependencies: {[str(path) for path in missing]}")
    return {"status": "PREFLIGHT_PASS", "required_source_files": [str(path) for path in required], "model_training_enabled": False, "fold0_evaluation_enabled": False, "gpu_parameter_present": False, "remote_submitted": False}


def validate_runtime_inputs(actions: Path, questions: Path) -> None:
    missing = [path for path in (actions, questions) if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing incoming relevance forensic runtime inputs: {[str(path) for path in missing]}")


@function(name="udsc-task1-incoming-relevance-signal-cpu", cpu=16, memory="64Gi", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=0, headless=True)
def run() -> None:
    sync_code()
    root = RUNTIME / "artifacts/task1/recovery_096/v3_residual"; policy = root / "policy"
    actions = root / "actions.jsonl" if (root / "actions.jsonl").is_file() else policy / "actions.jsonl"
    questions = RUNTIME / "data/raw/btc/LegalIR/train.json"; output = root / "incoming_relevance_signal"
    validate_runtime_inputs(actions, questions)
    output.mkdir(parents=True, exist_ok=True)
    command = ["python", "scripts/beam/task1_v3_residual/forensic_incoming_relevance_signal.py", "--actions", str(actions), "--questions", str(questions), "--output-dir", str(output)]
    if subprocess.run(command, cwd=str(RUNTIME)).returncode: raise RuntimeError(f"Incoming relevance forensic failed: {command}")
    print(json.dumps({"status": "INCOMING_RELEVANCE_SIGNAL_COMPLETE", "output_dir": str(output), "fold0_labels_read": False, "fold0_evaluated": False, "model_trained": False, "gpu_launched": False, "retrieval_rerun": False, "neural_scoring_rerun": False, "submission_created": False}), flush=True)


if __name__ == "__main__":
    if "--preflight" in sys.argv: print(json.dumps(preflight(), indent=2))
    else: run.remote()
