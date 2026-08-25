"""CPU-only Beam preparation for the immutable V2 fold0 docgroups.

This job performs the expensive payload scan and true-S2 materialization on
CPU memory.  It never loads the reranker and has no GPU resource parameter.
"""
from __future__ import annotations

import hashlib
import json
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
LOCAL_V2 = ROOT / "scripts/beam/task1_v2"
LOCAL_DATA = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v2/fold0_v2a/data"
REMOTE_DATA = RUNTIME / "artifacts/task1/recovery_096/inference_matched_reranker_v2/fold0_v2a/data"
QUESTIONS = ROOT / "data/raw/btc/LegalIR/train.json"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
CANDIDATES = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"
BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
PAYLOADS = ROOT / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"

IMAGE = Image(python_version="python3.11", python_packages=["pyvi"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_clean_data(data_dir: Path, candidates: Path, evidence: Path) -> dict:
    report_path = data_dir / "docgroups_report.json"
    train_path = data_dir / "train_docgroups.jsonl"
    eval_path = data_dir / "fold0_eval_docgroups.jsonl"
    required = [report_path, train_path, eval_path]
    if any(not path.is_file() for path in required):
        raise RuntimeError(f"prepared V2 data is incomplete: {data_dir}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "CLEAN":
        raise RuntimeError(f"prepared V2 data is not clean: {report.get('status')}")
    if int(report.get("eval_query_count", -1)) != 1400:
        raise RuntimeError("prepared V2 eval query count is not 1400")
    if [int(value) for value in report.get("train_folds", [])] != [1, 2, 3, 4]:
        raise RuntimeError("prepared V2 train folds are not [1,2,3,4]")
    if int(report.get("missing_evidence_count", report.get("missing_payload_or_raw_text", -1))) != 0:
        raise RuntimeError("prepared V2 data has missing evidence")
    if float(report.get("true_s2_selector_share", -1.0)) != 1.0:
        raise RuntimeError("prepared V2 true-S2 share is not 1.0")
    expected = {
        "candidate_refs_full_sha256": sha256(candidates),
        "train_docgroups_sha256": sha256(train_path),
        "fold0_eval_docgroups_sha256": sha256(eval_path),
        "evidence_py_sha256": sha256(evidence),
    }
    mismatches = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"prepared V2 hash handoff mismatch: {mismatches}")
    return report


def prepare_or_reuse(data_dir: Path, runtime: Path) -> dict:
    # Đồng bộ code V2 mới nhất (đã fix evidence.py) từ code được Beam upload
    # vào volume trước khi chạy subprocess. build_v2_docgroups.py chạy với cwd
    # là volume, nên file evidence.py trong volume phải là bản mới nhất.
    local_v2_dir = ROOT / "scripts/beam/task1_v2"
    remote_v2_dir = runtime / "scripts/beam/task1_v2"
    if local_v2_dir.is_dir() and remote_v2_dir.is_dir():
        for py in ("evidence.py", "build_v2_docgroups.py"):
            local_file = local_v2_dir / py
            remote_file = remote_v2_dir / py
            if local_file.is_file():
                remote_file.write_text(local_file.read_text(encoding="utf-8"), encoding="utf-8")

    questions = runtime / "data/raw/btc/LegalIR/train.json"
    folds = runtime / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
    candidates = runtime / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"
    baseline = runtime / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
    payloads = runtime / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"
    evidence = runtime / "scripts/beam/task1_v2/evidence.py"
    required = [questions, folds, candidates, baseline, payloads, evidence, runtime / "scripts/beam/task1_v2/build_v2_docgroups.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("CPU preparation input missing: " + ", ".join(missing))

    if data_dir.exists() and any(data_dir.iterdir()):
        return {"status": "REUSED_CLEAN", "report": validate_clean_data(data_dir, candidates, evidence)}
    data_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python", "scripts/beam/task1_v2/build_v2_docgroups.py",
        "--questions", "data/raw/btc/LegalIR/train.json",
        "--folds", "artifacts/task1/evaluation/strict_cv_v2/folds.json",
        "--candidates", "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl",
        "--baseline", "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl",
        "--payloads", "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json",
        "--output-dir", "artifacts/task1/recovery_096/inference_matched_reranker_v2/fold0_v2a/data",
        "--hard-negatives", "24", "--topk-evidence", "3",
    ]
    result = subprocess.run(command, cwd=str(runtime))
    if result.returncode:
        raise RuntimeError(f"CPU V2 preparation failed: {result.returncode}")
    return {"status": "CREATED_CLEAN", "report": validate_clean_data(data_dir, candidates, evidence)}


def preflight() -> dict[str, object]:
    required = [
        LOCAL_V2 / "evidence.py",
        LOCAL_V2 / "build_v2_docgroups.py",
        Path(__file__).resolve(),
        QUESTIONS,
        FOLDS,
        CANDIDATES,
        BASELINE,
        PAYLOADS,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("CPU prep preflight missing: " + ", ".join(missing))
    for path in [LOCAL_V2 / "evidence.py", LOCAL_V2 / "build_v2_docgroups.py", Path(__file__).resolve()]:
        subprocess.run([sys.executable, "-m", "py_compile", str(path)], check=True)
    source = Path(__file__).read_text(encoding="utf-8")
    decorator = source.split("def prepare_cpu", 1)[0]
    if "gpu=" in decorator:
        raise AssertionError("CPU preparation runner must not reserve a GPU")
    return {
        "status": "PREFLIGHT_PASS",
        "cpu": 16,
        "memory": "64Gi",
        "gpu_parameter_present": False,
        "model_loading": False,
        "output": "artifacts/task1/recovery_096/inference_matched_reranker_v2/fold0_v2a/data",
        "remote_submitted": False,
    }


@function(
    name="udsc-task1-v2-prepare-cpu",
    cpu=16,
    memory="64Gi",
    image=IMAGE,
    volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=1,
    headless=True,
)
def prepare_cpu():
    print(json.dumps(prepare_or_reuse(REMOTE_DATA, RUNTIME), indent=2), flush=True)


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        print(json.dumps(preflight(), indent=2))
    else:
        prepare_cpu.remote()
