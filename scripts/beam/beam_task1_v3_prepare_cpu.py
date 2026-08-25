"""CPU-only Beam materialization of V3 shortlist true-S2 evidence."""
from __future__ import annotations

import json
import hashlib
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
    def function(**kwargs):
        return lambda fn: fn

ROOT = Path(__file__).resolve().parents[2]
VOLUME_ROOT = Path("/workspace/p13"); RUNTIME = VOLUME_ROOT / "runtime"
OUT = RUNTIME / "artifacts/task1/recovery_096/v3_residual"
IMAGE = Image(python_version="python3.11", python_packages=["pyvi"])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def preflight() -> dict:
    required = [ROOT / "scripts/beam/task1_v3_residual/build_shortlist_features.py", ROOT / "scripts/beam/task1_v2/evidence.py"]
    if any(not path.is_file() for path in required): raise FileNotFoundError("V3 CPU source missing")
    source = Path(__file__).read_text(encoding="utf-8").split("def prepare_cpu", 1)[0]
    if "gpu=" in source or "models/reranker" in source: raise AssertionError("V3 CPU prep must not reserve GPU or load model")
    return {"status":"PREFLIGHT_PASS", "cpu":16, "memory":"64Gi", "gpu_parameter_present":False, "model_loading":False, "remote_submitted":False}

def _sync_v3_code() -> None:
    """Đồng bộ code V3 (và dependency evidence.py) từ code-upload vào volume.

    build_shortlist_features.py chạy qua subprocess với cwd là volume, nên mọi
    module nó import phải tồn tại trong volume. Beam mount volume tại VOLUME_ROOT
    nên ta có thể ghi đè trực tiếp bằng bản code upload (ROOT).
    """
    pairs = [
        ("scripts/beam/task1_v3_residual", "scripts/beam/task1_v3_residual"),
        ("scripts/beam/task1_v2/evidence.py", "scripts/beam/task1_v2/evidence.py"),
    ]
    for local_rel, remote_rel in pairs:
        local = ROOT / local_rel
        remote = RUNTIME / remote_rel
        if local.is_dir():
            remote.mkdir(parents=True, exist_ok=True)
            for src in local.rglob("*.py"):
                if "__pycache__" in src.parts:
                    continue
                dst = remote / src.relative_to(local)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        elif local.is_file():
            remote.parent.mkdir(parents=True, exist_ok=True)
            remote.write_text(local.read_text(encoding="utf-8"), encoding="utf-8")


@function(name="udsc-task1-v3-prepare-cpu", cpu=16, memory="64Gi", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=1, headless=True)
def prepare_cpu():
    _sync_v3_code()
    report = OUT / "shortlist_report.json"
    shortlist = OUT / "shortlist_evidence.jsonl"
    if report.is_file() and shortlist.is_file():
        value = json.loads(report.read_text(encoding="utf-8"))
        expected = {"candidate_refs_full_sha256":sha256(RUNTIME / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"), "evidence_py_sha256":sha256(RUNTIME / "scripts/beam/task1_v2/evidence.py"), "shortlist_evidence_sha256":sha256(shortlist)}
        if value.get("status") == "CLEAN" and all(value.get(key) == digest for key, digest in expected.items()):
            print(json.dumps({"status":"REUSED_CLEAN","report":value}, indent=2), flush=True); return
        raise RuntimeError("refusing to overwrite incompatible or non-clean V3 shortlist")
    if OUT.exists() and any(OUT.iterdir()): raise RuntimeError("refusing to overwrite non-empty V3 output")
    OUT.mkdir(parents=True, exist_ok=True)
    command = ["python","scripts/beam/task1_v3_residual/build_shortlist_features.py","--questions","data/raw/btc/LegalIR/train.json","--folds","artifacts/task1/evaluation/strict_cv_v2/folds.json","--candidates","artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl","--baseline","artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl","--payloads","data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json","--output-dir","artifacts/task1/recovery_096/v3_residual"]
    if subprocess.run(command, cwd=str(RUNTIME)).returncode: raise RuntimeError("V3 CPU shortlist materialization failed")

if __name__ == "__main__":
    print(json.dumps(preflight(), indent=2)) if "--preflight" in sys.argv else prepare_cpu.remote()
