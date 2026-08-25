"""Beam GPU launcher for V3 frozen feature scoring only; no fine-tuning."""
from __future__ import annotations

import json
import hashlib
import shutil
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
# Keep direct local ``python scripts/beam/...py --preflight`` invocation able
# to import the repository's namespace package; Beam remote execution already
# starts with the repository root on sys.path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VOLUME_ROOT = Path("/workspace/p13"); RUNTIME = VOLUME_ROOT / "runtime"
OUT = RUNTIME / "artifacts/task1/recovery_096/v3_residual"
PUBLIC_OUT = RUNTIME / "artifacts/task1/recovery_096/final_public_v3"
LOCAL_OUT = ROOT / "artifacts/task1/recovery_096/v3_residual"
LOCAL_PUBLIC_OUT = ROOT / "artifacts/task1/recovery_096/final_public_v3"
LOCAL_V3 = ROOT / "scripts/beam/task1_v3_residual"
REMOTE_V3 = RUNTIME / "scripts/beam/task1_v3_residual"
BATCH_SIZE = 32
TRAIN_GPU = "RTX4090"
PUBLIC_GPU = "A10G"
IMAGE = Image(
    python_version="python3.11",
    # Pin Torch so Beam does not pull a newer CUDA wheel than the node driver.
    python_packages=["torch==2.2.2", "transformers==5.0.0", "tokenizers", "safetensors", "pyvi"],
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def _model_files(model_root: Path) -> list[Path]:
    return [
        model_root / "config.json",
        model_root / "model.safetensors",
        model_root / "tokenizer.json",
    ]


def _public_shortlist_contract(shortlist_path: Path, report_path: Path) -> dict:
    """Validate public shortlist without loading labels or train artifacts."""
    missing = [str(path) for path in (shortlist_path, report_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "public frozen scoring requires public shortlist/evidence artifacts:\n"
            + "\n".join(missing)
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "CLEAN" or int(report.get("query_count", -1)) != 1000:
        raise RuntimeError("public shortlist report must be CLEAN with query_count=1000")
    if report.get("shortlist_evidence_sha256") != sha256(shortlist_path):
        raise RuntimeError("public shortlist hash mismatch")
    rows = []
    with shortlist_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if any(key in row for key in ("fold", "answer", "gold", "label")):
                raise RuntimeError("public shortlist contains train/label fields")
            rows.append(row)
    if len(rows) != 1000:
        raise RuntimeError(f"public shortlist must contain 1000 rows, got {len(rows)}")
    return {"status": "PASS", "query_count": len(rows), "shortlist": str(shortlist_path), "report": str(report_path)}


def _resolve_public_shortlist(base: Path) -> tuple[Path, str, str, dict]:
    """Resolve a public-safe shortlist, preferring the recovery path requested for hotfixes."""
    candidates = (
        (base / "artifacts/task1/recovery_096/v3_residual", "shortlist_evidence.jsonl", "shortlist_report.json"),
        (base / "artifacts/task1/recovery_096/final_public_v3", "public_shortlist_evidence.jsonl", "public_shortlist_report.json"),
    )
    errors: list[str] = []
    for root, shortlist_name, report_name in candidates:
        shortlist_path = root / shortlist_name
        report_path = root / report_name
        if not shortlist_path.is_file() and not report_path.is_file():
            continue
        try:
            contract = _public_shortlist_contract(shortlist_path, report_path)
            return root, shortlist_name, report_name, contract
        except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{shortlist_path}: {exc}")
    message = "public frozen scoring requires a public-safe 1000-query shortlist; checked:\n" + "\n".join(
        f"- {root / shortlist_name}\n- {root / report_name}" for root, shortlist_name, report_name in candidates
    )
    if errors:
        message += "\nValidation errors:\n" + "\n".join(errors)
    raise FileNotFoundError(message)


def preflight(mode: str = "train") -> dict:
    required_source_files = [
        LOCAL_V3 / "score_frozen_features.py",
        LOCAL_V3 / "common.py",
    ]
    missing = [str(path) for path in required_source_files if not path.is_file()]
    model_root = ROOT / "models/reranker"
    missing_model = [str(path) for path in _model_files(model_root) if not path.is_file()]
    if missing or missing_model:
        details = missing + missing_model
        raise FileNotFoundError("V3 frozen scorer/model config missing:\n" + "\n".join(details))
    if mode == "public":
        shortlist_root, shortlist_name, _, contract = _resolve_public_shortlist(ROOT)
        return {
            "status": "PREFLIGHT_PASS",
            "mode": "public",
            "selected_gpu": PUBLIC_GPU,
            "batch_size": BATCH_SIZE,
            "model_path": str(model_root),
            "input_shortlist_path": str(shortlist_root / shortlist_name),
            "output_frozen_path": str(LOCAL_PUBLIC_OUT / "public_frozen_features.jsonl"),
            "output_report_path": str(LOCAL_PUBLIC_OUT / "public_frozen_features_report.json"),
            "frozen_inference_only": True,
            "public_contract": contract,
            "no_public_labels_used": True,
            "remote_submitted": False,
        }
    from scripts.beam.task1_v3_residual.score_frozen_features import parity_smoke
    parity = parity_smoke(
        model_root,
        LOCAL_OUT / "frozen_features_report.json",
    )
    return {"status":"PREFLIGHT_PASS" if parity.get("status") == "PASS" else "PREFLIGHT_BLOCKED", "mode": mode, "gpu":"RTX4090", "default_batch_size":32, "frozen_inference_only":True, "required_source_files":[str(path) for path in required_source_files], "model":"models/reranker", "train_parity_smoke": parity, "remote_submitted":False}


def _sync_code() -> None:
    """Sync the scorer and its local helper into the mounted runtime."""

    for name in ("score_frozen_features.py", "common.py"):
        local = LOCAL_V3 / name
        remote = REMOTE_V3 / name
        remote.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, remote)

def _run(mode: str, shortlist_root: Path, output_root: Path, shortlist_name: str, shortlist_report_name: str, feature_name: str, report_name: str) -> None:
    _sync_code()
    shortlist_path = shortlist_root / shortlist_name
    shortlist_report_path = shortlist_root / shortlist_report_name
    required = [shortlist_path, shortlist_report_path, *_model_files(RUNTIME / "models/reranker"), RUNTIME / "scripts/beam/task1_v3_residual/score_frozen_features.py", RUNTIME / "scripts/beam/task1_v3_residual/common.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError("V3 frozen feature input missing:\n" + "\n".join(missing))
    shortlist_report = json.loads(shortlist_report_path.read_text(encoding="utf-8"))
    expected_count = 7000 if mode == "train" else 1000
    if shortlist_report.get("status") != "CLEAN" or int(shortlist_report.get("query_count", -1)) != expected_count: raise RuntimeError(f"V3 {mode} shortlist is not clean")
    hash_key = "shortlist_evidence_sha256"
    if shortlist_report.get(hash_key) != sha256(shortlist_path): raise RuntimeError("V3 shortlist hash mismatch")
    if mode == "public":
        _public_shortlist_contract(shortlist_path, shortlist_report_path)
    output_path = output_root / feature_name
    report_path = output_root / report_name
    if output_path.exists() or report_path.exists(): raise RuntimeError("refusing to overwrite existing V3 frozen features")
    shortlist_rel = shortlist_path.relative_to(RUNTIME).as_posix()
    output_rel = output_path.relative_to(RUNTIME).as_posix()
    report_rel = report_path.relative_to(RUNTIME).as_posix()
    command = ["python", "scripts/beam/task1_v3_residual/score_frozen_features.py", "--mode", mode, "--shortlist", shortlist_rel, "--model", "models/reranker", "--output", output_rel, "--report", report_rel, "--batch-size", str(BATCH_SIZE)]
    model_file = RUNTIME / "models/reranker/model.safetensors"; model_sha = sha256(model_file)
    if subprocess.run(command, cwd=str(RUNTIME)).returncode: raise RuntimeError("V3 frozen feature scoring failed")
    if sha256(model_file) != model_sha: raise RuntimeError("frozen base model was modified")


@function(name="udsc-task1-v3-frozen-features", cpu=8, memory="32Gi", gpu=TRAIN_GPU, image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=1, headless=True)
def run():
    _run("train", OUT, OUT, "shortlist_evidence.jsonl", "shortlist_report.json", "frozen_features.jsonl", "frozen_features_report.json")


@function(name="udsc-task1-v3-public-frozen-features", cpu=8, memory="32Gi", gpu=PUBLIC_GPU, image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=0, headless=True)
def run_public():
    shortlist_root, shortlist_name, shortlist_report_name, _ = _resolve_public_shortlist(RUNTIME)
    _run("public", shortlist_root, PUBLIC_OUT, shortlist_name, shortlist_report_name, "public_frozen_features.jsonl", "public_frozen_features_report.json")

if __name__ == "__main__":
    public = "--public" in sys.argv
    print(json.dumps(preflight("public" if public else "train"), indent=2)) if "--preflight" in sys.argv else (run_public.remote() if public else run.remote())
