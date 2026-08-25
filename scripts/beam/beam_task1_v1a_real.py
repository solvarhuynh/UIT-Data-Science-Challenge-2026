"""Beam launcher for the fold0 v1a inference-matched reranker training run.

Mirrors the pattern used by ``beam_bm25_grounded_bge_b4_v1.py``: a single
``@function``-decorated entrypoint that Beam creates, retries, and tears down
on its own. There is no manual Sandbox lifecycle management and no local
state file to fall out of sync with the remote side.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from beam import Image, Volume, function

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/dataset"
TRAIN = DATASET_DIR / "fold0_train_v1a.jsonl"
EVAL_MANIFEST = DATASET_DIR / "fold0_eval_candidate_manifest.jsonl"
TRAIN_SCRIPT = ROOT / "scripts/beam/beam_inference_matched_reranker_v1_fold0.py"
MATERIALIZE_SCRIPT = ROOT / "scripts/beam/materialize_fold0_inference_chunks.py"
SCORE_SCRIPT = ROOT / "scripts/beam/score_fold0_v1a.py"
EVAL_SCRIPT = ROOT / "scripts/beam/evaluate_inference_matched_reranker_v1_fold0.py"
TRAIN_SHA256 = "29a2d23d70da3565131afa823435f0a3715a05e8e8f4fb77fb0ee18dbc1acdc0"

VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"

image = Image(
    python_version="python3.11",
    python_packages=[
        "torch",
        "transformers==5.0.0",
        "tokenizers",
        "safetensors",
        "accelerate>=1.1,<2",
        "numpy",
        "tqdm",
        "pyvi",
    ],
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight() -> dict:
    """Local, no-GPU checks. Confirms everything is in place before you burn
    GPU minutes submitting the remote job."""
    required = [TRAIN_SCRIPT, MATERIALIZE_SCRIPT, SCORE_SCRIPT, EVAL_SCRIPT, TRAIN, EVAL_MANIFEST]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required file missing: " + ", ".join(missing))
    for path in (TRAIN_SCRIPT, MATERIALIZE_SCRIPT, SCORE_SCRIPT, EVAL_SCRIPT):
        subprocess.run(["python", "-m", "py_compile", str(path)], check=True)
    actual_sha = sha256(TRAIN)
    if actual_sha != TRAIN_SHA256:
        raise ValueError(f"TRAIN_SHA_MISMATCH expected={TRAIN_SHA256} actual={actual_sha}")
    return {"status": "PREFLIGHT_PASS", "train_sha256": actual_sha}


@function(
    name="udsc-task1-v1a-fold0",
    cpu=8,
    memory="32Gi",
    gpu="RTX5090",
    image=image,
    volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))],
    timeout=-1,
    retries=1,
    headless=True,
)
def run() -> None:
    print("[ENTRY] task1_v1a_fold0 run() reached", flush=True)
    cmd = [
        "python",
        "scripts/beam/beam_inference_matched_reranker_v1_fold0.py",
        "--train",
        "artifacts/task1/recovery_096/inference_matched_reranker_v1/dataset/fold0_train_v1a.jsonl",
        "--base-model",
        "models/reranker",
        "--output-dir",
        "artifacts/task1/recovery_096/models/inference_matched_reranker_v1_fold0",
        "--max-length",
        "512",
        "--learning-rate",
        "5e-7",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--allow-training",
    ]
    result = subprocess.run(cmd, cwd=str(RUNTIME))
    print(f"[EXIT] training process exited with code {result.returncode}", flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"training process failed with exit code {result.returncode}")


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        print(json.dumps(preflight(), ensure_ascii=False))
    else:
        run.remote()