"""Run LegalQA generation directly from a verified contexts JSONL on Modal.

This entrypoint deliberately does not call candidate generation or reranking.
It is for the already aggregated production context artifact only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

import modal

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").is_dir():
    PROJECT_ROOT = Path("/workspace")
VOLUME_NAME = "udsc"
VOLUME_PATH = "/mnt/udsc"
volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.1,<2", "modal==1.6.4", "nltk>=3.8,<4",
        "numpy>=1.26,<2", "pydantic>=2.8", "pyvi>=0.1,<1",
        "pyyaml>=6", "rapidfuzz>=3.9,<4", "sentencepiece>=0.2,<1",
        "torch==2.7.1",
        "torchvision==0.22.1", "tqdm>=4.66,<5", "transformers==5.23.0",
    )
    .run_commands(
        "python -c \"import nltk; nltk.download('wordnet', quiet=True); "
        "nltk.download('omw-1.4', quiet=True)\""
    )
    .env({
        "PYTHONPATH": "/workspace/src",
        "UDSC2026_CONFIG_DIR": "/workspace/configs",
        "UDSC2026_PROMPTS_PATH": "/workspace/prompts",
        "TOKENIZERS_PARALLELISM": "false",
        "LLM_DTYPE": "bfloat16",
    })
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/workspace/src")
    .add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/workspace/configs")
    .add_local_dir(str(PROJECT_ROOT / "prompts"), remote_path="/workspace/prompts")
    .add_local_dir(str(PROJECT_ROOT / "scripts" / "answers" / "inference"), remote_path="/workspace/scripts/answers/inference")
    .add_local_dir(str(PROJECT_ROOT / "scripts" / "answers" / "submission"), remote_path="/workspace/scripts/answers/submission")
)

app = modal.App("udsc-qa-production-contexts")


def _extract_archive(archive_path: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise ValueError(f"unsafe parent archive entry: {member.name}")
        archive.extractall(destination, filter="data")
    parents = destination / "parents"
    if not parents.is_dir() or not any(parents.glob("*.jsonl")):
        raise ValueError("parent archive has no parents/*.jsonl files")
    return parents


@app.function(
    image=image,
    gpu=["A10", "L4"],
    cpu=4,
    memory=12288,
    timeout=24 * 60 * 60,
    volumes={VOLUME_PATH: volume},
)
def run_production(
    questions: str = "data/predictions/selected/legalir/production/candidate_k500_rerank_k200/questions.json",
    contexts: str = "data/predictions/selected/legalir/production/candidate_k500_rerank_k200/contexts.jsonl",
    context_manifest: str = "data/predictions/selected/legalir/production/candidate_k500_rerank_k200/contexts.manifest.json",
    output: str = "data/predictions/selected/legalqa/production_contexts/answers.json",
    diagnostics: str = "data/predictions/selected/legalqa/production_contexts/answers.diagnostics.json",
    answer_manifest: str = "data/predictions/selected/legalqa/production_contexts/answers.manifest.json",
    package: str = "data/submissions/answers/production_contexts.zip",
    save_every: int = 100,
    resume: bool = False,
) -> None:
    if not 1 <= save_every <= 500:
        raise ValueError("save_every must be between 1 and 500")
    os.chdir(VOLUME_PATH)
    required = [Path(questions), Path(contexts), Path(context_manifest), Path("models/qwen3.5-2b"), Path("models/download_manifest.json"), Path("data/processed/parents_aiteam.tar.gz")]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"required production artifact is missing: {path}")
    extraction_root = Path("/tmp/udsc2026-production-parents")
    parents_dir = _extract_archive(Path("data/processed/parents_aiteam.tar.gz"), extraction_root)
    base = [
        sys.executable, "/workspace/scripts/answers/inference/generate_answers.py",
        "--questions", questions, "--contexts", contexts,
        "--context-manifest", context_manifest, "--output", output,
        "--diagnostics", diagnostics, "--manifest", answer_manifest,
        "--config", "/workspace/configs/gpu.yaml", "--top-n", "3",
        "--save-every", str(save_every), "--parents-dir", str(parents_dir),
        "--parent-archive", "data/processed/parents_aiteam.tar.gz",
        "--parent-max-tokens", "3072", "--parent-total-tokens", "10000",
        "--commit-modal-volume", "--modal-volume-name", VOLUME_NAME,
    ]
    if resume:
        base.append("--resume")
    subprocess.run(base, check=True, cwd=VOLUME_PATH)
    package_path = Path(package)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "/workspace/scripts/answers/submission/package_submission.py",
        "--input", output, "--questions", questions, "--output", package,
    ], check=True, cwd=VOLUME_PATH)
    subprocess.run([
        sys.executable, "/workspace/scripts/answers/submission/validate_package.py",
        "--input", package, "--questions", questions,
    ], check=True, cwd=VOLUME_PATH)
    volume.commit()


@app.local_entrypoint()
def main(
    questions: str = "data/predictions/selected/legalqa/public/candidate_k500_rerank_k200/questions.json",
    contexts: str = "data/predictions/selected/legalqa/public/candidate_k500_rerank_k200/contexts.jsonl",
    context_manifest: str = "data/predictions/selected/legalqa/public/candidate_k500_rerank_k200/contexts.manifest.json",
    output: str = "data/predictions/selected/legalqa/public/candidate_k500_rerank_k200/answers.public_contexts.json",
    diagnostics: str = "data/predictions/selected/legalqa/public/candidate_k500_rerank_k200/answers.public_contexts.diagnostics.json",
    answer_manifest: str = "data/predictions/selected/legalqa/public/candidate_k500_rerank_k200/answers.public_contexts.manifest.json",
    package: str = "data/submissions/answers/submission.public_contexts.zip",
    save_every: int = 100,
    resume: bool = False,
) -> None:
    run_production.remote(
        questions=questions,
        contexts=contexts,
        context_manifest=context_manifest,
        output=output,
        diagnostics=diagnostics,
        answer_manifest=answer_manifest,
        package=package,
        save_every=save_every,
        resume=resume,
    )
