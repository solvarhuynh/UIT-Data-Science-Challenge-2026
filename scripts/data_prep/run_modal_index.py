"""Build the production FAISS index from the configured GPU embedding model."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from pathlib import PurePosixPath

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
        "faiss-cpu>=1.8,<2",
        "modal==1.6.4",
        "numpy>=1.26,<2",
        "pydantic>=2.8",
        "pyvi>=0.1,<1",
        "pyyaml>=6",
        "sentence-transformers==5.6.0",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "tqdm>=4.66,<5",
        "transformers==5.23.0",
    )
    .env(
        {
            "PYTHONPATH": "/workspace/src",
            "UDSC2026_CONFIG_DIR": "/workspace/configs",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/workspace/src")
    .add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/workspace/configs")
    .add_local_dir(
        str(PROJECT_ROOT / "scripts" / "data_prep"),
        remote_path="/workspace/scripts/data_prep",
    )
)

app = modal.App("udsc-retrieval-index")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finalize_index_files() -> None:
    os.chdir(VOLUME_PATH)
    archive_path = Path(VOLUME_PATH) / "data/processed/chunks_aiteam.tar.gz"
    manifest_path = (
        Path(VOLUME_PATH)
        / "data/vector/faiss"
        / "legal_chunks_aiteam_vietnamese_embedding_v2_1024"
        / "manifest.json"
    )
    if not archive_path.is_file():
        raise FileNotFoundError(f"chunk archive is missing: {archive_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"completed index manifest is missing: {manifest_path}")

    index_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_count = index_manifest.get("chunk_count")
    if (
        index_manifest.get("complete") is not True
        or isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count <= 0
        or index_manifest.get("invalid_record_count") != 0
        or not index_manifest.get("corpus_hash")
    ):
        raise ValueError("FAISS manifest does not describe a complete, valid index")

    processing_manifest_path = (
        Path(VOLUME_PATH) / "data/processed/metadata/processing_manifest.json"
    )
    if processing_manifest_path.is_file():
        processing_manifest = json.loads(
            processing_manifest_path.read_text(encoding="utf-8")
        )
        expected_chunks = processing_manifest.get("counts", {}).get("chunks")
        if chunk_count != expected_chunks:
            raise ValueError(
                f"index covers {chunk_count} chunks; processed corpus has "
                f"{expected_chunks}"
            )

    index_path = manifest_path.with_name("index.faiss")
    payload_path = manifest_path.with_name("payloads.json")
    if not index_path.is_file() or not payload_path.is_file():
        raise FileNotFoundError("completed FAISS index or payload file is missing")

    index_manifest["source_archive"] = str(archive_path.relative_to(VOLUME_PATH))
    index_manifest["source_archive_sha256"] = _sha256(archive_path)
    index_manifest["index_sha256"] = _sha256(index_path)
    index_manifest["payload_sha256"] = _sha256(payload_path)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(index_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    volume.commit()


@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=60 * 60,
    volumes={VOLUME_PATH: volume},
)
def finalize_completed_index() -> None:
    """Validate and fingerprint an already completed FAISS index without a GPU."""

    _finalize_index_files()


@app.function(
    image=image,
    gpu=["A10", "L4"],
    cpu=4,
    memory=16384,
    timeout=24 * 60 * 60,
    volumes={VOLUME_PATH: volume},
)
def build_index(
    batch_size: int = 16,
    checkpoint_chunks: int = 100_000,
    resume: bool = False,
) -> None:
    if batch_size <= 0 or batch_size > 64:
        raise ValueError("batch-size must be from 1 to 64 on the selected GPU")
    if checkpoint_chunks <= 0:
        raise ValueError("checkpoint-chunks must be positive")

    os.chdir(VOLUME_PATH)
    archive_path = Path(VOLUME_PATH) / "data/processed/chunks_aiteam.tar.gz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"chunk archive is missing: {archive_path}")
    extraction_root = Path("/tmp/udsc2026-index")
    chunks_dir = extraction_root / "chunks"
    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    extraction_root.mkdir(parents=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"unsafe chunk archive entry: {member.name}")
        archive.extractall(extraction_root, filter="data")

    command = [
        sys.executable,
        "/workspace/scripts/data_prep/index_chunks.py",
        "--chunks-dir",
        str(chunks_dir),
        "--config-env",
        "gpu",
        "--vector-db-type",
        "faiss",
        "--batch-size",
        str(batch_size),
        "--checkpoint-chunks",
        str(checkpoint_chunks),
        "--commit-modal-volume",
        "--modal-volume-name",
        VOLUME_NAME,
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True, cwd=VOLUME_PATH)
    _finalize_index_files()


@app.local_entrypoint()
def main(
    batch_size: int = 16,
    checkpoint_chunks: int = 100_000,
    resume: bool = False,
    finalize_only: bool = False,
) -> None:
    if finalize_only:
        finalize_completed_index.remote()
        return
    build_index.remote(
        batch_size=batch_size,
        checkpoint_chunks=checkpoint_chunks,
        resume=resume,
    )
