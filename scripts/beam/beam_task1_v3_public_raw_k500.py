"""Beam GPU production runner for public raw K500 dense retrieval."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
import shutil
import tempfile
from pathlib import Path

try:
    from beam import Image, Volume, function
except ModuleNotFoundError:
    if "--preflight" not in sys.argv and "--self-test" not in sys.argv:
        raise
    class Image:
        def __init__(self, **kwargs): pass
    class Volume:
        def __init__(self, **kwargs): pass
    def function(**kwargs):
        return lambda fn: fn

ROOT = Path(__file__).resolve().parents[2]
VOLUME_ROOT = Path("/workspace/p13")
RUNTIME = VOLUME_ROOT / "runtime"
ASSET_ROOT = RUNTIME
OUT = RUNTIME / "artifacts/task1/recovery_096/final_public_v3/public_raw_k500.jsonl"
MANIFEST = RUNTIME / "artifacts/task1/recovery_096/final_public_v3/public_raw_k500_manifest.json"
IMAGE = Image(python_version="python3.11", python_packages=["torch", "transformers", "sentence-transformers", "tokenizers", "safetensors", "pyyaml", "faiss-cpu", "numpy", "pydantic", "scikit-learn", "tqdm", "rank-bm25", "pyvi"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight() -> dict[str, object]:
    required = [
        ROOT / "scripts/evaluation/generate_dense_candidates.py",
        ROOT / "configs/base.yaml",
        ROOT / "configs/gpu.yaml",
        ROOT / "data/raw/btc/LegalIR/public-official.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("public K500 source missing:\n" + "\n".join(missing))
    return {"status": "READY_GPU_PUBLIC_K500", "query_count": 1000, "candidate_k": 500, "gpu": "required_for_embedding_path", "public_answers_read": False, "remote_submitted": False}


def _sync_code() -> None:
    pairs = [
        ("scripts/evaluation/generate_dense_candidates.py", "scripts/evaluation/generate_dense_candidates.py"),
        ("configs/base.yaml", "configs/base.yaml"),
        ("configs/gpu.yaml", "configs/gpu.yaml"),
        ("data/raw/btc/LegalIR/public-official.json", "data/raw/btc/LegalIR/public-official.json"),
    ]
    for local_rel, remote_rel in pairs:
        local = ROOT / local_rel
        remote = RUNTIME / remote_rel
        if not local.is_file():
            if remote.is_file():
                continue
            archives = [RUNTIME.parent / "udsc_p13_code.zip", RUNTIME.parent / "udsc_p13_data.zip"]
            extracted = False
            for archive in archives:
                if not archive.is_file():
                    continue
                with zipfile.ZipFile(archive) as handle:
                    matches = [name for name in handle.namelist() if name.rstrip("/").endswith(remote_rel)]
                    if not matches:
                        continue
                    name = sorted(matches, key=len)[0]
                    remote.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(name) as source, remote.open("wb") as target:
                        while block := source.read(8 * 1024 * 1024):
                            target.write(block)
                    extracted = True
                    break
            if not extracted:
                raise FileNotFoundError(
                    f"public K500 source is absent locally, on the Beam volume, and in the bundled archives: {local_rel}"
                )
            continue
        remote.parent.mkdir(parents=True, exist_ok=True)
        remote.write_text(local.read_text(encoding="utf-8"), encoding="utf-8")

    local_src = ROOT / "src"
    remote_src = RUNTIME / "src"
    if not local_src.is_dir():
        if not remote_src.is_dir():
            raise FileNotFoundError(
                "public K500 source is absent locally and on the Beam volume: src"
            )
    else:
        remote_src.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_src, remote_src, dirs_exist_ok=True)


MODEL_REL = Path("models/dek21-v2")
VECTOR_REL = Path("data/vector_store/faiss/legal_chunks_dek21_v2_768")


def _retrieval_environment() -> dict[str, str]:
    """Keep native BLAS thread pools bounded in the Beam subprocess."""

    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def _required_asset_files(runtime_root: Path) -> tuple[Path, Path]:
    return (
        runtime_root / MODEL_REL / "model.safetensors",
        runtime_root / VECTOR_REL / "index.faiss",
    )


def _archive_candidates(archive_root: Path, name: str) -> list[Path]:
    candidates = [archive_root / name, archive_root / "runtime" / name]
    return list(dict.fromkeys(candidates))


def _extract_folder_if_needed(
    archive: Path,
    runtime_root: Path,
    source_prefixes: tuple[str, ...],
    destination_rel: Path,
) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"required asset archive missing: {archive}")
    with zipfile.ZipFile(archive) as handle:
        names = [name for name in handle.namelist() if not name.endswith("/")]
        prefix = next((candidate for candidate in source_prefixes if any(name.startswith(candidate) for name in names)), None)
        if prefix is None:
            raise FileNotFoundError(
                f"required folder missing from {archive}: {source_prefixes}"
            )
        for name in names:
            if not name.startswith(prefix):
                continue
            relative = Path(name[len(prefix):])
            if not relative.parts:
                continue
            destination = runtime_root / destination_rel / relative
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(name) as source, destination.open("wb") as target:
                while block := source.read(8 * 1024 * 1024):
                    target.write(block)


def _extract_required_assets(
    runtime_root: Path = ASSET_ROOT,
    archive_root: Path | None = None,
) -> dict[str, object]:
    archive_root = archive_root or runtime_root.parent
    required = _required_asset_files(runtime_root)
    if all(path.is_file() for path in required):
        return {"status": "EXISTING_ASSETS_PASS", "extracted": False}

    models_archive = None
    if not required[0].is_file():
        models_archive = next(
            (path for path in _archive_candidates(archive_root, "models.zip") if path.is_file()),
            None,
        )
    vector_archive = None
    if not required[1].is_file():
        vector_archive = next(
            (path for path in _archive_candidates(archive_root, "vector_store.zip") if path.is_file()),
            None,
        )
    if (
        (models_archive is None and not required[0].is_file())
        or (vector_archive is None and not required[1].is_file())
    ):
        missing = []
        if models_archive is None and not required[0].is_file():
            missing.append("models.zip")
        if vector_archive is None and not required[1].is_file():
            missing.append("vector_store.zip")
        raise FileNotFoundError(
            "required public K500 assets missing; archive(s) unavailable: "
            + ", ".join(missing)
        )

    if models_archive is not None:
        _extract_folder_if_needed(
            models_archive,
            runtime_root,
            ("models/dek21-v2/", "dek21-v2/"),
            MODEL_REL,
        )
    if vector_archive is not None:
        _extract_folder_if_needed(
            vector_archive,
            runtime_root,
            (
                "data/vector_store/faiss/legal_chunks_dek21_v2_768/",
                "vector_store/faiss/legal_chunks_dek21_v2_768/",
            ),
            VECTOR_REL,
        )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(
            "zip extraction did not produce required public K500 assets: "
            + ", ".join(str(path) for path in required if not path.is_file())
        )
    return {"status": "ZIP_EXTRACTION_PASS", "extracted": True}


def self_test() -> dict[str, bool]:
    with tempfile.TemporaryDirectory(prefix="public_k500_assets_") as temp_dir:
        root = Path(temp_dir)
        runtime = root / "runtime"
        missing_archive_failed_closed = False
        try:
            _extract_required_assets(runtime, root)
        except FileNotFoundError:
            missing_archive_failed_closed = True

        existing_runtime = root / "existing_runtime"
        existing_model = existing_runtime / MODEL_REL / "model.safetensors"
        existing_index = existing_runtime / VECTOR_REL / "index.faiss"
        existing_model.parent.mkdir(parents=True)
        existing_index.parent.mkdir(parents=True)
        existing_model.write_bytes(b"existing-model")
        existing_index.write_bytes(b"existing-index")
        existing_asset_passes = _extract_required_assets(existing_runtime, root)["status"] == "EXISTING_ASSETS_PASS"

        zip_root = root / "zip_runtime"
        models_zip = root / "models.zip"
        vector_zip = root / "vector_store.zip"
        with zipfile.ZipFile(models_zip, "w") as archive:
            archive.writestr("models/dek21-v2/model.safetensors", b"zip-model")
        with zipfile.ZipFile(vector_zip, "w") as archive:
            archive.writestr("vector_store/faiss/legal_chunks_dek21_v2_768/index.faiss", b"zip-index")
        _extract_required_assets(zip_root, root)
        zip_extraction_path_correct = all(path.is_file() for path in _required_asset_files(zip_root))

    return {
        "missing_archive_fails_closed": missing_archive_failed_closed,
        "existing_asset_passes": existing_asset_passes,
        "zip_extraction_path_correct": zip_extraction_path_correct,
    }


@function(name="udsc-task1-v3-public-raw-k500", cpu=8, memory="32Gi", gpu="RTX4090", image=IMAGE, volumes=[Volume(name="udsc-p13", mount_path=str(VOLUME_ROOT))], timeout=-1, retries=0, headless=True)
def run_public_raw_k500() -> None:
    _sync_code()
    _extract_required_assets()
    if OUT.exists() and MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if int(manifest.get("question_count", -1)) == 1000 and int(manifest.get("candidate_k", -1)) == 500 and manifest.get("complete") is True and sha256(OUT) == manifest.get("output_sha256"):
            print(json.dumps({"status": "REUSED_CLEAN", "output": str(OUT)}, indent=2), flush=True)
            return
        raise RuntimeError("refusing to overwrite incomplete or incompatible public K500 output")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    command = ["python", "scripts/evaluation/generate_dense_candidates.py", "--public-official", "data/raw/btc/LegalIR/public-official.json", "--config-env", "gpu", "--candidate-k", "500", "--output", "artifacts/task1/recovery_096/final_public_v3/public_raw_k500.jsonl", "--manifest", "artifacts/task1/recovery_096/final_public_v3/public_raw_k500_manifest.json"]
    if subprocess.run(
        command,
        cwd=str(RUNTIME),
        env=_retrieval_environment(),
    ).returncode:
        raise RuntimeError("public raw K500 retrieval failed")
    if not OUT.is_file() or not MANIFEST.is_file():
        raise RuntimeError("public raw K500 output missing")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["output_sha256"] = sha256(OUT)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if "--preflight" in sys.argv:
        print(json.dumps(preflight(), indent=2))
    elif "--self-test" in sys.argv:
        checks = {name: bool(value) for name, value in self_test().items()}
        if not all(checks.values()):
            raise AssertionError(checks)
        print(json.dumps({"status": "SELF_TEST_PASS", **checks}, indent=2))
    else:
        run_public_raw_k500.remote()
