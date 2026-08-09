"""Fail-fast checks for the rented Windows CUDA machine before long jobs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_MODELS = {
    "embedding": ROOT / "models" / "dek21-v2",
    "reranker": ROOT / "models" / "reranker",
    "llm": ROOT / "models" / "qwen3-legal",
}
_PROCESSED_TREE_LAYOUT = (
    ("documents", "*.json"),
    ("parents", "*.jsonl"),
    ("chunks", "*.jsonl"),
)


def _module_check(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report every environment failure
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "version": getattr(module, "__version__", "unknown")}


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object: {0}".format(path))
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _processed_corpus_tree_hash(processed_root: Path) -> str:
    """Recompute the immutable corpus hash using the disk-audit file order."""

    digest = hashlib.sha256()
    for directory_name, pattern in _PROCESSED_TREE_LAYOUT:
        directory = processed_root / directory_name
        for path in sorted(directory.glob(pattern)):
            digest.update(path.relative_to(processed_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def _data_check(processed_root: Path) -> dict[str, Any]:
    chunks_dir = processed_root / "chunks"
    parents_dir = processed_root / "parents"
    documents_dir = processed_root / "documents"
    benchmark = processed_root / "benchmarks" / "synthetic_qa.jsonl"
    processing_manifest_path = processed_root / "metadata" / "processing_manifest.json"
    disk_audit_path = processed_root / "metadata" / "disk_audit_report.json"
    chunk_files = list(chunks_dir.glob("*.jsonl")) if chunks_dir.is_dir() else []
    parent_files = list(parents_dir.glob("*.jsonl")) if parents_dir.is_dir() else []
    document_files = (
        list(documents_dir.glob("*.json")) if documents_dir.is_dir() else []
    )
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    if not processing_manifest_path.is_file():
        errors.append("missing_processing_manifest")
    else:
        try:
            manifest = _read_json_object(processing_manifest_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append("invalid_processing_manifest:{0}".format(exc))
    if not disk_audit_path.is_file():
        errors.append("missing_disk_audit")
    else:
        try:
            audit = _read_json_object(disk_audit_path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            errors.append("invalid_disk_audit:{0}".format(exc))
    if not benchmark.is_file():
        errors.append("missing_benchmark")

    raw_counts = manifest.get("counts")
    counts: dict[str, Any] = raw_counts if isinstance(raw_counts, dict) else {}
    expected_documents = counts.get("documents")
    expected_chunks = counts.get("chunks")
    expected_parents = counts.get("parents")
    if manifest and manifest.get("integrity_gate_passed") is not True:
        errors.append("processing_manifest_integrity_gate_failed")
    if audit and audit.get("integrity_gate_passed") is not True:
        errors.append("disk_audit_integrity_gate_failed")
    if audit and audit.get("integrity_failures") != []:
        errors.append("disk_audit_has_integrity_failures")
    if expected_documents != len(document_files):
        errors.append("document_file_count_mismatch")
    if audit and audit.get("document_count") != expected_documents:
        errors.append("audited_document_count_mismatch")
    if len(chunk_files) != len(document_files):
        errors.append("chunk_file_count_mismatch")
    if len(parent_files) != len(document_files):
        errors.append("parent_file_count_mismatch")
    if audit and audit.get("chunk_count") != expected_chunks:
        errors.append("audited_chunk_count_mismatch")
    if audit and audit.get("parent_count") != expected_parents:
        errors.append("audited_parent_count_mismatch")

    manifest_tree_hash = manifest.get("processed_corpus_tree_hash")
    audit_tree_hash = audit.get("corpus_tree_hash")
    if not isinstance(manifest_tree_hash, str) or not manifest_tree_hash:
        errors.append("missing_manifest_tree_hash")
    if not isinstance(audit_tree_hash, str) or not audit_tree_hash:
        errors.append("missing_audit_tree_hash")
    if (
        isinstance(manifest_tree_hash, str)
        and isinstance(audit_tree_hash, str)
        and manifest_tree_hash != audit_tree_hash
    ):
        errors.append("manifest_audit_tree_hash_mismatch")
    actual_tree_hash = None
    try:
        actual_tree_hash = _processed_corpus_tree_hash(processed_root)
    except OSError as exc:
        errors.append("processed_corpus_tree_hash_error:{0}".format(exc))
    expected_tree_hash = (
        manifest_tree_hash
        if isinstance(manifest_tree_hash, str) and manifest_tree_hash
        else audit_tree_hash
    )
    if expected_tree_hash and actual_tree_hash != expected_tree_hash:
        errors.append("processed_corpus_tree_hash_mismatch")

    benchmark_manifest = manifest.get("synthetic_benchmark")
    benchmark_sha256 = None
    benchmark_record_count = 0
    if benchmark.is_file():
        benchmark_sha256 = _sha256(benchmark)
        with benchmark.open("r", encoding="utf-8") as stream:
            benchmark_record_count = sum(bool(line.strip()) for line in stream)
    if not isinstance(benchmark_manifest, dict):
        errors.append("missing_benchmark_manifest")
    else:
        if benchmark_sha256 != benchmark_manifest.get("benchmark_sha256"):
            errors.append("benchmark_sha256_mismatch")
        if benchmark_record_count != benchmark_manifest.get("record_count"):
            errors.append("benchmark_record_count_mismatch")
        if benchmark_manifest.get("source_chunk_count") != expected_chunks:
            errors.append("benchmark_source_chunk_count_mismatch")

    return {
        "ok": not errors,
        "processed_root": str(processed_root),
        "chunk_file_count": len(chunk_files),
        "parent_file_count": len(parent_files),
        "document_file_count": len(document_files),
        "audited_chunk_count": audit.get("chunk_count"),
        "audited_parent_count": audit.get("parent_count"),
        "processed_corpus_tree_hash": actual_tree_hash,
        "expected_processed_corpus_tree_hash": expected_tree_hash,
        "benchmark": str(benchmark),
        "benchmark_exists": benchmark.is_file(),
        "benchmark_record_count": benchmark_record_count,
        "benchmark_sha256": benchmark_sha256,
        "processing_manifest_exists": processing_manifest_path.is_file(),
        "disk_audit_exists": disk_audit_path.is_file(),
        "integrity_gate_passed": audit.get("integrity_gate_passed"),
        "semantic_completeness_gate_passed": audit.get(
            "semantic_completeness_gate_passed"
        ),
        "semantic_issues": audit.get("semantic_issues", []),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed_v3")
    )
    args = parser.parse_args(argv)
    processed_root = (
        args.processed_root
        if args.processed_root.is_absolute()
        else ROOT / args.processed_root
    )
    checks: dict[str, Any] = {
        "python": {
            "ok": (3, 10) <= sys.version_info[:2] < (3, 13),
            "version": sys.version.split()[0],
        },
        "packages": {
            name: _module_check(name)
            for name in (
                "faiss",
                "huggingface_hub",
                "numpy",
                "pyvi",
                "sentence_transformers",
                "torch",
                "transformers",
            )
        },
    }
    free_bytes = shutil.disk_usage(ROOT).free
    checks["disk"] = {
        "ok": free_bytes >= 40 * 1024**3,
        "free_gib": round(free_bytes / 1024**3, 2),
    }
    checks["data"] = _data_check(processed_root)
    checks["models"] = {
        key: {
            "ok": path.is_dir() and (path / "config.json").is_file(),
            "path": str(path),
        }
        for key, path in REQUIRED_MODELS.items()
    }

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        checks["cuda"] = {
            "ok": cuda_ok,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if cuda_ok else None,
            "capability": (
                list(torch.cuda.get_device_capability(0)) if cuda_ok else None
            ),
            "vram_gib": (
                round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
                if cuda_ok
                else None
            ),
        }
    except Exception as exc:  # noqa: BLE001
        checks["cuda"] = {"ok": False, "error": str(exc)}

    failures: list[str] = []
    for key in ("python", "disk", "data", "cuda"):
        if not checks[key]["ok"]:
            failures.append(key)
    failures.extend(
        f"package:{name}"
        for name, result in checks["packages"].items()
        if not result["ok"]
    )
    failures.extend(
        f"model:{name}" for name, result in checks["models"].items() if not result["ok"]
    )
    checks["failures"] = failures
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
