"""Fail-fast checks for the rented Windows CUDA machine before long jobs."""

from __future__ import annotations

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


def _module_check(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report every environment failure
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "version": getattr(module, "__version__", "unknown")}


def main() -> int:
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
    chunks_dir = ROOT / "data" / "processed" / "chunks"
    chunk_files = list(chunks_dir.glob("*.jsonl")) if chunks_dir.is_dir() else []
    documents_dir = ROOT / "data" / "processed" / "documents"
    document_files = (
        list(documents_dir.glob("*.json")) if documents_dir.is_dir() else []
    )
    benchmark = ROOT / "data" / "processed" / "benchmarks" / "synthetic_qa.jsonl"
    checks["data"] = {
        "ok": bool(chunk_files) and bool(document_files) and benchmark.is_file(),
        "chunk_file_count": len(chunk_files),
        "document_file_count": len(document_files),
        "benchmark": str(benchmark),
        "benchmark_exists": benchmark.is_file(),
    }
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
