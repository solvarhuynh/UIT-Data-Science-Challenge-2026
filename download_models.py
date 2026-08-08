"""Download the three approved UDSC2026 models into reproducible local paths."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    local_dir: Path


MODELS = (
    ModelSpec(
        "embedding",
        "huyydangg/DEk21_hcmute_embedding_v2",
        Path("models/dek21-v2"),
    ),
    ModelSpec(
        "reranker",
        "BAAI/bge-reranker-v2-m3",
        Path("models/reranker"),
    ),
    ModelSpec(
        "llm",
        "thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2",
        Path("models/qwen3-legal"),
    ),
)

# Training checkpoints are not needed for inference and can more than double
# download/storage usage on some community model repositories.
IGNORE_PATTERNS = (
    "*.pt",
    "*.pth",
    "optimizer*",
    "scheduler*",
    "rng_state*",
    "training_args.bin",
    "onnx/**",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[spec.key for spec in MODELS],
        help="Download only selected model roles (default: all three).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub branch/tag/commit to resolve; the resolved SHA is recorded.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface-hub; run `python -m pip install -e .[gpu]` first."
        ) from exc
    selected = set(args.only or [spec.key for spec in MODELS])
    api = HfApi()
    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_revision": args.revision,
        "models": {},
    }
    model_records = cast(dict[str, object], manifest["models"])

    for spec in MODELS:
        if spec.key not in selected:
            continue
        print(f"Downloading {spec.key}: {spec.repo_id}")
        spec.local_dir.mkdir(parents=True, exist_ok=True)
        info = api.model_info(spec.repo_id, revision=args.revision)
        snapshot_download(
            repo_id=spec.repo_id,
            revision=info.sha,
            local_dir=spec.local_dir,
            ignore_patterns=list(IGNORE_PATTERNS),
        )
        model_records[spec.key] = {
            "repo_id": spec.repo_id,
            "resolved_revision": info.sha,
            "local_dir": str(spec.local_dir),
        }
        print(f"Ready: {spec.local_dir} @ {info.sha[:12]}")

    manifest_path = Path("models/download_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
