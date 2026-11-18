"""Download explicitly selected UDSC model checkpoints from Hugging Face."""

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
    role: str
    status: str


MODELS = (
    ModelSpec("embedding", "huyydangg/DEk21_hcmute_embedding_v2", Path("models/dek21-v2"), "embedding", "active anchor"),
    ModelSpec("bge_reranker", "BAAI/bge-reranker-v2-m3", Path("models/reranker"), "reranker", "active anchor"),
    ModelSpec("llm", "thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2", Path("models/qwen3-legal"), "generator", "current local generator"),
    ModelSpec("qwen_vl_reranker", "Qwen/Qwen3-VL-Reranker-2B", Path("models/qwen3-vl-reranker-2b"), "reranker", "optional; requires the Qwen3-VL scoring path"),
    ModelSpec("aiteam_embedding", "AITeamVN/Vietnamese_Embedding_v2", Path("models/aiteam-vietnamese-embedding-v2"), "embedding", "evaluation candidate"),
    ModelSpec("aiteam_reranker", "AITeamVN/Vietnamese_Reranker", Path("models/aiteam-vietnamese-reranker"), "reranker", "evaluation candidate"),
    ModelSpec("vnlegal_lal", "darklethelong/vnlegal-lal", Path("models/vnlegal-lal"), "embedding", "evaluation candidate; custom query prefix and last-token pooling required"),
    ModelSpec("vietlegal_harrier", "mainguyen9/vietlegal-harrier-0.6b", Path("models/vietlegal-harrier-0.6b"), "embedding", "evaluation candidate"),
    ModelSpec("vietlegal_e5", "mainguyen9/vietlegal-e5", Path("models/vietlegal-e5"), "embedding", "evaluation candidate; follow model-card input contract"),
)

IGNORE_PATTERNS = (
    "*.pt", "*.pth", "optimizer*", "scheduler*", "rng_state*",
    "training_args.bin", "onnx/**",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", choices=[model.key for model in MODELS],
        help="Model keys to download. Downloads nothing unless explicitly selected.",
    )
    parser.add_argument("--list", action="store_true", help="List models and exit without downloading.")
    parser.add_argument("--revision", default="main", help="Hub branch, tag, or commit; resolved SHA is recorded.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        for model in MODELS:
            print(f"{model.key:20} {model.role:10} {model.status}: {model.repo_id} -> {model.local_dir}")
        return 0
    if not args.only:
        raise SystemExit("Select models explicitly, for example: python scripts/download_models.py --only aiteam_embedding")
    selected = set(args.only)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("Missing huggingface-hub; install it with `python -m pip install -e '.[gpu]'`.") from exc

    api = HfApi()
    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_revision": args.revision,
        "models": {},
    }
    records = cast(dict[str, object], manifest["models"])
    for model in MODELS:
        if model.key not in selected:
            continue
        print(f"Downloading {model.key}: {model.repo_id}")
        info = api.model_info(model.repo_id, revision=args.revision)
        model.local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model.repo_id,
            revision=info.sha,
            local_dir=model.local_dir,
            ignore_patterns=list(IGNORE_PATTERNS),
        )
        records[model.key] = {
            "repo_id": model.repo_id,
            "resolved_revision": info.sha,
            "local_dir": str(model.local_dir),
            "role": model.role,
            "status": model.status,
        }
        print(f"Ready: {model.local_dir} @ {info.sha[:12]}")

    manifest_path = Path("models/download_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
