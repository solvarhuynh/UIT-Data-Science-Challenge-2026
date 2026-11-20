"""Download the three models selected for the production pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo_id: str
    destination: Path
    role: str
    status: str


# Only the selected production models are offered.
MODELS = (
    ModelSpec("embedding", "AITeamVN/Vietnamese_Embedding_v2", Path("models/aiteam-vietnamese-embedding-v2"), "embedding", "selected retrieval embedding"),
    ModelSpec("reranker", "AITeamVN/Vietnamese_Reranker", Path("models/aiteam-vietnamese-reranker"), "reranker", "selected retrieval reranker"),
    ModelSpec("answer", "Qwen/Qwen3.5-2B", Path("models/qwen3.5-2b"), "generator", "selected LegalQA generator"),
)

PROFILES = {
    "retrieval": ("embedding", "reranker"),
    "answers": ("answer",),
    "all": ("embedding", "reranker", "answer"),
}

# Pin the answer challenger so later downloads resolve to the same checkpoint.
PINNED_REVISIONS = {
    "answer": "15852e8c16360a2fea060d615a32b45270f8a8fc",
}

IGNORE_PATTERNS = ("*.pt", "*.pth", "optimizer*", "scheduler*", "rng_state*", "training_args.bin", "onnx/**")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the selected model set.")
    parser.add_argument("--profile", choices=PROFILES, help="Download one named comparison arm.")
    parser.add_argument("--only", nargs="+", choices=[model.key for model in MODELS], help="Download selected model keys.")
    parser.add_argument("--revision", help="Override Hub revision for selected models.")
    return parser


def _contains_weights(directory: Path) -> bool:
    return any(directory.glob("*.safetensors")) or any(directory.glob("*.bin"))


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        for model in MODELS:
            print(f"{model.key:18} {model.role:10} {model.status}: {model.repo_id} -> {model.destination}")
        print("Profiles:")
        for name, keys in PROFILES.items():
            print(f"  {name}: {', '.join(keys)}")
        return 0
    if args.profile and args.only:
        raise SystemExit("Choose either --profile or --only, not both.")
    selected = set(args.only or PROFILES.get(args.profile, ()))
    if not selected:
        raise SystemExit("Choose a model profile, for example: python scripts/shared/download_models.py --profile retrieval")

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("Missing huggingface-hub; install with `python -m pip install -e '.[models]'`.") from exc

    api = HfApi()
    manifest_path = Path("models/download_manifest.json")
    previous_records: dict[str, object] = {}
    if manifest_path.is_file():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_models = previous_manifest.get("models", {})
        if isinstance(previous_models, dict):
            previous_records.update(previous_models)
    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": previous_records,
    }
    records = cast(dict[str, object], manifest["models"])
    cache_root = Path("models/.download-cache")
    for model in MODELS:
        if model.key not in selected:
            continue
        requested_revision = args.revision or PINNED_REVISIONS.get(model.key, "main")
        print(f"Downloading {model.key}: {model.repo_id} @ {requested_revision}")
        info = api.model_info(model.repo_id, revision=requested_revision)
        previous = records.get(model.key)
        if (
            isinstance(previous, dict)
            and previous.get("resolved_revision") == info.sha
            and model.destination.is_dir()
            and _contains_weights(model.destination)
        ):
            print(f"Already ready: {model.destination} @ {info.sha}")
            continue
        if model.destination.exists() and any(model.destination.iterdir()):
            raise FileExistsError(
                f"Model destination is occupied by another revision: {model.destination}"
            )

        snapshot = Path(
            snapshot_download(
                repo_id=model.repo_id,
                revision=info.sha,
                cache_dir=str(cache_root),
                ignore_patterns=list(IGNORE_PATTERNS),
            )
        )
        staging = model.destination.with_name(f".{model.destination.name}.staging")
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(snapshot, staging)
        if not _contains_weights(staging):
            shutil.rmtree(staging)
            raise RuntimeError(f"Downloaded checkpoint has no weight files: {model.repo_id}")
        model.destination.parent.mkdir(parents=True, exist_ok=True)
        if model.destination.exists():
            model.destination.rmdir()
        staging.rename(model.destination)
        records[model.key] = {
            "repo_id": model.repo_id,
            "requested_revision": requested_revision,
            "resolved_revision": info.sha,
            "destination": str(model.destination),
            "role": model.role,
            "status": model.status,
        }
        print(f"Ready: {model.destination} @ {info.sha}")
        shutil.rmtree(cache_root, ignore_errors=True)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
