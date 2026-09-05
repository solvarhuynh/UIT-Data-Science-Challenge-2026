"""Package the small P15 overlay that sits on top of the verified P14 input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_P14_RUN = (
    PROJECT_ROOT
    / "artifacts/task2/modal_download/task2_p14_modal_result/results/runs"
    / "16d81ab92bd57c46050dc876"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        default=DEFAULT_P14_RUN / "qwen_lora",
    )
    parser.add_argument(
        "--adaptation-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task2/training/p15_adaptation",
    )
    parser.add_argument(
        "--nltk-data",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task2/nltk_data",
    )
    parser.add_argument(
        "--selector",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts/task2/evaluation/p15_answer_selector_meteor_fast"
            / "selector.joblib"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task2/task2_p15_modal_input.zip",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _members(args: argparse.Namespace) -> list[tuple[Path, str]]:
    members: list[tuple[Path, str]] = []
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_manifest.json",
    ):
        members.append((args.initial_adapter / name, f"initial_adapter/{name}"))
    adaptation_files = {
        "adaptation_train.jsonl": "training/adaptation_train.jsonl",
        "adaptation_all.jsonl": "training/adaptation_all.jsonl",
        "adaptation_dev_ids.json": "training/adaptation_dev_ids.json",
        "baseline_dev_predictions.json": "training/baseline_dev_predictions.json",
        "adaptation_manifest.json": "training/adaptation_manifest.json",
        "dev_rankings.jsonl": "rankings/dev_rankings.jsonl",
        "public_rankings.jsonl": "rankings/public_rankings.jsonl",
        "dev_extractive_predictions.json": "predictions/dev_extractive.json",
        "public_extractive_predictions.json": "predictions/public_extractive.json",
    }
    members.extend(
        (args.adaptation_dir / local, archive)
        for local, archive in adaptation_files.items()
    )
    for name in ("wordnet.zip", "omw-1.4.zip"):
        members.append(
            (args.nltk_data / "corpora" / name, f"nltk_data/corpora/{name}")
        )
    if args.selector is not None:
        members.append((args.selector, "selector/selector.joblib"))
    return members


def package(args: argparse.Namespace) -> tuple[Path, Path]:
    members = _members(args)
    missing = [str(path) for path, _ in members if not path.is_file()]
    if missing:
        raise FileNotFoundError("P15 bundle inputs are missing: " + ", ".join(missing))
    archive_names = [name for _, name in members]
    if len(archive_names) != len(set(archive_names)):
        raise ValueError("P15 bundle contains duplicate archive member names")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as bundle:
            for source, archive_name in members:
                bundle.write(source, archive_name)
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise ValueError("P15 bundle failed its integrity check")
            if sorted(bundle.namelist()) != sorted(archive_names):
                raise ValueError("P15 bundle member list changed while packaging")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest_path = args.output.with_suffix(".manifest.json")
    payload: dict[str, Any] = {
        "schema_version": "task2-p15-modal-input-v1",
        "archive": str(args.output),
        "archive_bytes": args.output.stat().st_size,
        "archive_sha256": _sha256(args.output),
        "members": [
            {
                "archive_path": archive_name,
                "source": str(source),
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
            for source, archive_name in members
        ],
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return args.output, manifest_path


def main() -> int:
    args = build_parser().parse_args()
    try:
        written = package(args)
    except (OSError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Task 2 P15 packaging error: {exc}")
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
