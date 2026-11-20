"""Merge complete, disjoint fold predictions into one OOF ranking file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.retrieval_metrics import (  # noqa: E402
    LegalIRPredictionSet,
    load_legal_ir_question_ids,
)
from udsc2026.evaluation.reporting import write_run_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--filename", default="predictions.json")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    if args.fold_count < 2:
        raise ValueError("--fold-count must be at least two")
    predictions = []
    seen: set[str] = set()
    input_paths: list[Path] = []
    for fold_index in range(args.fold_count):
        path = args.folds_dir / f"fold_{fold_index}" / args.filename
        input_paths.append(path)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        fold_predictions = LegalIRPredictionSet.model_validate(
            payload, strict=True
        ).root
        for prediction in fold_predictions:
            if prediction.id in seen:
                raise ValueError(f"duplicate out-of-fold prediction ID: {prediction.id}")
            seen.add(prediction.id)
            predictions.append(prediction)
    expected_ids = load_legal_ir_question_ids(args.questions)
    if seen != set(expected_ids):
        missing = sorted(set(expected_ids) - seen)[:5]
        extra = sorted(seen - set(expected_ids))[:5]
        raise ValueError(f"fold prediction coverage mismatch: missing={missing}, extra={extra}")

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    inputs = [args.questions, *input_paths]
    destinations = [args.output, manifest_path]
    if args.output.resolve() == manifest_path.resolve():
        raise ValueError("--output and --manifest must be different files")
    for destination in destinations:
        if any(destination.resolve() == source.resolve() for source in inputs):
            raise ValueError(f"output path collides with an input: {destination}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.model_dump(mode="json") for item in predictions]
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    write_run_manifest(
        {
            "schema_version": "out-of-fold-predictions-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "questions_sha256": _sha256(args.questions),
            "fold_predictions": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in input_paths
            ],
            "output_sha256": _sha256(args.output),
            "question_count": len(rows),
        },
        manifest_path,
    )
    print(f"complete_questions={len(rows)}")
    print(args.output)
    print(manifest_path)
    return args.output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"fold prediction merge error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
