"""Merge complete, disjoint LegalQA held-out predictions into one OOF file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.answer_metrics import (  # noqa: E402
    LegalQAPredictionSet,
    load_legal_qa_question_ids,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--filename", default="predictions.json")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.fold_count < 2:
        raise ValueError("--fold-count must be at least two")
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    input_paths: list[Path] = []
    for index in range(args.fold_count):
        path = args.folds_dir / f"fold_{index}" / args.filename
        input_paths.append(path)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        predictions = LegalQAPredictionSet.model_validate(payload, strict=True).root
        for prediction in predictions:
            if prediction.id in seen:
                raise ValueError(f"duplicate out-of-fold prediction ID: {prediction.id}")
            seen.add(prediction.id)
            rows.append(prediction.model_dump(mode="json"))

    expected_ids = set(load_legal_qa_question_ids(args.questions))
    if seen != expected_ids:
        raise ValueError(
            "fold prediction coverage mismatch: "
            f"missing={sorted(expected_ids - seen)[:5]}, "
            f"extra={sorted(seen - expected_ids)[:5]}"
        )

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    if args.output.resolve() == manifest_path.resolve():
        raise ValueError("--output and --manifest must be different files")
    input_paths.append(args.questions)
    for destination in (args.output, manifest_path):
        if any(destination.resolve() == source.resolve() for source in input_paths):
            raise ValueError(f"output path collides with an input: {destination}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "legal-qa-oof-predictions-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_count": len(rows),
        "questions_sha256": _sha256(args.questions),
        "fold_predictions": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in input_paths[:-1]
        ],
        "output_sha256": _sha256(args.output),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
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
        print(f"LegalQA fold merge error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
