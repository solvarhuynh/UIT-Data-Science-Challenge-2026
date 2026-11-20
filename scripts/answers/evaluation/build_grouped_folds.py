"""Build deterministic grouped folds for LegalQA answer evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.answer_metrics import (  # noqa: E402
    LegalQAWarmupSample,
    load_legal_qa_warmup,
)
from udsc2026.evaluation.retrieval_metrics import (  # noqa: E402
    normalize_legal_ir_matching_question,
)


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unavailable"


def _question_groups(
    samples: Sequence[LegalQAWarmupSample],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        key = normalize_legal_ir_matching_question(sample.question)
        if not key:
            raise ValueError(f"question {sample.id!r} normalizes to empty text")
        groups[key].append(sample.id)
    return {key: sorted(ids) for key, ids in groups.items()}


def _build_folds(
    samples: Sequence[LegalQAWarmupSample], *, count: int, seed: int
) -> list[list[str]]:
    if count < 2:
        raise ValueError("fold count must be at least two")
    groups = _question_groups(samples)
    if len(groups) < count:
        raise ValueError("fold count cannot exceed normalized-question groups")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    ordered = sorted(enumerate(keys), key=lambda item: (-len(groups[item[1]]), item[0]))
    folds: list[list[str]] = [[] for _ in range(count)]
    group_counts = [0] * count
    for _, key in ordered:
        target = min(range(count), key=lambda i: (len(folds[i]), group_counts[i], i))
        folds[target].extend(groups[key])
        group_counts[target] += 1
    return [sorted(ids) for ids in folds]


def _validate_folds(
    samples: Sequence[LegalQAWarmupSample], folds: Sequence[Sequence[str]]
) -> dict[str, Any]:
    expected = {sample.id for sample in samples}
    memberships: dict[str, int] = {}
    group_fold: dict[str, int] = {}
    overlap: list[str] = []
    question_by_id = {sample.id: sample.question for sample in samples}
    for fold_index, ids in enumerate(folds):
        for question_id in ids:
            if question_id not in expected:
                raise ValueError(f"unknown question ID in folds: {question_id}")
            if question_id in memberships:
                raise ValueError(f"question ID appears in multiple folds: {question_id}")
            memberships[question_id] = fold_index
            key = normalize_legal_ir_matching_question(question_by_id[question_id])
            previous = group_fold.setdefault(key, fold_index)
            if previous != fold_index:
                overlap.append(key)
    if set(memberships) != expected:
        raise ValueError(
            "fold coverage mismatch: "
            f"missing={sorted(expected - set(memberships))[:5]}"
        )
    if overlap:
        raise ValueError("normalized question groups cross folds")
    return {
        "question_count": len(expected),
        "normalized_question_group_count": len(group_fold),
        "folds_disjoint": True,
        "validation_union_equals_all_train": True,
        "normalized_question_groups_disjoint": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/btc/LegalQA/train.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits/answers"))
    parser.add_argument("--folds", type=_positive_int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    samples = sorted(load_legal_qa_warmup(args.input), key=lambda item: item.id)
    folds = _build_folds(samples, count=args.folds, seed=args.seed)
    validation = _validate_folds(samples, folds)
    input_hash = _sha256(args.input)
    fold_payload: dict[str, Any] = {
        "schema_version": "legal-qa-grouped-folds-v1",
        "seed": args.seed,
        "fold_count": args.folds,
        "normalization": "nfkc-casefold-nonword-collapse-v1",
        "input": str(args.input),
        "input_sha256": input_hash,
        "validation": validation,
        "folds": [],
    }
    rows_by_id = {sample.id: sample for sample in samples}
    full_questions = {
        sample.id: {"question": sample.question}
        for sample in samples
    }
    full_questions_path = args.output_dir / "questions.json"
    full_questions_path.parent.mkdir(parents=True, exist_ok=True)
    full_questions_path.write_text(
        json.dumps(full_questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    outputs: list[Path] = [full_questions_path]
    for index, ids in enumerate(folds):
        validation_ids = set(ids)
        training_ids = sorted(set(rows_by_id).difference(validation_ids))
        fold_payload["folds"].append(
            {
                "fold": index,
                "validation_ids": ids,
                "training_ids": training_ids,
            }
        )
        references = {
            question_id: {
                "question": rows_by_id[question_id].question,
                "answer": rows_by_id[question_id].answer,
            }
            for question_id in ids
        }
        fold_dir = args.output_dir / f"fold_{index}"
        question_mapping = {
            question_id: {"question": rows_by_id[question_id].question}
            for question_id in ids
        }
        question_path = fold_dir / "questions.json"
        question_path.parent.mkdir(parents=True, exist_ok=True)
        question_path.write_text(
            json.dumps(question_mapping, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        reference_path = fold_dir / "references.json"
        reference_path.write_text(
            json.dumps(references, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        outputs.extend((question_path, reference_path))

    folds_path = args.output_dir / "folds.json"
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    folds_path.write_text(
        json.dumps(fold_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "legal-qa-fold-manifest-v1",
        "git_commit": _git_commit(),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hash,
        "parameters": {"fold_count": args.folds, "seed": args.seed},
        "validation": validation,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return [folds_path, manifest_path, *outputs]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        for path in run(args):
            print(path)
    except (OSError, TypeError, ValueError) as exc:
        print(f"LegalQA fold generation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
