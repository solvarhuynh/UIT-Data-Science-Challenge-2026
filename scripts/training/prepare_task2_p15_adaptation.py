"""Prepare deterministic P15 continuation and preference datasets.

P15 starts from the already trained P14 adapter.  This command uses only the
previously untouched strict split for development: 80% adapts the model and
20% remains a real gate.  A second all-record file is emitted only for the
final fit after the development gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts.training.finetune_task2_qwen_lora import prepare_records  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--question-ids", type=Path, required=True)
    parser.add_argument("--labels", type=Path, action="append", required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contexts-per-question", type=int, default=3)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: Any, *, jsonl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            if jsonl:
                for row in payload:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
                    stream.write("\n")
            else:
                json.dump(
                    payload,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_dev(question_id: str, *, fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"task2-p15|{seed}|{question_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < fraction


def prepare(
    questions: dict[str, Any],
    question_ids: list[str],
    labels: list[dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    *,
    contexts_per_question: int,
    dev_fraction: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if contexts_per_question < 1:
        raise ValueError("contexts-per-question must be positive")
    if not 0.05 <= dev_fraction <= 0.5:
        raise ValueError("dev-fraction must be in [0.05, 0.5]")
    records, preparation = prepare_records(
        questions,
        question_ids,
        labels,
        contexts_per_question=contexts_per_question,
    )
    baseline: dict[str, str] = {}
    for row in baseline_predictions:
        if not isinstance(row, dict) or set(row) != {"id", "answer"}:
            raise ValueError("baseline rows must contain exactly id and answer")
        question_id = str(row["id"])
        answer = row["answer"]
        if question_id in baseline or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"invalid baseline prediction {question_id!r}")
        baseline[question_id] = answer.strip()
    missing_baselines = {str(row["id"]) for row in records} - set(baseline)
    if missing_baselines:
        raise ValueError(f"baseline is missing IDs: {sorted(missing_baselines)[:5]}")

    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    preferences_train: list[dict[str, Any]] = []
    preferences_all: list[dict[str, Any]] = []
    baseline_dev: list[dict[str, str]] = []
    for row in records:
        question_id = str(row["id"])
        preference = {
            "schema_version": 1,
            "id": question_id,
            "question": row["question"],
            "contexts": row["contexts"],
            "chosen": row["answer"],
            "rejected": baseline[question_id],
        }
        preferences_all.append(preference)
        if _is_dev(question_id, fraction=dev_fraction, seed=seed):
            dev.append(row)
            baseline_dev.append({"id": question_id, "answer": baseline[question_id]})
        else:
            train.append(row)
            preferences_train.append(preference)
    if not train or not dev:
        raise ValueError("deterministic split produced an empty partition")
    train_ids = {str(row["id"]) for row in train}
    dev_ids = {str(row["id"]) for row in dev}
    if train_ids & dev_ids:
        raise RuntimeError("adaptation train/dev ID overlap")
    outputs = {
        "adaptation_all": records,
        "adaptation_train": train,
        "adaptation_dev": dev,
        "preference_all": preferences_all,
        "preference_train": preferences_train,
        "baseline_dev": baseline_dev,
    }
    report = {
        "schema_version": "task2-p15-adaptation-v1",
        "requested_questions": len(question_ids),
        "usable_questions": len(records),
        "train_questions": len(train),
        "dev_questions": len(dev),
        "train_dev_overlap": 0,
        "dev_fraction": dev_fraction,
        "seed": seed,
        "preparation": preparation,
        "leakage_policy": (
            "development gate IDs are excluded from adaptation_train; "
            "adaptation_all is permitted only after the gate passes"
        ),
    }
    return outputs, report


def run(args: argparse.Namespace) -> list[Path]:
    questions = _load_json(args.questions)
    question_ids = [str(value) for value in _load_json(args.question_ids)]
    baseline = _load_json(args.baseline_predictions)
    if not isinstance(questions, dict) or not isinstance(baseline, list):
        raise ValueError("invalid questions or baseline payload")
    outputs, report = prepare(
        questions,
        question_ids,
        _read_jsonl(args.labels),
        baseline,
        contexts_per_question=args.contexts_per_question,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    paths = {
        "adaptation_all": args.output_dir / "adaptation_all.jsonl",
        "adaptation_train": args.output_dir / "adaptation_train.jsonl",
        "adaptation_dev": args.output_dir / "adaptation_dev.jsonl",
        "preference_all": args.output_dir / "preference_all.jsonl",
        "preference_train": args.output_dir / "preference_train.jsonl",
        "baseline_dev": args.output_dir / "baseline_dev_predictions.json",
    }
    for name, path in paths.items():
        _atomic_write(path, outputs[name], jsonl=name != "baseline_dev")
    dev_ids_path = args.output_dir / "adaptation_dev_ids.json"
    _atomic_write(
        dev_ids_path,
        [str(row["id"]) for row in outputs["adaptation_dev"]],
    )
    manifest_path = args.output_dir / "adaptation_manifest.json"
    report["inputs"] = {
        "questions": str(args.questions),
        "question_ids": str(args.question_ids),
        "labels": [str(path) for path in args.labels],
        "baseline_predictions": str(args.baseline_predictions),
    }
    report["artifacts"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in {**paths, "dev_ids": dev_ids_path}.items()
    }
    _atomic_write(manifest_path, report)
    return [*paths.values(), dev_ids_path, manifest_path]


def main() -> int:
    args = build_parser().parse_args()
    try:
        written = run(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Task 2 P15 preparation error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
