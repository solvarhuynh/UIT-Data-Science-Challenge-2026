"""Select one label-free answer profile on P15 dev and apply it to public."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from udsc2026.evaluation.legal_qa_candidates import (  # noqa: E402
    build_answer_candidates,
)
from udsc2026.evaluation.legal_qa_scoring import (  # noqa: E402
    official_meteor,
    official_rouge_l,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-questions", type=Path, required=True)
    parser.add_argument("--dev-ids", type=Path, required=True)
    parser.add_argument("--dev-qwen", type=Path, required=True)
    parser.add_argument("--dev-extractive", type=Path, required=True)
    parser.add_argument("--public-questions", type=Path)
    parser.add_argument("--public-qwen", type=Path)
    parser.add_argument("--public-extractive", type=Path)
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Score/select on dev without requiring or writing public predictions.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--meteor-weight", type=float, default=0.85)
    parser.add_argument("--rouge-weight", type=float, default=0.15)
    parser.add_argument("--maximum-rouge-drop", type=float, default=0.04)
    return parser


def _load_questions(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"questions must be a non-empty object: {path}")
    return payload


def _load_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"IDs must be a non-empty array: {path}")
    ids = [str(value) for value in payload]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs: {path}")
    return ids


def _load_predictions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"predictions must be an array: {path}")
    mapped: dict[str, str] = {}
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"id", "answer"}:
            raise ValueError(f"invalid prediction row: {path}")
        question_id = str(row["id"])
        answer = row["answer"]
        if question_id in mapped or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"duplicate or blank prediction {question_id!r}: {path}")
        mapped[question_id] = answer.strip()
    return mapped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
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


def _candidate_map(question: str, qwen: str, extractive: str) -> dict[str, str]:
    return {
        candidate.profile: candidate.answer
        for candidate in build_answer_candidates(question, qwen, extractive)
    }


def select_profile(
    questions: dict[str, Any],
    question_ids: list[str],
    qwen: dict[str, str],
    extractive: dict[str, str],
    *,
    meteor_weight: float,
    rouge_weight: float,
    maximum_rouge_drop: float,
) -> tuple[str, dict[str, dict[str, float]], list[dict[str, str]]]:
    if (
        not math.isfinite(meteor_weight)
        or not math.isfinite(rouge_weight)
        or meteor_weight < 0
        or rouge_weight < 0
        or meteor_weight + rouge_weight <= 0
    ):
        raise ValueError("metric weights must be finite, nonnegative, and nonzero")
    if not math.isfinite(maximum_rouge_drop) or maximum_rouge_drop < 0:
        raise ValueError("maximum-rouge-drop must be finite and nonnegative")
    sums: dict[str, dict[str, float]] = defaultdict(
        lambda: {"meteor": 0.0, "rouge_l": 0.0}
    )
    profiles: list[str] | None = None
    candidate_rows: list[dict[str, dict[str, str]]] = []
    for index, question_id in enumerate(question_ids, 1):
        source = questions.get(question_id)
        question = source.get("question") if isinstance(source, dict) else None
        reference = source.get("answer") if isinstance(source, dict) else None
        if not isinstance(question, str) or not isinstance(reference, str):
            raise ValueError(f"dev question {question_id!r} lacks text/answer")
        if question_id not in qwen or question_id not in extractive:
            raise ValueError(f"dev sources are missing {question_id!r}")
        candidates = _candidate_map(
            question, qwen[question_id], extractive[question_id]
        )
        current_profiles = list(candidates)
        if profiles is None:
            profiles = current_profiles
        elif current_profiles != profiles:
            raise ValueError(f"candidate profile order changed for {question_id!r}")
        candidate_rows.append({"id": question_id, "answers": candidates})
        for profile, answer in candidates.items():
            sums[profile]["meteor"] += official_meteor(reference, answer)
            sums[profile]["rouge_l"] += official_rouge_l(reference, answer)
        if index % 50 == 0 or index == len(question_ids):
            print(f"profile_scoring={index}/{len(question_ids)}", flush=True)
    assert profiles is not None
    metrics = {
        profile: {
            metric: value / len(question_ids) for metric, value in sums[profile].items()
        }
        for profile in profiles
    }
    qwen_rouge = metrics["qwen"]["rouge_l"]
    eligible = [
        profile
        for profile in profiles
        if metrics[profile]["rouge_l"] >= qwen_rouge - maximum_rouge_drop
    ]
    selected = max(
        eligible,
        key=lambda profile: (
            meteor_weight * metrics[profile]["meteor"]
            + rouge_weight * metrics[profile]["rouge_l"],
            metrics[profile]["meteor"],
            metrics[profile]["rouge_l"],
            profile,
        ),
    )
    selected_rows = [
        {"id": row["id"], "answer": row["answers"][selected]}
        for row in candidate_rows
    ]
    return selected, metrics, selected_rows


def apply_profile(
    questions: dict[str, Any],
    qwen: dict[str, str],
    extractive: dict[str, str],
    profile: str,
) -> list[dict[str, str]]:
    if set(qwen) != set(questions) or set(extractive) != set(questions):
        raise ValueError("public source coverage differs from public questions")
    output: list[dict[str, str]] = []
    for question_id, source in questions.items():
        question = source.get("question") if isinstance(source, dict) else None
        if not isinstance(question, str):
            raise ValueError(f"public question {question_id!r} lacks text")
        candidates = _candidate_map(
            question, qwen[question_id], extractive[question_id]
        )
        if profile not in candidates:
            raise ValueError(f"profile {profile!r} is unavailable for {question_id!r}")
        output.append({"id": question_id, "answer": candidates[profile]})
    return output


def run(args: argparse.Namespace) -> list[Path]:
    dev_questions = _load_questions(args.dev_questions)
    dev_ids = _load_ids(args.dev_ids)
    selected, metrics, dev_predictions = select_profile(
        dev_questions,
        dev_ids,
        _load_predictions(args.dev_qwen),
        _load_predictions(args.dev_extractive),
        meteor_weight=args.meteor_weight,
        rouge_weight=args.rouge_weight,
        maximum_rouge_drop=args.maximum_rouge_drop,
    )
    dev_path = args.output_dir / "dev_predictions.json"
    public_path = args.output_dir / "public_predictions.json"
    report_path = args.output_dir / "profile_report.json"
    _write_json(dev_path, dev_predictions)
    public_predictions: list[dict[str, str]] = []
    if not args.selection_only:
        public_inputs = (
            args.public_questions,
            args.public_qwen,
            args.public_extractive,
        )
        if any(path is None for path in public_inputs):
            raise ValueError(
                "public questions, Qwen, and extractive inputs are required "
                "unless --selection-only is used"
            )
        public_predictions = apply_profile(
            _load_questions(args.public_questions),
            _load_predictions(args.public_qwen),
            _load_predictions(args.public_extractive),
            selected,
        )
        _write_json(public_path, public_predictions)
    _write_json(
        report_path,
        {
            "schema_version": "task2-p15-profile-selection-v1",
            "selected_profile": selected,
            "selected_metrics": metrics[selected],
            "profile_metrics": metrics,
            "metric_weights": {
                "meteor": args.meteor_weight,
                "rouge_l": args.rouge_weight,
            },
            "maximum_rouge_drop": args.maximum_rouge_drop,
            "dev_question_count": len(dev_ids),
            "public_question_count": len(public_predictions),
            "dev_predictions_sha256": _sha256(dev_path),
            "public_predictions_sha256": (
                _sha256(public_path) if public_predictions else None
            ),
            "label_policy": "profile selected on dev; public application is label-free",
        },
    )
    print(
        "P15_PROFILE_SELECTED "
        f"profile={selected} meteor={metrics[selected]['meteor']:.6f} "
        f"rouge_l={metrics[selected]['rouge_l']:.6f}",
        flush=True,
    )
    return [dev_path, *([public_path] if public_predictions else []), report_path]


def main() -> int:
    args = build_parser().parse_args()
    try:
        written = run(args)
    except (
        ImportError,
        LookupError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Task 2 P15 profile selection error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
