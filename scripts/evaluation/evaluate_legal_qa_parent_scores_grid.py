"""Evaluate a source-only LegalQA parent-ranking grid with official metrics.

The scored parent JSONL is parsed once.  Each grid cell is written as an
internal prediction JSON plus a metric report in a fresh evaluation directory;
this command never creates or overwrites an official submission archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.submission.build_legal_qa_parent_scores import (  # noqa: E402
    _load_json,
    _write_json,
    build_predictions,
    load_parent_scores,
    load_question_ids,
)
from scripts.submission.supervised_legal_qa_third_parent import (  # noqa: E402
    _official_meteor,
    _official_rouge_l,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--question-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--top-parents",
        type=_positive_int,
        nargs="+",
        default=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--max-parent-tokens",
        type=_positive_int,
        nargs="+",
        default=[256, 384, 512],
    )
    parser.add_argument("--split-a-count", type=_positive_int, default=500)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("metric split must not be empty")
    result = sum(values) / len(values)
    if not math.isfinite(result):
        raise ValueError("metric mean is not finite")
    return result


def _metric_block(
    meteor: Sequence[float], rouge_l: Sequence[float]
) -> dict[str, float | int]:
    if len(meteor) != len(rouge_l):
        raise ValueError("METEOR and ROUGE-L arrays have different lengths")
    return {
        "count": len(meteor),
        "meteor": _mean(meteor),
        "rouge_l": _mean(rouge_l),
    }


def _load_labeled_questions(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("questions must be a non-empty JSON object")
    for question_id, record in payload.items():
        if not isinstance(question_id, str) or not isinstance(record, dict):
            raise TypeError("questions must map string IDs to objects")
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"question {question_id!r} has no reference answer")
    return payload


def _score_predictions(
    predictions: Sequence[dict[str, str]],
    questions: dict[str, dict[str, Any]],
    *,
    split_a_count: int,
) -> dict[str, dict[str, float | int]]:
    meteor: list[float] = []
    rouge_l: list[float] = []
    for prediction in predictions:
        question_id = prediction["id"]
        reference = questions[question_id]["answer"]
        answer = prediction["answer"]
        meteor.append(_official_meteor(reference, answer))
        rouge_l.append(_official_rouge_l(reference, answer))
    return {
        "split_a": _metric_block(meteor[:split_a_count], rouge_l[:split_a_count]),
        "split_b": _metric_block(meteor[split_a_count:], rouge_l[split_a_count:]),
        "pooled": _metric_block(meteor, rouge_l),
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    inputs = (args.scores, args.questions, args.question_ids)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output_dir}")
    output_root = args.output_dir.resolve()
    for input_path in inputs:
        input_resolved = input_path.resolve()
        if input_resolved == output_root or output_root in input_resolved.parents:
            raise ValueError("output directory must not contain an input")

    question_ids = load_question_ids(args.questions, args.question_ids)
    if args.split_a_count >= len(question_ids):
        raise ValueError("split-a-count must be smaller than question count")
    questions = _load_labeled_questions(args.questions)
    scores = load_parent_scores(args.scores)
    if set(scores) != set(question_ids):
        raise ValueError("scored question coverage differs from explicit IDs")

    top_values = list(dict.fromkeys(args.top_parents))
    token_values = list(dict.fromkeys(args.max_parent_tokens))
    if any(value > 4 for value in top_values):
        raise ValueError("top-parents values must not exceed four")

    results: list[dict[str, Any]] = []
    for top_parents in top_values:
        for parent_tokens in token_values:
            total_tokens = top_parents * parent_tokens
            cell_name = f"k{top_parents}_m{parent_tokens}_t{total_tokens}"
            cell_dir = args.output_dir / cell_name
            prediction_path = cell_dir / "predictions.json"
            report_path = cell_dir / "metrics.json"
            predictions, build_summary = build_predictions(
                question_ids,
                scores,
                top_parents=top_parents,
                max_parent_tokens=parent_tokens,
                max_total_tokens=total_tokens,
            )
            metrics = _score_predictions(
                predictions,
                questions,
                split_a_count=args.split_a_count,
            )
            _write_json(prediction_path, predictions)
            report: dict[str, Any] = {
                "cell": cell_name,
                "top_parents": top_parents,
                "max_parent_tokens": parent_tokens,
                "max_total_tokens": total_tokens,
                "build": build_summary,
                "metrics": metrics,
                "prediction_path": str(prediction_path),
                "prediction_sha256": _sha256(prediction_path),
                "constraints": {
                    "external_data": False,
                    "known_answer_overlay": False,
                    "synthetic_or_generated_text": False,
                    "submission_packaging": False,
                },
            }
            _write_json(report_path, report)
            results.append(report)
            print(
                json.dumps(
                    {
                        "cell": cell_name,
                        "meteor_a": metrics["split_a"]["meteor"],
                        "meteor_b": metrics["split_b"]["meteor"],
                        "meteor_pooled": metrics["pooled"]["meteor"],
                        "rouge_l_pooled": metrics["pooled"]["rouge_l"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    ranked = sorted(
        results,
        key=lambda row: (
            -row["metrics"]["pooled"]["meteor"],
            -min(
                row["metrics"]["split_a"]["meteor"],
                row["metrics"]["split_b"]["meteor"],
            ),
            -row["metrics"]["pooled"]["rouge_l"],
            row["top_parents"],
            row["max_parent_tokens"],
        ),
    )
    summary = {
        "schema_version": 1,
        "scorer": {
            "source": "docs/Scoring-Program-Task-LegalQA/scoring.py",
            "meteor": "nltk.translate.meteor_score on whitespace tokens",
            "rouge_l": "rouge_score RougeScorer rougeL use_stemmer=False",
            "macro_average": True,
        },
        "inputs": {
            "scores": {
                "path": str(args.scores),
                "sha256": _sha256(args.scores),
            },
            "questions": {
                "path": str(args.questions),
                "sha256": _sha256(args.questions),
            },
            "question_ids": {
                "path": str(args.question_ids),
                "sha256": _sha256(args.question_ids),
            },
        },
        "split_a_count": args.split_a_count,
        "split_b_count": len(question_ids) - args.split_a_count,
        "grid_cell_count": len(results),
        "ranking": [row["cell"] for row in ranked],
        "best": ranked[0],
        "cells": results,
    }
    _write_json(args.output_dir / "grid_summary.json", summary)
    return ranked


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ranked = run(args)
    except (OSError, TypeError, ValueError) as exc:
        print(f"LegalQA parent grid error: {exc}", file=sys.stderr)
        return 2
    print(f"best_cell={ranked[0]['cell']}")
    print(args.output_dir / "grid_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
