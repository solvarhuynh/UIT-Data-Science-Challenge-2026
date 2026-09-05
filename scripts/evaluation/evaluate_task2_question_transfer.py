"""Evaluate leakage-safe nearest-question answer transfer for Task 2.

The evaluator fits only on explicitly permitted organizer training IDs, finds
the nearest labelled question for each held-out question, and sweeps a cosine
similarity threshold.  References are used only for reporting the held-out
score; public application never reads public references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--eval-ids", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-threshold", type=float, default=0.35)
    parser.add_argument("--maximum-threshold", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser


def _load_questions(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError("questions must be a non-empty object")
    output: dict[str, dict[str, str]] = {}
    for question_id, row in payload.items():
        if not isinstance(row, dict):
            raise ValueError(f"invalid question row {question_id!r}")
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} has no text")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"question {question_id!r} has no reference answer")
        output[str(question_id)] = {
            "question": question.strip(),
            "answer": answer.strip(),
        }
    return output


def _load_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("eval IDs must be a non-empty array")
    ids = [str(value) for value in payload]
    if len(ids) != len(set(ids)):
        raise ValueError("eval IDs contain duplicates")
    return ids


def _load_predictions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("baseline predictions must be an array")
    output: dict[str, str] = {}
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"id", "answer"}:
            raise ValueError("invalid baseline prediction row")
        question_id = str(row["id"])
        answer = row["answer"]
        if question_id in output or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"duplicate or blank prediction {question_id!r}")
        output[question_id] = answer.strip()
    return output


def _normalize_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _nearest_questions(
    fit_texts: list[str], eval_texts: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    word = TfidfVectorizer(
        preprocessor=_normalize_question,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.995,
        sublinear_tf=True,
        max_features=180_000,
    )
    char = TfidfVectorizer(
        preprocessor=_normalize_question,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.999,
        sublinear_tf=True,
        max_features=220_000,
    )
    all_texts = fit_texts + eval_texts
    word_matrix = word.fit_transform(all_texts)
    char_matrix = char.fit_transform(all_texts)
    matrix = sparse.hstack(
        [word_matrix * math.sqrt(0.55), char_matrix * math.sqrt(0.45)],
        format="csr",
    )
    matrix = normalize(matrix, norm="l2", copy=False)
    fit_matrix = matrix[: len(fit_texts)]
    eval_matrix = matrix[len(fit_texts) :]
    nearest_indices: list[int] = []
    nearest_scores: list[float] = []
    for start in range(0, len(eval_texts), 128):
        similarities = eval_matrix[start : start + 128] @ fit_matrix.T
        indices = np.asarray(similarities.argmax(axis=1)).ravel()
        scores = np.asarray(similarities.max(axis=1).toarray()).ravel()
        nearest_indices.extend(int(value) for value in indices)
        nearest_scores.extend(float(value) for value in scores)
    return np.asarray(nearest_indices), np.asarray(nearest_scores)


def _thresholds(minimum: float, maximum: float, step: float) -> list[float]:
    if not all(math.isfinite(value) for value in (minimum, maximum, step)):
        raise ValueError("threshold values must be finite")
    if minimum < 0 or maximum > 1 or minimum > maximum or step <= 0:
        raise ValueError("invalid threshold range")
    count = int(math.floor((maximum - minimum) / step + 1e-9))
    return [round(minimum + index * step, 10) for index in range(count + 1)]


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
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from udsc2026.evaluation.legal_qa_scoring import official_meteor, official_rouge_l

    questions = _load_questions(args.questions)
    eval_ids = _load_ids(args.eval_ids)
    baseline = _load_predictions(args.baseline)
    missing = set(eval_ids) - set(questions)
    if missing:
        raise ValueError(f"eval IDs are absent from questions: {sorted(missing)[:5]}")
    if set(baseline) != set(eval_ids):
        raise ValueError("baseline coverage must exactly equal eval IDs")
    eval_set = set(eval_ids)
    fit_ids = [question_id for question_id in questions if question_id not in eval_set]
    if not fit_ids:
        raise ValueError("no leakage-safe fit questions remain")
    indices, similarities = _nearest_questions(
        [questions[question_id]["question"] for question_id in fit_ids],
        [questions[question_id]["question"] for question_id in eval_ids],
    )
    transferred = [questions[fit_ids[index]]["answer"] for index in indices]
    references = [questions[question_id]["answer"] for question_id in eval_ids]
    baseline_answers = [baseline[question_id] for question_id in eval_ids]

    # Score each possible answer once. A naive threshold sweep would call the
    # WordNet-backed METEOR implementation tens of thousands of times.
    baseline_meteor = np.asarray(
        [
            official_meteor(reference, answer)
            for reference, answer in zip(references, baseline_answers)
        ]
    )
    baseline_rouge = np.asarray(
        [
            official_rouge_l(reference, answer)
            for reference, answer in zip(references, baseline_answers)
        ]
    )
    transfer_meteor = np.asarray(
        [
            official_meteor(reference, answer)
            for reference, answer in zip(references, transferred)
        ]
    )
    transfer_rouge = np.asarray(
        [
            official_rouge_l(reference, answer)
            for reference, answer in zip(references, transferred)
        ]
    )
    baseline_metrics = {
        "meteor": float(baseline_meteor.mean()),
        "rouge_l": float(baseline_rouge.mean()),
    }
    rows: list[dict[str, Any]] = []
    for threshold in _thresholds(
        args.minimum_threshold, args.maximum_threshold, args.threshold_step
    ):
        overlay = similarities >= threshold
        metrics = {
            "meteor": float(
                np.where(overlay, transfer_meteor, baseline_meteor).mean()
            ),
            "rouge_l": float(
                np.where(overlay, transfer_rouge, baseline_rouge).mean()
            ),
        }
        rows.append(
            {
                "threshold": threshold,
                "overlay_count": int(overlay.sum()),
                **metrics,
                "meteor_gain": metrics["meteor"] - baseline_metrics["meteor"],
                "rouge_l_gain": metrics["rouge_l"] - baseline_metrics["rouge_l"],
            }
        )
    best = max(rows, key=lambda row: (row["meteor"], row["rouge_l"], row["threshold"]))
    payload = {
        "schema_version": "task2-question-transfer-eval-v1",
        "fit_count": len(fit_ids),
        "eval_count": len(eval_ids),
        "fit_eval_overlap": 0,
        "baseline": baseline_metrics,
        "best": best,
        "similarity": {
            "minimum": float(similarities.min()),
            "median": float(np.median(similarities)),
            "p90": float(np.quantile(similarities, 0.9)),
            "p95": float(np.quantile(similarities, 0.95)),
            "maximum": float(similarities.max()),
        },
        "sweep": rows,
        "inputs": {
            "questions": str(args.questions),
            "eval_ids": str(args.eval_ids),
            "baseline": str(args.baseline),
            "questions_sha256": _sha256(args.questions),
            "eval_ids_sha256": _sha256(args.eval_ids),
            "baseline_sha256": _sha256(args.baseline),
        },
        "leakage_policy": "eval IDs excluded from labelled nearest-neighbour fit",
    }
    _write_json(args.output, payload)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = evaluate(args)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        print(f"Task 2 question-transfer evaluation error: {exc}")
        return 2
    print(json.dumps(payload["best"], ensure_ascii=False, sort_keys=True))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
