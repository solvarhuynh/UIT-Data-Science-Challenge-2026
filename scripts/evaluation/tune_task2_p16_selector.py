"""Tune a leakage-safe semantic KNN selector over Task 2 answer candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--question-ids", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--extractive", type=Path, required=True)
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"expected a non-empty object: {path}")
    return payload


def _load_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"expected a non-empty ID array: {path}")
    ids = [str(value) for value in payload]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs: {path}")
    return ids


def _load_predictions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"predictions must be an array: {path}")
    output: dict[str, str] = {}
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"id", "answer"}:
            raise ValueError(f"invalid prediction row: {path}")
        question_id = str(row["id"])
        answer = row["answer"]
        if question_id in output or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"duplicate or blank prediction {question_id!r}")
        output[question_id] = answer.strip()
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("candidate bank is empty")
    return rows


def _fold(question_id: str, folds: int, seed: int) -> int:
    payload = f"task2-p16|{seed}|{question_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def _document(question: str, qwen: str, extractive: str) -> str:
    return (
        f"QUESTION {question.strip()}\n"
        f"QWEN {' '.join(qwen.split()[:192])}\n"
        f"EVIDENCE {' '.join(extractive.split()[:192])}"
    )


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=60_000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def _knn_predictions(
    similarities: np.ndarray,
    train_targets: np.ndarray,
    *,
    neighbors: int,
    power: float,
    prior_weight: float,
) -> np.ndarray:
    available = similarities.shape[1]
    selected_count = min(neighbors, available)
    indices = np.argpartition(
        similarities, available - selected_count, axis=1
    )[:, -selected_count:]
    selected = np.take_along_axis(similarities, indices, axis=1)
    weights = np.maximum(selected, 1e-6) ** power
    local = np.take(train_targets, indices, axis=0)
    prediction = (local * weights[..., None]).sum(axis=1) / weights.sum(
        axis=1, keepdims=True
    )
    if prior_weight:
        prior = train_targets.mean(axis=0, keepdims=True)
        prediction = (1.0 - prior_weight) * prediction + prior_weight * prior
    return prediction


def _atomic_json(path: Path, payload: Any) -> None:
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


def _metrics(
    selected: np.ndarray, meteor: np.ndarray, rouge: np.ndarray, folds: np.ndarray
) -> dict[str, Any]:
    rows = np.arange(len(selected))
    meteor_values = meteor[rows, selected]
    rouge_values = rouge[rows, selected]
    return {
        "meteor": float(meteor_values.mean()),
        "rouge_l": float(rouge_values.mean()),
        "fold_meteor": {
            str(fold): float(meteor_values[folds == fold].mean())
            for fold in sorted(set(int(value) for value in folds))
        },
    }


def tune(args: argparse.Namespace) -> dict[str, Any]:
    if args.folds < 3:
        raise ValueError("folds must be at least 3")
    questions = _load_object(args.questions)
    question_ids = _load_ids(args.question_ids)
    qwen = _load_predictions(args.qwen)
    extractive = _load_predictions(args.extractive)
    for name, source in (
        ("questions", questions),
        ("qwen", qwen),
        ("extractive", extractive),
    ):
        if missing := set(question_ids) - set(source):
            raise ValueError(f"{name} is missing IDs: {sorted(missing)[:5]}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(args.candidate_bank):
        grouped[str(row.get("id", ""))].append(row)
    profiles = [str(row["profile"]) for row in grouped[question_ids[0]]]
    if not profiles:
        raise ValueError("candidate profiles are empty")
    for question_id in question_ids:
        rows = grouped[question_id]
        if [str(row.get("profile", "")) for row in rows] != profiles:
            raise ValueError(f"profile coverage changed for {question_id!r}")
    meteor = np.asarray(
        [[float(row["meteor"]) for row in grouped[qid]] for qid in question_ids],
        dtype=np.float32,
    )
    rouge = np.asarray(
        [[float(row["rouge_l"]) for row in grouped[qid]] for qid in question_ids],
        dtype=np.float32,
    )
    documents = [
        _document(
            str(questions[qid]["question"]), qwen[qid], extractive[qid]
        )
        for qid in question_ids
    ]
    fold_array = np.asarray(
        [_fold(qid, args.folds, args.seed) for qid in question_ids],
        dtype=np.int16,
    )
    grid = [
        (neighbors, power, prior)
        for neighbors in (5, 10, 20, 40, 80, 160)
        for power in (1.0, 2.0, 4.0)
        for prior in (0.0, 0.20, 0.40)
    ]
    oof = {
        key: np.full_like(meteor, np.nan)
        for key in grid
    }
    for fold in range(args.folds):
        train = np.flatnonzero(fold_array != fold)
        valid = np.flatnonzero(fold_array == fold)
        vectorizer = _vectorizer()
        train_matrix = normalize(
            vectorizer.fit_transform([documents[index] for index in train]),
            copy=False,
        )
        valid_matrix = normalize(
            vectorizer.transform([documents[index] for index in valid]),
            copy=False,
        )
        similarities = (valid_matrix @ train_matrix.T).toarray()
        for neighbors, power, prior in grid:
            oof[(neighbors, power, prior)][valid] = _knn_predictions(
                similarities,
                meteor[train],
                neighbors=neighbors,
                power=power,
                prior_weight=prior,
            )
        print(
            f"p16_selector_fold={fold + 1}/{args.folds} "
            f"train={len(train)} valid={len(valid)} features={train_matrix.shape[1]}",
            flush=True,
        )
    results: list[dict[str, Any]] = []
    selected_by_key: dict[tuple[int, float, float], np.ndarray] = {}
    for key, predictions in oof.items():
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"OOF selector is incomplete: {key}")
        selected = predictions.argmax(axis=1)
        selected_by_key[key] = selected
        metrics = _metrics(selected, meteor, rouge, fold_array)
        results.append(
            {
                "neighbors": key[0],
                "power": key[1],
                "prior_weight": key[2],
                **metrics,
                "profile_counts": {
                    profiles[index]: int((selected == index).sum())
                    for index in sorted(set(int(value) for value in selected))
                },
            }
        )
    best = max(
        results,
        key=lambda row: (
            row["meteor"],
            min(row["fold_meteor"].values()),
            row["rouge_l"],
            -row["neighbors"],
        ),
    )
    best_key = (
        int(best["neighbors"]),
        float(best["power"]),
        float(best["prior_weight"]),
    )
    selected = selected_by_key[best_key]
    predictions = [
        {"id": qid, "answer": str(grouped[qid][int(index)]["answer"])}
        for qid, index in zip(question_ids, selected)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "selector_oof_predictions.json"
    model_path = args.output_dir / "selector.joblib"
    report_path = args.output_dir / "selector_report.json"
    _atomic_json(predictions_path, predictions)
    final_vectorizer = _vectorizer()
    final_matrix = normalize(final_vectorizer.fit_transform(documents), copy=False)
    joblib.dump(
        {
            "schema_version": "task2-p16-semantic-knn-selector-v1",
            "profiles": profiles,
            "neighbors": best_key[0],
            "power": best_key[1],
            "prior_weight": best_key[2],
            "vectorizer": final_vectorizer,
            "train_matrix": sparse.csr_matrix(final_matrix),
            "target_matrix": meteor,
        },
        model_path,
        compress=3,
    )
    payload = {
        "schema_version": "task2-p16-semantic-knn-selector-report-v1",
        "question_count": len(question_ids),
        "profiles": profiles,
        "best": best,
        "grid": sorted(results, key=lambda row: row["meteor"], reverse=True),
        "folds": args.folds,
        "seed": args.seed,
        "artifacts": {
            "oof_predictions": str(predictions_path),
            "selector": str(model_path),
        },
        "leakage_policy": (
            "question-grouped OOF; validation references never enter fit folds"
        ),
    }
    _atomic_json(report_path, payload)
    print("P16_SELECTOR_BEST " + json.dumps(best, ensure_ascii=False), flush=True)
    return payload


def main() -> int:
    args = build_parser().parse_args()
    try:
        tune(args)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Task 2 P16 selector error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
