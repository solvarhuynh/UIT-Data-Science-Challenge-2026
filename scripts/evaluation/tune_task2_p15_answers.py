"""Build and cross-validate a label-free Task 2 P15 answer selector."""

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
    answer_features,
    build_answer_candidates,
    validate_candidate_bank,
)
from udsc2026.evaluation.legal_qa_scoring import (  # noqa: E402
    fast_exact_meteor,
    official_meteor,
    official_rouge_l,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--question-ids", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--extractive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--metric",
        choices=("fast", "official"),
        default="fast",
        help="Use fast exact-METEOR for search or official WordNet METEOR.",
    )
    parser.add_argument("--meteor-weight", type=float, default=0.75)
    parser.add_argument("--rouge-weight", type=float, default=0.25)
    return parser


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"expected non-empty object: {path}")
    return payload


def _load_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"expected non-empty ID list: {path}")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scorers(metric: str) -> tuple[Any, Any, str]:
    if metric == "fast":
        return fast_exact_meteor, official_rouge_l, "fast-exact-meteor"
    # Validate resources before the long candidate pass. official_meteor uses
    # the same lazy cache on subsequent calls.
    official_meteor("resource check", "resource check")
    return official_meteor, official_rouge_l, "organizer-wordnet-omw"


def _fold_for_id(question_id: str, folds: int, seed: int) -> int:
    payload = f"task2-p15|{seed}|{question_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % folds


def _selector_document(question: str, qwen: str, extractive: str) -> str:
    """Build bounded, reference-free text used by the semantic selector."""

    qwen_prefix = " ".join(qwen.split()[:160])
    extractive_prefix = " ".join(extractive.split()[:160])
    return (
        f"QUESTION {question.strip()}\n"
        f"QWEN {qwen_prefix}\n"
        f"EVIDENCE {extractive_prefix}"
    )


def run(args: argparse.Namespace) -> list[Path]:
    if args.folds < 3:
        raise ValueError("folds must be at least 3")
    if (
        not math.isfinite(args.meteor_weight)
        or not math.isfinite(args.rouge_weight)
        or args.meteor_weight < 0
        or args.rouge_weight < 0
        or args.meteor_weight + args.rouge_weight <= 0
    ):
        raise ValueError("metric weights must be finite, nonnegative, and nonzero")
    questions = _load_object(args.questions)
    question_ids = _load_ids(args.question_ids)
    qwen = _load_predictions(args.qwen)
    extractive = _load_predictions(args.extractive)
    sources_to_check = (
        ("questions", questions),
        ("qwen", qwen),
        ("extractive", extractive),
    )
    for label, source in sources_to_check:
        missing = set(question_ids) - set(source)
        if missing:
            raise ValueError(f"{label} missing IDs: {sorted(missing)[:5]}")

    bank: list[dict[str, Any]] = []
    sources: dict[str, tuple[str, str, str]] = {}
    for index, question_id in enumerate(question_ids, 1):
        row = questions[question_id]
        question = row.get("question") if isinstance(row, dict) else None
        reference = row.get("answer") if isinstance(row, dict) else None
        if not isinstance(question, str) or not isinstance(reference, str):
            raise ValueError(f"question {question_id!r} lacks question/answer")
        sources[question_id] = (question, qwen[question_id], extractive[question_id])
        for candidate in build_answer_candidates(
            question, qwen[question_id], extractive[question_id]
        ):
            bank.append(
                {
                    "id": question_id,
                    "profile": candidate.profile,
                    "answer": candidate.answer,
                }
            )
        if index % 100 == 0 or index == len(question_ids):
            print(f"candidate_bank={index}/{len(question_ids)}", flush=True)
    bank_ids, profiles = validate_candidate_bank(bank)
    if bank_ids != question_ids:
        raise ValueError("candidate bank changed question order")

    meteor_scorer, rouge_scorer, metric_profile = _scorers(args.metric)
    scored: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, candidate in enumerate(bank, 1):
        question_id = candidate["id"]
        reference = str(questions[question_id]["answer"])
        meteor = float(meteor_scorer(reference, candidate["answer"]))
        rouge = float(rouge_scorer(reference, candidate["answer"]))
        target = (
            args.meteor_weight * meteor + args.rouge_weight * rouge
        ) / (args.meteor_weight + args.rouge_weight)
        row = {**candidate, "meteor": meteor, "rouge_l": rouge, "target": target}
        scored.append(row)
        grouped[question_id].append(row)
        if index % 1000 == 0 or index == len(bank):
            print(f"candidate_scores={index}/{len(bank)}", flush=True)

    import joblib
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    features: list[list[float]] = []
    targets: list[float] = []
    folds: list[int] = []
    for row in scored:
        question, qwen_answer, extractive_answer = sources[row["id"]]
        features.append(
            answer_features(
                question,
                row["answer"],
                qwen_answer,
                extractive_answer,
                profile=row["profile"],
                profile_names=profiles,
            )
        )
        targets.append(float(row["target"]))
        folds.append(_fold_for_id(row["id"], args.folds, args.seed))
    matrix = np.asarray(features, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    fold_array = np.asarray(folds, dtype=np.int16)
    oof_score = np.full(len(scored), np.nan, dtype=np.float32)
    for fold in range(args.folds):
        train_mask = fold_array != fold
        valid_mask = fold_array == fold
        model = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=args.seed + fold,
            n_jobs=-1,
        )
        model.fit(matrix[train_mask], target_array[train_mask])
        oof_score[valid_mask] = model.predict(matrix[valid_mask])
        print(
            f"selector_fold={fold + 1}/{args.folds} "
            f"train={int(train_mask.sum())} valid={int(valid_mask.sum())}",
            flush=True,
        )
    if not np.isfinite(oof_score).all():
        raise RuntimeError("selector did not score every OOF candidate")

    documents = [
        _selector_document(*sources[question_id]) for question_id in question_ids
    ]
    question_folds = np.asarray(
        [
            _fold_for_id(question_id, args.folds, args.seed)
            for question_id in question_ids
        ],
        dtype=np.int16,
    )
    target_matrix = np.asarray(
        [
            [float(row["target"]) for row in grouped[question_id]]
            for question_id in question_ids
        ],
        dtype=np.float32,
    )
    for question_id in question_ids:
        observed = [str(row["profile"]) for row in grouped[question_id]]
        if observed != profiles:
            raise ValueError(f"candidate profile order changed for {question_id!r}")
    text_oof = np.full_like(target_matrix, np.nan)
    for fold in range(args.folds):
        train_indices = np.flatnonzero(question_folds != fold)
        valid_indices = np.flatnonzero(question_folds == fold)
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=30_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        train_matrix = vectorizer.fit_transform(
            [documents[index] for index in train_indices]
        )
        valid_matrix = vectorizer.transform(
            [documents[index] for index in valid_indices]
        )
        text_model = Ridge(alpha=20.0)
        text_model.fit(train_matrix, target_matrix[train_indices])
        text_oof[valid_indices] = text_model.predict(valid_matrix)
        print(
            f"text_selector_fold={fold + 1}/{args.folds} "
            f"features={train_matrix.shape[1]}",
            flush=True,
        )
    if not np.isfinite(text_oof).all():
        raise RuntimeError("text selector did not score every OOF candidate")
    text_oof_flat = text_oof.reshape(-1)
    selector_scores = {
        "numeric_extra_trees": oof_score,
        "semantic_ridge": text_oof_flat,
        "hybrid_35_tree_65_text": 0.35 * oof_score + 0.65 * text_oof_flat,
    }

    def choose(
        scores: Any,
    ) -> tuple[list[dict[str, str]], dict[str, int]]:
        selected_rows: list[dict[str, str]] = []
        counts: dict[str, int] = defaultdict(int)
        offset = 0
        for question_id in question_ids:
            candidates = grouped[question_id]
            size = len(candidates)
            local_scores = scores[offset : offset + size]
            chosen_index = max(
                range(size), key=lambda idx: (float(local_scores[idx]), -idx)
            )
            chosen = candidates[chosen_index]
            selected_rows.append({"id": question_id, "answer": chosen["answer"]})
            counts[str(chosen["profile"])] += 1
            offset += size
        return selected_rows, counts

    oracle_scores = np.asarray(
        [float(row["target"]) for row in scored], dtype=np.float32
    )
    oracle, _ = choose(oracle_scores)

    def aggregate(rows: list[dict[str, str]]) -> dict[str, float]:
        meteor_values: list[float] = []
        rouge_values: list[float] = []
        for row in rows:
            reference = str(questions[row["id"]]["answer"])
            meteor_values.append(float(meteor_scorer(reference, row["answer"])))
            rouge_values.append(float(rouge_scorer(reference, row["answer"])))
        return {
            "meteor": sum(meteor_values) / len(meteor_values),
            "rouge_l": sum(rouge_values) / len(rouge_values),
        }

    profile_metrics: dict[str, dict[str, float]] = {}
    for profile in profiles:
        rows = [
            {"id": row["id"], "answer": row["answer"]}
            for row in scored
            if row["profile"] == profile
        ]
        profile_metrics[profile] = aggregate(rows)
    method_predictions: dict[str, list[dict[str, str]]] = {}
    method_counts: dict[str, dict[str, int]] = {}
    method_metrics: dict[str, dict[str, float]] = {}
    for method, scores in selector_scores.items():
        predictions, counts = choose(scores)
        method_predictions[method] = predictions
        method_counts[method] = dict(sorted(counts.items()))
        method_metrics[method] = aggregate(predictions)
    selected_method = max(
        method_metrics,
        key=lambda name: (
            args.meteor_weight * method_metrics[name]["meteor"]
            + args.rouge_weight * method_metrics[name]["rouge_l"],
            name,
        ),
    )
    selected = method_predictions[selected_method]
    profile_counts = method_counts[selected_method]
    selector_metrics = method_metrics[selected_method]
    oracle_metrics = aggregate(oracle)

    final_model = ExtraTreesRegressor(
        n_estimators=600,
        min_samples_leaf=8,
        max_features=0.8,
        random_state=args.seed,
        n_jobs=-1,
    )
    final_model.fit(matrix, target_array)
    final_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    final_text_matrix = final_vectorizer.fit_transform(documents)
    final_text_model = Ridge(alpha=20.0)
    final_text_model.fit(final_text_matrix, target_matrix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = args.output_dir / "candidate_bank.jsonl"
    selected_path = args.output_dir / "selector_oof_predictions.json"
    model_path = args.output_dir / "selector.joblib"
    report_path = args.output_dir / "selector_report.json"
    _atomic_write(bank_path, scored, jsonl=True)
    _atomic_write(selected_path, selected)
    joblib.dump(
        {
            "schema_version": "task2-p15-answer-selector-v1",
            "profiles": profiles,
            "selected_method": selected_method,
            "numeric_model": final_model,
            "text_vectorizer": final_vectorizer,
            "text_model": final_text_model,
        },
        model_path,
    )
    _atomic_write(
        report_path,
        {
            "schema_version": "task2-p15-answer-selector-report-v1",
            "question_count": len(question_ids),
            "candidate_count": len(scored),
            "profiles": profiles,
            "metric_profile": metric_profile,
            "metric_weights": {
                "meteor": args.meteor_weight,
                "rouge_l": args.rouge_weight,
            },
            "profile_metrics": profile_metrics,
            "selector_method": selected_method,
            "selector_method_metrics": method_metrics,
            "selector_oof_metrics": selector_metrics,
            "candidate_oracle_metrics": oracle_metrics,
            "selector_profile_counts": profile_counts,
            "selector_method_profile_counts": method_counts,
            "folds": args.folds,
            "seed": args.seed,
            "inputs": {
                "questions": str(args.questions),
                "question_ids": str(args.question_ids),
                "qwen": str(args.qwen),
                "extractive": str(args.extractive),
            },
            "artifacts": {
                "candidate_bank": str(bank_path),
                "candidate_bank_sha256": _sha256(bank_path),
                "selector_predictions": str(selected_path),
                "selector_predictions_sha256": _sha256(selected_path),
                "selector_model": str(model_path),
                "selector_model_sha256": _sha256(model_path),
            },
            "leakage_policy": (
                "selector features are reference-free; target scores are used only "
                "inside deterministic question-level cross-validation"
            ),
        },
    )
    print(
        "P15_SELECTOR_OOF "
        f"meteor={selector_metrics['meteor']:.6f} "
        f"rouge_l={selector_metrics['rouge_l']:.6f}",
        flush=True,
    )
    return [bank_path, selected_path, model_path, report_path]


def main() -> int:
    args = build_parser().parse_args()
    try:
        written = run(args)
    except (ImportError, LookupError, OSError, RuntimeError, ValueError) as exc:
        print(f"Task 2 P15 tuning error: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
