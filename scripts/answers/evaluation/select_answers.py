"""Build and cross-validate a reference-independent answer selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from udsc2026.evaluation.answer_candidates import (  # noqa: E402
    answer_features,
    build_answer_candidates,
    validate_candidate_bank,
)
from udsc2026.evaluation.answer_scoring import (  # noqa: E402
    fast_exact_meteor,
    official_meteor,
    official_rouge_l,
)
from udsc2026.evaluation.retrieval_metrics import (  # noqa: E402
    normalize_legal_ir_matching_question,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--extractive", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=("fast", "official"),
        default="official",
        help="Use fast exact-METEOR for search or official WordNet METEOR.",
    )
    parser.add_argument("--meteor-weight", type=float, default=1.0)
    parser.add_argument("--rouge-weight", type=float, default=0.0)
    return parser


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"expected non-empty object: {path}")
    return payload


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


def _load_grouped_folds(
    path: Path,
    question_ids: list[str],
    questions: dict[str, Any],
) -> tuple[dict[str, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "legal-qa-grouped-folds-v1":
        raise ValueError("fold manifest must use legal-qa-grouped-folds-v1")
    raw_folds = payload.get("folds")
    fold_count = payload.get("fold_count")
    if (
        not isinstance(raw_folds, list)
        or not isinstance(fold_count, int)
        or fold_count < 3
        or len(raw_folds) != fold_count
    ):
        raise ValueError("fold manifest has an invalid fold list or count")

    assignment: dict[str, int] = {}
    for expected_fold, item in enumerate(raw_folds):
        if not isinstance(item, dict) or item.get("fold") != expected_fold:
            raise ValueError("fold manifest indices must be ordered from zero")
        validation_ids = item.get("validation_ids")
        training_ids = item.get("training_ids")
        if not isinstance(validation_ids, list) or not isinstance(training_ids, list):
            raise ValueError("fold manifest must include validation and training IDs")
        if any(not isinstance(item, str) for item in validation_ids + training_ids):
            raise ValueError(f"fold {expected_fold} IDs must be strings")
        if len(validation_ids) != len(set(validation_ids)):
            raise ValueError(f"fold {expected_fold} contains duplicate validation IDs")
        if len(training_ids) != len(set(training_ids)):
            raise ValueError(f"fold {expected_fold} contains duplicate training IDs")
        for question_id in validation_ids:
            if not isinstance(question_id, str) or question_id in assignment:
                raise ValueError(f"fold manifest repeats or invalidates ID {question_id!r}")
            assignment[question_id] = expected_fold
        if set(training_ids) & set(validation_ids):
            raise ValueError(f"fold {expected_fold} training and validation IDs overlap")

    unknown = set(question_ids) - set(assignment)
    if unknown:
        raise ValueError(
            f"fold manifest is missing reference IDs: {sorted(unknown)[:5]}"
        )

    group_folds: dict[str, int] = {}
    for question_id in question_ids:
        record = questions.get(question_id)
        question = record.get("question") if isinstance(record, dict) else None
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question {question_id!r} lacks question text")
        group = normalize_legal_ir_matching_question(question)
        if not group:
            raise ValueError(f"question {question_id!r} normalizes to empty text")
        previous = group_folds.setdefault(group, assignment[question_id])
        if previous != assignment[question_id]:
            raise ValueError("normalized duplicate questions cross grouped folds")

    for fold_index, item in enumerate(raw_folds):
        expected_training = set(assignment) - set(item["validation_ids"])
        if set(item["training_ids"]) != expected_training:
            raise ValueError(f"fold {fold_index} training IDs do not match validation complement")

    metadata = {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": payload["schema_version"],
        "seed": payload.get("seed"),
        "fold_count": fold_count,
        "manifest_question_count": len(assignment),
        "question_count": len(question_ids),
        "normalized_question_group_count": len(group_folds),
        "population_match": set(assignment) == set(question_ids),
    }
    return assignment, metadata


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
    if (
        not math.isfinite(args.meteor_weight)
        or not math.isfinite(args.rouge_weight)
        or args.meteor_weight < 0
        or args.rouge_weight < 0
        or args.meteor_weight + args.rouge_weight <= 0
    ):
        raise ValueError("metric weights must be finite, nonnegative, and nonzero")
    questions = _load_object(args.questions)
    references = _load_object(args.references)
    question_ids = list(questions)
    if set(questions) != set(references):
        raise ValueError("question-only and reference mappings must have identical IDs")
    qwen = _load_predictions(args.qwen)
    extractive = _load_predictions(args.extractive)
    sources_to_check = (
        ("questions", questions),
        ("references", references),
        ("qwen", qwen),
        ("extractive", extractive),
    )
    for label, source in sources_to_check:
        missing = set(question_ids) - set(source)
        extra = set(source) - set(question_ids)
        if missing or extra:
            raise ValueError(
                f"{label} ID mismatch: missing={sorted(missing)[:5]}, "
                f"extra={sorted(extra)[:5]}"
            )
    fold_by_id, fold_metadata = _load_grouped_folds(
        args.fold_manifest,
        question_ids,
        questions,
    )

    bank: list[dict[str, Any]] = []
    sources: dict[str, tuple[str, str, str]] = {}
    for index, question_id in enumerate(question_ids, 1):
        row = questions[question_id]
        question = row.get("question") if isinstance(row, dict) else None
        reference_record = references[question_id]
        reference = (
            reference_record.get("answer")
            if isinstance(reference_record, dict)
            else None
        )
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
        reference = str(references[question_id]["answer"])
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
    matrix = np.asarray(features, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    documents = [
        _selector_document(*sources[question_id]) for question_id in question_ids
    ]
    fold_count = int(fold_metadata["fold_count"])
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

    if len(set(len(grouped[question_id]) for question_id in question_ids)) != 1:
        raise ValueError("each question must have the same answer candidate count")
    candidate_count = len(grouped[question_ids[0]])
    question_position = {question_id: index for index, question_id in enumerate(question_ids)}
    candidate_rows = {
        question_id: np.arange(
            question_position[question_id] * candidate_count,
            (question_position[question_id] + 1) * candidate_count,
        )
        for question_id in question_ids
    }

    def fit_scores(
        train_ids: list[str],
        valid_ids: list[str],
        *,
        random_state: int,
        n_estimators: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        train_rows = np.concatenate([candidate_rows[item] for item in train_ids])
        valid_rows = np.concatenate([candidate_rows[item] for item in valid_ids])
        numeric_model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=8,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
        numeric_model.fit(matrix[train_rows], target_array[train_rows])
        numeric_scores = numeric_model.predict(matrix[valid_rows]).reshape(
            len(valid_ids), candidate_count
        )

        train_positions = np.asarray(
            [question_position[item] for item in train_ids], dtype=np.int64
        )
        valid_positions = np.asarray(
            [question_position[item] for item in valid_ids], dtype=np.int64
        )
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=30_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        train_matrix = vectorizer.fit_transform(
            [documents[index] for index in train_positions]
        )
        valid_matrix = vectorizer.transform(
            [documents[index] for index in valid_positions]
        )
        text_model = Ridge(alpha=20.0)
        text_model.fit(train_matrix, target_matrix[train_positions])
        text_scores = text_model.predict(valid_matrix)
        return (
            {"numeric_extra_trees": numeric_scores, "semantic_ridge": text_scores},
            {
                "numeric_model": numeric_model,
                "text_vectorizer": vectorizer,
                "text_model": text_model,
            },
        )

    def combined_scores(score_parts: dict[str, Any]) -> dict[str, Any]:
        numeric = score_parts["numeric_extra_trees"]
        semantic = score_parts["semantic_ridge"]
        return {
            "numeric_extra_trees": numeric,
            "semantic_ridge": semantic,
            "hybrid_35_tree_65_text": 0.35 * numeric + 0.65 * semantic,
        }

    def choose(
        ids: list[str],
        scores: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        selected_rows: list[dict[str, Any]] = []
        counts: dict[str, int] = defaultdict(int)
        for index, question_id in enumerate(ids):
            candidates = grouped[question_id]
            batch_scores = scores[index]
            chosen_index = max(
                range(candidate_count),
                key=lambda idx: (float(batch_scores[idx]), -idx),
            )
            chosen = candidates[chosen_index]
            selected_rows.append(
                {
                    "id": question_id,
                    "answer": chosen["answer"],
                    "profile": chosen["profile"],
                }
            )
            counts[str(chosen["profile"])] += 1
        return selected_rows, counts

    def aggregate(rows: list[dict[str, str]]) -> dict[str, float]:
        meteor_values: list[float] = []
        rouge_values: list[float] = []
        for row in rows:
            reference = str(references[row["id"]]["answer"])
            meteor_values.append(float(meteor_scorer(reference, row["answer"])))
            rouge_values.append(float(rouge_scorer(reference, row["answer"])))
        return {
            "meteor": sum(meteor_values) / len(meteor_values),
            "rouge_l": sum(rouge_values) / len(rouge_values),
        }

    selected_rows_by_id: dict[str, dict[str, str]] = {}
    method_predictions: dict[str, list[dict[str, str]]] = {
        "numeric_extra_trees": [],
        "semantic_ridge": [],
        "hybrid_35_tree_65_text": [],
    }
    method_metrics: dict[str, dict[str, float]] = {}
    method_counts: dict[str, dict[str, int]] = {}
    selected_methods: list[str] = []
    fold_reports: list[dict[str, Any]] = []

    for outer_fold in range(fold_count):
        outer_train_ids = [
            question_id
            for question_id in question_ids
            if fold_by_id[question_id] != outer_fold
        ]
        outer_valid_ids = [
            question_id
            for question_id in question_ids
            if fold_by_id[question_id] == outer_fold
        ]
        inner_predictions: dict[str, list[dict[str, str]]] = {
            "numeric_extra_trees": [],
            "semantic_ridge": [],
            "hybrid_35_tree_65_text": [],
        }
        inner_fold_reports: list[dict[str, int]] = []
        for inner_fold in range(fold_count):
            if inner_fold == outer_fold:
                continue
            inner_train_ids = [
                question_id
                for question_id in outer_train_ids
                if fold_by_id[question_id] != inner_fold
            ]
            inner_valid_ids = [
                question_id
                for question_id in outer_train_ids
                if fold_by_id[question_id] == inner_fold
            ]
            if not inner_train_ids or not inner_valid_ids:
                raise ValueError(
                    f"fold {outer_fold} cannot form inner grouped fold {inner_fold}"
                )
            inner_parts, _ = fit_scores(
                inner_train_ids,
                inner_valid_ids,
                random_state=2026 + outer_fold * fold_count + inner_fold,
                n_estimators=120,
            )
            for method, scores in combined_scores(inner_parts).items():
                prediction_rows, _ = choose(inner_valid_ids, scores)
                inner_predictions[method].extend(prediction_rows)
            inner_fold_reports.append(
                {
                    "fold": inner_fold,
                    "train_count": len(inner_train_ids),
                    "validation_count": len(inner_valid_ids),
                }
            )
        inner_metrics = {
            method: aggregate(rows)
            for method, rows in inner_predictions.items()
        }
        selected_method = max(
            inner_metrics,
            key=lambda name: (
                args.meteor_weight * inner_metrics[name]["meteor"]
                + args.rouge_weight * inner_metrics[name]["rouge_l"],
                name,
            ),
        )
        selected_methods.append(selected_method)

        outer_parts, _ = fit_scores(
            outer_train_ids,
            outer_valid_ids,
            random_state=3026 + outer_fold,
            n_estimators=200,
        )
        outer_predictions_for_method: dict[str, list[dict[str, str]]] = {}
        for method, scores in combined_scores(outer_parts).items():
            prediction_rows, counts = choose(outer_valid_ids, scores)
            outer_predictions_for_method[method] = prediction_rows
            method_predictions[method].extend(prediction_rows)
            method_counts.setdefault(method, defaultdict(int))
            for profile, count in counts.items():
                method_counts[method][profile] += count
        chosen_outer = outer_predictions_for_method[selected_method]
        selected_rows_by_id.update(
            {row["id"]: row for row in chosen_outer}
        )
        fold_reports.append(
            {
                "fold": outer_fold,
                "outer_train_count": len(outer_train_ids),
                "outer_validation_count": len(outer_valid_ids),
                "inner_folds": inner_fold_reports,
                "selected_method": selected_method,
                "inner_validation_metrics": inner_metrics,
                "outer_metrics_by_method": {
                    method: aggregate(rows)
                    for method, rows in outer_predictions_for_method.items()
                },
                "outer_selected_metrics": aggregate(chosen_outer),
            }
        )
        print(
            f"nested_selector_fold={outer_fold + 1}/{fold_count} "
            f"train={len(outer_train_ids)} valid={len(outer_valid_ids)} "
            f"selected={selected_method}",
            flush=True,
        )

    selected = [selected_rows_by_id[question_id] for question_id in question_ids]
    method_metrics = {
        method: aggregate(rows) for method, rows in method_predictions.items()
    }
    deployment_method = max(
        Counter(selected_methods), key=lambda name: (Counter(selected_methods)[name], name)
    )
    selector_metrics = aggregate(selected)
    profile_counts: dict[str, int] = Counter(
        str(row["profile"]) for row in selected
    )

    selected_by_id = {str(row["id"]): row for row in selected}
    question_groups: dict[str, list[str]] = defaultdict(list)
    for question_id in question_ids:
        question = str(questions[question_id]["question"])
        question_groups[normalize_legal_ir_matching_question(question)].append(
            question_id
        )
    candidate_meteor_by_id = {
        str(row["id"]): float(row["meteor"])
        for row in scored
        if row["profile"] == selected_by_id[str(row["id"])]["profile"]
    }

    def paired_bootstrap(baseline: dict[str, str]) -> dict[str, Any]:
        baseline_scores = {
            question_id: float(
                meteor_scorer(
                    str(references[question_id]["answer"]), baseline[question_id]
                )
            )
            for question_id in question_ids
        }
        group_deltas = [
            (
                sum(
                    candidate_meteor_by_id[question_id]
                    - baseline_scores[question_id]
                    for question_id in ids
                ),
                len(ids),
            )
            for ids in question_groups.values()
        ]
        rng = random.Random(2026)
        bootstrap_deltas: list[float] = []
        for _ in range(10_000):
            sample = [
                group_deltas[rng.randrange(len(group_deltas))]
                for _ in range(len(group_deltas))
            ]
            bootstrap_deltas.append(
                sum(delta for delta, _ in sample) / sum(count for _, count in sample)
            )
        bootstrap_deltas.sort()
        baseline_mean = sum(baseline_scores.values()) / len(baseline_scores)
        probability = sum(value > 0 for value in bootstrap_deltas) / len(
            bootstrap_deltas
        )
        return {
            "metric": "organizer_meteor",
            "scope": "nested_grouped_oof_paired_bootstrap_internal",
            "baseline_mean": baseline_mean,
            "candidate_mean": selector_metrics["meteor"],
            "observed_delta": selector_metrics["meteor"] - baseline_mean,
            "bootstrap_probability_delta_positive": probability,
            "bootstrap_delta_interval_95": [
                bootstrap_deltas[int((len(bootstrap_deltas) - 1) * 0.025)],
                bootstrap_deltas[int((len(bootstrap_deltas) - 1) * 0.975)],
            ],
            "group_count": len(group_deltas),
            "draws": len(bootstrap_deltas),
            "seed": 2026,
            "minimum_groups_for_gate": 100,
            "passes_internal_70_percent_gate": (
                len(group_deltas) >= 100
                and selector_metrics["meteor"] > baseline_mean
                and probability >= 0.70
            ),
        }

    selector_vs_qwen = paired_bootstrap(qwen)
    selector_vs_extractive = paired_bootstrap(extractive)

    _, final_models = fit_scores(
        question_ids,
        question_ids,
        random_state=4026,
        n_estimators=300,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = args.output_dir / "candidate_bank.jsonl"
    selected_path = args.output_dir / "selector_oof_predictions.json"
    model_path = args.output_dir / "selector.joblib"
    report_path = args.output_dir / "selector_report.json"
    _atomic_write(bank_path, scored, jsonl=True)
    _atomic_write(
        selected_path,
        [{"id": row["id"], "answer": row["answer"]} for row in selected],
    )
    joblib.dump(
        {
            "schema_version": "answer-selector-v1",
            "profiles": profiles,
            "selected_method": deployment_method,
            **final_models,
        },
        model_path,
    )
    _atomic_write(
        report_path,
        {
            "schema_version": "answer-selector-report-v1",
            "evaluation_scope": "nested_grouped_oof",
            "question_count": len(question_ids),
            "candidate_count": len(scored),
            "population_match": fold_metadata["population_match"],
            "profiles": profiles,
            "metric_profile": metric_profile,
            "metric_weights": {
                "meteor": args.meteor_weight,
                "rouge_l": args.rouge_weight,
            },
            "selector_method": deployment_method,
            "selector_method_metrics": method_metrics,
            "selector_oof_metrics": selector_metrics,
            "selector_vs_qwen": selector_vs_qwen,
            "selector_vs_extractive": selector_vs_extractive,
            "nested_selected_methods": dict(Counter(selected_methods)),
            "selector_profile_counts": dict(sorted(profile_counts.items())),
            "selector_method_profile_counts": {
                method: dict(sorted(counts.items()))
                for method, counts in method_counts.items()
            },
            "fold_manifest": fold_metadata,
            "inputs": {
                "questions": str(args.questions),
                "questions_sha256": _sha256(args.questions),
                "references": str(args.references),
                "references_sha256": _sha256(args.references),
                "qwen": str(args.qwen),
                "qwen_sha256": _sha256(args.qwen),
                "extractive": str(args.extractive),
                "extractive_sha256": _sha256(args.extractive),
            },
            "fold_reports": fold_reports,
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
                "inside deterministic nested grouped cross-validation"
            ),
        },
    )
    print(
        "ANSWER_SELECTOR_OOF "
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
        print(f"answer selector error: {exc}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
