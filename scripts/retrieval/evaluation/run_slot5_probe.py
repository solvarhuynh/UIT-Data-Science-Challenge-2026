"""Probe a leakage-safe slot-five rescue on cached LegalIR rerankings.

The probe consumes completed reranker JSONL files only.  It never loads a
neural model, changes the corpus, or uses the held-out fold's references to
choose a rule.  For every outer fold, document priors are fitted on the other
four folds and the guard parameters are selected on the corresponding inner
folds.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation.models import PredictionSample
from udsc2026.evaluation.retrieval_candidates import (
    AggregatedDocument,
    aggregate_document_candidates,
)
from udsc2026.evaluation.retrieval_metrics import (
    LegalIRPrediction,
    LegalIRReference,
    evaluate_legal_ir,
    load_warmup,
)


METHOD = "rrf_cap2_dense_signal"
SCORE_FIELD = "rerank_score"
FINAL_DOCUMENTS = 5
PRIOR_ALPHA = 1.0
PRIOR_BETA = 5.0
EPSILON = 1e-12


@dataclass(frozen=True)
class QueryBundle:
    """Compact document features retained after one JSONL line is parsed."""

    question_id: str
    documents: tuple[AggregatedDocument, ...]


@dataclass(frozen=True)
class RuleParams:
    """One candidate rescue rule selected only on an inner validation split."""

    prior_weight: float
    raw_score_ratio: float
    minimum_prior: float
    maximum_candidate_rank: int

    def as_dict(self) -> dict[str, object]:
        return {
            "prior_weight": self.prior_weight,
            "raw_score_ratio": self.raw_score_ratio,
            "minimum_prior": self.minimum_prior,
            "maximum_candidate_rank": self.maximum_candidate_rank,
        }


@dataclass(frozen=True)
class PriorStats:
    candidate_count: dict[str, int]
    gold_count: dict[str, int]

    def value(self, document_id: str) -> float:
        candidate_count = self.candidate_count.get(document_id, 0)
        gold_count = self.gold_count.get(document_id, 0)
        return (gold_count + PRIOR_ALPHA) / (candidate_count + PRIOR_BETA)


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
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return "unavailable"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unavailable"


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc


def _load_folds(path: Path) -> dict[str, int]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != "legal-ir-strict-cv-v2":
        raise ValueError(f"{path} is not a strict grouped-fold artifact")
    rows = payload.get("folds")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError(f"{path} must contain exactly five folds")
    result: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path} fold rows must be objects")
        fold = row.get("fold")
        validation_ids = row.get("validation_ids")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
            raise ValueError(f"{path} has an invalid fold number")
        if not isinstance(validation_ids, list) or not all(
            isinstance(item, str) for item in validation_ids
        ):
            raise ValueError(f"{path} validation_ids must be string arrays")
        for question_id in validation_ids:
            if question_id in result:
                raise ValueError(f"question {question_id} appears in multiple folds")
            result[question_id] = fold
    if not result:
        raise ValueError(f"{path} contains no grouped-fold questions")
    return result


def _load_bundles(path: Path, expected_ids: set[str]) -> dict[str, QueryBundle]:
    if not path.is_file():
        raise FileNotFoundError(f"reranked input is missing: {path}")
    result: dict[str, QueryBundle] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                sample = PredictionSample.model_validate_json(line, strict=True)
            except ValueError as exc:
                raise ValueError(f"invalid reranked record at {path}:{line_number}") from exc
            if sample.question_id in result:
                raise ValueError(f"duplicate question ID {sample.question_id} in {path}")
            documents = tuple(
                aggregate_document_candidates(
                    sample,
                    method=METHOD,
                    evidence_limit=3,
                    rrf_k=60,
                    dense_rrf_k=30,
                    dense_signal_weight=0.30,
                    score_field=SCORE_FIELD,
                )
            )
            if len(documents) < FINAL_DOCUMENTS:
                raise ValueError(
                    f"{sample.question_id} has only {len(documents)} aggregated documents"
                )
            result[sample.question_id] = QueryBundle(sample.question_id, documents)
    if set(result) != expected_ids:
        missing = sorted(expected_ids - set(result))[:5]
        extra = sorted(set(result) - expected_ids)[:5]
        raise ValueError(f"reranked coverage mismatch in {path}: missing={missing}, extra={extra}")
    return result


def _fit_prior_stats(
    bundles: dict[str, QueryBundle],
    references: dict[str, LegalIRReference],
    question_ids: Iterable[str],
) -> PriorStats:
    candidate_count: Counter[str] = Counter()
    gold_count: Counter[str] = Counter()
    for question_id in question_ids:
        bundle = bundles[question_id]
        candidate_count.update(document.doc_id for document in bundle.documents)
        gold_count.update(references[question_id].gold_documents)
    return PriorStats(dict(candidate_count), dict(gold_count))


def _baseline_documents(bundle: QueryBundle) -> tuple[AggregatedDocument, ...]:
    return bundle.documents[:FINAL_DOCUMENTS]


def _rescue_documents(
    bundle: QueryBundle,
    stats: PriorStats,
    params: RuleParams,
) -> tuple[tuple[str, ...], dict[str, object]]:
    baseline = _baseline_documents(bundle)
    baseline_slot = baseline[-1]
    candidates: list[tuple[float, float, int, AggregatedDocument, float]] = []
    for rank, document in enumerate(bundle.documents[FINAL_DOCUMENTS:], start=FINAL_DOCUMENTS + 1):
        if rank > params.maximum_candidate_rank:
            break
        prior = stats.value(document.doc_id)
        raw_score = document.aggregation_score
        if raw_score + EPSILON < baseline_slot.aggregation_score * params.raw_score_ratio:
            continue
        if prior + EPSILON < params.minimum_prior:
            continue
        adjusted = raw_score + params.prior_weight * prior
        candidates.append((adjusted, raw_score, rank, document, prior))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3].doc_id))
    selected = baseline
    swap = None
    if candidates and candidates[0][0] > baseline_slot.aggregation_score + EPSILON:
        chosen = candidates[0]
        selected = (*baseline[: FINAL_DOCUMENTS - 1], chosen[3])
        swap = {
            "from": baseline_slot.doc_id,
            "to": chosen[3].doc_id,
            "from_aggregation_score": baseline_slot.aggregation_score,
            "to_aggregation_score": chosen[3].aggregation_score,
            "to_adjusted_score": chosen[0],
            "to_prior": chosen[4],
            "to_candidate_rank": chosen[2],
        }
    documents = tuple(item.doc_id for item in selected)
    if len(documents) != FINAL_DOCUMENTS or len(set(documents)) != FINAL_DOCUMENTS:
        raise AssertionError(f"invalid slot-five output for {bundle.question_id}")
    return documents, {"swapped": swap is not None, "swap": swap}


def _score_documents(
    question_ids: Sequence[str],
    bundles: dict[str, QueryBundle],
    references: dict[str, LegalIRReference],
    stats: PriorStats,
    params: RuleParams,
    *,
    collect_audit: bool = False,
) -> tuple[float, float, int, list[dict[str, object]]]:
    recalls: list[float] = []
    precisions: list[float] = []
    swap_count = 0
    audit: list[dict[str, object]] = []
    for question_id in question_ids:
        documents, info = _rescue_documents(bundles[question_id], stats, params)
        gold = set(references[question_id].gold_documents)
        matched = len(gold.intersection(documents))
        recalls.append(matched / len(gold))
        precisions.append(matched / len(documents))
        if info["swapped"]:
            swap_count += 1
        if collect_audit:
            audit.append(
                {
                    "id": question_id,
                    "documents": list(documents),
                    "baseline_documents": [
                        item.doc_id for item in _baseline_documents(bundles[question_id])
                    ],
                    **info,
                }
            )
    return (
        sum(recalls) / len(recalls),
        sum(precisions) / len(precisions),
        swap_count,
        audit,
    )


def _parameter_grid() -> tuple[RuleParams, ...]:
    return tuple(
        RuleParams(*values)
        for values in itertools.product(
            (0.0, 0.01, 0.02, 0.04),
            (0.70, 0.85, 0.95),
            (0.0, 0.005, 0.01),
            (10, 20, 40),
        )
    )


def _select_params(
    bundles: dict[str, QueryBundle],
    references: dict[str, LegalIRReference],
    fold_by_id: dict[str, int],
    *,
    outer_fold: int,
) -> tuple[RuleParams, list[dict[str, object]]]:
    inner_folds = [fold for fold in range(5) if fold != outer_fold]
    inner_data: list[tuple[list[str], PriorStats]] = []
    for inner_fold in inner_folds:
        train_ids = [
            question_id
            for question_id, fold in fold_by_id.items()
            if fold != outer_fold and fold != inner_fold
        ]
        validation_ids = [
            question_id
            for question_id, fold in fold_by_id.items()
            if fold == inner_fold
        ]
        inner_data.append(
            (
                validation_ids,
                _fit_prior_stats(bundles, references, train_ids),
            )
        )
    scores: list[dict[str, object]] = []
    for params in _parameter_grid():
        inner_recall: list[float] = []
        inner_precision: list[float] = []
        inner_swaps: list[int] = []
        for validation_ids, stats in inner_data:
            recall, precision, swaps, _ = _score_documents(
                validation_ids, bundles, references, stats, params
            )
            inner_recall.append(recall)
            inner_precision.append(precision)
            inner_swaps.append(swaps)
        scores.append(
            {
                "params": params.as_dict(),
                "recall": sum(inner_recall) / len(inner_recall),
                "precision": sum(inner_precision) / len(inner_precision),
                "swaps": sum(inner_swaps),
            }
        )
    scores.sort(
        key=lambda row: (
            -float(row["recall"]),
            -float(row["precision"]),
            int(row["swaps"]),
            json.dumps(row["params"], sort_keys=True),
        )
    )
    chosen = scores[0]["params"]
    if not isinstance(chosen, dict):
        raise AssertionError("selected parameter row is malformed")
    return (
        RuleParams(
            prior_weight=float(chosen["prior_weight"]),
            raw_score_ratio=float(chosen["raw_score_ratio"]),
            minimum_prior=float(chosen["minimum_prior"]),
            maximum_candidate_rank=int(chosen["maximum_candidate_rank"]),
        ),
        scores[:10],
    )


def _official_score(
    references: Sequence[LegalIRReference],
    bundles: dict[str, QueryBundle],
    stats: PriorStats,
    params: RuleParams | None,
) -> tuple[Any, list[dict[str, object]]]:
    predictions: list[LegalIRPrediction] = []
    audit: list[dict[str, object]] = []
    for reference in references:
        bundle = bundles[reference.id]
        if params is None:
            documents = tuple(item.doc_id for item in _baseline_documents(bundle))
            info: dict[str, object] = {"swapped": False, "swap": None}
        else:
            documents, info = _rescue_documents(bundle, stats, params)
        predictions.append(LegalIRPrediction(id=reference.id, documents=list(documents)))
        audit.append(
            {
                "id": reference.id,
                "documents": list(documents),
                "baseline_documents": [item.doc_id for item in _baseline_documents(bundle)],
                **info,
            }
        )
    return evaluate_legal_ir(references, predictions), audit


def _bootstrap_probability(
    by_fold: dict[int, list[tuple[float, float]]],
    *,
    seed: int = 2026,
    draws: int = 10000,
) -> dict[str, float]:
    import random

    rng = random.Random(seed)
    fold_ids = sorted(by_fold)
    if len(fold_ids) < 2:
        raise ValueError("paired grouped bootstrap requires at least two folds")
    positive_recall = 0
    positive_precision = 0
    for _ in range(draws):
        sampled = [rng.choice(fold_ids) for _ in fold_ids]
        deltas = [delta for fold in sampled for delta in by_fold[fold]]
        recall_delta = sum(item[0] for item in deltas) / len(deltas)
        precision_delta = sum(item[1] for item in deltas) / len(deltas)
        positive_recall += recall_delta > 0.0
        positive_precision += precision_delta > 0.0
    return {
        "draws": draws,
        "seed": seed,
        "grouping": "strict_validation_fold",
        "p_recall_delta_gt_zero": positive_recall / draws,
        "p_precision_delta_gt_zero": positive_precision / draws,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--reranked-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-name", default="reranked.jsonl")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Path:
    references_list = load_warmup(args.gold)
    references = {
        sample.id: LegalIRReference(id=sample.id, gold_documents=list(sample.gold_documents))
        for sample in references_list
    }
    fold_by_id = _load_folds(args.folds)
    if set(references) != set(fold_by_id):
        raise ValueError("gold and strict folds do not have identical question IDs")
    if args.output_root.exists():
        if not args.overwrite:
            raise ValueError(f"output already exists, pass --overwrite: {args.output_root}")
        if not args.output_root.is_dir() or args.output_root.resolve() == Path("/"):
            raise ValueError("refusing to remove an unsafe output path")
        import shutil

        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=False)

    bundles: dict[str, QueryBundle] = {}
    source_paths: list[Path] = []
    for fold in range(5):
        path = args.reranked_root / f"fold_{fold}" / args.input_name
        expected_ids = {question_id for question_id, value in fold_by_id.items() if value == fold}
        bundles.update(_load_bundles(path, expected_ids))
        source_paths.append(path)

    baseline_by_fold: dict[int, Any] = {}
    rescue_by_fold: dict[int, Any] = {}
    baseline_prediction_rows: dict[int, list[LegalIRPrediction]] = {}
    rescue_prediction_rows: dict[int, list[LegalIRPrediction]] = {}
    all_audit: list[dict[str, object]] = []
    fold_reports: list[dict[str, object]] = []
    paired_deltas: dict[int, list[tuple[float, float]]] = defaultdict(list)

    for outer_fold in range(5):
        validation_ids = [qid for qid, fold in fold_by_id.items() if fold == outer_fold]
        validation_references = [references[qid] for qid in validation_ids]
        params, inner_top = _select_params(
            bundles, references, fold_by_id, outer_fold=outer_fold
        )
        train_ids = [qid for qid, fold in fold_by_id.items() if fold != outer_fold]
        stats = _fit_prior_stats(bundles, references, train_ids)
        baseline_report, baseline_audit = _official_score(
            validation_references, bundles, stats, None
        )
        rescue_report, rescue_audit = _official_score(
            validation_references, bundles, stats, params
        )
        baseline_by_fold[outer_fold] = baseline_report
        rescue_by_fold[outer_fold] = rescue_report
        baseline_prediction_rows[outer_fold] = [
            LegalIRPrediction(id=row["id"], documents=list(row["documents"]))
            for row in baseline_audit
        ]
        rescue_prediction_rows[outer_fold] = [
            LegalIRPrediction(id=row["id"], documents=list(row["documents"]))
            for row in rescue_audit
        ]
        baseline_per_id = {item.id: item for item in baseline_report.per_query}
        rescue_per_id = {item.id: item for item in rescue_report.per_query}
        for question_id in validation_ids:
            paired_deltas[outer_fold].append(
                (
                    rescue_per_id[question_id].recall - baseline_per_id[question_id].recall,
                    rescue_per_id[question_id].precision - baseline_per_id[question_id].precision,
                )
            )
        for row in rescue_audit:
            row["strict_cv_validation_fold"] = outer_fold
            row["selected_params"] = params.as_dict()
            all_audit.append(row)
        output_fold = args.output_root / f"fold_{outer_fold}"
        _write_json(output_fold / "predictions.json", [
            {"id": row["id"], "documents": row["documents"]} for row in rescue_audit
        ])
        _write_json(output_fold / "manifest.json", {
            "schema_version": "legal-ir-slot5-fold-v1",
            "fold": outer_fold,
            "question_count": len(validation_ids),
            "selected_params": params.as_dict(),
            "source_reranked": str(source_paths[outer_fold]),
            "source_reranked_sha256": _sha256(source_paths[outer_fold]),
            "strict_folds_sha256": _sha256(args.folds),
            "gold_sha256": _sha256(args.gold),
        })
        fold_reports.append({
            "fold": outer_fold,
            "sample_count": len(validation_ids),
            "selected_params": params.as_dict(),
            "inner_top_candidates": inner_top,
            "baseline": {
                "recall": baseline_report.aggregate.recall,
                "precision": baseline_report.aggregate.precision,
            },
            "slot5": {
                "recall": rescue_report.aggregate.recall,
                "precision": rescue_report.aggregate.precision,
                "swap_count": sum(1 for row in rescue_audit if row["swapped"]),
            },
        })

    baseline_predictions = [
        item for fold in range(5) for item in baseline_prediction_rows[fold]
    ]
    rescue_predictions = [
        item for fold in range(5) for item in rescue_prediction_rows[fold]
    ]
    ordered_references = [references[qid] for qid in fold_by_id]
    baseline_report = evaluate_legal_ir(ordered_references, baseline_predictions)
    rescue_report = evaluate_legal_ir(ordered_references, rescue_predictions)
    bootstrap = _bootstrap_probability(paired_deltas)
    summary = {
        "schema_version": "legal-ir-slot5-probe-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "method": METHOD,
        "score_field": SCORE_FIELD,
        "final_documents": FINAL_DOCUMENTS,
        "question_count": len(ordered_references),
        "nested_grouped_cv": {
            "outer_folds": 5,
            "inner_folds": 4,
            "selection": "outer-train document prior, inner grouped-fold macro Recall then Precision",
            "gold_used_for_validation_rule": False,
        },
        "inputs": {
            "gold": str(args.gold),
            "gold_sha256": _sha256(args.gold),
            "folds": str(args.folds),
            "folds_sha256": _sha256(args.folds),
            "reranked": [
                {"path": str(path), "sha256": _sha256(path)} for path in source_paths
            ],
        },
        "baseline": baseline_report.model_dump(mode="json"),
        "slot5": rescue_report.model_dump(mode="json"),
        "delta": {
            "recall": rescue_report.aggregate.recall - baseline_report.aggregate.recall,
            "precision": rescue_report.aggregate.precision - baseline_report.aggregate.precision,
        },
        "paired_grouped_bootstrap": bootstrap,
        "folds": fold_reports,
        "swap_count": sum(1 for row in all_audit if row["swapped"]),
    }
    _write_json(args.output_root / "summary.json", summary)
    _write_jsonl(args.output_root / "audit.jsonl", all_audit)
    _write_json(args.output_root / "baseline.json", [
        item.model_dump(mode="json") for item in baseline_predictions
    ])
    _write_json(args.output_root / "predictions.json", [
        item.model_dump(mode="json") for item in rescue_predictions
    ])
    return args.output_root / "summary.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"LegalIR slot-five probe error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
