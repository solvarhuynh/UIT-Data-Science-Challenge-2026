"""Analyze cached LegalIR chunks and early document aggregation without models."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from udsc2026.evaluation import (  # noqa: E402
    AGGREGATION_METHODS,
    AggregatedDocument,
    LegalIRReference,
    PredictionSample,
    aggregate_document_candidates,
    aggregate_to_legal_ir_prediction,
    document_ids_from_hits,
    evaluate_legal_ir,
    load_predictions,
    load_warmup,
)

DEFAULT_K_VALUES = (5, 10, 20, 50, 100, 200)
DEFAULT_STRICT_FOLDS = Path("artifacts/task1/evaluation/strict_cv_v2/folds.json")


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _infer_manifest_path(predictions: Path) -> Path:
    if "_predictions" not in predictions.stem:
        raise ValueError(
            "cannot infer manifest; pass --manifest for predictions without "
            "'_predictions' in the filename"
        )
    return predictions.with_name(
        predictions.stem.replace("_predictions", "_manifest") + ".json"
    )


def _validate_cached_manifest(
    manifest_path: Path,
    predictions_path: Path,
    *,
    max_k: int,
    prediction_count: int,
) -> dict[str, Any]:
    """Validate that a cached candidate manifest still names this artifact."""

    manifest = _read_json(manifest_path)
    candidate_k = manifest.get("candidate_k")
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int):
        raise ValueError(f"{manifest_path} has no integer candidate_k")
    if candidate_k < max_k:
        raise ValueError(
            f"{manifest_path} candidate_k={candidate_k} is shallower than "
            f"requested K={max_k}"
        )
    if manifest.get("question_count") != prediction_count:
        raise ValueError(
            f"{manifest_path} question_count does not match predictions "
            f"({manifest.get('question_count')} != {prediction_count})"
        )
    output = manifest.get("output")
    if not isinstance(output, str):
        raise ValueError(f"{manifest_path} has no output path")
    if (PROJECT_ROOT / output).resolve() != predictions_path.resolve():
        raise ValueError(f"{manifest_path} output does not match --predictions")
    benchmark = manifest.get("benchmark")
    benchmark_hash = manifest.get("benchmark_sha256")
    if not isinstance(benchmark, str) or not isinstance(benchmark_hash, str):
        raise ValueError(f"{manifest_path} has no benchmark hash metadata")
    benchmark_path = (PROJECT_ROOT / benchmark).resolve()
    if not benchmark_path.is_file():
        raise FileNotFoundError(
            f"manifest benchmark required for hash validation is missing: "
            f"{benchmark_path}"
        )
    observed_hash = _sha256(benchmark_path)
    if observed_hash != benchmark_hash:
        raise ValueError(
            f"{manifest_path} benchmark hash mismatch: "
            f"{observed_hash} != {benchmark_hash}"
        )
    return manifest


def _load_strict_fold_map(path: Path) -> dict[str, int]:
    payload = _read_json(path)
    if payload.get("schema_version") != "legal-ir-strict-cv-v2":
        raise ValueError(f"{path} is not a P1 strict_cv_v2 folds artifact")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError(f"{path} must contain exactly five strict folds")
    mapping: dict[str, int] = {}
    for row in folds:
        if not isinstance(row, dict):
            raise ValueError(f"{path} fold records must be objects")
        fold = row.get("fold")
        validation_ids = row.get("validation_ids")
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError(f"{path} fold number must be an integer")
        if not isinstance(validation_ids, list) or not all(
            isinstance(item, str) for item in validation_ids
        ):
            raise ValueError(f"{path} validation_ids must be strings")
        for question_id in validation_ids:
            if question_id in mapping:
                raise ValueError(
                    f"{path} assigns question ID to multiple validation folds: "
                    f"{question_id}"
                )
            mapping[question_id] = fold
    return mapping


def _select_inputs(
    gold_path: Path,
    predictions_path: Path,
    *,
    strict_fold_map: dict[str, int],
    strict_fold: int | None,
    max_questions: int | None,
) -> tuple[list[LegalIRReference], list[PredictionSample], dict[str, int]]:
    predictions = load_predictions(predictions_path)
    if strict_fold is not None:
        predictions = [
            prediction
            for prediction in predictions
            if strict_fold_map.get(prediction.question_id) == strict_fold
        ]
    if max_questions is not None:
        predictions = predictions[:max_questions]
    if not predictions:
        raise ValueError("prediction selection is empty")

    gold_by_id = {
        sample.id: LegalIRReference(
            id=sample.id,
            gold_documents=list(sample.gold_documents),
        )
        for sample in load_warmup(gold_path)
    }
    missing_gold = sorted(
        prediction.question_id
        for prediction in predictions
        if prediction.question_id not in gold_by_id
    )
    if missing_gold:
        raise ValueError(
            "cached predictions contain question IDs absent from --gold: "
            + ", ".join(missing_gold[:5])
        )
    missing_folds = sorted(
        prediction.question_id
        for prediction in predictions
        if prediction.question_id not in strict_fold_map
    )
    if missing_folds:
        raise ValueError(
            "cached predictions are absent from P1 strict CV folds: "
            + ", ".join(missing_folds[:5])
        )
    references = [gold_by_id[prediction.question_id] for prediction in predictions]
    selected_folds = {
        prediction.question_id: strict_fold_map[prediction.question_id]
        for prediction in predictions
    }
    return references, predictions, selected_folds


def _first_gold_ranks(
    prediction: PredictionSample,
    gold_documents: Sequence[str],
) -> tuple[dict[str, int | None], dict[str, int | None]]:
    gold = set(gold_documents)
    chunk_rank: dict[str, int | None] = {document: None for document in gold_documents}
    document_rank: dict[str, int | None] = {
        document: None for document in gold_documents
    }
    seen_documents: set[str] = set()
    for list_rank, hit in enumerate(prediction.hits, start=1):
        rank = hit.rank if hit.rank is not None else list_rank
        if hit.doc_id in gold and chunk_rank[hit.doc_id] is None:
            chunk_rank[hit.doc_id] = rank
        if hit.doc_id not in seen_documents:
            seen_documents.add(hit.doc_id)
            if hit.doc_id in gold and document_rank[hit.doc_id] is None:
                document_rank[hit.doc_id] = len(seen_documents)
    return chunk_rank, document_rank


def _candidate_metrics(
    references: Sequence[LegalIRReference],
    predictions: Sequence[PredictionSample],
    k_values: Sequence[int],
) -> dict[str, Any]:
    by_prediction = {prediction.question_id: prediction for prediction in predictions}
    result: dict[str, Any] = {
        "candidate_doc_recall_at_k": {},
        "mean_unique_docs_at_chunk_depth": {},
        "candidate_depth_complete_query_count": {},
        "multi_gold_candidate_doc_recall_at_k": {},
        "multi_gold_all_gold_covered_query_count_at_k": {},
    }
    multi_gold_references = [
        reference for reference in references if len(reference.gold_documents) > 1
    ]
    for depth in k_values:
        recalls: list[float] = []
        unique_counts: list[int] = []
        complete_count = 0
        multi_recalls: list[float] = []
        multi_complete = 0
        for reference in references:
            prediction = by_prediction[reference.id]
            documents = document_ids_from_hits(prediction, chunk_depth=depth)
            gold = set(reference.gold_documents)
            recalls.append(len(gold.intersection(documents)) / len(gold))
            unique_counts.append(len(documents))
            if len(prediction.hits) >= depth:
                complete_count += 1
            if len(reference.gold_documents) > 1:
                multi_recalls.append(len(gold.intersection(documents)) / len(gold))
                if gold.issubset(documents):
                    multi_complete += 1
        key = str(depth)
        result["candidate_doc_recall_at_k"][key] = sum(recalls) / len(recalls)
        result["mean_unique_docs_at_chunk_depth"][key] = sum(unique_counts) / len(
            unique_counts
        )
        result["candidate_depth_complete_query_count"][key] = complete_count
        result["multi_gold_candidate_doc_recall_at_k"][key] = (
            sum(multi_recalls) / len(multi_recalls) if multi_recalls else None
        )
        result["multi_gold_all_gold_covered_query_count_at_k"][key] = multi_complete
    result["multi_gold_sample_count"] = len(multi_gold_references)
    return result


def _aggregation_reports(
    references: Sequence[LegalIRReference],
    predictions: Sequence[PredictionSample],
    *,
    final_documents: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[AggregatedDocument]]]]:
    reports: dict[str, dict[str, Any]] = {}
    documents_by_method: dict[str, dict[str, list[AggregatedDocument]]] = {}
    for method in AGGREGATION_METHODS:
        grouped = {
            prediction.question_id: aggregate_document_candidates(
                prediction,
                method=method,
            )
            for prediction in predictions
        }
        official_predictions = [
            aggregate_to_legal_ir_prediction(
                prediction,
                method=method,
                max_documents=final_documents,
            )
            for prediction in predictions
        ]
        report = evaluate_legal_ir(references, official_predictions)
        reports[method] = {
            "official_recall": report.aggregate.recall,
            "official_precision": report.aggregate.precision,
            "multi_gold_recall": (
                sum(
                    diagnostic.recall
                    for diagnostic in report.per_query
                    if len(diagnostic.gold_documents) > 1
                )
                / report.aggregate.multi_gold_sample_count
                if report.aggregate.multi_gold_sample_count
                else None
            ),
        }
        documents_by_method[method] = grouped
    return reports, documents_by_method


def _failure_records(
    references: Sequence[LegalIRReference],
    predictions: Sequence[PredictionSample],
    *,
    selected_documents: dict[str, list[AggregatedDocument]],
    strict_folds: dict[str, int],
    final_documents: int,
) -> list[dict[str, Any]]:
    by_prediction = {prediction.question_id: prediction for prediction in predictions}
    records: list[dict[str, Any]] = []
    for reference in references:
        prediction = by_prediction[reference.id]
        aggregated = selected_documents[reference.id]
        final = aggregated[:final_documents]
        candidate_documents = set(document_ids_from_hits(prediction))
        final_documents_set = {document.doc_id for document in final}
        gold = set(reference.gold_documents)
        candidate_matched = sorted(gold.intersection(candidate_documents))
        final_matched = sorted(gold.intersection(final_documents_set))
        missing = sorted(gold.difference(candidate_documents))
        chunk_ranks, document_ranks = _first_gold_ranks(
            prediction,
            reference.gold_documents,
        )
        if not candidate_matched:
            bucket = "retrieval_miss"
        elif len(reference.gold_documents) > 1 and 0 < len(final_matched) < len(
            reference.gold_documents
        ):
            bucket = "multi_gold_partial"
        elif not final_matched:
            bucket = "candidate_present_final_miss"
        else:
            bucket = "success"
        known_chunk_ranks = [rank for rank in chunk_ranks.values() if rank is not None]
        known_document_ranks = [
            rank for rank in document_ranks.values() if rank is not None
        ]
        records.append(
            {
                "id": reference.id,
                "strict_cv_validation_fold": strict_folds[reference.id],
                "bucket": bucket,
                "gold_documents": list(reference.gold_documents),
                "candidate_matched_gold_documents": candidate_matched,
                "final_matched_gold_documents": final_matched,
                "gold_missing_entirely_documents": missing,
                "gold_first_chunk_rank": chunk_ranks,
                "gold_first_document_rank": document_ranks,
                "first_gold_chunk_rank": min(known_chunk_ranks, default=None),
                "first_gold_document_rank": min(known_document_ranks, default=None),
                "candidate_chunk_count": len(prediction.hits),
                "candidate_unique_document_count": len(
                    document_ids_from_hits(prediction)
                ),
                "final_documents": [document.doc_id for document in final],
                "final_aggregated_documents": [
                    document.model_dump() for document in final
                ],
            }
        )
    return records


def _strict_fold_metrics(
    references: Sequence[LegalIRReference],
    predictions: Sequence[PredictionSample],
    *,
    strict_folds: dict[str, int],
    aggregation: str,
    final_documents: int,
) -> dict[str, Any]:
    predictions_by_id = {
        prediction.question_id: prediction for prediction in predictions
    }
    result: dict[str, Any] = {}
    for fold in range(5):
        fold_references = [
            reference for reference in references if strict_folds[reference.id] == fold
        ]
        if not fold_references:
            result[str(fold)] = {"sample_count": 0}
            continue
        fold_predictions = [
            aggregate_to_legal_ir_prediction(
                predictions_by_id[reference.id],
                method=aggregation,  # type: ignore[arg-type]
                max_documents=final_documents,
            )
            for reference in fold_references
        ]
        report = evaluate_legal_ir(fold_references, fold_predictions)
        result[str(fold)] = {
            "sample_count": len(fold_references),
            "official_recall": report.aggregate.recall,
            "official_precision": report.aggregate.precision,
        }
    return result


def _rank_diagnostics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize rank-space diagnostics while keeping per-gold values in JSONL."""

    chunk_ranks = [
        rank
        for record in records
        for rank in record["gold_first_chunk_rank"].values()
        if rank is not None
    ]
    document_ranks = [
        rank
        for record in records
        for rank in record["gold_first_document_rank"].values()
        if rank is not None
    ]
    missing_documents = sum(
        len(record["gold_missing_entirely_documents"]) for record in records
    )
    return {
        "gold_document_count": sum(len(record["gold_documents"]) for record in records),
        "gold_missing_entirely_document_count": missing_documents,
        "queries_with_any_gold_missing_entirely_count": sum(
            bool(record["gold_missing_entirely_documents"]) for record in records
        ),
        "mean_gold_first_chunk_rank": (
            sum(chunk_ranks) / len(chunk_ranks) if chunk_ranks else None
        ),
        "median_gold_first_chunk_rank": (
            statistics.median(chunk_ranks) if chunk_ranks else None
        ),
        "mean_gold_first_document_rank": (
            sum(document_ranks) / len(document_ranks) if document_ranks else None
        ),
        "median_gold_first_document_rank": (
            statistics.median(document_ranks) if document_ranks else None
        ),
    }


def _depth_recommendation(
    candidate_metrics: dict[str, Any],
    failure_counts: Counter[str],
    *,
    maximum_cached_depth: int,
) -> dict[str, str]:
    recall = candidate_metrics["candidate_doc_recall_at_k"][str(maximum_cached_depth)]
    retrieval_miss = failure_counts["retrieval_miss"]
    if maximum_cached_depth < 500 and (recall < 0.98 or retrieval_miss > 0):
        return {
            "decision": "recommended_for_gpu_probe",
            "reason": (
                f"CandidateDocRecall@{maximum_cached_depth}={recall:.4f} and "
                f"{retrieval_miss} queries have no gold document in the cached "
                "pool. A reranker cannot rescue those retrieval misses; run a "
                "small cached-compatible GPU probe at candidate_k=500 before 1000."
            ),
        }
    return {
        "decision": "not_priority_from_current_evidence",
        "reason": (
            f"Cached CandidateDocRecall@{maximum_cached_depth}={recall:.4f} "
            "does not by itself justify a deeper GPU search."
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )


def _summary_markdown(summary: dict[str, Any]) -> str:
    metrics = summary["candidate_metrics"]
    maximum_depth = str(summary["maximum_cached_depth"])
    failures = summary["failure_buckets"]
    aggregation = summary["early_document_aggregation"]
    recommendation = summary["depth_recommendation"]
    lines = [
        "# LegalIR P2 candidate coverage",
        "",
        f"- CandidateDocRecall@{maximum_depth}: "
        f"{metrics['candidate_doc_recall_at_k'][maximum_depth]:.6f}",
        f"- Mean unique documents in top-{maximum_depth} chunks: "
        f"{metrics['mean_unique_docs_at_chunk_depth'][maximum_depth]:.2f}",
        f"- Final failures not rescuable by a reranker (no gold candidate): "
        f"{failures['retrieval_miss']}",
        f"- Depth >{maximum_depth}: {recommendation['decision']}. "
        f"{recommendation['reason']}",
        "",
        "## Early document aggregation (official P1 evaluator, top 5 docs)",
        "",
        "| Method | Recall | Precision | Multi-gold Recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, values in aggregation.items():
        multi = values["multi_gold_recall"]
        multi_text = "n/a" if multi is None else f"{multi:.6f}"
        lines.append(
            f"| {method} | {values['official_recall']:.6f} | "
            f"{values['official_precision']:.6f} | {multi_text} |"
        )
    lines.extend(
        [
            "",
            "## Failure buckets",
            "",
            "| Bucket | Queries |",
            "| --- | ---: |",
        ]
    )
    for bucket in (
        "retrieval_miss",
        "candidate_present_final_miss",
        "multi_gold_partial",
        "success",
    ):
        lines.append(f"| {bucket} | {failures[bucket]} |")
    lines.extend(
        [
            "",
            "This is a cached-candidate diagnostic only. Strict-CV fold IDs are "
            "recorded per query; no label-derived method is selected or trained.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Cached dense-candidate manifest. Inferred from --predictions by default.",
    )
    parser.add_argument(
        "--k",
        type=_positive_int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="Chunk depths; future cached 500/1000 artifacts are supported.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strict-folds",
        type=Path,
        default=DEFAULT_STRICT_FOLDS,
        help="Required P1 strict_cv_v2 folds.json artifact.",
    )
    parser.add_argument(
        "--strict-fold",
        type=int,
        choices=range(5),
        help="Optionally analyze only one P1 validation fold.",
    )
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATION_METHODS,
        default="first_occurrence",
        help="Aggregation used for final failure buckets.",
    )
    parser.add_argument("--final-documents", type=_positive_int, default=5)
    parser.add_argument("--max-questions", type=_positive_int)
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    if args.final_documents > 5:
        raise ValueError("--final-documents must not exceed the official limit of 5")
    k_values = sorted(set(args.k))
    manifest_path = args.manifest or _infer_manifest_path(args.predictions)
    strict_fold_map = _load_strict_fold_map(args.strict_folds)
    all_predictions = load_predictions(args.predictions)
    dense_manifest = _validate_cached_manifest(
        manifest_path,
        args.predictions,
        max_k=max(k_values),
        prediction_count=len(all_predictions),
    )
    references, predictions, selected_folds = _select_inputs(
        args.gold,
        args.predictions,
        strict_fold_map=strict_fold_map,
        strict_fold=args.strict_fold,
        max_questions=args.max_questions,
    )
    candidate_metrics = _candidate_metrics(references, predictions, k_values)
    aggregation_reports, aggregation_documents = _aggregation_reports(
        references,
        predictions,
        final_documents=args.final_documents,
    )
    selected_aggregation = args.aggregation
    failure_records = _failure_records(
        references,
        predictions,
        selected_documents=aggregation_documents[selected_aggregation],
        strict_folds=selected_folds,
        final_documents=args.final_documents,
    )
    failure_counts: Counter[str] = Counter(
        record["bucket"] for record in failure_records
    )
    for bucket in (
        "retrieval_miss",
        "candidate_present_final_miss",
        "multi_gold_partial",
        "success",
    ):
        failure_counts.setdefault(bucket, 0)
    maximum_cached_depth = min(len(prediction.hits) for prediction in predictions)
    if max(k_values) > maximum_cached_depth:
        raise ValueError(
            f"selected predictions only contain {maximum_cached_depth} chunks, "
            f"but K={max(k_values)} was requested"
        )
    index_manifest = dense_manifest.get("index_manifest")
    corpus_hash = (
        index_manifest.get("corpus_hash")
        if isinstance(index_manifest, dict)
        else "unavailable"
    )
    summary: dict[str, Any] = {
        "schema_version": "legal-ir-p2-candidate-analysis-v1",
        "artifact_manifest": {
            "git_commit": _git_commit(),
            "config": {
                "gold": str(args.gold),
                "gold_sha256": _sha256(args.gold),
                "predictions": str(args.predictions),
                "predictions_sha256": _sha256(args.predictions),
                "dense_manifest": str(manifest_path),
                "strict_folds": str(args.strict_folds),
                "strict_folds_sha256": _sha256(args.strict_folds),
            },
            "model": {
                "model_id": dense_manifest.get("model_id"),
                "model_path": dense_manifest.get("model_path"),
                "loaded_for_p2": False,
            },
            "corpus": {
                "root": "data/processed_v3",
                "corpus_hash": corpus_hash,
                "modified": False,
            },
            "parameters": {
                "k_values": k_values,
                "final_documents": args.final_documents,
                "aggregation": selected_aggregation,
                "max_questions": args.max_questions,
                "strict_fold": args.strict_fold,
            },
        },
        "manifest_validation": {
            "status": "passed",
            "candidate_k": dense_manifest["candidate_k"],
            "benchmark_sha256": dense_manifest["benchmark_sha256"],
            "benchmark": dense_manifest["benchmark"],
        },
        "strict_cv": {
            "source": str(args.strict_folds),
            "schema_version": "legal-ir-strict-cv-v2",
            "selection": "validation-fold membership only; no labels used for training",
            "selected_question_count_by_fold": {
                str(fold): sum(value == fold for value in selected_folds.values())
                for fold in range(5)
            },
            "official_metrics_by_validation_fold": _strict_fold_metrics(
                references,
                predictions,
                strict_folds=selected_folds,
                aggregation=selected_aggregation,
                final_documents=args.final_documents,
            ),
        },
        "sample_count": len(predictions),
        "maximum_cached_depth": maximum_cached_depth,
        "candidate_metrics": candidate_metrics,
        "gold_rank_diagnostics": _rank_diagnostics(failure_records),
        "early_document_aggregation": aggregation_reports,
        "selected_failure_aggregation": selected_aggregation,
        "failure_buckets": dict(failure_counts),
        "final_failure_count": len(predictions) - failure_counts["success"],
        "final_failures_not_reranker_rescuable_count": failure_counts["retrieval_miss"],
        "depth_recommendation": _depth_recommendation(
            candidate_metrics,
            failure_counts,
            maximum_cached_depth=maximum_cached_depth,
        ),
    }
    outputs = [
        args.output_dir / "summary.json",
        args.output_dir / "summary.md",
        args.output_dir / "failure_cases.jsonl",
    ]
    _write_json(outputs[0], summary)
    outputs[1].parent.mkdir(parents=True, exist_ok=True)
    outputs[1].write_text(_summary_markdown(summary), encoding="utf-8", newline="\n")
    _write_jsonl(outputs[2], failure_records)
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"LegalIR candidate analysis error: {exc}", file=sys.stderr)
        return 2
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
