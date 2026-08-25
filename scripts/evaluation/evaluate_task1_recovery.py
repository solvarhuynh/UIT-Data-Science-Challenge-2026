"""Run strict five-fold OOF fusion, ablations, and candidate-oracle diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from udsc2026.evaluation.legal_ir_meta_fusion import nested_oof_meta_ranker
from udsc2026.evaluation.legal_ir_recovery import (
    metrics,
    nested_oof_fusion,
    query_score,
    strict_source_ablation,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_gold(path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    payload = _load_json(path)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("--gold must be a non-empty question mapping")
    gold, questions = {}, {}
    for raw_id, row in payload.items():
        if not isinstance(row, dict) or not isinstance(row.get("answer"), list):
            raise ValueError(f"invalid gold row {raw_id!r}")
        query_id = str(raw_id)
        documents = list(dict.fromkeys(str(value) for value in row["answer"]))
        question = str(row.get("question", "")).strip()
        if not documents or not question:
            raise ValueError(f"empty gold/question at {query_id!r}")
        gold[query_id], questions[query_id] = documents, question
    return gold, questions


def _load_folds(path: Path) -> dict[str, int]:
    payload = _load_json(path)
    output: dict[str, int] = {}
    for item in payload.get("folds", []):
        fold = int(item["fold"])
        for raw_id in item.get("validation_ids", []):
            query_id = str(raw_id)
            if query_id in output:
                raise ValueError(f"query {query_id!r} occurs in multiple folds")
            output[query_id] = fold
    if len(set(output.values())) != 5:
        raise ValueError("strict OOF evaluation requires exactly five folds")
    return output


def _ranking_from_row(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    query_id = str(
        row.get("query_id", row.get("question_id", row.get("id", "")))
    ).strip()
    raw_documents = row.get(
        "documents", row.get("top5", row.get("predicted_documents"))
    )
    if raw_documents is None and isinstance(row.get("hits"), list):
        raw_documents = [
            hit.get("doc_id", hit.get("document_id"))
            for hit in row["hits"]
            if isinstance(hit, dict)
        ]
    if not query_id or not isinstance(raw_documents, list):
        raise ValueError("ranking row needs an ID and documents/hits list")
    documents = list(
        dict.fromkeys(str(value) for value in raw_documents if value is not None)
    )
    if len(documents) < 5:
        raise ValueError(f"ranking for {query_id!r} contains fewer than five documents")
    return query_id, documents


def _load_rankings(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output: dict[str, list[str]] = {}
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            rows = (json.loads(line) for line in stream if line.strip())
            for row in rows:
                query_id, documents = _ranking_from_row(row)
                if query_id in output:
                    raise ValueError(f"duplicate query {query_id!r} in {path}")
                output[query_id] = documents
    else:
        payload = _load_json(path)
        rows = payload if isinstance(payload, list) else payload.get("predictions", [])
        for row in rows:
            query_id, documents = _ranking_from_row(row)
            if query_id in output:
                raise ValueError(f"duplicate query {query_id!r} in {path}")
            output[query_id] = documents
    if not output:
        raise ValueError(f"ranking file is empty: {path}")
    return output


def _load_meta_features(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id", row.get("question_id", "")))
            document_features = row.get("document_features")
            if isinstance(document_features, dict):
                output[query_id] = {
                    str(doc_id): {
                        str(key): float(value) for key, value in values.items()
                    }
                    for doc_id, values in document_features.items()
                }
                continue
            aggregates = row.get("document_aggregates", [])
            scores = [float(item["score"]) for item in aggregates[:2]]
            margin = scores[0] - scores[1] if len(scores) == 2 else 0.0
            output[query_id] = {
                str(item["doc_id"]): {
                    "aggregate_score": float(item["score"]),
                    "supporting_chunk_count": float(
                        item.get("supporting_chunk_count", 0)
                    ),
                    "query_top1_top2_margin": margin,
                }
                for item in aggregates
            }
    return output


def _source_arguments(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--source must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in output:
            raise ValueError(f"invalid/duplicate source name {name!r}")
        output[name] = Path(raw_path)
    return output


def _query_length_bucket(question: str) -> str:
    count = len(question.split())
    if count <= 10:
        return "short_1_10"
    if count <= 25:
        return "medium_11_25"
    return "long_26_plus"


def _oracle_for_file(
    path: Path,
    *,
    depth: int,
    gold: Mapping[str, Sequence[str]],
    questions: Mapping[str, str],
    recovery_sources: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[dict[str, object], dict[str, set[str]]]:
    recalls: list[float] = []
    pools: dict[str, set[str]] = {}
    category: dict[str, list[float]] = defaultdict(list)
    family: dict[str, list[float]] = defaultdict(list)
    recovery_counts: Counter[str] = Counter()
    missing_queries = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(
                row.get("query_id", row.get("question_id", row.get("id", "")))
            )
            if query_id not in gold:
                continue
            hits = [hit for hit in row.get("hits", [])[:depth] if isinstance(hit, dict)]
            docs = {
                str(hit.get("doc_id", hit.get("document_id")))
                for hit in hits
                if hit.get("doc_id", hit.get("document_id")) is not None
            }
            pools[query_id] = docs
            wanted = set(gold[query_id])
            recall = len(wanted & docs) / len(wanted)
            recalls.append(recall)
            category["single_gold" if len(wanted) == 1 else "multi_gold"].append(recall)
            category[_query_length_bucket(questions[query_id])].append(recall)
            gold_families = set()
            for hit in hits:
                doc_id = str(hit.get("doc_id", hit.get("document_id", "")))
                if doc_id not in wanted:
                    continue
                metadata = (
                    hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
                )
                value = metadata.get(
                    "source_family", hit.get("source_family", hit.get("law_name"))
                )
                if value:
                    gold_families.add(str(value))
            if not gold_families:
                gold_families.add("metadata_unavailable_or_gold_missing")
            for value in gold_families:
                family[value].append(recall)
            missing = wanted - docs
            if missing:
                missing_queries += 1
                for source_name, rankings in recovery_sources.items():
                    if missing & set(rankings.get(query_id, ())):
                        recovery_counts[source_name] += 1
    if set(gold) != set(pools):
        raise ValueError(f"oracle input {path} does not cover all gold queries")
    return (
        {
            "path": str(path),
            "raw_chunk_depth": depth,
            "query_count": len(recalls),
            "macro_gold_coverage": sum(recalls) / len(recalls),
            "queries_with_any_gold_missing": missing_queries,
            "breakdown": {
                key: {"query_count": len(values), "recall": sum(values) / len(values)}
                for key, values in sorted(category.items())
            },
            "law_document_family_when_available": {
                key: {"query_count": len(values), "recall": sum(values) / len(values)}
                for key, values in sorted(family.items())
            },
            "miss_recovered_by_source_query_count": dict(
                sorted(recovery_counts.items())
            ),
        },
        pools,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument(
        "--source", action="append", default=[], help="Repeat NAME=PATH"
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--raw-k200", type=Path)
    parser.add_argument("--raw-k500", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--run-meta-fusion", action="store_true")
    parser.add_argument("--meta-features", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    gold, questions = _load_gold(args.gold)
    folds = _load_folds(args.folds)
    if set(gold) != set(folds):
        raise ValueError("gold and strict-fold coverage differ")
    source_paths = _source_arguments(args.source)
    if not source_paths:
        raise ValueError("provide at least one --source NAME=PATH")
    sources = {name: _load_rankings(path) for name, path in source_paths.items()}
    for name, ranking in sources.items():
        if set(ranking) != set(gold):
            raise ValueError(f"source {name!r} coverage differs from gold")

    predictions, fold_results = nested_oof_fusion(
        gold=gold, folds=folds, sources=sources
    )
    query_ids = sorted(gold)
    pooled = metrics(query_ids, gold, predictions)
    baseline = _load_rankings(args.baseline) if args.baseline else None
    comparison: dict[str, object] | None = None
    if baseline is not None:
        if set(baseline) != set(gold):
            raise ValueError("baseline coverage differs from gold")
        improved, harmed = [], []
        for query_id in query_ids:
            new_recall = query_score(gold[query_id], predictions[query_id])[0]
            old_recall = query_score(gold[query_id], baseline[query_id])[0]
            record = {
                "query_id": query_id,
                "baseline_recall": old_recall,
                "oof_recall": new_recall,
                "gold_documents": gold[query_id],
                "baseline_top5": baseline[query_id][:5],
                "oof_top5": predictions[query_id],
            }
            if new_recall > old_recall:
                improved.append(record)
            elif new_recall < old_recall:
                harmed.append(record)
        comparison = {
            "baseline": metrics(query_ids, gold, baseline),
            "improved_query_count": len(improved),
            "harmed_query_count": len(harmed),
            "improved_queries": improved,
            "harmed_queries": harmed,
        }

    oracle_rows = []
    pools = {}
    for path, depth in ((args.raw_k200, 200), (args.raw_k500, 500)):
        if path is not None:
            row, pool = _oracle_for_file(
                path,
                depth=depth,
                gold=gold,
                questions=questions,
                recovery_sources=sources,
            )
            oracle_rows.append(row)
            pools[depth] = pool
    k500_recommendation: dict[str, object] | None = None
    if 200 in pools and 500 in pools:
        recall200 = float(oracle_rows[0]["macro_gold_coverage"])
        recall500 = float(oracle_rows[1]["macro_gold_coverage"])
        delta = recall500 - recall200
        k500_recommendation = {
            "coverage_delta": delta,
            "recommend_k500_reranking": delta >= 0.01 and recall200 < 0.98,
            "rule": "recommend only when delta>=0.01 and K=200 oracle<0.98",
        }

    output_dir = args.output_dir
    prediction_path = output_dir / "oof_predictions.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8", newline="\n") as stream:
        for query_id in query_ids:
            stream.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "fold": folds[query_id],
                        "gold_documents": gold[query_id],
                        "documents": predictions[query_id],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    meta_report: dict[str, object] | None = None
    if args.run_meta_fusion:
        external = (
            _load_meta_features(args.meta_features) if args.meta_features else None
        )
        if external is not None and set(external) != set(gold):
            raise ValueError("meta feature coverage differs from gold")
        meta_predictions, meta_folds = nested_oof_meta_ranker(
            gold=gold,
            folds=folds,
            sources=sources,
            external_features=external,
        )
        meta_path = output_dir / "meta_oof_predictions.jsonl"
        with meta_path.open("w", encoding="utf-8", newline="\n") as stream:
            for query_id in query_ids:
                stream.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "fold": folds[query_id],
                            "documents": meta_predictions[query_id],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        meta_report = {
            "confirmatory": True,
            "learner": "standardized L2 logistic regression, class-balanced",
            "pooled_strict_oof": metrics(query_ids, gold, meta_predictions),
            "folds": meta_folds,
            "predictions": str(meta_path),
        }
    report = {
        "schema_version": "task1-recovery-strict-oof-v1",
        "status": "complete",
        "metric": "official-compatible macro set Recall(primary)/Precision(secondary)",
        "confirmatory": True,
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "pooled_strict_oof": pooled,
        "folds": fold_results,
        "candidate_oracle": oracle_rows,
        "k500_recommendation": k500_recommendation,
        "baseline_comparison": comparison,
        "source_ablation": (
            None
            if args.skip_ablation
            else strict_source_ablation(gold=gold, folds=folds, sources=sources)
        ),
        "meta_fusion": meta_report,
        "oof_predictions": str(prediction_path),
    }
    _write_json(output_dir / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Task1 recovery evaluation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["pooled_strict_oof"], ensure_ascii=False))
    print(args.output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
