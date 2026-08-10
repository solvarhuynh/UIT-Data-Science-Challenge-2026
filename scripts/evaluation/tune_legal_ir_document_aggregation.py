"""Evaluate document-level aggregation for DSC2026 LegalIR chunk rankings.

This is an evaluation-only utility. It reads two disjoint validation splits,
uses only their organizer labels for scoring, and never writes a submission.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SplitPaths:
    name: str
    dense: Path
    bge: Path


def _load_gold(path: Path) -> dict[str, tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("gold root must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for question_id, record in payload.items():
        if not isinstance(record, dict) or not isinstance(record.get("answer"), list):
            raise TypeError(f"invalid gold record {question_id!r}")
        result[str(question_id)] = tuple(str(item) for item in record["answer"])
    return result


def _load_hits(path: Path) -> dict[str, list[tuple[str, float, float]]]:
    rows: dict[str, list[tuple[str, float, float]]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            if question_id in rows:
                raise ValueError(f"duplicate question {question_id!r} in {path}")
            hits: list[tuple[str, float, float]] = []
            for hit in row["hits"]:
                dense_score = float(hit.get("dense_score", hit.get("score", 0.0)))
                bge_score = float(
                    hit.get("rerank_score", hit.get("final_score", dense_score))
                )
                if not math.isfinite(dense_score) or not math.isfinite(bge_score):
                    raise ValueError(f"non-finite score at {path}:{line_number}")
                hits.append((str(hit["doc_id"]), dense_score, bge_score))
            if not hits:
                raise ValueError(f"empty hits for {question_id!r} in {path}")
            rows[question_id] = hits
    return rows


def _first_document_ranking(
    hits: Iterable[tuple[str, float, float]],
) -> list[str]:
    seen: set[str] = set()
    ranking: list[str] = []
    for document_id, _, _ in hits:
        if document_id not in seen:
            seen.add(document_id)
            ranking.append(document_id)
    return ranking


def _rank_aggregate(
    hits: list[tuple[str, float, float]], *, cap: int, rank_k: int
) -> list[str]:
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    first_rank: dict[str, int] = {}
    for rank, (document_id, _, _) in enumerate(hits, 1):
        first_rank.setdefault(document_id, rank)
        count = counts.get(document_id, 0)
        if count >= cap:
            continue
        counts[document_id] = count + 1
        scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (
            rank_k + rank
        )
    return sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id], first_rank[document_id], document_id
        ),
    )


def _score_aggregate(
    hits: list[tuple[str, float, float]], *, source: str, cap: int
) -> list[str]:
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    first_rank: dict[str, int] = {}
    for rank, (document_id, dense_score, bge_score) in enumerate(hits, 1):
        first_rank.setdefault(document_id, rank)
        count = counts.get(document_id, 0)
        if count >= cap:
            continue
        counts[document_id] = count + 1
        raw_score = dense_score if source == "dense" else bge_score
        value = raw_score if source == "dense" else 1.0 / (1.0 + math.exp(-raw_score))
        scores[document_id] = scores.get(document_id, 0.0) + value
    return sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id], first_rank[document_id], document_id
        ),
    )


def _weighted_rrf(
    rankings: list[list[str]], weights: list[float], rank_k: int
) -> list[str]:
    scores: dict[str, float] = {}
    rank_maps: list[dict[str, int]] = []
    for ranking, weight in zip(rankings, weights):
        rank_map = {document_id: rank for rank, document_id in enumerate(ranking, 1)}
        rank_maps.append(rank_map)
        if weight <= 0.0:
            continue
        for document_id, rank in rank_map.items():
            scores[document_id] = scores.get(document_id, 0.0) + weight / (
                rank_k + rank
            )
    return sorted(
        scores,
        key=lambda document_id: (
            -scores[document_id],
            *(rank_map.get(document_id, 10**9) for rank_map in rank_maps),
            document_id,
        ),
    )


def _metrics(
    predictions: dict[str, list[str]], gold: dict[str, tuple[str, ...]]
) -> dict[str, float]:
    if set(predictions) - set(gold):
        raise ValueError("predictions contain unknown question IDs")
    recalls: list[float] = []
    precisions: list[float] = []
    for question_id, predicted in predictions.items():
        truth = set(gold[question_id])
        selected = set(predicted[:5])
        overlap = len(truth & selected)
        recalls.append(overlap / len(truth))
        precisions.append(overlap / min(5, len(predicted)))
    return {
        "recall": sum(recalls) / len(recalls),
        "precision": sum(precisions) / len(precisions),
    }


def _build_sources(
    rows: dict[str, list[tuple[str, float, float]]], source: str
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {"first": {}}
    for cap in (2, 3, 5, 10):
        for rank_k in (0, 5, 10, 20, 40, 60):
            result[f"rank_cap{cap}_k{rank_k}"] = {}
        result[f"score_cap{cap}"] = {}
    for question_id, hits in rows.items():
        result["first"][question_id] = _first_document_ranking(hits)
        for cap in (2, 3, 5, 10):
            for rank_k in (0, 5, 10, 20, 40, 60):
                key = f"rank_cap{cap}_k{rank_k}"
                result[key][question_id] = _rank_aggregate(
                    hits, cap=cap, rank_k=rank_k
                )
            result[f"score_cap{cap}"][question_id] = _score_aggregate(
                hits, source=source, cap=cap
            )
    return result


def _score_config(
    split_predictions: dict[str, dict[str, list[str]]],
    split_ids: dict[str, list[str]],
    gold: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    by_split: dict[str, dict[str, float]] = {}
    pooled: dict[str, list[str]] = {}
    for split_name, question_ids in split_ids.items():
        predictions = split_predictions[split_name]
        if set(predictions) != set(question_ids):
            raise ValueError(f"prediction coverage mismatch for {split_name}")
        by_split[split_name] = _metrics(predictions, gold)
        pooled.update(predictions)
    return {"splits": by_split, "pooled": _metrics(pooled, gold)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--dense-a", type=Path, required=True)
    parser.add_argument("--bge-a", type=Path, required=True)
    parser.add_argument("--dense-b", type=Path, required=True)
    parser.add_argument("--bge-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    gold = _load_gold(args.gold)
    split_paths = (
        SplitPaths("a", args.dense_a, args.bge_a),
        SplitPaths("b", args.dense_b, args.bge_b),
    )
    split_ids: dict[str, list[str]] = {}
    dense_sources: dict[str, dict[str, dict[str, list[str]]]] = {}
    bge_sources: dict[str, dict[str, dict[str, list[str]]]] = {}
    for paths in split_paths:
        dense_rows = _load_hits(paths.dense)
        bge_rows = _load_hits(paths.bge)
        if list(dense_rows) != list(bge_rows):
            raise ValueError(f"dense/BGE question order differs for split {paths.name}")
        for question_id in dense_rows:
            dense_pool = [item[0] for item in dense_rows[question_id]]
            bge_pool = [item[0] for item in bge_rows[question_id]]
            if sorted(dense_pool) != sorted(bge_pool):
                raise ValueError(f"candidate pool mismatch at {question_id!r}")
        split_ids[paths.name] = list(dense_rows)
        dense_sources[paths.name] = _build_sources(dense_rows, "dense")
        bge_sources[paths.name] = _build_sources(bge_rows, "bge")

    source_scores: list[dict[str, Any]] = []
    for source_name, collection in (("dense", dense_sources), ("bge", bge_sources)):
        for config_name in collection["a"]:
            predictions = {
                split_name: collection[split_name][config_name]
                for split_name in split_ids
            }
            metrics = _score_config(predictions, split_ids, gold)
            source_scores.append(
                {"source": source_name, "config": config_name, **metrics}
            )
    source_scores.sort(key=lambda item: -item["pooled"]["recall"])

    dense_shortlist = [
        item["config"]
        for item in source_scores
        if item["source"] == "dense"
    ][:8]
    bge_shortlist = [
        item["config"]
        for item in source_scores
        if item["source"] == "bge"
    ][:8]
    fusion_scores: list[dict[str, Any]] = []
    for dense_name in dense_shortlist:
        for bge_name in bge_shortlist:
            for rank_k in (0, 5, 10, 20, 40, 60):
                for dense_tenths in range(1, 10):
                    dense_weight = dense_tenths / 10.0
                    bge_weight = 1.0 - dense_weight
                    split_predictions: dict[str, dict[str, list[str]]] = {}
                    for split_name, question_ids in split_ids.items():
                        predictions: dict[str, list[str]] = {}
                        for question_id in question_ids:
                            predictions[question_id] = _weighted_rrf(
                                [
                                    dense_sources[split_name][dense_name][question_id],
                                    bge_sources[split_name][bge_name][question_id],
                                ],
                                [dense_weight, bge_weight],
                                rank_k,
                            )[:5]
                        split_predictions[split_name] = predictions
                    metrics = _score_config(split_predictions, split_ids, gold)
                    fusion_scores.append(
                        {
                            "dense": dense_name,
                            "bge": bge_name,
                            "dense_weight": dense_weight,
                            "bge_weight": bge_weight,
                            "rrf_k": rank_k,
                            **metrics,
                        }
                    )
    fusion_scores.sort(
        key=lambda item: (
            -item["pooled"]["recall"],
            -min(value["recall"] for value in item["splits"].values()),
        )
    )

    report = {
        "question_counts": {name: len(ids) for name, ids in split_ids.items()},
        "source_scores": source_scores,
        "dense_shortlist": dense_shortlist,
        "bge_shortlist": bge_shortlist,
        "fusion_scores": fusion_scores[:200],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    best = fusion_scores[0]
    print(f"best_recall={best['pooled']['recall']:.9f}")
    print(f"best_precision={best['pooled']['precision']:.9f}")
    print(json.dumps({key: best[key] for key in (
        'dense', 'bge', 'dense_weight', 'bge_weight', 'rrf_k', 'splits'
    )}, ensure_ascii=False))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
