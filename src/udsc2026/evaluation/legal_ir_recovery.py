"""Strict out-of-fold fusion utilities for the Task 1 recovery experiment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

RankingMap = Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class FusionConfig:
    """A small, auditable weighted-RRF configuration."""

    weights: tuple[tuple[str, float], ...]
    rrf_k: int

    def as_dict(self) -> dict[str, object]:
        return {"weights": dict(self.weights), "rrf_k": self.rrf_k}


def query_score(gold: Sequence[str], predicted: Sequence[str]) -> tuple[float, float]:
    """Return official-compatible set recall and precision for one query."""

    relevant = set(gold)
    returned = set(predicted[:5])
    if not relevant:
        raise ValueError("gold document set must not be empty")
    if len(predicted[:5]) != len(returned):
        raise ValueError("predictions must contain distinct document IDs")
    return (
        len(relevant & returned) / len(relevant),
        len(relevant & returned) / len(returned) if returned else 0.0,
    )


def metrics(
    query_ids: Sequence[str],
    gold: Mapping[str, Sequence[str]],
    predictions: RankingMap,
) -> dict[str, float | int]:
    """Compute macro Recall/Precision using exact query-ID coverage."""

    if set(query_ids) - set(predictions):
        raise ValueError("prediction coverage is incomplete")
    values = [query_score(gold[qid], predictions[qid]) for qid in query_ids]
    return {
        "query_count": len(values),
        "recall": sum(value[0] for value in values) / len(values),
        "precision": sum(value[1] for value in values) / len(values),
    }


def weighted_rrf(
    rankings: Mapping[str, Sequence[str]],
    config: FusionConfig,
    *,
    limit: int = 5,
) -> list[str]:
    """Fuse independent document sources with deterministic tie-breaking."""

    scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    for source, weight in config.weights:
        ranking = rankings.get(source, ())
        rank_map = {
            doc_id: rank
            for rank, doc_id in enumerate(dict.fromkeys(map(str, ranking)), 1)
        }
        source_ranks[source] = rank_map
        for doc_id, rank in rank_map.items():
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (config.rrf_k + rank)
    ordered_sources = [source for source, _ in config.weights]
    return sorted(
        scores,
        key=lambda doc_id: (
            -scores[doc_id],
            *(source_ranks[source].get(doc_id, 10**9) for source in ordered_sources),
            doc_id,
        ),
    )[:limit]


def default_configs(source_names: Sequence[str]) -> list[FusionConfig]:
    """Build a bounded static grid suitable for nested selection on CPU."""

    names = tuple(dict.fromkeys(source_names))
    if not names:
        raise ValueError("at least one source is required")
    configs: list[FusionConfig] = []
    for rrf_k in (0, 10, 60):
        configs.append(FusionConfig(tuple((name, 1.0) for name in names), rrf_k))
        for emphasized in names:
            configs.append(
                FusionConfig(
                    tuple((name, 2.0 if name == emphasized else 1.0) for name in names),
                    rrf_k,
                )
            )
        for left, right in combinations(names, 2):
            configs.append(FusionConfig(((left, 1.0), (right, 1.0)), rrf_k))
    # Preserve the documented four-source baseline when those names exist.
    aliases = {
        "dense": 0.20,
        "bge": 0.50,
        "chunk_bge": 0.50,
        "knn_word": 0.20,
        "bm25": 0.10,
        "bm25_body": 0.10,
    }
    baseline = tuple((name, aliases[name]) for name in names if name in aliases)
    if len(baseline) >= 2:
        configs.append(FusionConfig(baseline, 60))
    unique = {(config.weights, config.rrf_k): config for config in configs}
    return list(unique.values())


def nested_oof_fusion(
    *,
    gold: Mapping[str, Sequence[str]],
    folds: Mapping[str, int],
    sources: Mapping[str, RankingMap],
    configs: Sequence[FusionConfig] | None = None,
) -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    """Select fusion only on four outer folds and predict the fifth."""

    query_ids = sorted(folds)
    if set(query_ids) != set(gold):
        raise ValueError("fold and gold query-ID coverage must match")
    for name, source in sources.items():
        if set(query_ids) - set(source):
            raise ValueError(f"source {name!r} is missing query IDs")
    candidates = list(configs or default_configs(tuple(sources)))
    if not candidates:
        raise ValueError("fusion config grid must not be empty")
    prediction_cache = [
        {
            qid: weighted_rrf(
                {name: source[qid] for name, source in sources.items()}, config
            )
            for qid in query_ids
        }
        for config in candidates
    ]
    predictions: dict[str, list[str]] = {}
    fold_results: list[dict[str, object]] = []
    for holdout in sorted(set(folds.values())):
        training_ids = [qid for qid in query_ids if folds[qid] != holdout]
        validation_ids = [qid for qid in query_ids if folds[qid] == holdout]
        best_index: int | None = None
        best_key: tuple[float, float, float, int] | None = None
        for index, config in enumerate(candidates):
            current = prediction_cache[index]
            score = metrics(training_ids, gold, current)
            key = (
                float(score["recall"]),
                float(score["precision"]),
                -sum(weight for _, weight in config.weights),
                -config.rrf_k,
            )
            if best_key is None or key > best_key:
                best_index, best_key = index, key
        assert best_index is not None
        best = candidates[best_index]
        heldout = {qid: prediction_cache[best_index][qid] for qid in validation_ids}
        predictions.update(heldout)
        fold_results.append(
            {
                "fold": holdout,
                "selected_config": best.as_dict(),
                "selection_query_count": len(training_ids),
                "heldout": metrics(validation_ids, gold, heldout),
            }
        )
    return predictions, fold_results


def strict_source_ablation(
    *,
    gold: Mapping[str, Sequence[str]],
    folds: Mapping[str, int],
    sources: Mapping[str, RankingMap],
) -> list[dict[str, object]]:
    """Re-select configs OOF after dropping each source independently."""

    rows: list[dict[str, object]] = []
    all_ids = sorted(folds)
    variants: list[tuple[str, dict[str, RankingMap]]] = [("none", dict(sources))]
    if len(sources) > 1:
        variants.extend(
            (name, {key: value for key, value in sources.items() if key != name})
            for name in sources
        )
    for removed, active in variants:
        prediction, _ = nested_oof_fusion(gold=gold, folds=folds, sources=active)
        rows.append(
            {
                "removed_source": removed,
                "remaining_sources": list(active),
                **metrics(all_ids, gold, prediction),
            }
        )
    return rows
