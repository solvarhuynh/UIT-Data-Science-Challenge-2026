"""Small interpretable OOF meta-ranker for LegalIR document fusion."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from udsc2026.evaluation.legal_ir_recovery import RankingMap, metrics

ExternalFeatures = Mapping[str, Mapping[str, Mapping[str, float]]]


def _feature_names(
    sources: Mapping[str, RankingMap], external: ExternalFeatures | None
) -> list[str]:
    names = [f"rr_{name}" for name in sorted(sources)] + ["source_support_count"]
    if external:
        names.extend(
            sorted(
                {
                    feature
                    for by_document in external.values()
                    for values in by_document.values()
                    for feature in values
                }
            )
        )
    return names


def _candidate_features(
    query_id: str,
    sources: Mapping[str, RankingMap],
    external: ExternalFeatures | None,
    names: Sequence[str],
    *,
    source_depth: int,
) -> dict[str, list[float]]:
    rank_maps = {
        name: {
            doc_id: rank
            for rank, doc_id in enumerate(
                dict.fromkeys(ranking[query_id][:source_depth]), 1
            )
        }
        for name, ranking in sources.items()
    }
    documents = {doc_id for ranks in rank_maps.values() for doc_id in ranks}
    if external:
        documents.update(external.get(query_id, {}))
    output = {}
    for doc_id in documents:
        values: dict[str, float] = {}
        for source, ranks in rank_maps.items():
            values[f"rr_{source}"] = 1.0 / ranks[doc_id] if doc_id in ranks else 0.0
        values["source_support_count"] = float(
            sum(doc_id in ranks for ranks in rank_maps.values())
        )
        if external:
            values.update(
                {
                    key: float(value)
                    for key, value in external.get(query_id, {}).get(doc_id, {}).items()
                }
            )
        output[doc_id] = [values.get(name, 0.0) for name in names]
    return output


def nested_oof_meta_ranker(
    *,
    gold: Mapping[str, Sequence[str]],
    folds: Mapping[str, int],
    sources: Mapping[str, RankingMap],
    external_features: ExternalFeatures | None = None,
    source_depth: int = 50,
    negatives_per_query: int = 30,
) -> tuple[dict[str, list[str]], list[dict[str, object]]]:
    """Train on four folds and score candidates from the held-out fifth fold."""

    if source_depth < 5 or negatives_per_query < 1:
        raise ValueError("invalid meta-ranker candidate/sampling depth")
    query_ids = sorted(folds)
    names = _feature_names(sources, external_features)
    feature_cache = {
        query_id: _candidate_features(
            query_id,
            sources,
            external_features,
            names,
            source_depth=source_depth,
        )
        for query_id in query_ids
    }
    predictions: dict[str, list[str]] = {}
    reports: list[dict[str, object]] = []
    for holdout in sorted(set(folds.values())):
        train_ids = [query_id for query_id in query_ids if folds[query_id] != holdout]
        test_ids = [query_id for query_id in query_ids if folds[query_id] == holdout]
        x_rows: list[list[float]] = []
        y_rows: list[int] = []
        for query_id in train_ids:
            wanted = set(gold[query_id])
            features = feature_cache[query_id]
            positives = sorted(wanted & set(features))
            negatives = sorted(
                (doc_id for doc_id in features if doc_id not in wanted),
                key=lambda doc_id: (-sum(features[doc_id]), doc_id),
            )[:negatives_per_query]
            for doc_id in positives + negatives:
                x_rows.append(features[doc_id])
                y_rows.append(int(doc_id in wanted))
        if len(set(y_rows)) != 2:
            raise ValueError(
                "meta-ranker training fold needs positive and negative rows"
            )
        learner = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.2,
                class_weight="balanced",
                max_iter=500,
                random_state=2026,
                solver="liblinear",
            ),
        )
        learner.fit(np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows))
        heldout: dict[str, list[str]] = {}
        for query_id in test_ids:
            features = feature_cache[query_id]
            documents = sorted(features)
            scores = learner.predict_proba(
                np.asarray([features[doc_id] for doc_id in documents], dtype=np.float64)
            )[:, 1]
            heldout[query_id] = [
                doc_id
                for doc_id, _ in sorted(
                    zip(documents, scores), key=lambda item: (-float(item[1]), item[0])
                )[:5]
            ]
        predictions.update(heldout)
        model = learner.named_steps["logisticregression"]
        reports.append(
            {
                "fold": holdout,
                "training_query_count": len(train_ids),
                "training_row_count": len(y_rows),
                "positive_row_count": sum(y_rows),
                "feature_coefficients": dict(
                    zip(names, (float(value) for value in model.coef_[0]))
                ),
                "heldout": metrics(test_ids, gold, heldout),
            }
        )
    return predictions, reports
