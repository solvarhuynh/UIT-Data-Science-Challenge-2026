"""CPU-only final guarded residual experiment for the frozen PV1 incumbent.

This module deliberately lives beside, rather than inside, the broad Phase 1
recovery script.  It reuses the already frozen K20 inputs and the repository's
existing LogisticRegression contract, but writes only to a new sprint48
experiment directory.  No model is loaded, no GPU/Modal call is made, and no
Private labels are read.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.analysis.phase1_step4_recovery import (  # noqa: E402
    EXPECTED_F1F4_QUERIES,
    EXPECTED_F1F4_QDOCS,
    EXPECTED_MODEL,
    EXPECTED_BASE_REV,
    EXPECTED_SELECTOR,
    EXPECTED_AGGREGATION,
    EXPECTED_FT_SHA,
    EXPECTED_FT_CONFIG_SHA,
    TARGET_FOLDS,
    PV1_CANDIDATES,
    PV1_WORKLIST,
    PV1_DELTA_SCORES,
    PV1_DELTA_METRICS,
    PV1_DENSE,
    PV1_BM25,
    PV1_BM25_MANIFEST,
    PV1_KNN_MANIFEST,
    PV1_K20_CANDIDATES,
    PV1_K20_WORKLIST,
    PV1_BGE_MERGED,
    PV1_BGE_MANIFEST,
    PV1_SUBMISSION_JSON,
    PV1_SUBMISSION_ZIP,
    PV1_SUBMISSION_MANIFEST,
    REUSABLE_SCORES,
    TRAIN,
    FOLDS,
    load_fold_map,
    load_gold,
    load_pv1_metadata,
    load_pv1_worklist,
    load_and_merge_scores,
    current_rrf_rank,
    metric,
    read_jsonl,
    key_of,
    qkey,
)
from udsc2026.evaluation.legal_ir_meta_fusion import (  # noqa: E402
    _candidate_features,
    _feature_names,
)

OUT = ROOT / "private_task1/experiments/sprint48_guarded_direct"
DIRECT_ARTIFACT = ROOT / "private_task1/experiments/sprint48_step4/fallback_direct_k20_top5.json"

EXPECTED_DIRECT = {
    "recall": 0.9293809523809524,
    "precision": 0.1980357142857143,
    "delta_recall": 0.003794642857142927,
    "delta_precision": 0.0009285714285714175,
    "queries_changed": 4683,
    "queries_improved": 68,
    "queries_harmed": 42,
    "relevant_docs_gained": 68,
    "relevant_docs_lost": 42,
    "net_relevant_doc_change": 26,
    "fold_recall_deltas": {
        "F1": 0.0011309523809523991,
        "F2": 0.00666666666666671,
        "F3": -0.00011904761904768524,
        "F4": 0.007499999999999951,
    },
}
QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
PRIVATE_CANDIDATES = ROOT / "private_task1/experiments/private_pv1/k20/private_pv1_k20_candidates.jsonl"
PRIVATE_SCORES = ROOT / "private_task1/experiments/private_pv1/bge/merged/private_pv1_bge_scores_merged.jsonl"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def freeze_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": rel(path), "exists": False}
    return {"path": rel(path), "exists": True, "bytes": path.stat().st_size, "sha256": sha256(path)}


def build_direct_inputs(
    docs_by_query: dict[str, list[dict[str, Any]]],
    score_by_key: dict[tuple[str, str], dict[str, Any]],
) -> tuple[
    dict[str, dict[str, list[str]]],
    dict[str, dict[str, dict[str, float]]],
    list[str],
    dict[str, dict[str, list[float]]],
]:
    """Build exactly the feature inputs used by nested_oof_meta_ranker."""

    source_names = ("dense", "bm25", "knn_word")
    source_rankings: dict[str, dict[str, list[str]]] = {name: {} for name in source_names}
    external: dict[str, dict[str, dict[str, float]]] = {}
    for qid in sorted(docs_by_query, key=qkey):
        docs = docs_by_query[qid]
        for source in source_names:
            source_rankings[source][qid] = [
                did
                for did, _rank in sorted(
                    (
                        (str(meta["document_id"]), int(meta["source_ranks"].get(source, 10**9)))
                        for meta in docs
                        if source in meta["source_ranks"]
                    ),
                    key=lambda item: (item[1], qkey(item[0])),
                )
            ]
        ft_values = {
            str(meta["document_id"]): float(score_by_key[(qid, str(meta["document_id"]))]["bge_ft_score"])
            for meta in docs
        }
        base_values = {
            str(meta["document_id"]): float(score_by_key[(qid, str(meta["document_id"]))]["bge_base_score"])
            for meta in docs
        }
        low_ft, high_ft = min(ft_values.values()), max(ft_values.values())
        low_base, high_base = min(base_values.values()), max(base_values.values())
        span_ft = max(1e-6, high_ft - low_ft)
        span_base = max(1e-6, high_base - low_base)
        ft_norm = {did: (value - low_ft) / span_ft for did, value in ft_values.items()}
        base_norm = {did: (value - low_base) / span_base for did, value in base_values.items()}
        max_rank = max(1, len(docs) - 1)
        external[qid] = {}
        for meta in docs:
            did = str(meta["document_id"])
            external[qid][did] = {
                "bge_ft_raw": ft_values[did],
                "bge_base_raw": base_values[did],
                "bge_ft_minmax": ft_norm[did],
                "bge_base_minmax": base_norm[did],
                "candidate_rank_norm": (len(docs) - int(meta["candidate_rank"])) / max_rank,
                "union_rrf": float(meta["rrf_score"]),
                "source_support": float(meta["source_support"]),
            }
    names = _feature_names(source_rankings, external)
    feature_cache = {
        qid: _candidate_features(
            qid,
            source_rankings,
            external,
            names,
            source_depth=20,
        )
        for qid in sorted(docs_by_query)
    }
    return source_rankings, external, names, feature_cache


def fit_direct_model(
    train_ids: list[str],
    feature_cache: dict[str, dict[str, list[float]]],
    gold: dict[str, dict[str, Any]],
) -> Any:
    """Fit the exact frozen estimator/sampling contract once."""

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for query_id in train_ids:
        wanted = {str(value) for value in gold[query_id].get("answer", [])}
        features = feature_cache[query_id]
        positives = sorted(wanted & set(features))
        negatives = sorted(
            (doc_id for doc_id in features if doc_id not in wanted),
            key=lambda doc_id: (-sum(features[doc_id]), doc_id),
        )[:30]
        for doc_id in positives + negatives:
            x_rows.append(features[doc_id])
            y_rows.append(int(doc_id in wanted))
    if len(set(y_rows)) != 2:
        raise ValueError("meta-ranker training fold needs positive and negative rows")
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
    return learner


def score_query_ids(
    model: Any,
    query_ids: Iterable[str],
    feature_cache: dict[str, dict[str, list[float]]],
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    score_map: dict[str, dict[str, float]] = {}
    ranking_map: dict[str, list[str]] = {}
    for query_id in query_ids:
        features = feature_cache[query_id]
        documents = sorted(features)
        scores = model.predict_proba(
            np.asarray([features[doc_id] for doc_id in documents], dtype=np.float64)
        )[:, 1]
        score_map[query_id] = {doc_id: float(score) for doc_id, score in zip(documents, scores)}
        ranking_map[query_id] = [
            doc_id
            for doc_id, _score in sorted(
                zip(documents, scores), key=lambda item: (-float(item[1]), item[0])
            )
        ]
    return score_map, ranking_map


def direct_outer_oof(
    folds: dict[str, int],
    gold: dict[str, dict[str, Any]],
    feature_cache: dict[str, dict[str, list[float]]],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, list[str]],
    list[dict[str, Any]],
]:
    """Reproduce the repository nested_oof_meta_ranker and retain scores."""

    query_ids = sorted(folds)
    score_map: dict[str, dict[str, float]] = {}
    ranking_map: dict[str, list[str]] = {}
    reports: list[dict[str, Any]] = []
    names: list[str] | None = None
    for holdout in sorted(set(folds.values())):
        train_ids = [query_id for query_id in query_ids if folds[query_id] != holdout]
        test_ids = [query_id for query_id in query_ids if folds[query_id] == holdout]
        learner = fit_direct_model(train_ids, feature_cache, gold)
        held_scores, held_rankings = score_query_ids(learner, test_ids, feature_cache)
        score_map.update(held_scores)
        ranking_map.update({qid: ranking[:5] for qid, ranking in held_rankings.items()})
        model = learner.named_steps["logisticregression"]
        # _feature_names is stable and is re-derived by the caller; this field
        # is filled there when reports are normalized.
        reports.append(
            {
                "fold": holdout,
                "training_query_count": len(train_ids),
                "training_row_count": int(sum(
                    len(
                        sorted(set(str(value) for value in gold[qid].get("answer", [])) & set(feature_cache[qid]))
                        + sorted(
                            (
                                doc_id
                                for doc_id in feature_cache[qid]
                                if doc_id not in {str(value) for value in gold[qid].get("answer", [])}
                            ),
                            key=lambda doc_id: (-sum(feature_cache[qid][doc_id]), doc_id),
                        )[:30]
                    )
                    for qid in train_ids
                )),
                "feature_coefficients": [float(value) for value in model.coef_[0]],
            }
        )
    return score_map, ranking_map, reports


def baseline_predictions(
    docs_by_query: dict[str, list[dict[str, Any]]],
    score_by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for qid in sorted(docs_by_query, key=qkey):
        rows = [{**meta, **score_by_key[(qid, str(meta["document_id"]))]} for meta in docs_by_query[qid]]
        result[qid] = current_rrf_rank(rows)[:5]
    return result


def fold_metrics(
    baseline: dict[str, list[str]],
    proposed: dict[str, list[str]],
    gold: dict[str, dict[str, Any]],
    folds: dict[str, int],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    proposed_metric = metric(proposed, gold)
    baseline_metric = metric(baseline, gold)
    by_fold: dict[str, dict[str, float]] = {}
    for fold in sorted({folds[qid] for qid in proposed}):
        ids = [qid for qid in proposed if folds[qid] == fold]
        b = metric({qid: baseline[qid] for qid in ids}, {qid: gold[qid] for qid in ids})
        p = metric({qid: proposed[qid] for qid in ids}, {qid: gold[qid] for qid in ids})
        by_fold[f"F{fold}"] = {
            "recall": float(p["recall"] - b["recall"]),
            "precision": float(p["precision"] - b["precision"]),
            "baseline_recall": float(b["recall"]),
            "proposed_recall": float(p["recall"]),
            "baseline_precision": float(b["precision"]),
            "proposed_precision": float(p["precision"]),
        }
    return {
        "baseline": baseline_metric,
        "proposed": proposed_metric,
        "delta": {
            "recall": float(proposed_metric["recall"] - baseline_metric["recall"]),
            "precision": float(proposed_metric["precision"] - baseline_metric["precision"]),
        },
    }, by_fold


def candidate_metadata(
    docs_by_query: dict[str, list[dict[str, Any]]],
    score_by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for qid, docs in docs_by_query.items():
        joined = [{**meta, **score_by_key[(qid, str(meta["document_id"]))]} for meta in docs]
        bge_order = [str(row["document_id"]) for row in sorted(
            joined,
            key=lambda row: (
                -float(row["bge_ft_score"]),
                -float(row["bge_base_score"]),
                int(row["candidate_rank"]),
                qkey(str(row["document_id"])),
            ),
        )]
        bge_rank = {did: index for index, did in enumerate(bge_order, 1)}
        result[qid] = {}
        for row in joined:
            did = str(row["document_id"])
            source_ranks = {str(key): int(value) for key, value in (row.get("source_ranks") or {}).items()}
            result[qid][did] = {
                "query_id": qid,
                "document_id": did,
                "candidate_rank": int(row["candidate_rank"]),
                "union_rank": int(row["candidate_rank"]),
                "source_ranks": source_ranks,
                "source_support": int(row["source_support"]),
                "bge_rank": int(bge_rank[did]),
                "bge_present": True,
                "selected_chunk_ids": [str(value) for value in row.get("selected_chunk_ids", [])],
            }
    return result


def doc_key_numeric(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def guard_safe(candidate: dict[str, Any], displaced: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(candidate["union_rank"]) <= int(displaced["union_rank"]):
        reasons.append("candidate_rank<=displaced_rank")
    if int(candidate["source_support"]) > int(displaced["source_support"]):
        reasons.append("source_support>displaced_source_support")
    # BGE scores every member of the frozen K20 candidate universe, so BGE
    # membership is explicitly counted as one of the four retrieval lists.
    source_presence = len(candidate["source_ranks"]) + int(bool(candidate.get("bge_present")))
    if source_presence >= 2:
        reasons.append("at_least_2_of_dense_bm25_knn_bge")
    return bool(reasons), reasons


def apply_guarded_policy(
    qid: str,
    baseline_top5: list[str],
    direct_scores: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    *,
    margin_threshold: float = 0.0,
) -> tuple[list[str], dict[str, Any]]:
    """Apply the single frozen residual rule, with at most one new document."""

    base = [str(value) for value in baseline_top5]
    if len(base) != 5 or len(set(base)) != 5:
        raise RuntimeError(f"invalid incumbent Top5 for {qid}")
    if any(doc_id not in metadata for doc_id in base):
        raise RuntimeError(f"incumbent document outside K20 for {qid}")
    candidate_order = sorted(
        (doc_id for doc_id in direct_scores if doc_id not in set(base) and int(metadata[doc_id]["union_rank"]) <= 20),
        key=lambda doc_id: (-float(direct_scores[doc_id]), int(metadata[doc_id]["candidate_rank"]), doc_key_numeric(doc_id)),
    )

    def find_swap(position: int) -> dict[str, Any] | None:
        displaced_id = base[position - 1]
        displaced = metadata[displaced_id]
        displaced_score = float(direct_scores[displaced_id])
        for candidate_id in candidate_order:
            candidate = metadata[candidate_id]
            candidate_score = float(direct_scores[candidate_id])
            margin = candidate_score - displaced_score
            if not candidate_score > displaced_score:
                continue
            if not margin > 0.0 or margin < float(margin_threshold):
                continue
            safe, reasons = guard_safe(candidate, displaced)
            if not safe:
                continue
            return {
                "pos": position,
                "added": candidate_id,
                "dropped": displaced_id,
                "direct_score": candidate_score,
                "displaced_direct_score": displaced_score,
                "direct_score_margin": margin,
                "candidate_rank": int(candidate["candidate_rank"]),
                "displaced_candidate_rank": int(displaced["candidate_rank"]),
                "source_support": int(candidate["source_support"]),
                "displaced_source_support": int(displaced["source_support"]),
                "safety_reasons": reasons,
                "bge_rank": int(candidate["bge_rank"]),
                "displaced_bge_rank": int(displaced["bge_rank"]),
            }
        return None

    # Rank 5 is always attempted first.  Rank 4 is considered only when rank
    # 5 has no eligible candidate; this enforces the max-one-new-doc rule.
    swap = find_swap(5)
    fallback_reason = "rank5_eligible"
    if swap is None:
        fallback_reason = "rank5_no_eligible_candidate"
        swap = find_swap(4)
    proposed = list(base)
    if swap is not None:
        proposed[int(swap["pos"]) - 1] = str(swap["added"])
        if len(set(proposed)) != 5:
            raise RuntimeError(f"guarded policy duplicate for {qid}")
    trace = {
        "query_id": qid,
        "baseline_top5": base,
        "proposed_top5": proposed,
        "swap": swap,
        "margin_threshold": float(margin_threshold),
        "rank4_fallback_reason": fallback_reason,
        "top1_top3_locked": proposed[:3] == base[:3],
        "candidate_pool_size": len(candidate_order),
    }
    return proposed, trace


def evaluate_guarded(
    baseline: dict[str, list[str]],
    proposed: dict[str, list[str]],
    traces: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    folds: dict[str, int],
    provenance_pass: bool,
) -> dict[str, Any]:
    overall, by_fold = fold_metrics(baseline, proposed, gold, folds)
    changed = improved = harmed = neutral = gained = lost = 0
    rank_changes: Counter[str] = Counter()
    top13_changed = 0
    for qid, new_top5 in proposed.items():
        old_top5 = baseline[qid]
        if new_top5 == old_top5:
            continue
        changed += 1
        wanted = {str(value) for value in gold[qid].get("answer", [])}
        old_hits = len(wanted & set(old_top5))
        new_hits = len(wanted & set(new_top5))
        if new_hits > old_hits:
            improved += 1
            gained += new_hits - old_hits
        elif new_hits < old_hits:
            harmed += 1
            lost += old_hits - new_hits
        else:
            neutral += 1
        for index, (old_doc, new_doc) in enumerate(zip(old_top5, new_top5), 1):
            if old_doc != new_doc:
                rank_changes[str(index)] += 1
                if index <= 3:
                    top13_changed += 1
    recall_delta = float(overall["delta"]["recall"])
    precision_delta = float(overall["delta"]["precision"])
    gate = (
        recall_delta >= 0.0015
        and all(float(value["recall"]) >= 0 for value in by_fold.values())
        and precision_delta >= -0.001
        and top13_changed == 0
        and provenance_pass
    )
    return {
        **overall,
        "delta_by_fold": by_fold,
        "queries_changed": changed,
        "queries_improved": improved,
        "queries_harmed": harmed,
        "queries_neutral": neutral,
        "relevant_docs_gained": gained,
        "relevant_docs_lost": lost,
        "net_relevant_doc_change": gained - lost,
        "rank4_changes": int(rank_changes["4"]),
        "rank5_changes": int(rank_changes["5"]),
        "top1_top3_changed": top13_changed,
        "top1_top3_lock_pass": top13_changed == 0,
        "provenance_pass": provenance_pass,
        "gate": "PASS" if gate else "FAIL",
    }


def direct_replay_diagnostics(
    baseline: dict[str, list[str]],
    direct_rankings: dict[str, list[str]],
    direct_scores: dict[str, dict[str, float]],
    metadata: dict[str, dict[str, dict[str, Any]]],
    gold: dict[str, dict[str, Any]],
    folds: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proposed = {qid: direct_rankings[qid][:5] for qid in baseline}
    overall, by_fold = fold_metrics(baseline, proposed, gold, folds)
    changed = improved = harmed = neutral = gained = lost = 0
    gains_top13 = gains_rank45 = harms_top13 = harms_rank45 = 0
    replacement_rows: list[dict[str, Any]] = []
    for qid in sorted(baseline, key=qkey):
        old = baseline[qid]
        new = proposed[qid]
        wanted = {str(value) for value in gold[qid].get("answer", [])}
        old_hits = len(wanted & set(old))
        new_hits = len(wanted & set(new))
        impact = "NEUTRAL"
        if new_hits > old_hits:
            impact = "HELP"
            improved += 1
            gained += new_hits - old_hits
        elif new_hits < old_hits:
            impact = "HARM"
            harmed += 1
            lost += old_hits - new_hits
        elif old != new:
            neutral += 1
        if old == new:
            continue
        changed += 1
        changed_positions = [index for index, pair in enumerate(zip(old, new), 1) if pair[0] != pair[1]]
        if impact == "HELP":
            if any(index <= 3 for index in changed_positions):
                gains_top13 += 1
            else:
                gains_rank45 += 1
        elif impact == "HARM":
            if any(index <= 3 for index in changed_positions):
                harms_top13 += 1
            else:
                harms_rank45 += 1
        full_rank = {doc_id: index for index, doc_id in enumerate(direct_rankings[qid], 1)}
        for index, (old_doc, new_doc) in enumerate(zip(old, new), 1):
            if old_doc == new_doc:
                continue
            incoming = metadata[qid][new_doc]
            dropped = metadata[qid][old_doc]
            replacement_rows.append({
                "query_id": qid,
                "fold": folds[qid],
                "impact": impact,
                "position": index,
                "position_label": f"Top{index}",
                "baseline_document_id": old_doc,
                "direct_document_id": new_doc,
                "baseline_rank": index,
                "direct_rank": full_rank[new_doc],
                "direct_score": float(direct_scores[qid][new_doc]),
                "displaced_direct_score": float(direct_scores[qid][old_doc]),
                "direct_score_margin_vs_displaced": float(direct_scores[qid][new_doc] - direct_scores[qid][old_doc]),
                "candidate_union_rank": int(incoming["candidate_rank"]),
                "dense_rank": incoming["source_ranks"].get("dense"),
                "bm25_rank": incoming["source_ranks"].get("bm25"),
                "knn_word_rank": incoming["source_ranks"].get("knn_word"),
                "bge_rank": int(incoming["bge_rank"]),
                "source_support": int(incoming["source_support"]),
                "baseline_candidate_union_rank": int(dropped["candidate_rank"]),
                "baseline_dense_rank": dropped["source_ranks"].get("dense"),
                "baseline_bm25_rank": dropped["source_ranks"].get("bm25"),
                "baseline_knn_word_rank": dropped["source_ranks"].get("knn_word"),
                "baseline_bge_rank": int(dropped["bge_rank"]),
                "baseline_source_support": int(dropped["source_support"]),
                "direct_document_relevant": new_doc in wanted,
                "baseline_document_relevant": old_doc in wanted,
                "new_to_baseline_top5": new_doc not in set(old),
            })
    summary = {
        **overall,
        "delta_by_fold": by_fold,
        "queries_changed": changed,
        "queries_improved": improved,
        "queries_harmed": harmed,
        "queries_neutral_changed": neutral,
        "relevant_docs_gained": gained,
        "relevant_docs_lost": lost,
        "net_relevant_doc_change": gained - lost,
        "gains_requiring_top1_3_change": gains_top13,
        "gains_only_rank4_5": gains_rank45,
        "harms_from_top1_3_change": harms_top13,
        "harms_only_rank4_5": harms_rank45,
        "direct_top1_top3_changed_queries": sum(
            any(index <= 3 for index, (old_doc, new_doc) in enumerate(zip(baseline[qid], proposed[qid]), 1) if old_doc != new_doc)
            for qid in baseline
        ),
    }
    return summary, replacement_rows


def make_guarded_contract() -> dict[str, Any]:
    return {
        "policy_name": "GUARDED_DIRECT_K20_RESIDUAL_V1",
        "status": "FROZEN_BEFORE_METRIC_EVALUATION",
        "incumbent": "RETRIEVAL_RRF_NO_LABEL",
        "candidate_depth": 20,
        "top1_top3_locked": True,
        "allowed_positions": [4, 5],
        "max_new_documents_per_query": 1,
        "position_preference": "rank5_then_rank4_only_if_rank5_has_no_eligible_candidate",
        "candidate_conditions": {
            "not_in_incumbent_top5": True,
            "direct_score_strictly_greater_than_displaced": True,
            "retrieval_safety_any_of": [
                "candidate_union_rank<=displaced_union_rank",
                "candidate_source_support>displaced_source_support",
                "candidate_present_in_at_least_2_of_dense_bm25_knn_bge",
            ],
        },
        "direct_score_margin_threshold": 0.0,
        "tie_break": ["direct_score_desc", "candidate_rank_asc", "numeric_document_id_asc"],
        "ambiguous_safety": "NO_OP",
        "estimator": {
            "family": "StandardScaler+LogisticRegression",
            "C": 0.2,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 500,
            "random_state": 2026,
        },
        "source_depth": 20,
        "negatives_per_query": 30,
        "outer_oof": True,
        "fold0_used": False,
        "private_labels_used": False,
        "gpu_runs": 0,
        "modal_runs": 0,
    }


def select_inner_threshold(
    outer_fold: int,
    train_ids: list[str],
    folds: dict[str, int],
    gold: dict[str, dict[str, Any]],
    baseline: dict[str, list[str]],
    metadata: dict[str, dict[str, dict[str, Any]]],
    feature_cache: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    """Select one threshold using only inner-OOF predictions in outer train."""

    inner_fold_values = sorted({folds[qid] for qid in train_ids})
    inner_scores: dict[str, dict[str, float]] = {}
    for inner_holdout in inner_fold_values:
        inner_train = [qid for qid in sorted(train_ids) if folds[qid] != inner_holdout]
        inner_test = [qid for qid in sorted(train_ids) if folds[qid] == inner_holdout]
        model = fit_direct_model(inner_train, feature_cache, gold)
        scores, _ranking = score_query_ids(model, inner_test, feature_cache)
        inner_scores.update(scores)

    provisional_margins: list[float] = []
    provisional: dict[str, dict[str, Any]] = {}
    for qid in sorted(train_ids, key=qkey):
        proposed, trace = apply_guarded_policy(qid, baseline[qid], inner_scores[qid], metadata[qid], margin_threshold=0.0)
        provisional[qid] = trace
        if trace["swap"] is not None:
            provisional_margins.append(float(trace["swap"]["direct_score_margin"]))
    if not provisional_margins:
        return {
            "outer_fold": outer_fold,
            "status": "NO_INNER_PROPOSALS",
            "selected_threshold": None,
            "quantile_candidates": [],
            "inner_fold_values": inner_fold_values,
        }

    quantile_candidates = [
        {
            "quantile": quantile,
            "threshold": float(np.quantile(np.asarray(provisional_margins, dtype=np.float64), quantile, method="linear")),
        }
        for quantile in QUANTILES
    ]
    evaluated: list[dict[str, Any]] = []
    for candidate in quantile_candidates:
        threshold = float(candidate["threshold"])
        proposed: dict[str, list[str]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for qid in sorted(train_ids, key=qkey):
            proposed[qid], traces[qid] = apply_guarded_policy(
                qid,
                baseline[qid],
                inner_scores[qid],
                metadata[qid],
                margin_threshold=threshold,
            )
        overall, by_fold = fold_metrics(
            {qid: baseline[qid] for qid in train_ids},
            proposed,
            {qid: gold[qid] for qid in train_ids},
            folds,
        )
        valid = (
            all(float(value["recall"]) >= 0 for value in by_fold.values())
            and float(overall["delta"]["precision"]) >= -0.001
        )
        evaluated.append({
            **candidate,
            "valid": valid,
            "recall_delta": float(overall["delta"]["recall"]),
            "precision_delta": float(overall["delta"]["precision"]),
            "fold_recall_deltas": {key: float(value["recall"]) for key, value in by_fold.items()},
            "interventions": sum(trace["swap"] is not None for trace in traces.values()),
        })
    valid = [row for row in evaluated if row["valid"]]
    if not valid:
        return {
            "outer_fold": outer_fold,
            "status": "NO_VALID_THRESHOLD",
            "selected_threshold": None,
            "quantile_candidates": evaluated,
            "inner_fold_values": inner_fold_values,
            "inner_provisional_proposals": len(provisional_margins),
        }
    selected = sorted(valid, key=lambda row: (-float(row["recall_delta"]), -float(row["threshold"]), -float(row["quantile"])))[0]
    return {
        "outer_fold": outer_fold,
        "status": "SELECTED",
        "selected_threshold": float(selected["threshold"]),
        "selected_quantile": float(selected["quantile"]),
        "quantile_candidates": evaluated,
        "inner_fold_values": inner_fold_values,
        "inner_provisional_proposals": len(provisional_margins),
    }


def nested_outer_policy(
    folds: dict[str, int],
    gold: dict[str, dict[str, Any]],
    baseline: dict[str, list[str]],
    metadata: dict[str, dict[str, dict[str, Any]]],
    feature_cache: dict[str, dict[str, list[float]]],
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    query_ids = sorted(folds)
    outer_scores: dict[str, dict[str, float]] = {}
    thresholds: dict[int, dict[str, Any]] = {}
    proposed: dict[str, list[str]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for outer_fold in sorted(set(folds.values())):
        train_ids = [qid for qid in query_ids if folds[qid] != outer_fold]
        test_ids = [qid for qid in query_ids if folds[qid] == outer_fold]
        calibration = select_inner_threshold(
            outer_fold,
            train_ids,
            folds,
            gold,
            baseline,
            metadata,
            feature_cache,
        )
        thresholds[outer_fold] = calibration
        model = fit_direct_model(train_ids, feature_cache, gold)
        scores, _rankings = score_query_ids(model, test_ids, feature_cache)
        outer_scores.update(scores)
        threshold = calibration.get("selected_threshold")
        if threshold is None:
            threshold = float("inf")
        for qid in test_ids:
            proposed[qid], traces[qid] = apply_guarded_policy(
                qid,
                baseline[qid],
                scores[qid],
                metadata[qid],
                margin_threshold=float(threshold),
            )
    return thresholds, proposed, traces, outer_scores


def final_global_threshold_from_strict_oof(
    outer_scores: dict[str, dict[str, float]],
    baseline: dict[str, list[str]],
    metadata: dict[str, dict[str, dict[str, Any]]],
    folds: dict[str, int],
    gold: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    margins: list[float] = []
    for qid in sorted(outer_scores, key=qkey):
        _proposed, trace = apply_guarded_policy(qid, baseline[qid], outer_scores[qid], metadata[qid], margin_threshold=0.0)
        if trace["swap"] is not None:
            margins.append(float(trace["swap"]["direct_score_margin"]))
    if not margins:
        return {"status": "NO_PROPOSALS", "selected_threshold": None, "candidates": []}
    candidates: list[dict[str, Any]] = []
    for quantile in QUANTILES:
        threshold = float(np.quantile(np.asarray(margins, dtype=np.float64), quantile, method="linear"))
        proposed: dict[str, list[str]] = {}
        for qid in sorted(outer_scores, key=qkey):
            proposed[qid], _trace = apply_guarded_policy(
                qid, baseline[qid], outer_scores[qid], metadata[qid], margin_threshold=threshold
            )
        overall, by_fold = fold_metrics(baseline, proposed, gold, folds)
        candidates.append({
            "quantile": quantile,
            "threshold": threshold,
            "valid": all(float(value["recall"]) >= 0 for value in by_fold.values()) and float(overall["delta"]["precision"]) >= -0.001,
            "recall_delta": float(overall["delta"]["recall"]),
            "precision_delta": float(overall["delta"]["precision"]),
            "fold_recall_deltas": {key: float(value["recall"]) for key, value in by_fold.items()},
        })
    valid = [row for row in candidates if row["valid"]]
    if not valid:
        return {"status": "NO_VALID_THRESHOLD", "selected_threshold": None, "candidates": candidates}
    selected = sorted(valid, key=lambda row: (-float(row["recall_delta"]), -float(row["threshold"]), -float(row["quantile"])))[0]
    return {
        "status": "SELECTED",
        "selected_threshold": float(selected["threshold"]),
        "selected_quantile": float(selected["quantile"]),
        "candidates": candidates,
    }


def build_private_inputs() -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], dict[str, Any]]]:
    candidate_rows = read_jsonl(PRIVATE_CANDIDATES)
    score_rows = read_jsonl(PRIVATE_SCORES)
    score_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in score_rows:
        key = key_of(row)
        if key in score_by_key:
            raise RuntimeError(f"Private score duplicate: {key}")
        score_by_key[key] = row
    docs_by_query: dict[str, list[dict[str, Any]]] = {}
    expected: set[tuple[str, str]] = set()
    for row in candidate_rows:
        qid = str(row["query_id"])
        docs: list[dict[str, Any]] = []
        for candidate in row.get("candidates", []):
            did = str(candidate["document_id"])
            source_ranks = {str(source): int(rank) for source, rank in (candidate.get("source_ranks") or {}).items()}
            docs.append({
                "query_id": qid,
                "document_id": did,
                "candidate_rank": int(candidate["candidate_rank"]),
                "union_rank": int(candidate["candidate_rank"]),
                "rrf_score": float(candidate.get("rrf_score", 0.0)),
                "source_support": len(source_ranks),
                "source_ranks": source_ranks,
            })
            expected.add((qid, did))
        docs_by_query[qid] = docs
    if len(docs_by_query) != 2080 or any(len(rows) != 20 for rows in docs_by_query.values()):
        raise RuntimeError("Private K20 candidate universe is not 2080 x 20")
    if expected != set(score_by_key):
        raise RuntimeError("Private K20 candidate/score identity mismatch")
    return docs_by_query, score_by_key


def write_private_submission(
    proposed: dict[str, list[str]],
    final_threshold: dict[str, Any],
    private_summary: dict[str, Any],
    final_model_info: dict[str, Any],
) -> dict[str, Any]:
    output_dir = ROOT / "private_task1/submissions/sprint48_guarded_direct"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "submission_private_guarded_direct.json"
    zip_path = output_dir / "submission_private_guarded_direct.zip"
    manifest_path = output_dir / "manifest.json"
    validation_path = output_dir / "validation_report.json"
    payload = {qid: {"answer": docs} for qid, docs in sorted(proposed.items(), key=lambda item: qkey(item[0]))}
    if len(payload) != 2080 or any(len(row["answer"]) != 5 or len(set(row["answer"])) != 5 for row in payload.values()):
        raise RuntimeError("invalid guarded Private submission shape")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="submission.json")
    validation = {
        "status": "PENDING_CANONICAL_VALIDATOR",
        "validator": "scripts/submission/validate_legal_ir_submission.py",
        "json": rel(json_path),
        "json_sha256": sha256(json_path),
        "zip": rel(zip_path),
        "zip_sha256": sha256(zip_path),
        "private_labels_used": False,
        "private_correctness_claim": False,
    }
    write_json(validation_path, validation)
    manifest = {
        "status": "READY_FOR_MANUAL_REVIEW",
        "policy": "GUARDED_DIRECT_K20_RESIDUAL_V1_OR_V2_NESTED",
        "queries": 2080,
        "top_k": 5,
        "source_submission": rel(PV1_SUBMISSION_JSON),
        "source_submission_sha256": sha256(PV1_SUBMISSION_JSON),
        "json": rel(json_path),
        "json_sha256": sha256(json_path),
        "zip": rel(zip_path),
        "zip_sha256": sha256(zip_path),
        "final_threshold": final_threshold,
        "final_model": final_model_info,
        "private_summary": private_summary,
        "private_labels_used": False,
        "fold0_used": False,
        "gpu_runs": 0,
        "modal_runs": 0,
        "validation_report": rel(validation_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Stage 0 is a recorded freeze, not a mutation of historical artifacts.
    stage0 = {
        "STEP4_K20_PORT": "CLOSED",
        "DIRECT_FULL_TOP5": "SCIENTIFIC_FAIL_AS_FULL_REPLACEMENT",
        "BGE_FT_V2": "CLOSED",
        "BGE_FT_V3_CONFLICT_ANCHOR": "CLOSED",
        "v3_f1_recall_delta": -0.0013095238095237605,
        "historical_artifacts_mutated": False,
    }

    expected_artifact = json.loads(DIRECT_ARTIFACT.read_text(encoding="utf-8"))
    direct_signal_provenance = (
        expected_artifact.get("outer_oof") is True
        and int(expected_artifact.get("source_depth", -1)) == 20
        and int(expected_artifact.get("negatives_per_query", -1)) == 30
        and expected_artifact.get("private_application") == "NOT_RUN"
        and expected_artifact.get("fallback_gate") is False
        and all(name in expected_artifact.get("feature_names", []) for name in (
            "rr_bm25", "rr_dense", "rr_knn_word", "source_support_count",
            "bge_base_minmax", "bge_base_raw", "bge_ft_minmax", "bge_ft_raw",
            "candidate_rank_norm", "source_support", "union_rrf",
        ))
    )
    if not direct_signal_provenance:
        write_json(OUT / "guarded_direct_result.json", {
            "DIRECT_SIGNAL_GATE": "FAIL",
            "DIRECT_REPLAY_GATE": "NOT_RUN",
            "reason": "existing direct artifact provenance contract failed",
            "gpu_runs": 0,
            "modal_runs": 0,
            "stage0": stage0,
        })
        return 2

    all_folds = load_fold_map()
    target_qids = {qid for qid, fold in all_folds.items() if fold in TARGET_FOLDS}
    # The candidate/worklist artifacts are the strict F1--F4 population.  Do
    # not pass Fold0 IDs into the OOF implementation; the original recovery
    # call also supplied a fold map restricted to the authorized population.
    folds = {qid: all_folds[qid] for qid in target_qids}
    gold_all = load_gold()
    if len(target_qids) != EXPECTED_F1F4_QUERIES or any(qid not in gold_all for qid in target_qids):
        raise RuntimeError("F1-F4 fold/gold population mismatch")
    gold = {qid: gold_all[qid] for qid in target_qids}
    _candidate_rows, _metadata, docs_by_query = load_pv1_metadata()
    _worklist_rows, worklist_by_key = load_pv1_worklist()
    reconstructed, merge_info = load_and_merge_scores(worklist_by_key)
    score_by_key = {key_of(row): row for row in reconstructed}
    baseline = baseline_predictions(docs_by_query, score_by_key)
    metadata = candidate_metadata(docs_by_query, score_by_key)
    source_rankings, external, feature_names, feature_cache = build_direct_inputs(docs_by_query, score_by_key)
    if set(docs_by_query) != target_qids or len(score_by_key) != EXPECTED_F1F4_QDOCS:
        raise RuntimeError("direct input universe mismatch")

    contract = make_guarded_contract()
    contract_path = OUT / "guarded_direct_k20_residual_v1_contract.json"
    write_json(contract_path, contract)
    contract_sha = sha256(contract_path)

    direct_scores, direct_rankings, fold_reports = direct_outer_oof(folds, gold, feature_cache)
    # Add exact feature names after the model replay; coefficients retain the
    # model's native ordering from _feature_names.
    for report in fold_reports:
        report["feature_names"] = feature_names
        report["heldout"] = metric(
            {qid: direct_rankings[qid][:5] for qid in folds if folds[qid] == report["fold"]},
            {qid: gold[qid] for qid in folds if folds[qid] == report["fold"]},
        )
        report["feature_coefficients"] = dict(zip(feature_names, report["feature_coefficients"]))
    direct_summary, replacement_rows = direct_replay_diagnostics(
        baseline, direct_rankings, direct_scores, metadata, gold, folds
    )
    exact_values = (
        abs(float(direct_summary["proposed"]["recall"]) - EXPECTED_DIRECT["recall"]) <= 1e-14
        and abs(float(direct_summary["proposed"]["precision"]) - EXPECTED_DIRECT["precision"]) <= 1e-14
        and abs(float(direct_summary["delta"]["recall"]) - EXPECTED_DIRECT["delta_recall"]) <= 1e-14
        and abs(float(direct_summary["delta"]["precision"]) - EXPECTED_DIRECT["delta_precision"]) <= 1e-14
        and direct_summary["queries_changed"] == EXPECTED_DIRECT["queries_changed"]
        and direct_summary["queries_improved"] == EXPECTED_DIRECT["queries_improved"]
        and direct_summary["queries_harmed"] == EXPECTED_DIRECT["queries_harmed"]
        and direct_summary["relevant_docs_gained"] == EXPECTED_DIRECT["relevant_docs_gained"]
        and direct_summary["relevant_docs_lost"] == EXPECTED_DIRECT["relevant_docs_lost"]
        and direct_summary["net_relevant_doc_change"] == EXPECTED_DIRECT["net_relevant_doc_change"]
        and all(
            abs(float(direct_summary["delta_by_fold"][fold]["recall"]) - value) <= 1e-14
            for fold, value in EXPECTED_DIRECT["fold_recall_deltas"].items()
        )
    )
    coefficient_diffs: list[float] = []
    for actual, expected in zip(fold_reports, expected_artifact.get("fold_reports", [])):
        expected_coeff = expected.get("feature_coefficients", {})
        coefficient_diffs.extend(
            abs(float(actual["feature_coefficients"][name]) - float(expected_coeff[name]))
            for name in feature_names
            if name in expected_coeff
        )
    direct_replay_gate = exact_values and bool(coefficient_diffs) and max(coefficient_diffs) <= 1e-12
    if not direct_replay_gate:
        write_json(OUT / "guarded_direct_result.json", {
            "DIRECT_SIGNAL_GATE": "PASS",
            "DIRECT_REPLAY_GATE": "FAIL",
            "direct_summary": direct_summary,
            "max_fold_coefficient_abs_diff": max(coefficient_diffs) if coefficient_diffs else None,
            "gpu_runs": 0,
            "modal_runs": 0,
            "stage0": stage0,
        })
        return 3

    write_jsonl(OUT / "direct_oof_scores.jsonl", [
        {
            "query_id": qid,
            "fold": folds[qid],
            "baseline_top5": baseline[qid],
            "direct_top5": direct_rankings[qid][:5],
            "scores": [
                {"document_id": did, "direct_rank": index, "direct_score": float(direct_scores[qid][did])}
                for index, did in enumerate(sorted(direct_scores[qid], key=lambda doc_id: (-direct_scores[qid][doc_id], doc_id)), 1)
            ],
        }
        for qid in sorted(direct_scores, key=qkey)
    ])
    write_jsonl(OUT / "direct_full_top5_replacements.jsonl", replacement_rows)
    write_json(OUT / "direct_replay_summary.json", {
        **direct_summary,
        "policy": "DIRECT_K20_PLUS_ANCHORS_DOCUMENT_TOP5",
        "outer_oof": True,
        "fold0_used": False,
        "private_labels_used": False,
        "feature_names": feature_names,
        "fold_reports": fold_reports,
        "max_fold_coefficient_abs_diff": max(coefficient_diffs),
    })

    provenance_pass = (
        direct_signal_provenance
        and direct_replay_gate
        and merge_info["selected_chunk_mismatches"] == 0
        and len(score_by_key) == EXPECTED_F1F4_QDOCS
        and all(folds[qid] in TARGET_FOLDS for qid in target_qids)
    )

    v1_proposed: dict[str, list[str]] = {}
    v1_traces: dict[str, dict[str, Any]] = {}
    for qid in sorted(baseline, key=qkey):
        v1_proposed[qid], v1_traces[qid] = apply_guarded_policy(
            qid, baseline[qid], direct_scores[qid], metadata[qid], margin_threshold=0.0
        )
    v1_result = evaluate_guarded(baseline, v1_proposed, v1_traces, gold, folds, provenance_pass)
    write_json(OUT / "guarded_direct_v1_result.json", {
        **v1_result,
        "policy": "GUARDED_DIRECT_K20_RESIDUAL_V1",
        "contract": rel(contract_path),
        "contract_sha256": contract_sha,
        "fold0_used": False,
        "private_labels_used": False,
        "gpu_runs": 0,
        "modal_runs": 0,
    })
    write_jsonl(OUT / "guarded_direct_v1_swaps.jsonl", [
        {"query_id": qid, "fold": folds[qid], **trace["swap"]}
        for qid, trace in sorted(v1_traces.items(), key=lambda item: qkey(item[0]))
        if trace["swap"] is not None
    ])

    nested_run = False
    nested_result: dict[str, Any] | None = None
    nested_thresholds: dict[int, dict[str, Any]] | None = None
    nested_outer_scores: dict[str, dict[str, float]] | None = None
    if float(v1_result["delta"]["recall"]) > 0 and v1_result["gate"] != "PASS":
        nested_run = True
        nested_thresholds, nested_proposed, nested_traces, nested_outer_scores = nested_outer_policy(
            folds, gold, baseline, metadata, feature_cache
        )
        nested_result = evaluate_guarded(
            baseline, nested_proposed, nested_traces, gold, folds, provenance_pass
        )
        nested_contract = {
            "policy_name": "GUARDED_DIRECT_K20_RESIDUAL_V2_NESTED",
            "status": "FROZEN_AFTER_PRIMARY_POSITIVE_FAIL",
            "base_contract": "GUARDED_DIRECT_K20_RESIDUAL_V1",
            "outer_fold_threshold_selection": "inner-OOF-only",
            "quantile_grid": list(QUANTILES),
            "selection": "maximize recall delta; require every inner fold recall delta>=0 and precision delta>=-0.001; ties choose higher threshold",
            "outer_labels_used_for_threshold": False,
            "top1_top3_locked": True,
            "max_new_documents_per_query": 1,
            "source_depth": 20,
            "negatives_per_query": 30,
            "fold0_used": False,
            "private_labels_used": False,
            "gpu_runs": 0,
            "modal_runs": 0,
        }
        nested_contract_path = OUT / "guarded_direct_k20_residual_v2_nested_contract.json"
        write_json(nested_contract_path, nested_contract)
        nested_contract_sha = sha256(nested_contract_path)
        write_json(OUT / "guarded_direct_nested_result.json", {
            **nested_result,
            "policy": "GUARDED_DIRECT_K20_RESIDUAL_V2_NESTED",
            "contract": rel(nested_contract_path),
            "contract_sha256": nested_contract_sha,
            "thresholds_by_outer_fold": nested_thresholds,
            "fold0_used": False,
            "private_labels_used": False,
            "gpu_runs": 0,
            "modal_runs": 0,
        })
        write_jsonl(OUT / "guarded_direct_nested_swaps.jsonl", [
            {"query_id": qid, "fold": folds[qid], **trace["swap"]}
            for qid, trace in sorted(nested_traces.items(), key=lambda item: qkey(item[0]))
            if trace["swap"] is not None
        ])

    selected_policy = "NONE"
    selected_result: dict[str, Any] | None = None
    if v1_result["gate"] == "PASS":
        selected_policy = "V1"
        selected_result = v1_result
    elif nested_result is not None and nested_result["gate"] == "PASS":
        selected_policy = "V2_NESTED"
        selected_result = nested_result

    private_application: dict[str, Any] = {"status": "NOT_RUN"}
    submission: dict[str, Any] = {"status": "NOT_CREATED"}
    final_threshold: dict[str, Any] | None = None
    final_model_info: dict[str, Any] = {"status": "NOT_CREATED"}
    if selected_policy != "NONE":
        private_docs, private_score_by_key = build_private_inputs()
        private_metadata = candidate_metadata(private_docs, private_score_by_key)
        _private_sources, _private_external, _private_names, private_feature_cache = build_direct_inputs(private_docs, private_score_by_key)
        final_model = fit_direct_model(sorted(target_qids), feature_cache, gold)
        scaler = final_model.named_steps["standardscaler"]
        logistic = final_model.named_steps["logisticregression"]
        final_model_path = OUT / "final_direct_model.json"
        write_json(final_model_path, {
            "status": "FROZEN_FINAL_CPU_MODEL",
            "policy": selected_policy,
            "estimator": {
                "family": "StandardScaler+LogisticRegression",
                "C": 0.2,
                "class_weight": "balanced",
                "solver": "liblinear",
                "max_iter": 500,
                "random_state": 2026,
            },
            "feature_names": feature_names,
            "feature_schema_sha256": hashlib.sha256(
                json.dumps(feature_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "standard_scaler": {
                "mean": [float(value) for value in scaler.mean_],
                "scale": [float(value) for value in scaler.scale_],
                "var": [float(value) for value in scaler.var_],
            },
            "logistic_regression": {
                "classes": [int(value) for value in logistic.classes_],
                "coef": [[float(value) for value in row] for row in logistic.coef_],
                "intercept": [float(value) for value in logistic.intercept_],
            },
            "training_population": {
                "qdocs": EXPECTED_F1F4_QUERIES,
                "folds": [1, 2, 3, 4],
                "fold0_used": False,
                "private_labels_used": False,
            },
            "contract_sha256": contract_sha,
            "direct_artifact_sha256": sha256(DIRECT_ARTIFACT),
            "input_worklist_sha256": sha256(PV1_WORKLIST),
            "gpu_runs": 0,
            "modal_runs": 0,
        })
        final_model_info = {
            "status": "FROZEN",
            "path": rel(final_model_path),
            "sha256": sha256(final_model_path),
            "feature_schema_sha256": hashlib.sha256(
                json.dumps(feature_names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        private_scores, _private_rankings = score_query_ids(final_model, sorted(private_docs, key=qkey), private_feature_cache)
        if selected_policy == "V1":
            final_threshold = {"status": "V1_FIXED", "selected_threshold": 0.0}
        else:
            if nested_outer_scores is None:
                raise RuntimeError("nested scores missing for selected V2 policy")
            final_threshold = final_global_threshold_from_strict_oof(
                nested_outer_scores, baseline, metadata, folds, gold
            )
        private_payload = json.loads(PV1_SUBMISSION_JSON.read_text(encoding="utf-8"))
        private_baseline = {str(qid): [str(value) for value in row["answer"]] for qid, row in private_payload.items()}
        private_proposed: dict[str, list[str]] = {}
        private_traces: dict[str, dict[str, Any]] = {}
        threshold = final_threshold.get("selected_threshold")
        if threshold is None:
            raise RuntimeError("selected policy has no deployable threshold")
        for qid in sorted(private_docs, key=qkey):
            private_proposed[qid], private_traces[qid] = apply_guarded_policy(
                qid,
                private_baseline[qid],
                private_scores[qid],
                private_metadata[qid],
                margin_threshold=float(threshold),
            )
        private_top13_changed = sum(
            any(old != new for old, new in zip(private_baseline[qid][:3], private_proposed[qid][:3]))
            for qid in private_baseline
        )
        private_application = {
            "status": "RUN",
            "queries": len(private_proposed),
            "queries_changed": sum(private_proposed[qid] != private_baseline[qid] for qid in private_proposed),
            "rank4_changes": sum(
                private_proposed[qid][3] != private_baseline[qid][3] for qid in private_proposed
            ),
            "rank5_changes": sum(
                private_proposed[qid][4] != private_baseline[qid][4] for qid in private_proposed
            ),
            "top1_3_changed": private_top13_changed,
            "max_new_documents_per_query": max(
                len(set(private_proposed[qid]) - set(private_baseline[qid])) for qid in private_proposed
            ),
            "top5_jaccard_mean": float(np.mean([
                len(set(private_proposed[qid]) & set(private_baseline[qid])) / len(set(private_proposed[qid]) | set(private_baseline[qid]))
                for qid in private_proposed
            ])),
            "candidate_rank_distribution": dict(Counter(
                trace["swap"]["candidate_rank"] for trace in private_traces.values() if trace["swap"] is not None
            )),
            "source_support_distribution": dict(Counter(
                trace["swap"]["source_support"] for trace in private_traces.values() if trace["swap"] is not None
            )),
            "private_labels_used": False,
            "correctness_claim": False,
        }
        if private_top13_changed != 0:
            raise RuntimeError("guarded Private application changed Top1-3")
        submission = write_private_submission(private_proposed, final_threshold, private_application, final_model_info)

    final_status = "READY_FOR_MANUAL_SUBMISSION" if selected_policy != "NONE" and submission.get("status") != "NOT_CREATED" else "DIRECT_RESIDUAL_SCIENTIFIC_FAIL"
    result = {
        "DIRECT_SIGNAL_GATE": "PASS",
        "DIRECT_REPLAY_GATE": "PASS",
        "FULL_DIRECT_RECALL": direct_summary["proposed"]["recall"],
        "FULL_DIRECT_F3_DELTA": direct_summary["delta_by_fold"]["F3"]["recall"],
        "GAINS_REQUIRING_TOP1_3_CHANGE": direct_summary["gains_requiring_top1_3_change"],
        "HARMS_FROM_TOP1_3_CHANGE": direct_summary["harms_from_top1_3_change"],
        "PRIMARY_POLICY": "GUARDED_DIRECT_K20_RESIDUAL_V1",
        "PRIMARY_RECALL": v1_result["proposed"]["recall"],
        "PRIMARY_DELTA": v1_result["delta"]["recall"],
        "PRIMARY_FOLD_DELTAS": {key: value["recall"] for key, value in v1_result["delta_by_fold"].items()},
        "PRIMARY_CHANGED_QUERIES": v1_result["queries_changed"],
        "PRIMARY_IMPROVED": v1_result["queries_improved"],
        "PRIMARY_HARMED": v1_result["queries_harmed"],
        "PRIMARY_GATE": v1_result["gate"],
        "NESTED_RUN": "YES" if nested_run else "NO",
        "NESTED_RECALL": None if nested_result is None else nested_result["proposed"]["recall"],
        "NESTED_DELTA": None if nested_result is None else nested_result["delta"]["recall"],
        "NESTED_FOLD_DELTAS": None if nested_result is None else {key: value["recall"] for key, value in nested_result["delta_by_fold"].items()},
        "NESTED_GATE": "NOT_RUN" if nested_result is None else nested_result["gate"],
        "SELECTED_POLICY": selected_policy,
        "PRIVATE_APPLICATION": private_application["status"],
        "PRIVATE_CHANGED_QUERIES": private_application.get("queries_changed"),
        "PRIVATE_TOP1_3_CHANGED": private_application.get("top1_3_changed"),
        "SUBMISSION_GATE": "PASS" if submission.get("status") != "NOT_CREATED" else "NOT_CREATED",
        "SUBMISSION_ZIP": submission.get("zip"),
        "SUBMISSION_SHA256": submission.get("zip_sha256"),
        "GPU_RUNS": 0,
        "MODAL_RUNS": 0,
        "stage0": stage0,
        "provenance": {
            "outer_oof": True,
            "fold0_used": False,
            "private_labels_used": False,
            "public_labels_used": False,
            "model_loaded": False,
            "gpu_runs": 0,
            "modal_runs": 0,
            "contract_sha256": contract_sha,
        },
        "direct_replay": direct_summary,
        "v1": v1_result,
        "nested": nested_result,
        "private_application_detail": private_application,
        "final_threshold": final_threshold,
        "final_model": final_model_info,
        "submission": submission,
        "input_hashes": {
            name: freeze_file(path)
            for name, path in {
                "direct_artifact": DIRECT_ARTIFACT,
                "pv1_candidates": PV1_CANDIDATES,
                "pv1_worklist": PV1_WORKLIST,
                "pv1_delta_scores": PV1_DELTA_SCORES,
                "pv1_delta_metrics": PV1_DELTA_METRICS,
                "pv1_dense": PV1_DENSE,
                "pv1_bm25": PV1_BM25,
                "pv1_bm25_manifest": PV1_BM25_MANIFEST,
                "pv1_knn_manifest": PV1_KNN_MANIFEST,
                "pv1_k20_candidates": PV1_K20_CANDIDATES,
                "pv1_k20_worklist": PV1_K20_WORKLIST,
                "pv1_bge_merged": PV1_BGE_MERGED,
                "pv1_bge_manifest": PV1_BGE_MANIFEST,
                "pv1_submission": PV1_SUBMISSION_JSON,
                "pv1_submission_zip": PV1_SUBMISSION_ZIP,
                "pv1_submission_manifest": PV1_SUBMISSION_MANIFEST,
                "reusable_scores": REUSABLE_SCORES,
                "train_f1_f4": TRAIN,
                "folds": FOLDS,
                "v1_contract": contract_path,
            }.items()
        },
        "selected_policy": selected_policy,
        "FINAL_STATUS": final_status,
        "NEXT_ACTION": "Run canonical validator on the isolated guarded submission, then manually review; no automatic upload." if selected_policy != "NONE" else "Close guarded Direct residual branch; next exact branch is offline corrected historical Qwen single-rescue audit using existing scores.",
    }
    write_json(OUT / "guarded_direct_result.json", result)
    report_lines = [
        "# GUARDED_DIRECT_K20_RESIDUAL — CPU-only final experiment",
        "",
        "No GPU, Modal, neural inference, Fold0, public labels, or Private labels were used.",
        "",
        "## Gates",
        "",
        f"- DIRECT_SIGNAL_GATE: **{result['DIRECT_SIGNAL_GATE']}**.",
        f"- DIRECT_REPLAY_GATE: **{result['DIRECT_REPLAY_GATE']}**; full-direct Recall `{result['FULL_DIRECT_RECALL']:.15f}`.",
        f"- Full-direct F3 delta: `{result['FULL_DIRECT_F3_DELTA']:+.15f}`.",
        f"- Gains requiring Top1–3 changes: `{result['GAINS_REQUIRING_TOP1_3_CHANGE']}`; harms from Top1–3 changes: `{result['HARMS_FROM_TOP1_3_CHANGE']}`.",
        "",
        "## Guarded V1",
        "",
        f"- Recall `{v1_result['proposed']['recall']:.15f}`; delta `{v1_result['delta']['recall']:+.15f}`; Precision delta `{v1_result['delta']['precision']:+.15f}`.",
        f"- Fold Recall deltas: `{ {key: value['recall'] for key, value in v1_result['delta_by_fold'].items()} }`.",
        f"- Changed/improved/harmed: `{v1_result['queries_changed']}/{v1_result['queries_improved']}/{v1_result['queries_harmed']}`.",
        f"- Rank4/rank5 changes: `{v1_result['rank4_changes']}/{v1_result['rank5_changes']}`; Top1–3 changed: `{v1_result['top1_top3_changed']}`.",
        f"- PRIMARY_GATE: **{v1_result['gate']}**.",
        "",
        "## Nested calibration",
        "",
        f"- Run: **{'YES' if nested_run else 'NO'}**; gate: **{'NOT_RUN' if nested_result is None else nested_result['gate']}**.",
        f"- Recall/delta: `{None if nested_result is None else nested_result['proposed']['recall']}` / `{None if nested_result is None else nested_result['delta']['recall']}`.",
        "",
        "## Deployment decision",
        "",
        f"- Selected policy: **{selected_policy}**.",
        f"- Private application: **{private_application['status']}**; submission: **{submission.get('status')}**.",
        f"- FINAL_STATUS: **{final_status}**.",
        f"- Next action: `{result['NEXT_ACTION']}`.",
        "",
        "## Frozen provenance",
        "",
        f"- V1 contract: `{rel(contract_path)}`; SHA256 `{contract_sha}`.",
        "- Closed branches remain untouched: STEP4_K20_PORT, full Direct Top5, BGE FT V2, BGE FT V3 conflict-anchor.",
        "- GPU runs: `0`; Modal runs: `0`; model loads: `0`.",
    ]
    (OUT / "guarded_direct_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "FINAL_STATUS": final_status,
        "DIRECT_SIGNAL_GATE": result["DIRECT_SIGNAL_GATE"],
        "DIRECT_REPLAY_GATE": result["DIRECT_REPLAY_GATE"],
        "PRIMARY_GATE": result["PRIMARY_GATE"],
        "NESTED_RUN": result["NESTED_RUN"],
        "NESTED_GATE": result["NESTED_GATE"],
        "SELECTED_POLICY": selected_policy,
        "PRIVATE_APPLICATION": private_application["status"],
        "SUBMISSION_GATE": result["SUBMISSION_GATE"],
        "GPU_RUNS": 0,
        "MODAL_RUNS": 0,
        "report": rel(OUT / "guarded_direct_report.md"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
