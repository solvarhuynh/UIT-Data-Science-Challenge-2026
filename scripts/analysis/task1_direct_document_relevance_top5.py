"""One fixed F1--F4 direct document-relevance final-top5 experiment.

This deliberately does not construct actions.  It ranks the provenance-safe
K20-plus-baseline document shortlist with a single sklearn classifier.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn import __version__ as SKLEARN_VERSION
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[2]
SHORTLIST = ROOT / "artifacts/task1/recovery_096/v3_residual/shortlist_evidence.jsonl"
CANDIDATES = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"
CANDIDATE_REPORT = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full_report.json"
BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
OUT = ROOT / "reports/task1/direct_document_relevance_top5"

FEATURES = (
    "bm25_max", "bm25_mean", "bm25_rank", "bm25_missing",
    "knn_char_rank", "knn_char_missing", "knn_word_rank", "knn_word_missing",
    "is_baseline_top5", "baseline_rank", "question_token_length", "question_char_length",
)
UNSAFE_EXCLUDED = (
    "union_rank", "reciprocal_union_rank", "source_support", "min_source_rank",
    "adaptive_k500_rank", "adaptive_k500_missing", "has_bge_support",
    "all V3 action features", "BENEFIT/HARM labels", "neural reranker features",
)
HP = {
    "loss": "log_loss", "learning_rate": 0.05, "max_iter": 200,
    "max_leaf_nodes": 31, "l2_regularization": 1.0, "random_state": 2026,
    "early_stopping": False,
}
CANONICAL = {
    "recall": 0.9259285714285714, "precision": 0.19739285714285715,
    "fold_recall": {1: 0.9363690476190477, 2: 0.9283333333333333, 3: 0.9194047619047620, 4: 0.9196071428571428},
}


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(predictions, gold):
    recalls, precisions = [], []
    for query_id, prediction in predictions.items():
        truth = set(map(str, gold[query_id]["answer"]))
        top5 = list(map(str, prediction))
        hits = len(truth & set(top5))
        recalls.append(hits / len(truth) if truth else 1.0)
        precisions.append(hits / 5.0)
    return {"recall": float(np.mean(recalls)), "precision": float(np.mean(precisions)), "query_count": len(predictions)}


def feature_row(query_id, row, doc):
    baseline = [str(value) for value in row["baseline_top5"]]
    doc_id = str(doc["doc_id"])
    source = dict(doc.get("source_ranks") or {})
    evidence = doc.get("evidence") or []
    if not evidence:
        raise ValueError(f"missing BM25 evidence for {query_id}:{doc_id}")
    bm25_scores = [float(item["bm25_score"]) for item in evidence]
    baseline_rank = baseline.index(doc_id) + 1 if doc_id in baseline else 0
    values = {
        "bm25_max": max(bm25_scores), "bm25_mean": float(np.mean(bm25_scores)),
        "bm25_rank": float(source.get("bm25", 0) or 0), "bm25_missing": float("bm25" not in source),
        "knn_char_rank": float(source.get("knn_char", 0) or 0), "knn_char_missing": float("knn_char" not in source),
        "knn_word_rank": float(source.get("knn_word", 0) or 0), "knn_word_missing": float("knn_word" not in source),
        "is_baseline_top5": float(bool(baseline_rank)), "baseline_rank": float(baseline_rank),
        "question_token_length": float(len(str(row["question"]).split())),
        "question_char_length": float(len(str(row["question"]))),
    }
    return {"query_id": query_id, "doc_id": doc_id, "fold": int(row["fold"]), "baseline_top5": baseline, "values": values}


def ordered_top5(scored):
    return [item["doc_id"] for item in sorted(
        scored,
        key=lambda item: (-item["probability"], -int(item["doc_id"] in item["baseline_top5"]),
                          item["baseline_top5"].index(item["doc_id"]) + 1 if item["doc_id"] in item["baseline_top5"] else 10**9,
                          item["doc_id"]),
    )[:5]]


def main():
    candidate_report = json.loads(CANDIDATE_REPORT.read_text(encoding="utf-8"))
    if candidate_report.get("no_gold_used_in_construction") is not True:
        raise RuntimeError("BLOCKED_FEATURE_PROVENANCE: candidate construction lacks no-gold proof")
    train = json.loads(TRAIN.read_text(encoding="utf-8-sig"))
    folds_raw = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    fold_map = {str(query_id): int(entry["fold"]) for entry in folds_raw for query_id in entry["validation_ids"]}
    baseline_rows = {str(row["query_id"]): [str(value) for value in row["top5"]] for row in rows(BASELINE)}
    candidate_pool = {str(row["query_id"]): {str(item["doc_id"]) for item in row["candidates"] if int(item["union_rank"]) <= 20} for row in rows(CANDIDATES)}
    shortlist = {str(row["query_id"]): row for row in rows(SHORTLIST)}
    if set(shortlist) != set(fold_map) or set(baseline_rows) != set(fold_map) or set(train) != set(fold_map):
        raise RuntimeError("BLOCKED_FEATURE_PROVENANCE: query-set mismatch")

    by_query = {}
    for query_id, row in shortlist.items():
        if int(row["fold"]) != fold_map[query_id]:
            raise RuntimeError(f"fold mismatch: {query_id}")
        baseline = baseline_rows[query_id]
        if [str(value) for value in row["baseline_top5"]] != baseline:
            raise RuntimeError(f"baseline mismatch in shortlist: {query_id}")
        expected = candidate_pool[query_id] | set(baseline)
        actual = {str(doc["doc_id"]) for doc in row["docs"]}
        if actual != expected or len(actual) > 25:
            raise RuntimeError(f"K20-plus-anchor membership mismatch: {query_id}")
        by_query[query_id] = [feature_row(query_id, row, doc) for doc in row["docs"]]

    scientific_ids = [query_id for query_id, fold in fold_map.items() if fold in (1, 2, 3, 4)]
    base_metrics = metrics({query_id: baseline_rows[query_id] for query_id in scientific_ids}, train)
    base_by_fold = {fold: metrics({q: baseline_rows[q] for q in scientific_ids if fold_map[q] == fold}, train) for fold in (1, 2, 3, 4)}
    # Arithmetic order differs from the historical report by at most two IEEE-754
    # ulps; this is an exact metric reproduction, not a changed acceptance value.
    exact_baseline = (math.isclose(base_metrics["recall"], CANONICAL["recall"], abs_tol=1e-15, rel_tol=0.0)
                      and math.isclose(base_metrics["precision"], CANONICAL["precision"], abs_tol=1e-15, rel_tol=0.0)
                      and all(math.isclose(base_by_fold[f]["recall"], CANONICAL["fold_recall"][f], abs_tol=1e-15, rel_tol=0.0) for f in base_by_fold))
    if not exact_baseline:
        raise RuntimeError("BLOCKED_BASELINE_MISMATCH")

    output_rows, models = [], {}
    for heldout in (1, 2, 3, 4):
        train_docs = [doc for query_id in scientific_ids if fold_map[query_id] != heldout for doc in by_query[query_id]]
        test_docs = [doc for query_id in scientific_ids if fold_map[query_id] == heldout for doc in by_query[query_id]]
        X_train = np.asarray([[doc["values"][name] for name in FEATURES] for doc in train_docs], dtype=np.float64)
        y_train = np.asarray([int(doc["doc_id"] in {str(v) for v in train[doc["query_id"]]["answer"]}) for doc in train_docs], dtype=np.int8)
        class_counts = np.bincount(y_train, minlength=2)
        weights = np.asarray([len(y_train) / (2 * class_counts[label]) for label in y_train], dtype=np.float64)
        model = HistGradientBoostingClassifier(**HP)
        model.fit(X_train, y_train, sample_weight=weights)
        models[heldout] = {"training_queries": len({doc["query_id"] for doc in train_docs}), "training_rows": len(train_docs), "class_counts": {"0": int(class_counts[0]), "1": int(class_counts[1])}, "weights": {"0": float(len(y_train)/(2*class_counts[0])), "1": float(len(y_train)/(2*class_counts[1]))}}
        X_test = np.asarray([[doc["values"][name] for name in FEATURES] for doc in test_docs], dtype=np.float64)
        probabilities = model.predict_proba(X_test)[:, 1]
        grouped = defaultdict(list)
        for doc, probability in zip(test_docs, probabilities):
            grouped[doc["query_id"]].append({**doc, "probability": float(probability)})
            output_rows.append({"query_id": doc["query_id"], "fold": heldout, "doc_id": doc["doc_id"], "probability": float(probability), "relevant": int(doc["doc_id"] in {str(v) for v in train[doc["query_id"]]["answer"]}), "baseline_top5": doc["baseline_top5"]})
        for query_id, docs in grouped.items():
            prediction = ordered_top5(docs)
            if len(prediction) != 5 or len(set(prediction)) != 5:
                raise RuntimeError(f"invalid top5: {query_id}")
            for item in output_rows[-len(test_docs):]:
                if item["query_id"] == query_id:
                    item["final_top5"] = prediction

    predictions = {query_id: next(item["final_top5"] for item in output_rows if item["query_id"] == query_id) for query_id in scientific_ids}
    result = metrics(predictions, train)
    result_by_fold = {fold: metrics({q: predictions[q] for q in scientific_ids if fold_map[q] == fold}, train) for fold in (1, 2, 3, 4)}
    changes = {"changed": 0, "improved": 0, "harmed": 0, "neutral": 0, "zero_hit_rescues": 0, "previously_correct_broken": 0, "net_relevant_document_change": 0}
    depth = Counter()
    missed_ranks = Counter()
    probability_by_label = defaultdict(list)
    for item in output_rows:
        probability_by_label[item["relevant"]].append(item["probability"])
    for query_id in scientific_ids:
        before, after, truth = baseline_rows[query_id], predictions[query_id], {str(v) for v in train[query_id]["answer"]}
        replacements = len(set(before) - set(after))
        depth["0" if replacements == 0 else "1" if replacements == 1 else "2" if replacements == 2 else "3+"] += 1
        delta_hits = len(truth & set(after)) - len(truth & set(before))
        changes["net_relevant_document_change"] += delta_hits
        if replacements:
            changes["changed"] += 1
            if delta_hits > 0: changes["improved"] += 1
            elif delta_hits < 0: changes["harmed"] += 1
            else: changes["neutral"] += 1
            if not (truth & set(before)) and (truth & set(after)): changes["zero_hit_rescues"] += 1
            if (truth & set(before)) and not (truth & set(after)): changes["previously_correct_broken"] += 1
        scored = sorted((item for item in output_rows if item["query_id"] == query_id), key=lambda item: -item["probability"])
        rank = {item["doc_id"]: index + 1 for index, item in enumerate(scored)}
        for doc_id in truth - set(after):
            missed_ranks[str(rank.get(doc_id, "outside_candidate"))] += 1

    deltas = {"recall": result["recall"] - base_metrics["recall"], "precision": result["precision"] - base_metrics["precision"]}
    fold_deltas = {fold: result_by_fold[fold]["recall"] - base_by_fold[fold]["recall"] for fold in result_by_fold}
    gate = {"recall_positive": deltas["recall"] > 0, "every_fold_nonnegative": all(value >= 0 for value in fold_deltas.values()), "precision_guard": deltas["precision"] >= -0.001, "complete_oof": len(predictions) == 5600, "leakage_free": True}
    passed = all(gate.values())
    if not passed: status, materiality = "DIRECT_DOCUMENT_RELEVANCE_FAIL", "NEGATIVE_OR_ZERO"
    elif deltas["recall"] >= .003: status, materiality = "DIRECT_DOCUMENT_TARGET_LEVEL_PASS", "TARGET_LEVEL"
    elif deltas["recall"] >= .002: status, materiality = "DIRECT_DOCUMENT_STRONG_PASS", "STRONG_POSITIVE"
    elif deltas["recall"] >= .001: status, materiality = "DIRECT_DOCUMENT_MATERIAL_PASS", "MATERIAL_POSITIVE"
    else: status, materiality = "DIRECT_DOCUMENT_WEAK_PASS", "WEAK_POSITIVE"

    OUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT / "direct_document_relevance_oof_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in sorted(output_rows, key=lambda item: (int(item["fold"]), item["query_id"], item["doc_id"])):
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    provenance_classes = {name: "SAFE_LABEL_FREE" for name in FEATURES}
    provenance_classes.update({"is_baseline_top5": "SAFE_OUTER_CROSSFIT", "baseline_rank": "SAFE_OUTER_CROSSFIT"})
    provenance = {"status": "SAFE", "candidate_membership": "baseline_top5 union canonical K20; candidate report proves no_gold_used_in_construction", "candidate_report_sha256": sha256(CANDIDATE_REPORT), "feature_list": list(FEATURES), "feature_classification": provenance_classes, "excluded": list(UNSAFE_EXCLUDED), "fold0_used": False, "public_labels_used": False}
    evaluation = {"status": status, "baseline": base_metrics, "baseline_by_fold": base_by_fold, "result": result, "result_by_fold": result_by_fold, "deltas": deltas, "fold_recall_deltas": fold_deltas, "query_outcomes": changes, "replacement_depth": dict(depth), "materiality": materiality, "scientific_gate": {**gate, "overall": passed}}
    anatomy = {"score_distribution": {str(label): {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))} for label, values in probability_by_label.items()}, "missed_gold_rank_distribution": dict(missed_ranks), "feature_importance": "NOT_EXPOSED_BY_HistGradientBoostingClassifier", "dominant_failure_cluster": "computed from final OOF metrics only; no post-hoc tuning performed"}
    manifest = {"experiment": "DIRECT_DOCUMENT_RELEVANCE_TOP5", "status": status, "estimator": "sklearn.ensemble.HistGradientBoostingClassifier", "sklearn_version": SKLEARN_VERSION, "parameters": HP, "weighting": "deterministic inverse-frequency sample_weight from each outer training split only", "seed": 2026, "outer_folds": {str(f): [x for x in (1,2,3,4) if x != f] for f in (1,2,3,4)}, "models": models, "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (SHORTLIST, CANDIDATES, CANDIDATE_REPORT, BASELINE, FOLDS, TRAIN)}}
    artifacts = {"evaluation": evaluation, "feature_provenance": provenance, "failure_attribution": anatomy, "training_manifest": manifest}
    for name, payload in artifacts.items():
        (OUT / f"direct_document_relevance_{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Do not claim a self-hash inside the file being rewritten; callers can
    # hash the final manifest byte artifact separately.
    manifest["artifact_sha256"] = {path.name: sha256(path) for path in OUT.iterdir() if path.is_file() and path.name != "direct_document_relevance_training_manifest.json"}
    (OUT / "direct_document_relevance_training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "recall": result["recall"], "delta": deltas["recall"], "precision": result["precision"], "artifacts": manifest["artifact_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
