"""Nested OOF change-policy V2 cho candidate V3-delta, không dùng public gold."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from udsc2026.evaluation.legal_ir_lexical import parse_legal_citations  # noqa: E402


FEATURE_NAMES = (
    "set_overlap",
    "documents_changed",
    "unique_docs_raw_top200",
    "bge_top_margin",
    "dense_top_margin",
    "bge_doc_entropy",
    "query_token_count",
    "citation_cue",
    "added_bge_max_mean",
    "added_dense_max_mean",
    "added_support_count_mean",
    "added_source_agreement_mean",
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _doc_features(hits: Sequence[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        grouped[str(hit["doc_id"])].append(hit)
    docs: dict[str, dict[str, float]] = {}
    bge: list[float] = []
    dense: list[float] = []
    for doc, values in grouped.items():
        b = max(float(value["bge_score"]) for value in values)
        d = max(float(value["dense_score"]) for value in values)
        docs[doc] = {"bge_max": b, "dense_max": d, "support_count": float(len(values))}
        bge.append(b)
        dense.append(d)
    bge.sort(reverse=True)
    dense.sort(reverse=True)
    probabilities = np.exp(np.asarray(bge) - bge[0])
    probabilities /= probabilities.sum()
    return docs, {
        "unique_docs_raw_top200": float(len(docs)),
        "bge_top_margin": bge[0] - bge[1] if len(bge) > 1 else 0.0,
        "dense_top_margin": dense[0] - dense[1] if len(dense) > 1 else 0.0,
        "bge_doc_entropy": float(-(probabilities * np.log(probabilities + 1e-12)).sum() / math.log(len(probabilities))),
    }


def _recall(gold: set[str], top: Sequence[str]) -> float:
    return len(gold & set(top)) / len(gold)


def _metric(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float | int]:
    recalls = [_recall(set(row["gold_documents"]), row[key]) for row in rows]
    precisions = [len(set(row["gold_documents"]) & set(row[key])) / len(set(row[key])) for row in rows]
    return {"macro_recall": float(mean(recalls)), "macro_precision": float(mean(precisions)), "query_count": len(rows)}


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _features(row: dict[str, Any]) -> list[float]:
    return [float(row["features"][name]) for name in FEATURE_NAMES]


def _rule_candidates(train: Sequence[dict[str, Any]]) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    output: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [("KEEP_ALL", lambda row: False)]
    for name in ("set_overlap", "bge_top_margin", "dense_top_margin", "bge_doc_entropy", "added_bge_max_mean", "added_support_count_mean", "added_source_agreement_mean"):
        values = sorted({float(row["features"][name]) for row in train})
        if not values:
            continue
        for quantile in (0.25, 0.5, 0.75, 0.9):
            threshold = values[min(len(values) - 1, int((len(values) - 1) * quantile))]
            output.append((f"{name}_ge_{threshold:.6g}", lambda row, n=name, t=threshold: float(row["features"][n]) >= t))
            output.append((f"{name}_le_{threshold:.6g}", lambda row, n=name, t=threshold: float(row["features"][n]) <= t))
    return output


def _policy_score(rows: Sequence[dict[str, Any]], mask: Sequence[bool]) -> tuple[float, float, int, int, int]:
    recalls = []
    precisions = []
    changes = better = worse = 0
    for row, use_candidate in zip(rows, mask):
        top = row["candidate_top5"] if use_candidate else row["baseline_top5"]
        gold = set(row["gold_documents"])
        recalls.append(_recall(gold, top))
        precisions.append(len(gold & set(top)) / len(set(top)))
        if use_candidate:
            changes += 1
            delta = _recall(gold, row["candidate_top5"]) - _recall(gold, row["baseline_top5"])
            better += delta > 0
            worse += delta < 0
    return mean(recalls), mean(precisions), changes, better, worse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/raw/btc/LegalIR/train.json")
    parser.add_argument("--fold0-cache", type=Path, required=True)
    parser.add_argument("--fold1to4-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train = json.loads(args.train.read_text(encoding="utf-8-sig"))
    baseline = {str(row["query_id"]): row for row in _jsonl(args.baseline)}
    candidate = {str(row["query_id"]): row for row in _jsonl(args.candidate)}
    compact = {str(row["query_id"]): row for row in _jsonl(args.fold0_cache) + _jsonl(args.fold1to4_cache)}
    if set(baseline) != set(candidate) or set(baseline) != set(compact):
        raise ValueError("Coverage baseline/candidate/cache không khớp")
    rows: list[dict[str, Any]] = []
    for qid, base in baseline.items():
        docs, qfeat = _doc_features(compact[qid]["hits"])
        dense_top = {doc for doc, _ in sorted(docs.items(), key=lambda item: (-item[1]["dense_max"], item[0]))[:5]}
        bge_top = {doc for doc, _ in sorted(docs.items(), key=lambda item: (-item[1]["bge_max"], item[0]))[:5]}
        base_top = [str(value) for value in base["top5"]]
        candidate_top = [str(value) for value in candidate[qid]["top5"]]
        added = [doc for doc in candidate_top if doc not in base_top]
        added_docs = [docs[doc] for doc in added if doc in docs]
        agreement = []
        for doc in added:
            agreement.append(sum((doc in base_top, doc in candidate_top, doc in dense_top, doc in bge_top)))
        question = str(train[qid]["question"])
        feature = {
            "set_overlap": float(len(set(base_top) & set(candidate_top))),
            "documents_changed": float(len(added)),
            **qfeat,
            "query_token_count": float(len(re.findall(r"(?u)\b\w+\b", question))),
            "citation_cue": float(parse_legal_citations(question).explicit),
            "added_bge_max_mean": mean([item["bge_max"] for item in added_docs]),
            "added_dense_max_mean": mean([item["dense_max"] for item in added_docs]),
            "added_support_count_mean": mean([item["support_count"] for item in added_docs]),
            "added_source_agreement_mean": mean(agreement),
        }
        gold = [str(value) for value in base["gold_documents"]]
        base_recall = _recall(set(gold), base_top)
        candidate_recall = _recall(set(gold), candidate_top)
        rows.append({
            "query_id": qid,
            "fold": int(base["fold"]),
            "gold_documents": gold,
            "baseline_top5": base_top,
            "candidate_top5": candidate_top,
            "features": feature,
            "candidate_better_label": int(candidate_recall > base_recall),
            "candidate_worse_label": int(candidate_recall < base_recall),
        })

    outputs: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for holdout in range(5):
        outer_train = [row for row in rows if row["fold"] != holdout]
        outer_test = [row for row in rows if row["fold"] == holdout]
        candidates: list[tuple[tuple[float, float, int, int, int], str, Callable[[dict[str, Any]], bool], dict[str, Any]]] = []
        for name, rule in _rule_candidates(outer_train):
            mask = [rule(row) and row["candidate_top5"] != row["baseline_top5"] for row in outer_train]
            recall, precision, changes, better, worse = _policy_score(outer_train, mask)
            candidates.append(((recall, precision, -changes, -worse, better), f"rule:{name}", rule, {"training_changes": changes, "training_better": better, "training_worse": worse}))

        x_train = np.asarray([_features(row) for row in outer_train], dtype=np.float64)
        y_train = np.asarray([row["candidate_better_label"] for row in outer_train], dtype=np.int64)
        logistic_coefficients: dict[str, float] = {}
        try:
            estimator = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=2026))])
            calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
            calibrated.fit(x_train, y_train)
            inner_model = calibrated.calibrated_classifiers_[0].estimator
            logistic_coefficients = {name: float(value) for name, value in zip(FEATURE_NAMES, inner_model.named_steps["model"].coef_[0])}
            probabilities = calibrated.predict_proba(x_train)[:, 1]
            for threshold in np.linspace(0.05, 0.95, 19):
                rule = lambda row, t=float(threshold), model=calibrated: bool(model.predict_proba(np.asarray([_features(row)]))[:, 1][0] >= t)
                mask = [value >= threshold and row["candidate_top5"] != row["baseline_top5"] for row, value in zip(outer_train, probabilities)]
                recall, precision, changes, better, worse = _policy_score(outer_train, mask)
                candidates.append(((recall, precision, -changes, -worse, better), f"logistic_calibrated:p>={threshold:.2f}", rule, {"training_changes": changes, "training_better": better, "training_worse": worse, "threshold": float(threshold)}))
        except ValueError as exc:
            logistic_coefficients = {"error": str(exc)}

        score, selected_name, selected_rule, selected_detail = max(candidates, key=lambda item: item[0])
        test_probability: dict[str, float | None] = {}
        for row in outer_test:
            probability: float | None = None
            if selected_name.startswith("logistic"):
                probability = float(selected_rule.__defaults__[1].predict_proba(np.asarray([_features(row)]))[:, 1][0])  # type: ignore[index]
            apply = selected_rule(row) and row["candidate_top5"] != row["baseline_top5"]
            test_probability[row["query_id"]] = probability
            outputs.append({
                "query_id": row["query_id"],
                "fold": holdout,
                "gold_documents": row["gold_documents"],
                "top5": row["candidate_top5"] if apply else row["baseline_top5"],
                "decision": "APPLY_CANDIDATE_CHANGE" if apply else "KEEP_BASELINE",
                "selected_policy": selected_name,
                "candidate_better_probability": probability,
                "features": row["features"],
            })
        fold_reports.append({"fold": holdout, "selected_policy": selected_name, "training_macro_recall": score[0], "training_macro_precision": score[1], "training_details": selected_detail, "logistic_coefficients": logistic_coefficients})

    outputs_by_id = {row["query_id"]: row for row in outputs}
    outputs = [outputs_by_id[row["query_id"]] for row in rows]
    output_top = {row["query_id"]: row["top5"] for row in outputs}
    for row in rows:
        row["policy_top5"] = output_top[row["query_id"]]
    baseline_metric = _metric(rows, "baseline_top5")
    candidate_metric = _metric(rows, "candidate_top5")
    policy_metric = _metric(rows, "policy_top5")
    per_fold = {}
    for fold in range(5):
        subset = [row for row in rows if row["fold"] == fold]
        per_fold[str(fold)] = {"baseline": _metric(subset, "baseline_top5"), "candidate": _metric(subset, "candidate_top5"), "policy": _metric(subset, "policy_top5"), "policy_delta": _metric(subset, "policy_top5")["macro_recall"] - _metric(subset, "baseline_top5")["macro_recall"]}
    better = worse = same = 0
    for row in rows:
        delta = _recall(set(row["gold_documents"]), row["policy_top5"]) - _recall(set(row["gold_documents"]), row["baseline_top5"])
        better += delta > 0
        worse += delta < 0
        same += delta == 0
    buckets: dict[str, dict[str, Any]] = {}
    for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        selected = [row for row in outputs if row["candidate_better_probability"] is not None and low <= float(row["candidate_better_probability"]) < high]
        if not selected:
            buckets[f"[{low:.2f},{min(high,1.0):.2f})"] = {"query_count": 0, "mean_probability": None, "empirical_candidate_better_rate": None}
            continue
        labels = {row["query_id"]: source["candidate_better_label"] for row, source in zip(outputs, rows)}
        buckets[f"[{low:.2f},{min(high,1.0):.2f})"] = {"query_count": len(selected), "mean_probability": mean([float(row["candidate_better_probability"]) for row in selected]), "empirical_candidate_better_rate": mean([float(labels[row["query_id"]]) for row in selected])}
    changed_rows = [row for row in outputs if row["decision"] == "APPLY_CANDIDATE_CHANGE"]
    changed = len(changed_rows)
    changed_better = changed_worse = changed_same = 0
    source_rows = {row["query_id"]: row for row in rows}
    for item in changed_rows:
        source = source_rows[item["query_id"]]
        delta = _recall(set(source["gold_documents"]), source["candidate_top5"]) - _recall(set(source["gold_documents"]), source["baseline_top5"])
        changed_better += delta > 0
        changed_worse += delta < 0
        changed_same += delta == 0
    high_confidence = {
        "query_count": changed,
        "candidate_better": changed_better,
        "candidate_same": changed_same,
        "candidate_worse": changed_worse,
        "benefit_rate": changed_better / changed if changed else 0.0,
        "non_neutral_win_rate": changed_better / (changed_better + changed_worse) if changed_better + changed_worse else 0.0,
        "required_non_neutral_win_rate": 0.80,
    }
    gate = {
        "pooled_delta_positive": policy_metric["macro_recall"] > baseline_metric["macro_recall"],
        "no_negative_fold": all(value["policy_delta"] >= 0 for value in per_fold.values()),
        "harms_not_exceed_improvements": worse <= better,
        "high_confidence_bucket_clean": high_confidence["non_neutral_win_rate"] >= high_confidence["required_non_neutral_win_rate"],
        "changed_query_count": changed,
    }
    gate["promote"] = bool(gate["pooled_delta_positive"] and gate["no_negative_fold"] and gate["harms_not_exceed_improvements"] and gate["high_confidence_bucket_clean"] and changed > 0)
    report = {"schema_version": "public-093-change-policy-v2-v1", "status": "PROMOTE_CHANGE_POLICY_V2" if gate["promote"] else "REJECT_CHANGE_POLICY_V2", "feature_names": list(FEATURE_NAMES), "feature_note": "BM25/KNN score chi tiết được dùng cho 35-query audit; policy full-OOF chỉ dùng feature có cache phủ 7.000 query.", "baseline": baseline_metric, "candidate": candidate_metric, "policy": policy_metric, "pooled_policy_delta": policy_metric["macro_recall"] - baseline_metric["macro_recall"], "per_fold": per_fold, "outcomes_vs_baseline": {"better": better, "same": same, "worse": worse}, "changed_query_count": changed, "confidence_buckets": buckets, "high_confidence_selected_rule": high_confidence, "fold_selections": fold_reports, "gate": gate, "constraints": {"public_gold_as_feature": False, "neural_training": False, "public_submission_created": False}}
    _write_jsonl(args.output_dir / "policy_v2_predictions.jsonl", outputs)
    _write_json(args.output_dir / "policy_v2_report.json", report)
    print(json.dumps({"status": report["status"], "baseline": baseline_metric, "candidate": candidate_metric, "policy": policy_metric, "gate": gate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
