"""Strict F1--F4 V3A/V3B exact-action consensus veto experiment.

The experiment intentionally has one policy: execute a V3A swap only when
the independently reproduced V3B policy selects the exact same action.
Fold0 and public data are excluded from all scientific calculations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V3_SOURCE = ROOT / "scripts" / "beam" / "task1_v3_residual"
if str(V3_SOURCE) not in sys.path:
    sys.path.insert(0, str(V3_SOURCE))

from common import jsonl, sort_query_ids, write_jsonl  # noqa: E402
from train_residual_policy import choose, feature_names, fit_model, utilities  # noqa: E402
from train_residual_policy_v3b import (  # noqa: E402
    FOLDS,
    build_predictions,
    fit_stage2,
    gate_probabilities,
    outer_oof_from_stage1_cache,
    outer_stage1_cache,
)


EXPECTED_V3A_RECALL = 0.9296488095238095
EXPECTED_V3B_RECALL = 0.9297976190476189
V3A_CONFIG = {"max_leaf_nodes": 15, "l2_regularization": 1.0, "harm_weight": 1.5, "threshold": 0.0}
V3B_CONFIG = {
    "max_leaf_nodes": 7,
    "benefit_multiplier": 4.0,
    "harm_train_multiplier": 1.0,
    "harm_utility_weight": 3.0,
    "stage2_positive_multiplier": 1.0,
    "gate_threshold": None,
    "action_threshold": 0.026512249187954383,
    "margin_threshold": None,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def action_identity(action: dict[str, Any] | None) -> tuple[str, str, str] | None:
    if action is None:
        return None
    return (str(action["query_id"]), str(action["incoming_doc_id"]), str(action["dropped_doc_id"]))


def v3a_selections(actions: list[dict[str, Any]], names: list[str]) -> dict[str, dict[str, Any] | None]:
    selected: dict[str, dict[str, Any] | None] = {}
    for held_out in FOLDS:
        train = [row for row in actions if int(row["fold"]) in set(FOLDS) - {held_out}]
        valid = [row for row in actions if int(row["fold"]) == held_out]
        model = fit_model(train, names, V3A_CONFIG["max_leaf_nodes"], V3A_CONFIG["l2_regularization"])
        chosen = choose(valid, utilities(model, valid, names, V3A_CONFIG["harm_weight"]), V3A_CONFIG["threshold"])
        selected.update(chosen)
    return selected


def v3b_selections(actions: list[dict[str, Any]], names: list[str]) -> dict[str, dict[str, Any] | None]:
    stage1_config = {key: V3B_CONFIG[key] for key in ("max_leaf_nodes", "benefit_multiplier", "harm_train_multiplier", "harm_utility_weight")}
    cache = outer_stage1_cache(actions, names, stage1_config)
    rows = outer_oof_from_stage1_cache(cache, stage1_config, V3B_CONFIG["stage2_positive_multiplier"])
    predictions = build_predictions(rows, V3B_CONFIG)
    return {str(row["query_id"]): row["selected_action"] for row in predictions}


def load_target_gold(actions: list[dict[str, Any]], train_path: Path) -> dict[str, set[str]]:
    target_ids = {str(row["query_id"]) for row in actions}
    with train_path.open(encoding="utf-8") as handle:
        corpus = json.load(handle)
    missing = target_ids - set(corpus)
    if missing:
        raise ValueError(f"target labels missing for {len(missing)} queries")
    return {query_id: {str(doc) for doc in corpus[query_id]["answer"]} for query_id in target_ids}


def base_by_query(actions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for action in actions:
        query_id = str(action["query_id"])
        existing = result.setdefault(query_id, action)
        if existing["baseline_top5"] != action["baseline_top5"] or int(existing["fold"]) != int(action["fold"]):
            raise ValueError(f"inconsistent baseline representation for {query_id}")
    return result


def prediction_top5(base: dict[str, Any], selected: dict[str, Any] | None) -> list[str]:
    top5 = [str(doc) for doc in base["baseline_top5"]]
    if selected is not None:
        top5[int(selected["drop_rank"]) - 1] = str(selected["incoming_doc_id"])
    if len(top5) != 5 or len(set(top5)) != 5 or top5[:3] != [str(doc) for doc in base["baseline_top5"][:3]]:
        raise ValueError(f"invalid protected top5 for {base['query_id']}")
    return top5


def evaluate(name: str, base: dict[str, dict[str, Any]], selections: dict[str, dict[str, Any] | None], gold: dict[str, set[str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_fold: dict[int, list[tuple[float, float]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    swaps = beneficial = harmful = neutral = rescues = broken = 0
    for query_id in sort_query_ids(base):
        source, selected = base[query_id], selections.get(query_id)
        baseline = [str(doc) for doc in source["baseline_top5"]]
        top5 = prediction_top5(source, selected)
        answers = gold[query_id]
        recall = len(answers & set(top5)) / len(answers)
        precision = len(answers & set(top5)) / 5.0
        baseline_recall = len(answers & set(baseline)) / len(answers)
        gain = recall - baseline_recall
        if selected is not None:
            swaps += 1
            beneficial += gain > 0.0
            harmful += gain < 0.0
            neutral += gain == 0.0
            rescues += baseline_recall == 0.0 and gain > 0.0
            broken += baseline_recall > 0.0 and gain < 0.0
        per_fold[int(source["fold"])].append((recall, precision))
        rows.append({
            "query_id": query_id, "fold": int(source["fold"]), "top5": top5,
            "selected_action": None if selected is None else {
                "query_id": query_id, "incoming_doc_id": str(selected["incoming_doc_id"]),
                "dropped_doc_id": str(selected["dropped_doc_id"]), "drop_rank": int(selected["drop_rank"]),
            },
        })
    metrics = {
        "policy": name,
        "query_count": len(rows),
        "pooled_macro_recall": mean(value[0] for values in per_fold.values() for value in values),
        "pooled_macro_precision": mean(value[1] for values in per_fold.values() for value in values),
        "per_fold_recall": {str(fold): mean(value[0] for value in values) for fold, values in sorted(per_fold.items())},
        "per_fold_precision": {str(fold): mean(value[1] for value in values) for fold, values in sorted(per_fold.items())},
        "swaps_executed": swaps, "beneficial_swaps": beneficial, "harmful_swaps": harmful,
        "neutral_swaps": neutral, "zero_hit_rescues": rescues, "previously_correct_queries_broken": broken,
    }
    return metrics, rows


def agreement_analysis(v3a: dict[str, dict[str, Any] | None], v3b: dict[str, dict[str, Any] | None], base: dict[str, dict[str, Any]], gold: dict[str, set[str]]) -> tuple[dict[str, dict[str, Any] | None], dict[str, Any], dict[str, Any]]:
    consensus: dict[str, dict[str, Any] | None] = {}
    counts: dict[str, int] = defaultdict(int)
    attribution: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query_id in sort_query_ids(base):
        v3a_action, v3b_action = v3a[query_id], v3b[query_id]
        exact = v3a_action is not None and action_identity(v3a_action) == action_identity(v3b_action)
        consensus[query_id] = v3a_action if exact else None
        if v3a_action is None:
            continue
        baseline = [str(doc) for doc in base[query_id]["baseline_top5"]]
        candidate = prediction_top5(base[query_id], v3a_action)
        gain = len(gold[query_id] & set(candidate)) / len(gold[query_id]) - len(gold[query_id] & set(baseline)) / len(gold[query_id])
        label = "beneficial" if gain > 0 else "harmful" if gain < 0 else "neutral"
        counts["v3a_swaps"] += 1
        counts[f"v3a_{label}"] += 1
        counts["exact_agreements" if exact else "disagreements"] += 1
        counts[("retained_" if exact else "vetoed_") + label] += 1
        if not exact and label in {"beneficial", "harmful"}:
            attribution[f"v3a_{label}_vetoed"].append({"query_id": query_id, "fold": int(base[query_id]["fold"]), "action": action_identity(v3a_action), "recall_gain": gain})
        if exact and label == "harmful":
            attribution["harmful_exact_agreements"].append({"query_id": query_id, "fold": int(base[query_id]["fold"]), "action": action_identity(v3a_action), "recall_gain": gain})
    retained_benefit = counts["retained_beneficial"]
    retained_harm = counts["retained_harmful"]
    benefit_total, harm_total = counts["v3a_beneficial"], counts["v3a_harmful"]
    summary = {key: int(value) for key, value in counts.items()}
    summary.update({
        "beneficial_retention_rate": retained_benefit / benefit_total if benefit_total else 0.0,
        "harmful_retention_rate": retained_harm / harm_total if harm_total else 0.0,
    })
    return consensus, summary, {key: value for key, value in attribution.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("use --execute")
    actions_path = ROOT / "artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl"
    features_path = ROOT / "artifacts/task1/recovery_096/v3_residual/frozen_features_report.json"
    v3a_report = ROOT / "artifacts/task1/recovery_096/v3_residual/policy/policy_training_report.json"
    v3b_report = ROOT / "artifacts/task1/recovery_096/v3_residual/v3b_policy/v3b_policy_training_report.json"
    folds_path = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
    train_path = ROOT / "data/raw/btc/LegalIR/train.json"
    inputs = (actions_path, features_path, v3a_report, v3b_report, folds_path, train_path)
    if any(not path.is_file() for path in inputs):
        raise FileNotFoundError("one or more exact reproduction inputs are unavailable")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite experiment directory: {args.output_dir}")
    all_actions = jsonl(actions_path)
    actions = [row for row in all_actions if int(row["fold"]) in FOLDS]
    if {int(row["fold"]) for row in actions} != set(FOLDS) or any("label" not in row for row in actions):
        raise ValueError("strict F1-F4 labeled action population is unavailable")
    if any(int(row["fold"]) == 0 and "label" in row for row in all_actions):
        raise ValueError("Fold0 label contract violated by input")
    base = base_by_query(actions)
    if len(base) != 5600 or {int(row["fold"]) for row in base.values()} != set(FOLDS):
        raise ValueError("F1-F4 population mismatch")
    names = feature_names(actions)
    gold = load_target_gold(actions, train_path)
    v3a = v3a_selections(actions, names)
    v3b = v3b_selections(actions, names)
    baseline = {query_id: None for query_id in base}
    v3a_metrics, _ = evaluate("V3A", base, v3a, gold)
    v3b_metrics, _ = evaluate("V3B", base, v3b, gold)
    baseline_metrics, _ = evaluate("Baseline", base, baseline, gold)
    if abs(v3a_metrics["pooled_macro_recall"] - EXPECTED_V3A_RECALL) > 1e-12:
        raise AssertionError(f"V3A reproduction mismatch: {v3a_metrics['pooled_macro_recall']}")
    if abs(v3b_metrics["pooled_macro_recall"] - EXPECTED_V3B_RECALL) > 1e-12:
        raise AssertionError(f"V3B reproduction mismatch: {v3b_metrics['pooled_macro_recall']}")
    consensus, agreement, attribution = agreement_analysis(v3a, v3b, base, gold)
    quick_kill = agreement["beneficial_retention_rate"] <= agreement["harmful_retention_rate"]
    if quick_kill:
        print(json.dumps({"status": "CONSENSUS_QUICK_KILL_FAIL", "reproduction": {"v3a": v3a_metrics, "v3b": v3b_metrics}, "agreement": agreement, "fold0_used": False}, indent=2, sort_keys=True))
        return
    consensus_metrics, consensus_predictions = evaluate("Consensus", base, consensus, gold)
    recall_delta = consensus_metrics["pooled_macro_recall"] - v3a_metrics["pooled_macro_recall"]
    precision_delta = consensus_metrics["pooled_macro_precision"] - v3a_metrics["pooled_macro_precision"]
    per_fold_delta = {fold: consensus_metrics["per_fold_recall"][fold] - v3a_metrics["per_fold_recall"][fold] for fold in consensus_metrics["per_fold_recall"]}
    gate = {
        "recall_strictly_improved": consensus_metrics["pooled_macro_recall"] > EXPECTED_V3A_RECALL,
        "every_fold_non_negative": all(value >= 0.0 for value in per_fold_delta.values()),
        "precision_non_decreasing": consensus_metrics["pooled_macro_precision"] >= v3a_metrics["pooled_macro_precision"],
    }
    gate["overall_pass"] = all(gate.values())
    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "v3a_v3b_exact_consensus_oof_predictions.jsonl"
    evaluation_path = args.output_dir / "v3a_v3b_exact_consensus_evaluation.json"
    attribution_path = args.output_dir / "v3a_v3b_exact_consensus_failure_attribution.json"
    manifest_path = args.output_dir / "v3a_v3b_exact_consensus_manifest.json"
    write_jsonl(predictions_path, consensus_predictions)
    evaluation = {"status": "CONSENSUS_EXPERIMENT_PASS" if gate["overall_pass"] else "CONSENSUS_EXPERIMENT_FAIL", "population": {"folds": list(FOLDS), "query_count": 5600, "fold0_used": False, "public_labels_used": False}, "policies": {"baseline": baseline_metrics, "v3a": v3a_metrics, "v3b": v3b_metrics, "consensus": consensus_metrics}, "consensus_vs_v3a": {"pooled_recall_delta": recall_delta, "pooled_precision_delta": precision_delta, "per_fold_recall_delta": per_fold_delta, "v3a_actions_vetoed": agreement["disagreements"], "beneficial_v3a_actions_lost": agreement["vetoed_beneficial"], "harmful_v3a_actions_prevented": agreement["vetoed_harmful"], "net_relevant_document_change": recall_delta * 5600}, "frozen_gate": gate}
    attribution_payload = {"agreement": agreement, "beneficial_v3a_actions_vetoed": attribution.get("v3a_beneficial_vetoed", []), "harmful_v3a_actions_vetoed": attribution.get("v3a_harmful_vetoed", []), "harmful_exact_agreements": attribution.get("harmful_exact_agreements", []), "k20_opportunity_queries_affected": None, "note": "No query-level K20 opportunity membership artifact was supplied to this experiment; value is intentionally not inferred."}
    evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    attribution_path.write_text(json.dumps(attribution_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"experiment": "V3A_V3B_EXACT_ACTION_CONSENSUS_VETO", "script_sha256": sha256(Path(__file__)), "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs}, "outputs": {str(path.name): sha256(path) for path in (predictions_path, evaluation_path, attribution_path)}, "v3a_config": V3A_CONFIG, "v3b_config": V3B_CONFIG, "exact_action_identity": ["query_id", "incoming_doc_id", "dropped_doc_id"], "fold0_scientific_use": False, "public_labels_used": False, "k77": False, "full_pool": False, "gpu": False, "modal": False}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": evaluation["status"], "output_dir": str(args.output_dir), "gate": gate, "agreement": agreement}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
