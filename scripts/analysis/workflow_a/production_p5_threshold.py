"""Authorized pooled F1-F4 inner-OOF threshold build for frozen Workflow-A P5."""
from __future__ import annotations

import hashlib
import json
import math
import runpy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "reports/task1"
PREFLIGHT = R / "production_threshold_preflight.json"
SCORES = R / "production_threshold_inner_oof_scores.jsonl"
SELECTION = R / "production_threshold_selection_report.json"
THRESHOLD = R / "production_threshold.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_matrix(rows: list[dict], columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    order = {"BENEFICIAL": 2, "NEUTRAL": 1, "HARMFUL": 0}
    total = 0
    for values in grouped.values():
        counts = Counter(value["label"] for value in values)
        total += 2 * (counts["BENEFICIAL"] * counts["NEUTRAL"] + counts["BENEFICIAL"] * counts["HARMFUL"] + counts["NEUTRAL"] * counts["HARMFUL"])
    x = np.empty((total, len(columns)), dtype=np.float32)
    y = np.empty(total, dtype=np.uint8)
    position = 0
    for values in grouped.values():
        values = sorted(values, key=lambda value: (value["incoming_union_rank"], value["incoming_doc_id"], value["drop_rank"]))
        for index, left in enumerate(values):
            for right in values[index + 1:]:
                if left["label"] == right["label"]:
                    continue
                high, low = (left, right) if order[left["label"]] > order[right["label"]] else (right, left)
                difference = np.asarray([high["features"][column] - low["features"][column] for column in columns], dtype=np.float32)
                x[position], y[position] = difference, 1
                x[position + 1], y[position + 1] = -difference, 0
                position += 2
    if position != total or int(y.sum()) * 2 != total:
        raise RuntimeError("P1 pair construction mismatch")
    return x, y


def score_p1(model: HistGradientBoostingClassifier, rows: list[dict], columns: list[str]) -> dict[str, list[tuple[dict, float]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    scored = {}
    for query_id, values in grouped.items():
        n = len(values)
        x = np.empty((n * (n - 1), len(columns)), dtype=np.float32)
        owners = np.empty(n * (n - 1), dtype=np.int32)
        position = 0
        for left_index, left in enumerate(values):
            for right_index, right in enumerate(values):
                if left_index == right_index:
                    continue
                x[position] = [left["features"][column] - right["features"][column] for column in columns]
                owners[position] = left_index
                position += 1
        probabilities = model.predict_proba(x)[:, 1]
        totals = np.zeros(n, dtype=np.float64)
        np.add.at(totals, owners, probabilities)
        scored[query_id] = [(row, float(totals[index] / (n - 1))) for index, row in enumerate(values)]
    return scored


def percentile_and_top(scored: list[tuple[dict, float]], proxy: dict[tuple, float]) -> dict:
    rows = []
    for row, p1_score in scored:
        identity = (row["incoming_doc_id"], row["incoming_union_rank"], row["drop_rank"], row["dropped_doc_id"])
        rows.append({"row": row, "p1_score": p1_score, "p_hat_beneficial": proxy[identity]})
    values = sorted(value["p_hat_beneficial"] for value in rows)
    positions = {}
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        positions[values[start]] = start + 0.5 * (end - start - 1)
        start = end
    for value in rows:
        value["proxy_percentile"] = positions[value["p_hat_beneficial"]] / (len(rows) - 1)
        value["fused_score"] = value["p1_score"] + 0.05 * value["proxy_percentile"]
    top = sorted(rows, key=lambda value: (-value["fused_score"], -value["row"]["drop_rank"], value["row"]["incoming_union_rank"], value["row"]["incoming_doc_id"]))[0]
    row = top["row"]
    return {"query_id": row["query_id"], "fold": row["fold"], "top_action_identity": {"incoming_doc_id": row["incoming_doc_id"], "incoming_union_rank": row["incoming_union_rank"], "drop_rank": row["drop_rank"], "dropped_doc_id": row["dropped_doc_id"]}, "top_fused_score": top["fused_score"], "P1_score": top["p1_score"], "p_hat_beneficial": top["p_hat_beneficial"], "proxy_percentile": top["proxy_percentile"], "query_action_count": len(rows)}


def main() -> None:
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    p1_contract = json.loads((R / "step2p1_model_contract.json").read_text(encoding="utf-8"))
    p4_contract = json.loads((R / "step2p4_discriminability_report.json").read_text(encoding="utf-8"))
    p1_production = json.loads((R / "production_p1_fit_config.json").read_text(encoding="utf-8"))
    p4_production = json.loads((R / "production_p4_fit_config.json").read_text(encoding="utf-8"))
    if preflight["status"] != "PASS" or p1_contract["status"] != "FROZEN_PRE_FIT" or p4_contract["status"] != "PASS":
        raise RuntimeError("preflight contract mismatch")
    columns, hp = p1_contract["feature_columns"], p1_contract["hyperparameters"]
    if len(columns) != 36 or len(set(columns)) != 36 or hp != p4_contract["model"]["hyperparameters"] or p1_production["feature_names"] != columns or p4_production["feature_names"] != columns:
        raise RuntimeError("canonical model/schema contract mismatch")
    parent = runpy.run_path(str(ROOT / "scripts/analysis/step2_k77_oof_policy_realizability.py"), run_name="production_threshold_parent")
    folds = parent["target_folds"]()
    records, gold = parent["load_gold"](set(folds))
    baseline, _ = parent["load_baseline"](set(folds), folds)
    candidates, _ = parent["load_candidates"](set(folds), folds)
    actions, incoming = parent["make_actions"](folds, records, gold, baseline, candidates, columns)
    all_rows = [row for fold in range(1, 5) for row in actions[fold]]
    if incoming != 403322 or len(all_rows) != 806644 or len({row["query_id"] for row in all_rows}) != 5600:
        raise RuntimeError("K77 all-development population mismatch")
    frozen_rows = []
    split_reports = []
    temporary_p1 = temporary_p4 = 0
    for held in range(1, 5):
        train_folds = [fold for fold in range(1, 5) if fold != held]
        train_rows = [row for fold in train_folds for row in actions[fold]]
        held_rows = actions[held]
        x_pair, y_pair = pair_matrix(train_rows, columns)
        p1 = HistGradientBoostingClassifier(**hp)
        p1.fit(x_pair, y_pair, sample_weight=np.ones(len(y_pair), dtype=np.float64))
        temporary_p1 += 1
        y_proxy = np.asarray([int(row["truth_delta"] > 1e-12) for row in train_rows], dtype=np.uint8)
        p4 = HistGradientBoostingClassifier(**hp)
        p4.fit(np.asarray([[row["features"][column] for column in columns] for row in train_rows], dtype=np.float32), y_proxy)
        temporary_p4 += 1
        proxy_values = p4.predict_proba(np.asarray([[row["features"][column] for column in columns] for row in held_rows], dtype=np.float32))[:, 1]
        proxy = {(row["incoming_doc_id"], row["incoming_union_rank"], row["drop_rank"], row["dropped_doc_id"]): float(value) for row, value in zip(held_rows, proxy_values)}
        scored = score_p1(p1, held_rows, columns)
        held_frozen = [percentile_and_top(values, proxy) for values in scored.values()]
        if len(held_frozen) != 1400:
            raise RuntimeError("held-out coverage mismatch")
        for row in held_frozen:
            row["held_out_fold"] = held
            row["training_folds"] = train_folds
        frozen_rows.extend(held_frozen)
        split_reports.append({"held_out_fold": held, "training_folds": train_folds, "training_query_count": len({row["query_id"] for row in train_rows}), "held_out_query_count": len(held_frozen), "train_held_out_query_overlap": 0, "training_action_count": len(train_rows), "held_out_action_count": len(held_rows), "P1_training_pair_count_after_mirroring": len(y_pair), "P4_training_action_count": len(train_rows)})
        del p1, p4, x_pair, y_pair, y_proxy
    with SCORES.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(frozen_rows, key=lambda value: (value["held_out_fold"], value["query_id"])):
            handle.write(json.dumps(row) + "\n")
    ids = [row["query_id"] for row in frozen_rows]
    if len(ids) != 5600 or len(set(ids)) != 5600 or set(ids) != set(folds):
        raise RuntimeError("pooled 5600-query truth-free freeze gate failed")
    # Only after the truth-free artifact has been persisted, join the canonical development truth.
    by_query = {row["query_id"]: row for row in frozen_rows}
    for query_id, frozen in by_query.items():
        identity = frozen["top_action_identity"]
        source = next(row for row in actions[frozen["held_out_fold"]] if row["query_id"] == query_id and row["incoming_doc_id"] == identity["incoming_doc_id"] and row["incoming_union_rank"] == identity["incoming_union_rank"] and row["drop_rank"] == identity["drop_rank"] and row["dropped_doc_id"] == identity["dropped_doc_id"])
        frozen["truth_delta"] = source["truth_delta"]
        frozen["label"] = source["label"]
    candidates_tau = [float("inf")] + sorted({row["top_fused_score"] for row in frozen_rows})
    options = []
    for tau in candidates_tau:
        taken = [row for row in frozen_rows if row["top_fused_score"] >= tau]
        gain = sum(row["truth_delta"] for row in taken)
        precision_delta = sum((1 if row["label"] == "BENEFICIAL" else -1 if row["label"] == "HARMFUL" else 0) / 5 for row in taken)
        options.append((gain, precision_delta, tau, taken))
    selected = max(options, key=lambda value: (value[0], value[1], value[2]))
    max_gain = selected[0]
    tied = [value[2] for value in options if value[0] == max_gain and value[1] == selected[1]]
    score_hash = sha256(SCORES)
    historical = json.loads((R / "step2p5_threshold_report.json").read_text(encoding="utf-8"))
    selection_report = {"status": "PASS", "preflight": "PASS", "canonical_sources": ["scripts/analysis/step2p1c_nested_oof_threshold_experiment.py", "scripts/analysis/step2p5_percentile_fusion.py"], "inner_splits": split_reports, "temporary_P1_model_count": temporary_p1, "temporary_P4_model_count": temporary_p4, "pooled_held_out_action_rows": 806644, "coverage_gate": {"query_rows": 5600, "unique_query_ids": 5600, "duplicate_query_ids": 0, "missing_query_ids": 0, "each_query_held_out_exactly_once": True}, "truth_freeze_gate": {"persisted_before_truth_join": True, "truth_join_timing": "AFTER_ALL_HELD_OUT_TOP_ACTIONS_AND_SCORES_FROZEN"}, "selected_threshold": selected[2], "selected_gain_sum": selected[0], "selected_precision_delta": selected[1], "candidate_threshold_count": len(candidates_tau), "tied_threshold_candidates_after_gain_and_precision": tied, "tie_break_resolution": "max(gain_sum, precision_delta, threshold): largest threshold selected", "historical_outer_thresholds_reference_only": {fold: info["selected_threshold"] for fold, info in historical["folds"].items()}, "historical_outer_thresholds_aggregated": False, "Fold0_used": False, "public_labels_used": False, "beta_sweep": False, "K_sweep": False, "feature_change": False, "model_family_change": False, "public_inference": False, "submission": False}
    SELECTION.write_text(json.dumps(selection_report, indent=2) + "\n", encoding="utf-8")
    output = {"status": "PASS", "operation": "PRODUCTION_THRESHOLD_SELECTION", "workflow": "A", "policy": "P5", "threshold": selected[2], "objective": "canonical_gain_sum", "population": "ALL_F1_F4_DEVELOPMENT_QUERIES", "query_count": 5600, "oracle_positive_only": False, "K": 77, "feature_count": 36, "beta": 0.05, "p4_target": "BENEFICIAL_vs_NEUTRAL_HARMFUL", "inner_oof": True, "truth_joined_after_scores_frozen": True, "fold0_used": False, "public_labels_used": False, "public_inference_executed": False, "threshold_sweep_beyond_canonical_algorithm": False, "old_outer_thresholds_aggregated": False, "canonical_threshold_algorithm_match": True, "inner_split_count": 4, "pooled_query_count": 5600, "pooled_unique_query_count": 5600, "duplicate_query_count": 0, "missing_query_count": 0, "selected_gain_sum": selected[0], "tie_break": "max(gain_sum, precision_delta, threshold)", "provenance": ["reports/task1/production_threshold_preflight.json", "reports/task1/production_p1_fit_config.json", "reports/task1/production_p4_fit_config.json", "reports/task1/production_threshold_inner_oof_scores.jsonl"], "sha256_of_inner_oof_scores": score_hash}
    THRESHOLD.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "threshold": selected[2], "gain_sum": selected[0], "queries": 5600, "score_sha256": score_hash}, indent=2))


if __name__ == "__main__":
    main()
