"""CPU-only, inner-CV selected residual swap policy for V3.

Legacy validation consumes only action labels from folds 1--4.  Fold0 action
rows remain label-free in the legacy path.  The explicit ``--final-fit`` path
may label Fold0 only after exact historical replay has passed, and performs no
new model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import jsonl, recall, sort_query_ids, write_jsonl
    from .build_actions import action_label
except ImportError:
    from common import jsonl, recall, sort_query_ids, write_jsonl
    from build_actions import action_label


LABELS = {"NEUTRAL": 0, "BENEFIT": 1, "HARM": 2}
INNER_CV_FOLDS = (1, 2, 3, 4)
THRESHOLDS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
HARM_WEIGHTS = (1.0, 1.5, 2.0, 3.0)
MAX_LEAF_NODES = (7, 15)


def feature_names(actions: list[dict[str, Any]]) -> list[str]:
    names = sorted(actions[0]["features"])
    if any(sorted(row["features"]) != names for row in actions):
        raise ValueError("V3 actions have inconsistent feature columns")
    return names


def matrix(actions: list[dict[str, Any]], names: list[str]):
    import numpy as np
    return np.asarray([[float(row["features"].get(name, 0.0)) for name in names] for row in actions], dtype="float32")


def query_weights(actions: list[dict[str, Any]]):
    import numpy as np
    counts = Counter(str(row["query_id"]) for row in actions)
    return np.asarray([1.0 / counts[str(row["query_id"])] for row in actions], dtype="float32")


def fit_model(actions: list[dict[str, Any]], names: list[str], max_leaf_nodes: int, l2: float):
    from sklearn.ensemble import HistGradientBoostingClassifier
    x = matrix(actions, names)
    y = [LABELS[str(row["label"])] for row in actions]
    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.06, max_leaf_nodes=max_leaf_nodes, l2_regularization=l2, random_state=2026)
    model.fit(x, y, sample_weight=query_weights(actions))
    return model


def utilities(model, actions: list[dict[str, Any]], names: list[str], harm_weight: float) -> list[float]:
    probabilities = model.predict_proba(matrix(actions, names))
    index = {int(label): position for position, label in enumerate(model.classes_)}
    benefit = probabilities[:, index[1]] if 1 in index else 0.0
    harm = probabilities[:, index[2]] if 2 in index else 0.0
    return [float(value) for value in benefit - float(harm_weight) * harm]


def choose(actions: list[dict[str, Any]], utility_values: list[float], threshold: float) -> dict[str, dict[str, Any] | None]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for action, utility in zip(actions, utility_values):
        grouped[str(action["query_id"])].append((action, utility))
    result: dict[str, dict[str, Any] | None] = {}
    for query_id, items in grouped.items():
        action, utility = sorted(items, key=lambda item: (-item[1], -int(item[0]["drop_rank"]), str(item[0]["incoming_doc_id"])))[0]
        result[query_id] = {**action, "predicted_utility": utility} if utility >= threshold else None
    return result


def labeled_metrics(actions: list[dict[str, Any]], chosen: dict[str, dict[str, Any] | None]) -> dict[str, float | int]:
    per_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        per_query[str(action["query_id"])].append(action)
    before = after = 0.0
    swaps = benefit = harm = 0
    for query_id, rows in per_query.items():
        baseline = float(rows[0]["baseline_recall"])
        selected = chosen.get(query_id)
        gain = float(selected["gain"]) if selected is not None else 0.0
        before += baseline
        after += baseline + gain
        swaps += selected is not None
        benefit += gain > 0
        harm += gain < 0
    count = len(per_query)
    return {"query_count": count, "baseline_macro_recall": before / count, "policy_macro_recall": after / count, "delta": (after - before) / count, "swaps": swaps, "beneficial_swaps": benefit, "harmful_swaps": harm}


def make_predictions(chosen: dict[str, dict[str, Any] | None], actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        by_query[str(action["query_id"])].append(action)
    rows = []
    for query_id in sort_query_ids(by_query):
        baseline = [str(doc_id) for doc_id in by_query[query_id][0]["baseline_top5"]]
        selected = chosen.get(query_id)
        if selected is None:
            top5 = baseline
        else:
            top5 = list(baseline)
            top5[int(selected["drop_rank"]) - 1] = str(selected["incoming_doc_id"])
        if len(top5) != 5 or len(set(top5)) != 5:
            raise ValueError(f"V3 policy emitted invalid top5 for {query_id}")
        rows.append({"query_id": query_id, "fold": int(by_query[query_id][0]["fold"]), "top5": top5, "selected_action": selected})
    return rows


def no_swap_metrics(actions: list[dict[str, Any]]) -> dict[str, float | int]:
    """Return the exact baseline result without fitting or applying a model."""
    return labeled_metrics(actions, {str(row["query_id"]): None for row in actions})


def summarize_cv_folds(fold_metrics: dict[int, dict[str, float | int]]) -> dict[str, float | int]:
    query_count = sum(int(metrics["query_count"]) for metrics in fold_metrics.values())
    if not query_count:
        raise ValueError("inner CV produced no validation queries")
    baseline_total = sum(float(metrics["baseline_macro_recall"]) * int(metrics["query_count"]) for metrics in fold_metrics.values())
    policy_total = sum(float(metrics["policy_macro_recall"]) * int(metrics["query_count"]) for metrics in fold_metrics.values())
    return {
        "query_count": query_count,
        "baseline_macro_recall": baseline_total / query_count,
        "policy_macro_recall": policy_total / query_count,
        "delta": (policy_total - baseline_total) / query_count,
        "swaps": sum(int(metrics["swaps"]) for metrics in fold_metrics.values()),
        "beneficial_swaps": sum(int(metrics["beneficial_swaps"]) for metrics in fold_metrics.values()),
        "harmful_swaps": sum(int(metrics["harmful_swaps"]) for metrics in fold_metrics.values()),
        "pooled_cv_delta": (policy_total - baseline_total) / query_count,
        "min_fold_delta": min(float(metrics["delta"]) for metrics in fold_metrics.values()),
        "negative_fold_count": sum(float(metrics["delta"]) < 0.0 for metrics in fold_metrics.values()),
    }


def selection_key(row: dict[str, Any]) -> tuple[float, int, int, int, float, float, int]:
    """Maximization key: preserve the baseline when efficacy is tied."""
    return (
        float(row["policy_macro_recall"]),
        -int(row["harmful_swaps"]),
        int(row["beneficial_swaps"]) - int(row["harmful_swaps"]),
        -int(row["swaps"]),
        float(row["threshold"]),
        float(row["harm_weight"]),
        -int(row["max_leaf_nodes"]),
    )


def select_policy(cv_results: list[dict[str, Any]], no_swap: dict[str, Any]) -> dict[str, Any]:
    """Choose stable gains first; otherwise require a strict pooled improvement."""
    stable = [
        row for row in cv_results
        if float(row["pooled_cv_delta"]) > 0.0 and int(row["negative_fold_count"]) == 0
    ]
    if stable:
        return max(stable, key=selection_key)
    best_learned = max(cv_results, key=selection_key)
    # NO_SWAP is exact baseline.  A tied or worse learned policy must not alter it.
    return best_learned if float(best_learned["pooled_cv_delta"]) > 0.0 else no_swap


def class_imbalance(actions: list[dict[str, Any]]) -> dict[str, Any]:
    by_fold: dict[str, dict[str, Any]] = {}
    pooled = Counter()
    for fold in INNER_CV_FOLDS:
        counts = Counter(str(row["label"]) for row in actions if int(row["fold"]) == fold)
        pooled.update(counts)
        total = sum(counts.values())
        by_fold[str(fold)] = {
            "counts": {label: int(counts[label]) for label in ("BENEFIT", "NEUTRAL", "HARM")},
            "benefit_fraction": float(counts["BENEFIT"]) / total if total else 0.0,
            "harm_fraction": float(counts["HARM"]) / total if total else 0.0,
        }
    total = sum(pooled.values())
    return {
        "weighting": "query_balanced_only",
        "optional_class_reweighting": False,
        "by_train_fold": by_fold,
        "pooled": {
            "counts": {label: int(pooled[label]) for label in ("BENEFIT", "NEUTRAL", "HARM")},
            "benefit_fraction": float(pooled["BENEFIT"]) / total if total else 0.0,
            "harm_fraction": float(pooled["HARM"]) / total if total else 0.0,
        },
    }


def validate_label_split(train_actions: list[dict[str, Any]], test_actions: list[dict[str, Any]]) -> None:
    if any("label" not in row for row in train_actions) or any("label" in row for row in test_actions):
        raise ValueError("V3 action label split violates fold0 leakage contract")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fold0_replay(model: Any, actions: list[dict[str, Any]], names: list[str], report_path: Path) -> dict[str, Any]:
    """Replay the serialized final-fit decision against the historical Fold0 output."""
    expected_path = report_path.parent / "fold0_v3_policy_predictions.jsonl"
    if not expected_path.is_file():
        raise FileNotFoundError(f"historical Fold0 V3A predictions missing: {expected_path}")
    fold0_actions = [row for row in actions if int(row["fold"]) == 0]
    expected = jsonl(expected_path)
    generated = make_predictions(
        choose(fold0_actions, utilities(model, fold0_actions, names, 1.5), 0.0),
        fold0_actions,
    )
    expected_by_id = {str(row["query_id"]): row for row in expected}
    generated_by_id = {str(row["query_id"]): row for row in generated}
    mismatches = []
    if set(expected_by_id) != set(generated_by_id):
        mismatches.append("query_id_set")
    if [str(row["query_id"]) for row in expected] != [str(row["query_id"]) for row in generated]:
        mismatches.append("query_id_order")
    for query_id in sorted(set(expected_by_id) & set(generated_by_id)):
        left = expected_by_id[query_id]
        right = generated_by_id[query_id]
        left_action = left.get("selected_action")
        right_action = right.get("selected_action")
        left_identity = None if left_action is None else (str(left_action.get("incoming_doc_id")), str(left_action.get("dropped_doc_id")), int(left_action.get("drop_rank")))
        right_identity = None if right_action is None else (str(right_action.get("incoming_doc_id")), str(right_action.get("dropped_doc_id")), int(right_action.get("drop_rank")))
        if [str(value) for value in left.get("top5", [])] != [str(value) for value in right.get("top5", [])]:
            mismatches.append(f"top5:{query_id}")
        if left_identity != right_identity:
            mismatches.append(f"action:{query_id}")
    return {"status": "PASS" if not mismatches else "FAIL", "expected_path": str(expected_path), "query_count": len(generated), "mismatch_count": len(mismatches), "mismatch_sample": mismatches[:10]}


def _label_fold0_after_replay(actions: list[dict[str, Any]], train_questions_path: Path) -> list[dict[str, Any]]:
    """Create temporary Fold0 labels only after replay parity has passed."""
    questions = json.loads(train_questions_path.read_text(encoding="utf-8-sig"))
    labeled: list[dict[str, Any]] = []
    for row in actions:
        if int(row["fold"]) != 0:
            continue
        query_id = str(row["query_id"])
        if query_id not in questions:
            raise ValueError(f"Fold0 query missing from train questions: {query_id}")
        copy = dict(row)
        gold = {str(value) for value in questions[query_id].get("answer", [])}
        gain, label = action_label(gold, [str(value) for value in row["baseline_top5"]], str(row["incoming_doc_id"]), int(row["drop_rank"]))
        copy.update({"baseline_recall": recall(gold, row["baseline_top5"]), "gain": gain, "label": label})
        labeled.append(copy)
    if not labeled:
        raise ValueError("no Fold0 actions available for production final fit")
    return labeled


def _final_fit_preflight(actions_path: Path, report_path: Path, train_questions_path: Path) -> dict[str, Any]:
    actions = jsonl(actions_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    questions = json.loads(train_questions_path.read_text(encoding="utf-8-sig"))
    train_actions = [row for row in actions if int(row["fold"]) in INNER_CV_FOLDS]
    fold0_actions = [row for row in actions if int(row["fold"]) == 0]
    selected = report.get("selected_config") or {}
    try:
        selected_contract = report.get("selected_policy_type") == "LEARNED" and float(selected.get("max_leaf_nodes")) == 15.0 and float(selected.get("harm_weight")) == 1.5 and float(selected.get("threshold")) == 0.0
    except (TypeError, ValueError):
        selected_contract = False
    return {
        "actions_present": bool(actions),
        "query_count": len({str(row["query_id"]) for row in actions}),
        "train_query_count": len({str(row["query_id"]) for row in train_actions}),
        "fold0_query_count": len({str(row["query_id"]) for row in fold0_actions}),
        "fold0_unlabeled_before_replay": all("label" not in row for row in fold0_actions),
        "fold1_4_labeled": bool(train_actions) and all("label" in row for row in train_actions),
        "historical_fold0_predictions_present": (report_path.parent / "fold0_v3_policy_predictions.jsonl").is_file(),
        "train_questions_7000": isinstance(questions, dict) and len(questions) == 7000,
        "selected_v3a_contract": selected_contract,
    }


def final_fit_model(actions: list[dict[str, Any]], actions_path: Path, report_path: Path, model_path: Path, manifest_path: Path, train_questions_path: Path) -> dict[str, Any]:
    """Replay legacy Fold0 first, then serialize an all-fold public model."""
    import joblib

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("selected_policy_type") != "LEARNED":
        raise ValueError("V3A final-fit requires the historically selected learned policy")
    selected = report.get("selected_config") or {}
    expected = {"max_leaf_nodes": 15, "harm_weight": 1.5, "threshold": 0.0}
    for key, value in expected.items():
        if float(selected.get(key)) != float(value):
            raise ValueError(f"selected V3A hyperparameter mismatch for {key}: {selected.get(key)!r}")
    replay_actions = [row for row in actions if int(row["fold"]) in INNER_CV_FOLDS and "label" in row]
    if not replay_actions:
        raise ValueError("no eligible labeled train actions for V3A replay")
    names = feature_names(replay_actions)
    replay_model = fit_model(replay_actions, names, 15, float(selected["l2_regularization"]))
    fold0_replay = _fold0_replay(replay_model, actions, names, report_path)
    if int(fold0_replay["mismatch_count"]) != 0:
        raise RuntimeError(f"V3A serialized final-fit Fold0 replay mismatch: {fold0_replay}")
    fold0_labeled = _label_fold0_after_replay(actions, train_questions_path)
    production_actions = replay_actions + fold0_labeled
    if {int(row["fold"]) for row in production_actions} != {0, 1, 2, 3, 4} or len({str(row["query_id"]) for row in production_actions}) != 7000:
        raise ValueError("all-fold production final fit does not cover exactly 7,000 train queries")
    production_model = fit_model(production_actions, names, 15, float(selected["l2_regularization"]))
    bundle = {"model": production_model, "feature_names": names, "harm_weight": 1.5, "threshold": 0.0, "model_family": "HistGradientBoostingClassifier", "max_leaf_nodes": 15, "l2_regularization": float(selected["l2_regularization"]), "deployment_stage": "production_final_fit_all_folds"}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    manifest = {"schema_version": "v3a-final-fit-v2", "feature_names": names, "feature_count": len(names), "hyperparameters": {"model_family": "HistGradientBoostingClassifier", "max_iter": 100, "learning_rate": 0.06, "max_leaf_nodes": 15, "l2_regularization": float(selected["l2_regularization"]), "harm_weight": 1.5, "threshold": 0.0, "random_state": 2026, "query_weighting": "one total weight per query"}, "validation_replay": {"training_folds": [1, 2, 3, 4], "holdout_fold": 0, "mismatch_count": 0, "status": "PASS", "expected_predictions": str(report_path.parent / "fold0_v3_policy_predictions.jsonl")}, "production_final_fit": {"training_folds": [0, 1, 2, 3, 4], "training_query_count": 7000, "no_model_selection_performed": True, "no_public_training": True}, "training_action_count": len(production_actions), "training_query_count": len({str(row["query_id"]) for row in production_actions}), "train_actions_sha256": _sha256(actions_path), "selected_validation_report_sha256": _sha256(report_path), "source_code_sha256": _sha256(Path(__file__)), "model_sha256": _sha256(model_path), "fold0_replay": fold0_replay, "no_public_training": True, "validation_note": "Fold0 is labeled only after exact historical replay; no in-sample metric is reported."}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"model": str(model_path), "manifest": str(manifest_path), "feature_count": len(names), "training_action_count": len(production_actions), "validation_replay": "V3A_REPLAY_PASS", "production_final_fit": "V3A_ALL_FOLD_FINAL_FIT_PASS", "production_model_ready": "V3A_PRODUCTION_MODEL_READY"}


def apply_public_model(features_path: Path, model_path: Path, baseline_path: Path, output_path: Path, report_path: Path, expected_ids_path: Path) -> dict[str, Any]:
    """Apply a serialized V3A model to public features without labels/folds."""
    import joblib
    try:
        from .build_actions import build_public_actions
    except ImportError:
        from build_actions import build_public_actions

    manifest_path = model_path.with_name("v3a_final_fit_manifest.json")
    if output_path.exists() or report_path.exists():
        raise RuntimeError("refusing to overwrite existing public V3A output/report")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"public V3A final-fit manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay = manifest.get("validation_replay", manifest.get("fold0_replay", {}))
    production = manifest.get("production_final_fit", {})
    if replay.get("status") != "PASS" or int(replay.get("mismatch_count", -1)) != 0:
        raise RuntimeError("public V3A requires exact Fold0 replay PASS with mismatch_count=0")
    if production.get("training_folds") != [0, 1, 2, 3, 4] or production.get("no_model_selection_performed") is not True or production.get("no_public_training") is not True:
        raise RuntimeError("public V3A requires the all-fold final-fit deployment manifest")
    if manifest.get("model_sha256") != _sha256(model_path):
        raise RuntimeError("public V3A model hash does not match final-fit manifest")
    rows = jsonl(features_path)
    questions = json.loads(expected_ids_path.read_text(encoding="utf-8-sig"))
    expected_ids = {str(value) for value in questions}
    with zipfile.ZipFile(baseline_path) as archive:
        baseline_payload = json.loads(archive.read("submission.json"))
    baseline_map = {str(qid): [str(value) for value in item["answer"]] for qid, item in baseline_payload.items()}
    if set(baseline_map) != expected_ids or any(len(docs) != 5 or len(set(docs)) != 5 for docs in baseline_map.values()):
        raise ValueError("public baseline does not have exactly five unique docs for every public ID")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "fold" in row or set(row) & {"answer", "gold", "label", "gain", "baseline_recall"}:
            raise ValueError("public frozen features contain train-only fields")
        grouped[str(row["query_id"])].append(row)
    if set(grouped) != expected_ids or len(grouped) != 1000:
        raise ValueError("public feature IDs do not match public-official IDs")
    public_rows = [{"query_id": qid, "baseline_top5": baseline_map[qid], "docs": grouped[qid]} for qid in grouped]
    if any([str(value) for value in grouped[qid][0]["baseline_top5"]] != baseline_map[qid] for qid in grouped):
        raise ValueError("public frozen feature baseline differs from verified baseline")
    actions = build_public_actions(public_rows)
    bundle = joblib.load(model_path)
    if bundle.get("deployment_stage") != "production_final_fit_all_folds":
        raise ValueError("public V3A application requires the all-fold production final-fit model")
    names = list(bundle["feature_names"])
    if len(names) != 58:
        raise ValueError("serialized V3A model does not contain 58 features")
    if feature_names(actions) != names:
        raise ValueError("public action feature ordering differs from serialized V3A model")
    chosen = choose(actions, utilities(bundle["model"], actions, names, float(bundle["harm_weight"])), float(bundle["threshold"]))
    by_query = {str(row["query_id"]): row for row in public_rows}
    outputs = []
    swapped = rank4 = rank5 = 0
    for qid in sorted(expected_ids, key=lambda value: (int(value) if value.isdigit() else value)):
        baseline = [str(value) for value in by_query[qid]["baseline_top5"]]
        selected = chosen.get(qid)
        top5 = list(baseline)
        if selected is not None:
            drop_rank = int(selected["drop_rank"])
            if drop_rank not in (4, 5):
                raise ValueError("public V3A attempted to modify protected rank")
            top5[drop_rank - 1] = str(selected["incoming_doc_id"])
            swapped += 1
            rank4 += drop_rank == 4
            rank5 += drop_rank == 5
        if len(top5) != 5 or len(set(top5)) != 5 or top5[:3] != baseline[:3]:
            raise ValueError(f"public V3A constraint failure: {qid}")
        outputs.append({"id": qid, "documents": top5})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"status": "PUBLIC_V3A_POLICY_PASS", "query_count": len(outputs), "swapped_query_count": swapped, "unchanged_query_count": len(outputs) - swapped, "rank4_drop_count": rank4, "rank5_drop_count": rank5, "model_sha256": _sha256(model_path), "features_sha256": _sha256(features_path), "baseline_sha256": _sha256(baseline_path), "feature_schema_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode("utf-8")).hexdigest(), "no_public_labels_used": True, "no_fold_used": True, "max_one_swap_per_query": True, "rank1_3_protected": True}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_self_tests() -> dict[str, bool]:
    baseline = {
        "selected_policy_type": "NO_SWAP_BASELINE", "threshold": float("inf"), "harm_weight": float("inf"),
        "max_leaf_nodes": 7, "policy_macro_recall": 0.5, "pooled_cv_delta": 0.0,
        "harmful_swaps": 0, "beneficial_swaps": 0, "swaps": 0,
    }
    worse = {**baseline, "selected_policy_type": "LEARNED", "threshold": 0.2, "harm_weight": 1.0, "policy_macro_recall": 0.4, "pooled_cv_delta": -0.1, "negative_fold_count": 4}
    no_swap_wins = select_policy([worse], baseline)["selected_policy_type"] == "NO_SWAP_BASELINE"
    stable_gain = {**worse, "threshold": 0.5, "harm_weight": 2.0, "policy_macro_recall": 0.6, "pooled_cv_delta": 0.1, "negative_fold_count": 0}
    learned_wins = select_policy([worse, stable_gain], baseline) is stable_gain
    tied_lower_threshold = {**stable_gain, "threshold": 0.3, "harm_weight": 1.0, "harmful_swaps": 2, "beneficial_swaps": 5, "swaps": 7}
    tied_conservative = {**stable_gain, "threshold": 0.7, "harm_weight": 3.0, "harmful_swaps": 1, "beneficial_swaps": 4, "swaps": 5}
    conservative_tiebreak = max([tied_lower_threshold, tied_conservative], key=selection_key) is tied_conservative
    try:
        validate_label_split([], [{"fold": 0, "label": "BENEFIT"}])
    except ValueError:
        fold0_label_guard = True
    else:
        fold0_label_guard = False
    return {"no_swap_when_all_learned_worse": no_swap_wins, "stable_learned_gain_selected": learned_wins, "conservative_tiebreak": conservative_tiebreak, "fold0_label_guard": fold0_label_guard}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--final-fit", action="store_true")
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit.joblib"))
    parser.add_argument("--manifest-output", type=Path, default=Path("artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit_manifest.json"))
    parser.add_argument("--selected-report", type=Path, default=Path("artifacts/task1/recovery_096/v3_residual/policy/policy_training_report.json"))
    parser.add_argument("--train-questions", type=Path, default=Path("data/raw/btc/LegalIR/train.json"))
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--public-model", type=Path, default=Path("artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit.joblib"))
    parser.add_argument("--public-baseline", type=Path, default=Path("artifacts/task1/recovery_096/public_anchor_093/reproduced_093_submission.zip"))
    parser.add_argument("--public-output", type=Path, default=Path("artifacts/task1/recovery_096/final_public_v3/public_v3a_predictions.json"))
    parser.add_argument("--public-report", type=Path, default=Path("artifacts/task1/recovery_096/final_public_v3/public_v3a_policy_report.json"))
    parser.add_argument("--public-questions", type=Path, default=Path("data/raw/btc/LegalIR/public-official.json"))
    args = parser.parse_args()
    if args.final_fit and args.public:
        raise ValueError("choose exactly one of --final-fit or --public")
    if args.preflight:
        if args.final_fit:
            required = [args.actions, args.selected_report, args.train_questions, args.selected_report.parent / "fold0_v3_policy_predictions.jsonl"]
            missing = [str(path) for path in required if not path.is_file()]
            checks = _final_fit_preflight(args.actions, args.selected_report, args.train_questions) if not missing else {}
            passed = not missing and all(bool(value) for value in checks.values())
            print(json.dumps({"status": "PREFLIGHT_PASS" if passed else "PREFLIGHT_BLOCKED", "final_fit_mode": True, "required": [str(path) for path in required], "missing": missing, "checks": checks, "replay_training_folds": [1, 2, 3, 4], "production_training_folds": [0, 1, 2, 3, 4], "gpu_launched": False}, indent=2))
        else:
            print(json.dumps({"status": "PREFLIGHT_PASS", "inner_cv_folds": [1, 2, 3, 4], "fold0_labels_read": False, "gpu_launched": False}, indent=2))
        return
    if args.self_test:
        checks = run_self_tests()
        if not all(checks.values()):
            raise AssertionError(f"residual policy self-test failed: {checks}")
        print(json.dumps({"status": "SELF_TEST_PASS", **checks, "fold0_labels_read": False, "gpu_launched": False}, indent=2))
        return
    if args.final_fit:
        actions = jsonl(args.actions)
        result = final_fit_model(actions, args.actions, args.selected_report, args.model_output, args.manifest_output, args.train_questions)
        print(json.dumps({"status": "V3A_PRODUCTION_MODEL_READY", **result, "gpu_launched": False}, indent=2))
        return
    if args.public:
        result = apply_public_model(args.actions, args.public_model, args.public_baseline, args.public_output, args.public_report, args.public_questions)
        print(json.dumps(result, indent=2))
        return
    actions = jsonl(args.actions)
    train_actions = [row for row in actions if int(row["fold"]) in {1, 2, 3, 4}]
    test_actions = [row for row in actions if int(row["fold"]) == 0]
    validate_label_split(train_actions, test_actions)
    if {int(row["fold"]) for row in train_actions} != set(INNER_CV_FOLDS) or not test_actions:
        raise ValueError("V3 requires train folds 1--4 and unlabeled fold0 actions")
    if any(str(row["label"]) not in LABELS for row in train_actions):
        raise ValueError("V3 train actions contain an unknown label")
    names = feature_names(actions)
    configs = [{"max_leaf_nodes": leaves, "l2_regularization": l2, "harm_weight": harm, "threshold": threshold} for leaves in MAX_LEAF_NODES for l2 in (1.0,) for harm in HARM_WEIGHTS for threshold in THRESHOLDS]
    cv_results = []
    for config in configs:
        fold_metrics: dict[int, dict[str, float | int]] = {}
        for fold in INNER_CV_FOLDS:
            train = [row for row in train_actions if int(row["fold"]) != fold]
            valid = [row for row in train_actions if int(row["fold"]) == fold]
            model = fit_model(train, names, config["max_leaf_nodes"], config["l2_regularization"])
            fold_metrics[fold] = labeled_metrics(valid, choose(valid, utilities(model, valid, names, config["harm_weight"]), config["threshold"]))
        cv_results.append({"selected_policy_type": "LEARNED", **config, **summarize_cv_folds(fold_metrics), "per_fold": {str(fold): fold_metrics[fold] for fold in INNER_CV_FOLDS}})
    baseline_fold_metrics = {fold: no_swap_metrics([row for row in train_actions if int(row["fold"]) == fold]) for fold in INNER_CV_FOLDS}
    no_swap = {"selected_policy_type": "NO_SWAP_BASELINE", "max_leaf_nodes": 7, "l2_regularization": 1.0, "harm_weight": float("inf"), "threshold": float("inf"), **summarize_cv_folds(baseline_fold_metrics), "per_fold": {str(fold): baseline_fold_metrics[fold] for fold in INNER_CV_FOLDS}}
    selected = select_policy(cv_results, no_swap)
    if selected["selected_policy_type"] == "NO_SWAP_BASELINE":
        predictions = make_predictions({}, test_actions)
    else:
        model = fit_model(train_actions, names, int(selected["max_leaf_nodes"]), float(selected["l2_regularization"]))
        fold0_utilities = utilities(model, test_actions, names, float(selected["harm_weight"]))
        predictions = make_predictions(choose(test_actions, fold0_utilities, float(selected["threshold"])), test_actions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "fold0_v3_policy_predictions.jsonl", predictions)
    no_swap_report = {**no_swap, "max_leaf_nodes": None, "l2_regularization": None, "harm_weight": "not_applicable", "threshold": "+inf"}
    selected_config = {key: selected[key] for key in ("max_leaf_nodes", "l2_regularization", "harm_weight", "threshold")}
    if selected["selected_policy_type"] == "NO_SWAP_BASELINE":
        selected_config = {key: no_swap_report[key] for key in selected_config}
    report = {"status": "POLICY_SELECTED_FOLD0_UNLABELED", "sklearn_model": "HistGradientBoostingClassifier" if selected["selected_policy_type"] == "LEARNED" else "not_fit_no_swap_baseline", "feature_count": len(names), "feature_names": names, "train_folds": list(INNER_CV_FOLDS), "fold0_labels_read": False, "query_weighting": "one total weight per query", "class_imbalance": class_imbalance(train_actions), "inner_cv": {"no_swap_baseline": no_swap_report, "learned_configs": cv_results}, "selected_policy_type": selected["selected_policy_type"], "selected_config": selected_config, "selected_inner_cv_metrics": {key: selected[key] for key in ("baseline_macro_recall", "policy_macro_recall", "delta", "pooled_cv_delta", "min_fold_delta", "negative_fold_count", "swaps", "beneficial_swaps", "harmful_swaps")}, "fold0_prediction_query_count": len(predictions), "diagnostic_only": False, "no_submission_created": True, "feature_importance": "not computed: HistGradientBoosting has no native feature_importances_; optional permutation analysis intentionally omitted"}
    (args.output_dir / "policy_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
