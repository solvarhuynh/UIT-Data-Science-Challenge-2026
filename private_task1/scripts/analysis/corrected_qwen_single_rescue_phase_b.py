"""Strict nested-OOF Qwen single-slot5 rescue evaluation (Phase B).

This is CPU-only and label-scoped to F1--F4.  It reconstructs the frozen
V2_NESTED incumbent exactly, evaluates only the predeclared eight Qwen rules,
and writes a closed-branch report when the hard deployment gate fails.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.phase1_step4_recovery import load_fold_map, load_gold, metric, qkey  # noqa: E402


OUT = ROOT / "private_task1/experiments/qwen_single_rescue/phase_b"
K20 = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_worklist_k20.jsonl"
QWEN = ROOT / "private_task1/experiments/qwen_single_rescue/validation_complete/qwen_validation_k20_complete.jsonl"
INCUMBENT_BASELINE_OOF = ROOT / "private_task1/experiments/sprint48_guarded_direct/direct_oof_scores.jsonl"
INCUMBENT_SWAPS = ROOT / "private_task1/experiments/sprint48_guarded_direct/guarded_direct_nested_swaps.jsonl"
INCUMBENT_RESULT = ROOT / "private_task1/experiments/sprint48_guarded_direct/guarded_direct_nested_result.json"
FINAL_MODEL = ROOT / "private_task1/experiments/sprint48_guarded_direct/final_direct_model.json"

OUT_PREDICTIONS = OUT / "qwen_single_rescue_oof_predictions.jsonl"
OUT_REPORT_JSON = OUT / "phase_b_report.json"
OUT_REPORT_MD = OUT / "phase_b_report.md"

EXPECTED_K20_SHA = "5bb1f804b629a65611ee0f9e6b11c4b95026a42c3303b395b3dbc0d92ab8a2e2"
EXPECTED_QWEN_SHA = "d66ad20546baa7641ddc39f558481c00bec56aa3ad897d8f75b82a9727c87b79"
EXPECTED_MODEL_SHA = "4b962c56570e6315876aeb6eea26d8877a8007e636d83c70ac2e84a13c2a175c"
EXPECTED_INCUMBENT_RECALL = 0.9300505952380952
EXPECTED_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
EXPECTED_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
EXPECTED_SCORER = "e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe"
RANK_CAPS = (10, 20)
QUANTILES = (0.70, 0.80, 0.90, 0.95)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"non-object JSONL row at {path}:{line_no}")
            yield row


def doc_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def quantile(values: list[float], requested: float) -> float:
    """Numpy-compatible linear quantile without a numerical dependency."""
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("cannot determine quantile of empty margin population")
    position = (len(ordered) - 1) * requested
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def load_k20() -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, int]]:
    rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    counts: Counter[str] = Counter()
    for row in read_jsonl(K20):
        qid, did = str(row["query_id"]), str(row["document_id"])
        if did in rows[qid]:
            raise RuntimeError(f"duplicate K20 identity: {(qid, did)}")
        rank = int(row["candidate_rank"])
        selected = tuple(str(value) for value in row.get("selected_chunk_ids", []))
        if rank < 1 or rank > 20 or not selected or len(selected) != int(row["expected_inference_units"]):
            raise RuntimeError(f"invalid K20 row: {(qid, did)}")
        if row.get("selector") != "true_s2_bm25_within_document_v2" or row.get("aggregation") != "MAX":
            raise RuntimeError(f"K20 contract mismatch: {(qid, did)}")
        rows[qid][did] = {"candidate_rank": rank, "selected_chunk_ids": selected}
        counts[qid] += 1
    if len(rows) != 5600 or sum(counts.values()) != 112000 or any(value != 20 for value in counts.values()):
        raise RuntimeError("current PV1 K20 cardinality is not 5600/112000/20")
    if any(sorted(item["candidate_rank"] for item in docs.values()) != list(range(1, 21)) for docs in rows.values()):
        raise RuntimeError("current K20 rank set is not exactly 1..20")
    return dict(rows), dict(counts)


def load_qwen(k20: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(QWEN):
        qid, did = str(row["query_id"]), str(row["document_id"])
        if did in rows[qid]:
            raise RuntimeError(f"duplicate Qwen identity: {(qid, did)}")
        if row.get("model") != EXPECTED_MODEL or row.get("revision") != EXPECTED_REVISION:
            raise RuntimeError(f"Qwen model/revision mismatch: {(qid, did)}")
        if row.get("scorer_sha256") != EXPECTED_SCORER:
            raise RuntimeError(f"Qwen scorer mismatch: {(qid, did)}")
        if row.get("selector") != "true_s2_bm25_within_document_v2" or row.get("aggregation") != "MAX":
            raise RuntimeError(f"Qwen selector/aggregation mismatch: {(qid, did)}")
        score = float(row["qwen_score"])
        if not math.isfinite(score):
            raise RuntimeError(f"nonfinite Qwen score: {(qid, did)}")
        source = str(row["score_source"])
        if source not in {"HISTORICAL_CANONICAL", "PHASE_A6_DELTA"}:
            raise RuntimeError(f"invalid Qwen score source: {(qid, did)}")
        if qid not in k20 or did not in k20[qid] or int(row["current_candidate_rank"]) != k20[qid][did]["candidate_rank"]:
            raise RuntimeError(f"Qwen/K20 identity or rank mismatch: {(qid, did)}")
        rows[qid][did] = {"score": score, "source": source}
    if set(rows) != set(k20) or any(set(rows[qid]) != set(k20[qid]) for qid in k20):
        raise RuntimeError("complete Qwen cache does not exactly cover current K20")
    split = Counter(item["source"] for docs in rows.values() for item in docs.values())
    if split != Counter({"HISTORICAL_CANONICAL": 111816, "PHASE_A6_DELTA": 184}):
        raise RuntimeError(f"unexpected Qwen source split: {dict(split)}")
    return dict(rows)


def load_incumbent(folds: dict[str, int]) -> dict[str, list[str]]:
    baseline: dict[str, list[str]] = {}
    for row in read_jsonl(INCUMBENT_BASELINE_OOF):
        qid = str(row["query_id"])
        if qid in baseline:
            raise RuntimeError(f"duplicate direct OOF baseline query: {qid}")
        if qid not in folds or int(row["fold"]) != folds[qid]:
            raise RuntimeError(f"direct OOF fold mismatch: {qid}")
        baseline[qid] = [str(value) for value in row["baseline_top5"]]
    if set(baseline) != set(folds) or any(len(value) != 5 or len(set(value)) != 5 for value in baseline.values()):
        raise RuntimeError("baseline incumbent submission is not a complete unique Top5 F1-F4 population")
    result = {qid: list(answer) for qid, answer in baseline.items()}
    seen: set[str] = set()
    for row in read_jsonl(INCUMBENT_SWAPS):
        qid = str(row["query_id"])
        if qid not in result or qid in seen:
            raise RuntimeError(f"invalid or duplicate nested incumbent swap: {qid}")
        seen.add(qid)
        if int(row["fold"]) != folds[qid]:
            raise RuntimeError(f"nested incumbent fold mismatch: {qid}")
        pos = int(row["pos"])
        if pos not in {4, 5}:
            raise RuntimeError(f"nested incumbent changed forbidden position: {qid}:{pos}")
        if result[qid][pos - 1] != str(row["dropped"]):
            raise RuntimeError(f"nested incumbent drop mismatch: {qid}")
        result[qid][pos - 1] = str(row["added"])
        if len(set(result[qid])) != 5:
            raise RuntimeError(f"nested incumbent created duplicate Top5: {qid}")
    if len(seen) != 1101:
        raise RuntimeError(f"nested incumbent swap count is {len(seen)}, expected 1101")
    return result


def metrics_by_fold(predictions: dict[str, list[str]], gold: dict[str, dict[str, Any]], folds: dict[str, int]) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    overall = metric(predictions, gold)
    per_fold: dict[str, dict[str, float]] = {}
    for fold in sorted({folds[qid] for qid in predictions}):
        ids = [qid for qid in predictions if folds[qid] == fold]
        per_fold[f"F{fold}"] = metric({qid: predictions[qid] for qid in ids}, {qid: gold[qid] for qid in ids})  # type: ignore[assignment]
    return overall, per_fold


def best_candidate(qid: str, incumbent: list[str], k20: dict[str, dict[str, dict[str, Any]]], qwen: dict[str, dict[str, dict[str, Any]]], rank_cap: int) -> dict[str, Any] | None:
    eligible = [
        did for did, meta in k20[qid].items()
        if 6 <= meta["candidate_rank"] <= rank_cap and did not in incumbent
    ]
    if not eligible:
        return None
    did = sorted(eligible, key=lambda value: (-qwen[qid][value]["score"], k20[qid][value]["candidate_rank"], doc_key(value)))[0]
    d5 = incumbent[4]
    if d5 not in qwen[qid]:
        raise RuntimeError(f"incumbent rank5 is outside current K20: {qid}:{d5}")
    return {
        "candidate": did,
        "candidate_rank": k20[qid][did]["candidate_rank"],
        "candidate_qwen_score": qwen[qid][did]["score"],
        "candidate_source": qwen[qid][did]["source"],
        "d5": d5,
        "d5_qwen_score": qwen[qid][d5]["score"],
        "d5_source": qwen[qid][d5]["source"],
        "margin": qwen[qid][did]["score"] - qwen[qid][d5]["score"],
    }


def apply_policy(qids: Iterable[str], incumbent: dict[str, list[str]], candidates: dict[str, dict[str, Any] | None], threshold: float) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    predictions: dict[str, list[str]] = {}
    actions: dict[str, dict[str, Any]] = {}
    for qid in qids:
        base = incumbent[qid]
        choice = candidates[qid]
        proposed = list(base)
        changed = choice is not None and float(choice["margin"]) >= threshold
        if changed:
            proposed[4] = str(choice["candidate"])
        if proposed[:4] != base[:4] or len(proposed) != 5 or len(set(proposed)) != 5:
            raise RuntimeError(f"single-slot policy invariant failed: {qid}")
        predictions[qid] = proposed
        actions[qid] = {"changed": changed, **(choice or {})}
    return predictions, actions


def change_stats(baseline: dict[str, list[str]], proposed: dict[str, list[str]], gold: dict[str, dict[str, Any]], actions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values: Counter[str] = Counter()
    depth: Counter[str] = Counter()
    source_involvement: dict[str, Counter[str]] = {"changed": Counter(), "improved": Counter(), "harmed": Counter()}
    for qid, predicted in proposed.items():
        if predicted[:4] != baseline[qid][:4]:
            values["top1_4_changed"] += 1
        if predicted == baseline[qid]:
            continue
        values["changed"] += 1
        action = actions[qid]
        rank = int(action["candidate_rank"])
        depth["6-10" if rank <= 10 else "11-20"] += 1
        wanted = {str(value) for value in gold[qid].get("answer", [])}
        old_hits = len(wanted & set(baseline[qid]))
        new_hits = len(wanted & set(predicted))
        outcome = "neutral"
        if new_hits > old_hits:
            outcome = "improved"
            values["improved"] += 1
            values["gained"] += new_hits - old_hits
        elif new_hits < old_hits:
            outcome = "harmed"
            values["harmed"] += 1
            values["lost"] += old_hits - new_hits
        else:
            values["neutral"] += 1
        involved = {str(action["candidate_source"]), str(action["d5_source"])}
        for source in involved:
            source_involvement["changed"][source] += 1
            if outcome in {"improved", "harmed"}:
                source_involvement[outcome][source] += 1
    return {
        "changed_queries": values["changed"],
        "improved_queries": values["improved"],
        "harmed_queries": values["harmed"],
        "neutral_changed_queries": values["neutral"],
        "net_relevant_document_change": values["gained"] - values["lost"],
        "top1_4_changed": values["top1_4_changed"],
        "candidate_rank_split": dict(depth),
        "qwen_source_involvement": {key: dict(value) for key, value in source_involvement.items()},
    }


def evaluate_configuration(train_ids: list[str], rank_cap: int, requested_quantile: float, incumbent: dict[str, list[str]], k20: dict[str, dict[str, dict[str, Any]]], qwen: dict[str, dict[str, dict[str, Any]]], gold: dict[str, dict[str, Any]], folds: dict[str, int]) -> dict[str, Any]:
    candidates = {qid: best_candidate(qid, incumbent[qid], k20, qwen, rank_cap) for qid in train_ids}
    margins = [float(value["margin"]) for value in candidates.values() if value is not None]
    threshold = quantile(margins, requested_quantile)
    proposed, actions = apply_policy(train_ids, incumbent, candidates, threshold)
    incumbent_train = {qid: incumbent[qid] for qid in train_ids}
    before, before_fold = metrics_by_fold(incumbent_train, gold, folds)
    after, after_fold = metrics_by_fold(proposed, gold, folds)
    delta_by_fold = {fold: after_fold[fold]["recall"] - before_fold[fold]["recall"] for fold in before_fold}
    stats = change_stats(incumbent_train, proposed, gold, actions)
    mutation_rate = stats["changed_queries"] / len(train_ids)
    eligible = (
        all(value >= 0.0 for value in delta_by_fold.values())
        and after["precision"] - before["precision"] >= -0.0002
        and mutation_rate <= 0.10
        and stats["top1_4_changed"] == 0
        and stats["changed_queries"] == stats["improved_queries"] + stats["harmed_queries"] + stats["neutral_changed_queries"]
    )
    return {
        "rank_cap": rank_cap,
        "quantile": requested_quantile,
        "threshold": threshold,
        "margin_population": len(margins),
        "training_metrics": {"incumbent": before, "proposed": after, "recall_delta": after["recall"] - before["recall"], "precision_delta": after["precision"] - before["precision"]},
        "training_fold_recall_delta": delta_by_fold,
        "mutation_rate": mutation_rate,
        "stats": stats,
        "train_eligible": eligible,
    }


def select_configuration(configs: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [item for item in configs if item["train_eligible"]]
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: (
        -float(item["training_metrics"]["recall_delta"]),
        -min(float(value) for value in item["training_fold_recall_delta"].values()),
        int(item["stats"]["changed_queries"]),
        -float(item["threshold"]),
        int(item["rank_cap"]),
    ))[0]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> int:
    input_hashes = {"k20": sha256(K20), "qwen": sha256(QWEN), "final_model": sha256(FINAL_MODEL)}
    if input_hashes["k20"] != EXPECTED_K20_SHA or input_hashes["qwen"] != EXPECTED_QWEN_SHA or input_hashes["final_model"] != EXPECTED_MODEL_SHA:
        raise RuntimeError("frozen input SHA mismatch")
    k20, k20_counts = load_k20()
    qwen = load_qwen(k20)
    folds = load_fold_map()
    target = {qid: fold for qid, fold in folds.items() if fold in {1, 2, 3, 4}}
    if set(target) != set(k20) or Counter(target.values()) != Counter({1: 1400, 2: 1400, 3: 1400, 4: 1400}):
        raise RuntimeError("F1-F4 fold manifest is not exactly 1400 each / Fold0 excluded")
    gold_all = load_gold()
    gold = {qid: gold_all[qid] for qid in target}
    incumbent = load_incumbent(target)
    incumbent_metric, incumbent_by_fold = metrics_by_fold(incumbent, gold, target)
    stored = json.loads(INCUMBENT_RESULT.read_text(encoding="utf-8"))
    replay_pass = (
        abs(float(incumbent_metric["recall"]) - EXPECTED_INCUMBENT_RECALL) <= 1e-12
        and abs(float(incumbent_metric["recall"]) - float(stored["proposed"]["recall"])) <= 1e-12
        and abs(float(incumbent_metric["precision"]) - float(stored["proposed"]["precision"])) <= 1e-12
    )
    if not replay_pass:
        raise RuntimeError(f"INCUMBENT_REPLAY_GATE failed: {incumbent_metric}")

    all_ids = sorted(target, key=qkey)
    oof_predictions: dict[str, list[str]] = {}
    oof_actions: dict[str, dict[str, Any]] = {}
    fold_reports: dict[str, dict[str, Any]] = {}
    for outer_fold in (1, 2, 3, 4):
        train_ids = [qid for qid in all_ids if target[qid] != outer_fold]
        test_ids = [qid for qid in all_ids if target[qid] == outer_fold]
        configurations = [
            evaluate_configuration(train_ids, cap, requested, incumbent, k20, qwen, gold, target)
            for cap in RANK_CAPS for requested in QUANTILES
        ]
        selected = select_configuration(configurations)
        if selected is None:
            proposed = {qid: list(incumbent[qid]) for qid in test_ids}
            actions = {qid: {"changed": False} for qid in test_ids}
            heldout_policy: dict[str, Any] = {"name": "NO_CHANGE", "rank_cap": None, "quantile": None, "threshold": None}
        else:
            candidates = {qid: best_candidate(qid, incumbent[qid], k20, qwen, int(selected["rank_cap"])) for qid in test_ids}
            proposed, actions = apply_policy(test_ids, incumbent, candidates, float(selected["threshold"]))
            heldout_policy = {"name": "QWEN_SINGLE_RESCUE", "rank_cap": selected["rank_cap"], "quantile": selected["quantile"], "threshold": selected["threshold"]}
        before, _ = metrics_by_fold({qid: incumbent[qid] for qid in test_ids}, gold, target)
        after, _ = metrics_by_fold(proposed, gold, target)
        fold_reports[f"F{outer_fold}"] = {
            "outer_fold": outer_fold,
            "train_query_count": len(train_ids),
            "heldout_query_count": len(test_ids),
            "configurations": configurations,
            "selected": heldout_policy,
            "heldout_metrics": {"incumbent": before, "proposed": after, "recall_delta": after["recall"] - before["recall"], "precision_delta": after["precision"] - before["precision"]},
            "heldout_stats": change_stats({qid: incumbent[qid] for qid in test_ids}, proposed, gold, actions),
        }
        oof_predictions.update(proposed)
        oof_actions.update(actions)

    if set(oof_predictions) != set(incumbent) or any(len(value) != 5 or len(set(value)) != 5 for value in oof_predictions.values()):
        raise RuntimeError("strict OOF output is not complete unique Top5")
    overall, per_fold = metrics_by_fold(oof_predictions, gold, target)
    stats = change_stats(incumbent, oof_predictions, gold, oof_actions)
    fold_delta = {fold: per_fold[fold]["recall"] - incumbent_by_fold[fold]["recall"] for fold in per_fold}
    recall_delta = overall["recall"] - incumbent_metric["recall"]
    precision_delta = overall["precision"] - incumbent_metric["precision"]
    positive_folds = sum(value > 0.0 for value in fold_delta.values())
    hard_gate = (
        stats["top1_4_changed"] == 0
        and all(value >= 0.0 for value in fold_delta.values())
        and positive_folds >= 3
        and recall_delta >= 0.0015
        and precision_delta >= -0.0002
        and stats["changed_queries"] <= 560
        and stats["improved_queries"] > stats["harmed_queries"]
        and stats["changed_queries"] == stats["improved_queries"] + stats["harmed_queries"] + stats["neutral_changed_queries"]
    )

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for qid in all_ids:
        action = oof_actions[qid]
        rows.append({
            "query_id": qid,
            "fold": target[qid],
            "incumbent_top5": incumbent[qid],
            "qwen_single_rescue_top5": oof_predictions[qid],
            "changed": bool(action["changed"]),
            "candidate_document_id": action.get("candidate"),
            "candidate_rank": action.get("candidate_rank"),
            "qwen_margin": action.get("margin"),
            "candidate_qwen_source": action.get("candidate_source"),
            "rank5_qwen_source": action.get("d5_source"),
        })
    write_jsonl(OUT_PREDICTIONS, rows)
    report = {
        "status": "CORRECTED_QWEN_SINGLE_RESCUE_PHASE_B_COMPLETE",
        "input_gates": {
            "complete_qwen_cache_sha256": input_hashes["qwen"],
            "complete_qwen_cache_gate": "PASS",
            "current_k20_sha256": input_hashes["k20"],
            "final_model_sha256": input_hashes["final_model"],
            "validation_queries": len(k20),
            "validation_qdocs": sum(k20_counts.values()),
            "qdocs_per_query": 20,
            "fold_counts": {f"F{fold}": count for fold, count in sorted(Counter(target.values()).items())},
            "fold0_used": False,
            "public_labels_used": False,
            "private_labels_used": False,
        },
        "incumbent": {
            "policy": "GUARDED_DIRECT_K20_RESIDUAL_V2_NESTED",
            "baseline_oof_source": rel(INCUMBENT_BASELINE_OOF),
            "nested_swaps_source": rel(INCUMBENT_SWAPS),
            "nested_result_source": rel(INCUMBENT_RESULT),
            "replay_gate": "PASS",
            "metrics": incumbent_metric,
            "per_fold": incumbent_by_fold,
        },
        "policy_family": {"rank_caps": list(RANK_CAPS), "training_margin_quantiles": list(QUANTILES), "control": "NO_CHANGE", "single_slot": 5, "top1_4_immutable": True},
        "nested_oof": {"fold_reports": fold_reports, "metrics": overall, "per_fold": per_fold, "recall_delta": recall_delta, "precision_delta": precision_delta, "fold_recall_delta": fold_delta, "positive_folds": positive_folds, "stats": stats, "oof_predictions_path": rel(OUT_PREDICTIONS), "oof_predictions_sha256": sha256(OUT_PREDICTIONS)},
        "hard_gate": {
            "top1_4_unchanged": stats["top1_4_changed"] == 0,
            "every_fold_nonnegative": all(value >= 0.0 for value in fold_delta.values()),
            "positive_folds_at_least_3": positive_folds >= 3,
            "pooled_recall_delta_at_least_0_0015": recall_delta >= 0.0015,
            "pooled_precision_delta_at_least_minus_0_0002": precision_delta >= -0.0002,
            "changed_queries_at_most_560": stats["changed_queries"] <= 560,
            "improved_gt_harmed": stats["improved_queries"] > stats["harmed_queries"],
            "duplicate_top5": 0,
            "fold0_used": False,
            "public_labels_used": False,
            "private_labels_used": False,
            "gate": "PASS" if hard_gate else "FAIL",
        },
        "decision": "OPEN_PRIVATE_PREP" if hard_gate else "CLOSE_NO_PRIVATE_GPU",
        "execution": {"gpu_runs_this_phase": 0, "modal_gpu_runs_this_phase": 0, "qwen_inference_this_phase": 0},
    }
    OUT_REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = [
        "# Corrected Qwen single-rescue — Phase B",
        "",
        f"- Incumbent replay: **PASS**; Recall `{incumbent_metric['recall']}`, Precision `{incumbent_metric['precision']}`.",
        f"- Strict nested OOF Qwen Recall `{overall['recall']}` (delta `{recall_delta}`); Precision delta `{precision_delta}`.",
        f"- Fold Recall deltas: `{fold_delta}`. Changed/improved/harmed: `{stats['changed_queries']}/{stats['improved_queries']}/{stats['harmed_queries']}`.",
        f"- Top1–4 changes: `{stats['top1_4_changed']}`; duplicate Top5: `0`.",
        f"- Hard scientific gate: **{'PASS' if hard_gate else 'FAIL'}**. Decision: **{report['decision']}**.",
        "- GPU/Modal GPU/Qwen inference in this phase: `0/0/0`; Fold0, Public labels, and Private labels unused.",
    ]
    OUT_REPORT_MD.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("COMPLETE_QWEN_CACHE_GATE: PASS")
    print("INCUMBENT_REPLAY_GATE: PASS")
    print(f"INCUMBENT_OOF_RECALL: {incumbent_metric['recall']}")
    print(f"QWEN_OOF_RECALL: {overall['recall']}")
    print(f"QWEN_OOF_RECALL_DELTA: {recall_delta}")
    print(f"QWEN_OOF_PRECISION_DELTA: {precision_delta}")
    for fold in ("F1", "F2", "F3", "F4"):
        print(f"{fold}_DELTA: {fold_delta[fold]}")
    print(f"OOF_CHANGED_QUERIES: {stats['changed_queries']}")
    print(f"OOF_IMPROVED_QUERIES: {stats['improved_queries']}")
    print(f"OOF_HARMED_QUERIES: {stats['harmed_queries']}")
    print(f"OOF_TOP1_4_CHANGED: {stats['top1_4_changed']}")
    print(f"PHASE_B_GATE: {'PASS' if hard_gate else 'FAIL'}")
    if hard_gate:
        print("QWEN_BRANCH_DECISION: OPEN_PRIVATE_PREP")
        print("PRIVATE_GPU_PREP: REQUIRED_NEXT_STAGE")
        print("NEXT_ACTION: CONTINUE_PRIVATE_PREP")
    else:
        print("QWEN_BRANCH_DECISION: CLOSE_NO_PRIVATE_GPU")
        print("PRIVATE_GPU_PREP: NOT_RUN")
        print("NEXT_ACTION: CLOSE_QWEN_BRANCH")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
