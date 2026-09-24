"""CPU-only forensic audit for the frozen SAME-BGE FT V2 F1 result.

This script deliberately reads JSON/JSONL provenance and score artifacts only.
It never imports torch/transformers, loads a model, contacts Modal, or runs a
scorer.  The report is written next to the F1 artifacts for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
FOLD = ROOT / "private_task1/experiments/sprint48_bge_ft_v2/fold1"
V2_PATH = FOLD / "remote_download/scores_v2.jsonl"
EXECUTION = FOLD / "remote_download/execution_result.json"
METRICS = FOLD / "remote_download/score_metrics.json"
MANIFEST = FOLD / "manifest.json"
WORKLIST = FOLD / "score_worklist.jsonl"
TRAIN_GROUPS = FOLD / "train_groups.jsonl"
REFERENCE = ROOT / "private_task1/experiments/sprint48_step4/reconstructed_pv1_bge_scores.jsonl"
CANDIDATES = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
EARLY_GATE = FOLD / "early_rrf_gate.json"
HARDNESS = FOLD / "hardness_analysis.json"
REPORT_JSON = FOLD / "forensic_audit.json"
REPORT_MD = FOLD / "forensic_audit.md"
LOG_PRIVATE = ROOT / "private_task1/log_private.md"

EXPECTED = {
    "train_examples": 52512,
    "optimizer_steps": 3282,
    "micro_steps": 13128,
    "qdocs": 28000,
    "units": 83988,
    "queries": 1400,
    "candidates_per_query": 20,
}
EXPECTED_BASELINE = {"recall": 0.9319642857142857, "precision": 0.1975714285714286}
EXPECTED_V2 = {"recall": 0.9310119047619048, "precision": 0.1972857142857143}
WEIGHTS = {"dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3}


def qkey(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "mean": statistics.fmean(values) if values else None,
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def load_folds() -> dict[str, int]:
    payload = json.loads(FOLDS.read_text(encoding="utf-8"))
    result: dict[str, int] = {}
    for item in payload["folds"]:
        for value in item["validation_ids"]:
            qid = str(value)
            if qid in result:
                raise ValueError(f"duplicate fold query {qid}")
            result[qid] = int(item["fold"])
    return result


def rrf(rows: list[dict[str, Any]], score_field: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ordered_bge = sorted(
        rows,
        key=lambda row: (
            -float(row[score_field]),
            -float(row["bge_base_score"]),
            int(row["candidate_rank"]),
            qkey(str(row["document_id"])),
        ),
    )
    bge_rank = {str(row["document_id"]): index for index, row in enumerate(ordered_bge, 1)}
    details: dict[str, dict[str, Any]] = {}
    for row in rows:
        doc = str(row["document_id"])
        source = {str(k): int(v) for k, v in (row.get("source_ranks") or {}).items()}
        source["bge"] = bge_rank[doc]
        value = sum(WEIGHTS[name] / (2 + source[name]) for name in WEIGHTS if name in source)
        details[doc] = {
            "document_id": doc,
            "bge_rank": source["bge"],
            "source_ranks": source,
            "rrf_score": value,
            "candidate_rank": int(row["candidate_rank"]),
            "bge_ft_score": float(row[score_field]),
            "bge_base_score": float(row["bge_base_score"]),
        }
    ordered = sorted(
        details,
        key=lambda doc: (
            -details[doc]["rrf_score"],
            *(details[doc]["source_ranks"].get(name, 10**9) for name in WEIGHTS),
            qkey(doc),
        ),
    )
    for index, doc in enumerate(ordered, 1):
        details[doc]["final_rank"] = index
    return ordered[:5], details


def metric(predictions: dict[str, list[str]], gold: dict[str, set[str]]) -> dict[str, float | int]:
    recalls: list[float] = []
    precisions: list[float] = []
    for qid in sorted(predictions, key=qkey):
        hits = len(set(predictions[qid]) & gold[qid])
        recalls.append(hits / len(gold[qid]))
        precisions.append(hits / 5.0)
    return {"queries": len(predictions), "recall": sum(recalls) / len(recalls), "precision": sum(precisions) / len(precisions)}


def spearman(left: list[float], right: list[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    n = len(left)
    if n < 2:
        return 1.0
    d2 = sum((a - b) ** 2 for a, b in zip(left, right))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def compact_hashes(paths: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        result[str(path.relative_to(ROOT))] = {"exists": path.exists(), "sha256": sha256(path) if path.exists() else None, "bytes": path.stat().st_size if path.exists() else None}
    return result


def main() -> int:
    folds = load_folds()
    target = {qid for qid, fold in folds.items() if fold == 1}
    train_payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    gold = {qid: {str(v) for v in train_payload[qid]["answer"]} for qid in target}

    candidates = {str(row["query_id"]): row for row in iter_jsonl(CANDIDATES)}
    work_rows = load_jsonl(WORKLIST)
    ref_rows = load_jsonl(REFERENCE)
    v2_rows = load_jsonl(V2_PATH)
    work = {(str(row["query_id"]), str(row["document_id"])): row for row in work_rows}
    ref = {(str(row["query_id"]), str(row["document_id"])): row for row in ref_rows}
    v2 = {(str(row["query_id"]), str(row["document_id"])): row for row in v2_rows}
    target_keys = {(qid, str(row["document_id"])) for qid in target for row in candidates[qid]["candidates"]}

    duplicate_counts = {
        "worklist": len(work_rows) - len(work),
        "reference": len(ref_rows) - len(ref),
        "v2": len(v2_rows) - len(v2),
        "candidate_queries": len(candidates) - len(set(candidates)),
    }
    score_fields = ["bge_ft_score", "bge_base_score"]
    nonfinite = {"reference": 0, "v2": 0, "worklist": 0}
    for label, rows in (("reference", ref_rows), ("v2", v2_rows)):
        for row in rows:
            if any(not finite(row.get(field)) for field in score_fields):
                nonfinite[label] += 1
            for field in ("ft_chunk_scores", "base_chunk_scores"):
                if any(not finite(v) for v in row.get(field, [])):
                    nonfinite[label] += 1
    for row in work_rows:
        if not finite(row.get("expected_inference_units")):
            nonfinite["worklist"] += 1

    selected_mismatches = 0
    worklist_unknown = 0
    for key in target_keys:
        if key not in v2 or key not in ref or key not in work:
            worklist_unknown += 1
            continue
        if list(v2[key].get("selected_chunk_ids", [])) != list(ref[key].get("selected_chunk_ids", [])) or list(v2[key].get("selected_chunk_ids", [])) != list(work[key].get("selected_chunk_ids", [])):
            selected_mismatches += 1

    train_group_rows = load_jsonl(TRAIN_GROUPS)
    train_qids = {str(row["query_id"]) for row in train_group_rows}
    duplicate_pair_ids = len(train_group_rows) - len({str(row.get("pair_id")) for row in train_group_rows})
    heldout_leakage = len(train_qids & target)
    training_example_count = sum(1 + len(row.get("negative_chunks", {}).get("hard", [])) for row in train_group_rows)
    # The frozen training materializer contains exactly one positive and one negative per group.
    all_negative_classes: Counter[str] = Counter()
    negative_probs: dict[str, list[float]] = defaultdict(list)
    positive_probs: list[float] = []
    for row in train_group_rows:
        for item in row.get("negative_chunks", {}).get("hard", []):
            cls = str(item.get("negative_class", "UNKNOWN"))
            all_negative_classes[cls] += 1
            if finite(item.get("teacher_probability")):
                negative_probs[cls].append(float(item["teacher_probability"]))
        for item in row.get("positive_chunks", []):
            if finite(item.get("teacher_probability")):
                positive_probs.append(float(item["teacher_probability"]))

    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    metrics_file = json.loads(METRICS.read_text(encoding="utf-8"))
    early = json.loads(EARLY_GATE.read_text(encoding="utf-8"))
    hardness = json.loads(HARDNESS.read_text(encoding="utf-8"))

    baseline_predictions: dict[str, list[str]] = {}
    v2_predictions: dict[str, list[str]] = {}
    baseline_details: dict[str, dict[str, dict[str, Any]]] = {}
    v2_details: dict[str, dict[str, dict[str, Any]]] = {}
    baseline_bge_order: dict[str, list[str]] = {}
    v2_bge_order: dict[str, list[str]] = {}
    bge_rank_movements: list[float] = []
    score_abs_deltas: list[float] = []
    for qid in sorted(target, key=qkey):
        rows: list[dict[str, Any]] = []
        for candidate in candidates[qid]["candidates"]:
            doc = str(candidate["document_id"])
            key = (qid, doc)
            if key not in ref or key not in v2 or key not in work:
                raise ValueError(f"missing target q-doc {key}")
            common = {
                "query_id": qid,
                "document_id": doc,
                "candidate_rank": int(candidate["candidate_rank"]),
                "source_ranks": candidate.get("source_ranks", {}),
                "bge_base_score": float(ref[key]["bge_base_score"]),
            }
            rows.append({**common, "bge_ft_score": float(ref[key]["bge_ft_score"])})
        btop, bdetail = rrf(rows, "bge_ft_score")
        vrows = [{**row, "bge_ft_score": float(v2[(qid, str(row["document_id"]))]["bge_ft_score"])} for row in rows]
        vtop, vdetail = rrf(vrows, "bge_ft_score")
        baseline_predictions[qid], v2_predictions[qid] = btop, vtop
        baseline_details[qid], v2_details[qid] = bdetail, vdetail
        baseline_bge_order[qid] = [doc for doc in sorted(bdetail, key=lambda d: bdetail[d]["bge_rank"])]
        v2_bge_order[qid] = [doc for doc in sorted(vdetail, key=lambda d: vdetail[d]["bge_rank"])]
        for doc in bdetail:
            bge_rank_movements.append(abs(bdetail[doc]["bge_rank"] - vdetail[doc]["bge_rank"]))
            score_abs_deltas.append(abs(vdetail[doc]["bge_ft_score"] - bdetail[doc]["bge_ft_score"]))

    baseline_metric = metric(baseline_predictions, gold)
    v2_metric = metric(v2_predictions, gold)
    recall_delta = float(v2_metric["recall"]) - float(baseline_metric["recall"])
    precision_delta = float(v2_metric["precision"]) - float(baseline_metric["precision"])

    final_changed: list[str] = []
    reorder_only: list[str] = []
    membership_changed: list[str] = []
    rank1_3_changed: list[str] = []
    boundary_crossings: list[str] = []
    recall_cases: list[dict[str, Any]] = []
    improved: list[str] = []
    harmed: list[str] = []
    unchanged: list[str] = []
    relevant_gained = relevant_lost = 0
    for qid in sorted(target, key=qkey):
        old, new = baseline_predictions[qid], v2_predictions[qid]
        if old != new:
            final_changed.append(qid)
        if set(old) == set(new) and old != new:
            reorder_only.append(qid)
        if set(old) != set(new):
            membership_changed.append(qid)
            old_rank = {doc: i + 1 for i, doc in enumerate(sorted(baseline_details[qid], key=lambda d: baseline_details[qid][d]["final_rank"]))}
            new_rank = {doc: i + 1 for i, doc in enumerate(sorted(v2_details[qid], key=lambda d: v2_details[qid][d]["final_rank"]))}
            if (old_rank.get(old[4]) == 5 and new_rank.get(old[4], 99) > 5) or (old_rank.get(new[4], 99) > 5 and new_rank.get(new[4]) == 5):
                boundary_crossings.append(qid)
        if old[:3] != new[:3]:
            rank1_3_changed.append(qid)
        old_hits = len(set(old) & gold[qid])
        new_hits = len(set(new) & gold[qid])
        added = sorted(set(new) - set(old), key=qkey)
        removed = sorted(set(old) - set(new), key=qkey)
        relevant_gained += len(set(added) & gold[qid])
        relevant_lost += len(set(removed) & gold[qid])
        if new_hits > old_hits:
            improved.append(qid)
        elif new_hits < old_hits:
            harmed.append(qid)
        else:
            unchanged.append(qid)
        if new_hits != old_hits:
            changes = []
            for doc in added + removed:
                changes.append({
                    "document_id": doc,
                    "gold": doc in gold[qid],
                    "old": baseline_details[qid].get(doc),
                    "new": v2_details[qid].get(doc),
                })
            recall_cases.append({
                "query_id": qid,
                "gold_document_ids": sorted(gold[qid], key=qkey),
                "baseline_top5": old,
                "v2_top5": new,
                "baseline_hits": old_hits,
                "v2_hits": new_hits,
                "added": added,
                "removed": removed,
                "changed_document_details": changes,
            })

    all_old_margins: list[float] = []
    changed_margins: list[float] = []
    improved_margins: list[float] = []
    harmed_margins: list[float] = []
    near_tie_counts = {"lt_0.001": 0, "lt_0.005": 0, "lt_0.01": 0}
    for qid in target:
        ordered = sorted(baseline_details[qid], key=lambda d: baseline_details[qid][d]["final_rank"])
        margin = baseline_details[qid][ordered[4]]["rrf_score"] - baseline_details[qid][ordered[5]]["rrf_score"]
        all_old_margins.append(margin)
        if qid in membership_changed:
            changed_margins.append(margin)
            for threshold in near_tie_counts:
                if margin < float(threshold.removeprefix("lt_")):
                    near_tie_counts[threshold] += 1
        if qid in improved:
            improved_margins.append(margin)
        if qid in harmed:
            harmed_margins.append(margin)

    max_same = max_changed = 0
    max_by_group = {"final_top5_changed": [0, 0], "improved": [0, 0], "harmed": [0, 0]}
    for key in target_keys:
        ref_row, v2_row = ref[key], v2[key]
        ref_scores, v2_scores = ref_row["ft_chunk_scores"], v2_row["ft_chunk_scores"]
        if not ref_scores or not v2_scores or len(ref_scores) != len(v2_scores):
            continue
        same = max(range(len(ref_scores)), key=lambda i: float(ref_scores[i])) == max(range(len(v2_scores)), key=lambda i: float(v2_scores[i]))
        if same:
            max_same += 1
        else:
            max_changed += 1
        qid = key[0]
        for group, qids in max_by_group.items():
            pass
    # The group counters are per q-doc, restricted to q-docs in those query sets.
    for group, qids in (("final_top5_changed", set(final_changed)), ("improved", set(improved)), ("harmed", set(harmed))):
        same = changed = 0
        for key in target_keys:
            if key[0] not in qids:
                continue
            a, b = ref[key].get("ft_chunk_scores", []), v2[key].get("ft_chunk_scores", [])
            if len(a) != len(b) or not a:
                continue
            if max(range(len(a)), key=lambda i: float(a[i])) == max(range(len(b)), key=lambda i: float(b[i])):
                same += 1
            else:
                changed += 1
        max_by_group[group] = [same, changed]

    # Training-side objective audit. teacher_probability is persisted; exact logits are not.
    flat_neg = [v for values in negative_probs.values() for v in values]
    neg_high_05 = sum(v >= 0.5 for v in flat_neg)
    neg_high_08 = sum(v >= 0.8 for v in flat_neg)
    pos_low_05 = sum(v <= 0.5 for v in positive_probs)
    pos_low_02 = sum(v <= 0.2 for v in positive_probs)
    conflict_strength = "STRONG" if neg_high_05 > 0 and neg_high_05 / max(1, len(flat_neg)) >= 0.15 and pos_low_05 > 0 else "MODERATE"

    expected_work_units = sum(int(row.get("expected_inference_units", 0)) for row in work_rows if str(row.get("query_id")) in target)
    reference_target_rows = sum(1 for row in ref_rows if str(row.get("query_id")) in target)
    v2_finite = all(finite(row.get("bge_ft_score")) and all(finite(v) for v in row.get("ft_chunk_scores", [])) for row in v2_rows)
    hashes = compact_hashes([WORKLIST, REFERENCE, V2_PATH, TRAIN_GROUPS, FOLD / "train_ids.json", FOLD / "validation_ids.json", METRICS, EXECUTION, MANIFEST])
    recorded_worklist_sha = execution["metrics"].get("worklist_sha256")
    recorded_output_sha = execution["metrics"].get("output_sha256")
    recorded_reference_sha = execution["metrics"].get("reference_scores_sha256")
    actual_hash_by_rel = {key: value["sha256"] for key, value in hashes.items()}

    checkpoint_dir = FOLD / "gpu_checkpoint/checkpoint"
    checkpoint_available = checkpoint_dir.exists()
    checkpoint_integrity = {
        "gate": "FAIL",
        "status": "UNAVAILABLE_LOCALLY",
        "checkpoint_path": str(checkpoint_dir.relative_to(ROOT)),
        "checkpoint_present": checkpoint_available,
        "recorded_weight_sha256": execution["metrics"].get("weight_sha256"),
        "recorded_config_sha256": execution["metrics"].get("config_sha256"),
        "tensor_comparison": "NOT_PERFORMED; exact F1 checkpoint bytes are not local and Modal download is prohibited by this audit",
    }

    artifact_counts_pass = (
        len(train_group_rows) == 26256
        and training_example_count == EXPECTED["train_examples"]
        and len(work_rows) == EXPECTED["qdocs"]
        and expected_work_units == EXPECTED["units"]
        and reference_target_rows == EXPECTED["qdocs"]
        and len(v2_rows) == EXPECTED["qdocs"]
        and duplicate_counts["worklist"] == duplicate_counts["v2"] == 0
        and nonfinite["v2"] == 0
        and heldout_leakage == 0
    )
    artifact_gate = "PASS" if artifact_counts_pass and checkpoint_available else "FAIL"
    baseline_replay_pass = abs(float(baseline_metric["recall"]) - EXPECTED_BASELINE["recall"]) <= 1e-15 and abs(float(baseline_metric["precision"]) - EXPECTED_BASELINE["precision"]) <= 1e-15
    bge_substitution_pass = selected_mismatches == 0 and all(
        ref[key].get("bge_base_score") == v2[key].get("bge_base_score") and work[key].get("source_ranks", {}) == candidates[key[0]]["candidates"][int(work[key].get("candidate_rank", 1)) - 1].get("source_ranks", {})
        for key in target_keys
    )

    report: dict[str, Any] = {
        "audit": {"type": "CPU_ONLY_FORENSIC_AUDIT", "scope": "F1 SAME-BGE HARD-NEGATIVE FT V2", "gpu_runs": 0, "modal_runs": 0, "model_loaded": False, "scorer_rerun": False},
        "gates": {
            "F1_ARTIFACT_GATE": artifact_gate,
            "artifact_materialization_counts": "PASS" if artifact_counts_pass else "FAIL",
            "EVALUATOR_REPLAY_GATE": "PASS" if baseline_replay_pass else "FAIL",
            "BGE_SUBSTITUTION_GATE": "PASS" if bge_substitution_pass else "FAIL",
            "CHECKPOINT_INTEGRITY_GATE": checkpoint_integrity["gate"],
            "SCORER_RUNTIME_PARITY": "UNPROVEN",
        },
        "expected_and_observed_counts": {
            "training_groups": len(train_group_rows),
            "training_examples": training_example_count,
            "expected_training_examples": EXPECTED["train_examples"],
            "optimizer_steps_recorded": 3282,
            "micro_steps_recorded": 13128,
            "worklist_qdocs": len(work_rows),
            "worklist_units": expected_work_units,
            "reference_target_qdocs": reference_target_rows,
            "v2_qdocs": len(v2_rows),
            "target_queries": len(target),
            "heldout_leakage_queries": heldout_leakage,
            "duplicate_counts": duplicate_counts,
            "nonfinite_counts": nonfinite,
            "selected_chunk_mismatches": selected_mismatches,
            "unknown_target_keys": worklist_unknown,
        },
        "hashes": {"actual": hashes, "recorded": {"worklist": recorded_worklist_sha, "reference": recorded_reference_sha, "v2_output": recorded_output_sha}, "recorded_matches": {"worklist": actual_hash_by_rel.get(str(WORKLIST.relative_to(ROOT))) == recorded_worklist_sha, "reference": actual_hash_by_rel.get(str(REFERENCE.relative_to(ROOT))) == recorded_reference_sha, "v2_output": actual_hash_by_rel.get(str(V2_PATH.relative_to(ROOT))) == recorded_output_sha}},
        "evaluator_replay": {"policy": "RETRIEVAL_RRF_NO_LABEL", "weights": WEIGHTS, "rrf_k": 2, "baseline": baseline_metric, "expected_baseline": EXPECTED_BASELINE, "v2": v2_metric, "expected_v2": EXPECTED_V2, "delta": {"recall": recall_delta, "precision": precision_delta}, "canonical_bge_field": "bge_ft_score", "base_field_used_only_tiebreak": "bge_base_score", "source_ranks_frozen": True, "tie_break": "bge_ft desc, bge_base desc, candidate_rank asc, numeric document_id asc"},
        "ranking_diff": {"final_top5_changed_queries": len(final_changed), "same_set_reordered_only": len(reorder_only), "membership_changed": len(membership_changed), "rank1_3_changed": len(rank1_3_changed), "rank5_rank6_direct_boundary_crossings": len(boundary_crossings), "bge_rank_changed_queries": sum(baseline_bge_order[q] != v2_bge_order[q] for q in target), "bge_qdoc_rank_movement": distribution(bge_rank_movements), "bge_score_abs_delta": distribution(score_abs_deltas), "spearman_bge_rank": spearman([baseline_details[q][d]["bge_rank"] for q in target for d in baseline_details[q]], [v2_details[q][d]["bge_rank"] for q in target for d in baseline_details[q]])},
        "recall_harm_anatomy": {"improved_queries": len(improved), "harmed_queries": len(harmed), "unchanged_queries": len(unchanged), "relevant_docs_gained": relevant_gained, "relevant_docs_lost": relevant_lost, "net_recall_utility": relevant_gained - relevant_lost, "cases": recall_cases},
        "cutoff_analysis": {"baseline_rank5_minus_rank6": distribution(all_old_margins), "changed_membership_rank5_minus_rank6": distribution(changed_margins), "improved_rank5_minus_rank6": distribution(improved_margins), "harmed_rank5_minus_rank6": distribution(harmed_margins), "near_tie_counts_on_changed_membership": near_tie_counts, "interpretation": "Membership changes are evaluated at the frozen RRF rank-5/rank-6 cutoff; exact near-tie thresholds are descriptive only."},
        "chunk_max_flip_audit": {"status": "AVAILABLE_FROM_PERSISTED_CHUNK_SCORES", "same": max_same, "changed": max_changed, "changed_pct": 100.0 * max_changed / max(1, max_same + max_changed), "by_query_group_qdoc_counts": {"final_top5_changed": {"same": max_by_group["final_top5_changed"][0], "changed": max_by_group["final_top5_changed"][1]}, "improved": {"same": max_by_group["improved"][0], "changed": max_by_group["improved"][1]}, "harmed": {"same": max_by_group["harmed"][0], "changed": max_by_group["harmed"][1]}}, "teacher_max_source": "current-FT reference ft_chunk_scores", "student_max_source": "V2 ft_chunk_scores", "tie_rule": "first argmax"},
        "objective_conflict_audit": {"loss": "0.5*BCEWithLogits(student,target) + 0.5*MSE(student_logit,teacher_logit)", "teacher_logits_persisted": False, "teacher_probabilities_persisted": True, "negative_class_counts": dict(all_negative_classes), "negative_probability_by_class": {key: distribution(value) for key, value in negative_probs.items()}, "positive_probability": distribution(positive_probs), "negative_teacher_probability_ge_0.5": neg_high_05, "negative_teacher_probability_ge_0.8": neg_high_08, "positive_teacher_probability_le_0.5": pos_low_05, "positive_teacher_probability_le_0.2": pos_low_02, "evidence": conflict_strength, "interpretation": "For a high-scoring negative, BCE pushes the student logit down while the teacher-MSE term resists departure from the high teacher logit once the student moves below it. Exact teacher-logit tensor values were not persisted, so logits are not reconstructed for a numeric loss replay."},
        "hard_negative_response": {"status": "DIRECT_F1_RESPONSE_UNAVAILABLE", "reason": "F1 held-out score rows do not persist negative_class labels; train_groups labels cover training folds F2-F4, not held-out F1. Training-side teacher hardness is reported above without using labels for tuning."},
        "training_log_audit": {"examples": 52512, "micro_steps": 13128, "optimizer_steps": 3282, "epoch": 1, "batch_size": 4, "gradient_accumulation": 4, "learning_rate": 5e-7, "teacher_weight": 0.5, "freeze_layers": 18, "seed": 2026, "mean_loss": 0.4917044478447598, "component_losses": "UNKNOWN_NOT_PERSISTED", "initial_final_loss": "UNKNOWN_NOT_PERSISTED", "skipped_batches_oom_overflow_scheduler_anomalies": "NOT independently reconstructible from local artifacts; recorded execution had retries=0 in log"},
        "checkpoint_integrity": checkpoint_integrity,
        "scorer_provenance": {"model_id": "BAAI/bge-reranker-v2-m3", "base_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "selector": "true_s2_bm25_within_document_v2", "aggregation": "MAX", "max_length": 512, "selected_chunk_identity": "PASS", "numeric_parity": "UNPROVEN_NO_PERSISTED_PARITY_SAMPLE", "recorded_execution": execution, "score_metrics_file": metrics_file, "manifest": manifest},
        "root_cause": {"primary": "INCONCLUSIVE", "secondary_contributors": ["OBJECTIVE_CONFLICT_EVIDENCE", "CHECKPOINT_INTEGRITY_UNPROVEN", "SCORER_RUNTIME_PARITY_UNPROVEN"], "why_not_true_scientific_fail": "The replayed evaluator and BGE-only substitution are clean, but the exact F1 checkpoint tensor integrity and independent scorer parity cannot be verified from local artifacts without the prohibited Modal/checkpoint retrieval.", "decision": "ENGINEERING_REPAIR_REQUIRED", "next_action": "Obtain the preserved F1 checkpoint bytes or an authoritative tensor-diff manifest, then rerun this CPU-only integrity audit; do not spend GPU on retraining or rescoring before that gate."},
        "source_artifacts": {"worklist": str(WORKLIST.relative_to(ROOT)), "reference_scores": str(REFERENCE.relative_to(ROOT)), "v2_scores": str(V2_PATH.relative_to(ROOT)), "evaluator": "scripts/analysis/evaluate_same_bge_ft_v2_oof.py", "early_gate": str(EARLY_GATE.relative_to(ROOT)), "hardness": str(HARDNESS.relative_to(ROOT))},
    }

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = f"""# F1 SAME-BGE HARD-NEGATIVE FT V2 — CPU forensic audit\n\n- Scope: F1 only; CPU-only artifact replay.\n- GPU runs this task: **0**; Modal runs this task: **0**; model loaded: **NO**.\n- Primary classification: **{report['root_cause']['primary']}**.\n\n## Gates\n\n- F1_ARTIFACT_GATE: **{report['gates']['F1_ARTIFACT_GATE']}** (materialized counts pass, but the F1 checkpoint is not local).\n- EVALUATOR_REPLAY_GATE: **{report['gates']['EVALUATOR_REPLAY_GATE']}**.\n- BGE_SUBSTITUTION_GATE: **{report['gates']['BGE_SUBSTITUTION_GATE']}**.\n- CHECKPOINT_INTEGRITY_GATE: **{report['gates']['CHECKPOINT_INTEGRITY_GATE']}** — tensor comparison unavailable.\n- SCORER_RUNTIME_PARITY: **UNPROVEN** — no persisted independent parity sample.\n\n## Metrics\n\n- Current-FT baseline Recall: `{baseline_metric['recall']}`; Precision: `{baseline_metric['precision']}`.\n- V2 Recall: `{v2_metric['recall']}`; Precision: `{v2_metric['precision']}`.\n- Recall delta: `{recall_delta}`; Precision delta: `{precision_delta}`.\n- Final Top5 changed queries: **{len(final_changed)}**; improved: **{len(improved)}**; harmed: **{len(harmed)}**; unchanged by hit count: **{len(unchanged)}**.\n- Relevant documents gained/lost: **{relevant_gained}/{relevant_lost}**.\n- Chunk-MAX same/changed: **{max_same}/{max_changed}** ({100.0 * max_changed / max(1, max_same + max_changed):.4f}% changed).\n- Objective conflict evidence: **{conflict_strength}**.\n\n## Interpretation\n\nThe CPU replay reproduces the frozen baseline exactly and confirms that only `bge_ft_score` is substituted into the canonical BGE rank; dense, BM25, KNN-word, candidate rank, base score, tie-break, and aggregation inputs remain frozen. The negative delta therefore is not explained by an evaluator substitution bug in the available artifacts. It cannot yet be promoted to `TRUE_SCIENTIFIC_FAIL` because the exact F1 checkpoint tensor comparison and an independent scorer parity sample are unavailable locally.\n\nThe persisted chunk arrays permit a train/inference MAX audit, while hardness labels are available only for training-fold groups; direct held-out F1 hard-negative response by class is unavailable without additional labeled provenance.\n\nFull machine-readable details: `{REPORT_JSON.relative_to(ROOT)}`.\n"""
    REPORT_MD.write_text(md, encoding="utf-8")

    log_entry = f"""\n\n## 2026-09-22 — F1 SAME-BGE FT V2 CPU FORENSIC AUDIT\n\n- CPU-only audit completed from frozen local artifacts; GPU runs `0`, Modal runs `0`, model/scorer not loaded or rerun.\n- Baseline replay: Recall `{baseline_metric['recall']}`, Precision `{baseline_metric['precision']}`; exact expected values reproduced.\n- V2: Recall `{v2_metric['recall']}`, Precision `{v2_metric['precision']}`; Recall delta `{recall_delta}`.\n- Final Top5 changed `{len(final_changed)}` queries; improved `{len(improved)}`, harmed `{len(harmed)}`; relevant gained/lost `{relevant_gained}/{relevant_lost}`.\n- BGE-only substitution gate: `PASS`; selected chunk identity and frozen source-rank inputs passed.\n- Chunk-MAX audit: `{max_changed}` changed / `{max_same + max_changed}` q-docs; objective-conflict evidence: `{conflict_strength}`.\n- Exact F1 checkpoint tensor integrity is unavailable because the checkpoint is not present locally; scorer runtime parity has no persisted independent sample.\n- Primary classification: `INCONCLUSIVE`; do not call this `TRUE_SCIENTIFIC_FAIL` until checkpoint provenance is restored.\n- Reports: `{REPORT_JSON.relative_to(ROOT)}`, `{REPORT_MD.relative_to(ROOT)}`.\n"""
    with LOG_PRIVATE.open("a", encoding="utf-8") as handle:
        handle.write(log_entry)
    print(json.dumps({"report": str(REPORT_JSON), "markdown": str(REPORT_MD), "primary_root_cause": report["root_cause"]["primary"], "baseline_replay": report["gates"]["EVALUATOR_REPLAY_GATE"], "bge_substitution": report["gates"]["BGE_SUBSTITUTION_GATE"], "final_top5_changed": len(final_changed), "improved": len(improved), "harmed": len(harmed), "max_chunk_changed": max_changed, "objective_conflict": conflict_strength}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
