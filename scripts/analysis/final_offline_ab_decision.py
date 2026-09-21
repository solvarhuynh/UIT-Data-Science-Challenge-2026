"""Finalize the CPU-only F1-F4 A/B decision after slot4/5 BGE completion.

This script reads the frozen historical BGE TOP5 artifact and the separately
completed slot4/5 anchor artifact.  It never loads a model, calls Modal, or
reads Fold0/public/private labels.  The F1-F4 gold fields are used only for
the already-authorized offline validation population.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import evaluate_bge_top5_f1_f4 as evaluator


BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
FOLDS = ROOT / "artifacts/task1/handoff/step4_bge_handoff/candidates/folds.json"
UNION = ROOT / "artifacts/task1/handoff/step4_bge_handoff/candidates/f1_f4_candidate_union.jsonl"
CURRENT = ROOT / "artifacts/task1/qwen_to_bge_minimal/bge_top5_production_canonical.jsonl"
CURRENT_MANIFEST = ROOT / "artifacts/task1/qwen_to_bge_minimal/bge_top5_production_manifest.json"
ANCHOR = ROOT / "artifacts/task1/qwen_to_bge_minimal/bge_baseline_slot45_download/results/production/scores.jsonl"
ANCHOR_MANIFEST = ROOT / "artifacts/task1/qwen_to_bge_minimal/bge_baseline_slot45_download/manifests/production.json"
ANCHOR_WORKLIST = ROOT / "artifacts/task1/qwen_to_bge_minimal/baseline_slot45_missing_current_bge.jsonl"

OUT_DIR = ROOT / "private_task1/experiments/f1_f4"
DECISION = OUT_DIR / "final_ab_decision.json"
REPORT = OUT_DIR / "final_ab_report.md"

EXPECTED_ANCHOR_WORKLIST_SHA = "8f8ee85d4e6244bd93e96943c75bc942cc9c41984fd792bf70bc866518e4d7e7"
EXPECTED_ANCHOR_OUTPUT_SHA = "0eb9952329cae0f7a8a09b48cb2b9b0e2a88f3e1a6edb3e7f770e560479bcb75"
EXPECTED_MODEL = "BAAI/bge-reranker-v2-m3"
EXPECTED_BASE_REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
EXPECTED_SELECTOR = "true_s2_bm25_within_document_v2"
EXPECTED_AGGREGATION = "MAX"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                row["_line"] = line_no
                rows.append(row)
    return rows


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_score_rows(rows: list[dict], expected_count: int, label: str) -> set[tuple[str, str]]:
    if len(rows) != expected_count:
        raise RuntimeError(f"{label}_ROW_COUNT:{len(rows)}!={expected_count}")
    keys = set()
    for row in rows:
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in keys:
            raise RuntimeError(f"{label}_DUPLICATE:{key}")
        keys.add(key)
        if row.get("model_id") != EXPECTED_MODEL:
            raise RuntimeError(f"{label}_MODEL:{key}")
        if row.get("base_model_revision") != EXPECTED_BASE_REV:
            raise RuntimeError(f"{label}_REVISION:{key}")
        if row.get("selector") != EXPECTED_SELECTOR or row.get("aggregation") != EXPECTED_AGGREGATION:
            raise RuntimeError(f"{label}_CONTRACT:{key}")
        if not finite(row.get("bge_ft_score")) or not finite(row.get("bge_base_score")):
            raise RuntimeError(f"{label}_NONFINITE_DOCUMENT_SCORE:{key}")
        selected = row.get("selected_chunk_ids")
        ft = row.get("ft_chunk_scores")
        base = row.get("base_chunk_scores")
        if not isinstance(selected, list) or not (1 <= len(selected) <= 3) or len(set(map(str, selected))) != len(selected):
            raise RuntimeError(f"{label}_CHUNK_SELECTION:{key}")
        if not isinstance(ft, list) or not isinstance(base, list) or len(ft) != len(selected) or len(base) != len(selected):
            raise RuntimeError(f"{label}_CHUNK_SCORE_LENGTH:{key}")
        if any(not finite(x) for x in list(ft) + list(base)):
            raise RuntimeError(f"{label}_NONFINITE_CHUNK_SCORE:{key}")
    return keys


def clean(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "_line"}


def paired_stats(predictions: dict[str, list[str]], baseline: dict[str, dict]) -> dict:
    changed = unchanged = a_wrong_b_correct = a_correct_b_wrong = 0
    gained = lost = 0
    for qid, b_top in predictions.items():
        a_top = [str(x) for x in baseline[qid]["top5"]]
        if b_top == a_top:
            unchanged += 1
        else:
            changed += 1
        gold = {str(x) for x in baseline[qid]["gold_documents"]}
        a_hits = len(gold & set(a_top))
        b_hits = len(gold & set(b_top))
        if a_hits == 0 and b_hits > 0:
            a_wrong_b_correct += 1
        if a_hits > 0 and b_hits == 0:
            a_correct_b_wrong += 1
        delta = b_hits - a_hits
        gained += max(delta, 0)
        lost += max(-delta, 0)
    return {
        "changed_queries": changed,
        "unchanged_queries": unchanged,
        "A_wrong_B_correct": a_wrong_b_correct,
        "A_correct_B_wrong": a_correct_b_wrong,
        "relevant_docs_gained": gained,
        "relevant_docs_lost": lost,
        "net_relevant_doc_gain": gained - lost,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    current_manifest = json.loads(CURRENT_MANIFEST.read_text(encoding="utf-8"))
    anchor_manifest = json.loads(ANCHOR_MANIFEST.read_text(encoding="utf-8"))
    current_sha = sha256(CURRENT)
    anchor_sha = sha256(ANCHOR)
    worklist_sha = sha256(ANCHOR_WORKLIST)
    if current_sha != current_manifest["output_sha256"]:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED CURRENT_OUTPUT_SHA_MISMATCH")
    if anchor_sha != anchor_manifest["output_sha256"] or anchor_sha != EXPECTED_ANCHOR_OUTPUT_SHA:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED ANCHOR_OUTPUT_SHA_MISMATCH")
    if worklist_sha != anchor_manifest["worklist_sha256"] or worklist_sha != EXPECTED_ANCHOR_WORKLIST_SHA:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED WORKLIST_SHA_MISMATCH")
    if anchor_manifest.get("status") != "COMPLETE":
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED GPU_OUTPUT_NOT_COMPLETE")
    if int(anchor_manifest.get("q_doc_count", -1)) != 10737 or int(anchor_manifest.get("inference_unit_count", -1)) != 32210:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED ANCHOR_COUNTS_MISMATCH")
    models = anchor_manifest.get("models", {})
    if models.get("base", {}).get("model_id") != EXPECTED_MODEL or models.get("base", {}).get("revision") != EXPECTED_BASE_REV:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED_BASE_MODEL_PROVENANCE")
    if models.get("ft", {}).get("weight_sha256") != "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c":
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED_FT_PROVENANCE")
    runner_text = (ROOT / "scripts/modal/task1_bge_subset_gpu.py").read_text(encoding="utf-8")
    if "MAX_LENGTH = 512" not in runner_text or "max_length=MAX_LENGTH" not in runner_text:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED_MAX_LENGTH_PROVENANCE")

    current_rows = read_jsonl(CURRENT)
    anchor_rows = read_jsonl(ANCHOR)
    current_keys = validate_score_rows(current_rows, int(current_manifest["q_doc_count"]), "CURRENT")
    anchor_keys = validate_score_rows(anchor_rows, 10737, "ANCHOR")
    if current_keys & anchor_keys:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED_CURRENT_ANCHOR_OVERLAP")

    baseline_rows = {str(row["query_id"]): row for row in read_jsonl(BASELINE)}
    folds = evaluator.load_folds()
    scientific = {q: row for q, row in baseline_rows.items() if folds[q] in evaluator.TARGET_FOLDS}
    if len(scientific) != 5600:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED_F1_F4_QUERY_COUNT")
    baseline_anchor_rank: dict[tuple[str, str], int] = {}
    for qid, row in scientific.items():
        top5 = [str(x) for x in row["top5"]]
        if len(top5) != 5 or len(set(top5)) != 5:
            raise RuntimeError(f"FINAL_OFFLINE_DECISION=BLOCKED_BASELINE_TOP5:{qid}")
        baseline_anchor_rank[(qid, top5[3])] = 4
        baseline_anchor_rank[(qid, top5[4])] = 5
    if len(baseline_anchor_rank) != 11200:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED_BASELINE_ANCHOR_UNIVERSE")
    worklist_rows = read_jsonl(ANCHOR_WORKLIST)
    worklist_keys = {(str(r["query_id"]), str(r["document_id"])) for r in worklist_rows}
    if len(worklist_rows) != 10737 or len(worklist_keys) != 10737 or worklist_keys != anchor_keys:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED_WORKLIST_IDENTITY_COVERAGE")
    existing_anchor_keys = current_keys & set(baseline_anchor_rank)
    if len(existing_anchor_keys) != 463:
        raise RuntimeError(f"FINAL_OFFLINE_DECISION=BLOCKED_EXISTING_ANCHOR_COUNT:{len(existing_anchor_keys)}")
    all_anchor_keys = existing_anchor_keys | anchor_keys
    if all_anchor_keys != set(baseline_anchor_rank):
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED_BASELINE_ANCHOR_COVERAGE")
    rank4 = sum(baseline_anchor_rank[k] == 4 for k in all_anchor_keys)
    rank5 = sum(baseline_anchor_rank[k] == 5 for k in all_anchor_keys)
    if rank4 != 5600 or rank5 != 5600:
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED_RANK4_RANK5_COVERAGE")

    baseline_predictions = {q: [str(x) for x in row["top5"]] for q, row in scientific.items()}
    baseline_metric = evaluator.metric(baseline_predictions, scientific)
    expected = {"recall": 0.9259285714285714, "precision": 0.19739285714285715}
    for key, value in expected.items():
        if abs(baseline_metric[key] - value) > 1e-15:
            raise RuntimeError(f"FINAL_OFFLINE_DECISION=BLOCKED_BASELINE_METRIC:{key}")
    baseline_by_fold = {
        str(f): evaluator.metric(
            {q: baseline_predictions[q] for q in scientific if folds[q] == f},
            {q: scientific[q] for q in scientific if folds[q] == f},
        )
        for f in evaluator.TARGET_FOLDS
    }

    merged_rows = [clean(row) for row in current_rows + anchor_rows]
    metadata = evaluator.build_meta(read_jsonl(UNION), worklist_rows)
    # Qwen-discovered TOP5 rows can be absent from the dense-union metadata.
    # Their retrieval features are neutral; Qwen scores/ranks are never read.
    for row in merged_rows:
        metadata.setdefault((str(row["query_id"]), str(row["document_id"])), {"union_rank": 999, "rrf_score": 0.0, "source_support": 1})
    scored_all: dict[str, list[dict]] = defaultdict(list)
    for row in merged_rows:
        qid = str(row["query_id"])
        if qid in scientific:
            scored_all[qid].append(row)
    scored = {q: rows for q, rows in scored_all.items()}
    if sum(len(v) for v in scored.values()) != len(merged_rows):
        raise RuntimeError("FINAL_OFFLINE_DECISION=BLOCKED_SCORE_POPULATION")

    policy_labels = {"FT_ONLY": "B_FT_ONLY", "BASE_ONLY": "B_BASE_ONLY", "FT_BASE_EQUAL": "B_FT_BASE_EQUAL", "HISTORICAL_STEP4": "B_STEP4", "BALANCED": "B_BALANCED", "FT_HEAVY": "B_FT_HEAVY"}
    results = []
    prediction_map = {}
    for policy in evaluator.POLICIES:
        for slot4 in evaluator.SLOT_MARGINS["slot4"]:
            for slot5 in evaluator.SLOT_MARGINS["slot5"]:
                for rank_safe in (False, True):
                    result, predictions = evaluator.evaluate_policy(policy, slot4, slot5, rank_safe, scientific, scored, metadata, folds)
                    result["policy_label"] = policy_labels[policy]
                    result["paired"] = paired_stats(predictions, scientific)
                    gate = dict(result["scientific_gate"])
                    gate["net_relevant_gain_positive"] = result["paired"]["net_relevant_doc_gain"] > 0
                    gate["overall"] = all(gate.values())
                    result["scientific_gate"] = gate
                    results.append(result)
                    prediction_map[(policy, slot4, slot5, rank_safe)] = predictions

    def simplicity(result: dict) -> tuple:
        weights = evaluator.POLICIES[result["policy"]]
        return (sum(abs(x) > 0 for x in weights), sum(abs(x) for x in weights), int(result["rank_safe"]), result["slot4_margin"], result["slot5_margin"], result["policy"])

    best_b = sorted(results, key=lambda r: (-r["metrics"]["recall"], -r["metrics"]["precision"], r["paired"]["changed_queries"], simplicity(r)))[0]
    admissible = [r for r in results if r["scientific_gate"]["overall"]]
    selected = sorted(admissible, key=lambda r: (-r["metrics"]["recall"], -r["metrics"]["precision"], r["paired"]["changed_queries"], simplicity(r)))[0] if admissible else None
    decision = "USE_B" if selected is not None else "USE_A"
    selected_name = selected["policy_label"] if selected else "A_BASELINE"
    selected_metric = selected["metrics"] if selected else baseline_metric
    selected_fold = selected["metrics_by_fold"] if selected else baseline_by_fold
    selected_pair = selected["paired"] if selected else {"changed_queries": 0, "unchanged_queries": 5600, "A_wrong_B_correct": 0, "A_correct_B_wrong": 0, "relevant_docs_gained": 0, "relevant_docs_lost": 0, "net_relevant_doc_gain": 0}
    selected_deltas = selected["delta_by_fold"] if selected else {str(f): {"recall": 0.0, "precision": 0.0} for f in evaluator.TARGET_FOLDS}

    payload = {
        "status": "FINAL_OFFLINE_DECISION_COMPLETE",
        "population": 5600,
        "gpu_output_manifest": str(ANCHOR_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "gpu_output_scores": str(ANCHOR.relative_to(ROOT)).replace("\\", "/"),
        "gpu_output_status": anchor_manifest["status"],
        "new_anchor_rows": len(anchor_rows),
        "new_anchor_unique": len(anchor_keys),
        "worklist_sha256": worklist_sha,
        "worklist_hash_match": True,
        "baseline_existing_current_anchors": len(existing_anchor_keys),
        "baseline_slot45_coverage": "11200/11200",
        "rank4_coverage": rank4,
        "rank5_coverage": rank5,
        "baseline_reproduced": True,
        "baseline_recall": baseline_metric["recall"],
        "baseline_precision": baseline_metric["precision"],
        "selected_policy": selected_name,
        "selected_recall": selected_metric["recall"],
        "selected_precision": selected_metric["precision"],
        "delta_recall": selected_metric["recall"] - baseline_metric["recall"],
        "delta_precision": selected_metric["precision"] - baseline_metric["precision"],
        "fold_recalls": {f"F{f}": selected_fold[str(f)]["recall"] for f in evaluator.TARGET_FOLDS},
        "fold_deltas": {f"F{f}": selected_deltas[str(f)]["recall"] for f in evaluator.TARGET_FOLDS},
        **selected_pair,
        "slot4_replacements": selected.get("slot_changes", {}).get("slot4_replacements", 0) if selected else 0,
        "slot5_replacements": selected.get("slot_changes", {}).get("slot5_replacements", 0) if selected else 0,
        "decision": decision,
        "policy_frozen": True,
        "best_b": best_b,
        "policy_grid": results,
        "provenance": {
            "gpu_output_manifest": str(ANCHOR_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "gpu_output_scores": str(ANCHOR.relative_to(ROOT)).replace("\\", "/"),
            "gpu_output_status": anchor_manifest["status"],
            "new_anchor_rows": len(anchor_rows),
            "new_anchor_unique": len(anchor_keys),
            "worklist_sha256": worklist_sha,
            "worklist_hash_match": True,
            "baseline_existing_current_anchors": len(existing_anchor_keys),
            "rank4_coverage": rank4,
            "rank5_coverage": rank5,
            "current_output_sha256": current_sha,
            "new_output_sha256": anchor_sha,
            "model": EXPECTED_MODEL,
            "base_revision": EXPECTED_BASE_REV,
            "selector": EXPECTED_SELECTOR,
            "aggregation": EXPECTED_AGGREGATION,
            "max_length": 512,
            "fold0_used": False,
            "public_labels_used": False,
            "private_labels_used": False,
            "qwen_score_used_for_final_rank": False,
            "old_unproven_bge_cache_used": False,
        },
        "execution": {"gpu_runs_this_task": 0, "modal_inference_this_task": 0, "model_loads_this_task": 0, "private_worklist_created": False, "submission_created": False},
    }
    DECISION.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def fmt(x: float) -> str:
        return f"{x:.15f}"
    report = "\n".join([
        "# FINAL OFFLINE DECISION — Task1 LegalIR",
        "",
        "CPU-only final paired A/B on the same 5,600 F1–F4 queries. No GPU, Modal inference, Qwen inference, Private labels, Fold0, or old unproven BGE cache was used.",
        "",
        f"- GPU output status: **{anchor_manifest['status']}**; new anchors: **{len(anchor_rows)}/10737**; unique: **{len(anchor_keys)}/10737**.",
        f"- Worklist SHA match: **YES** (`{worklist_sha}`).",
        f"- Existing current-BGE baseline anchors: **{len(existing_anchor_keys)}/463**.",
        f"- Baseline slot4/5 coverage: **11200/11200**; rank4 **{rank4}/5600**; rank5 **{rank5}/5600**.",
        f"- Baseline reproduced: **YES**; Recall `{fmt(baseline_metric['recall'])}`; Precision `{fmt(baseline_metric['precision'])}`.",
        "- Fold baseline recall: " + ", ".join(f"F{f}={fmt(baseline_by_fold[str(f)]['recall'])}" for f in evaluator.TARGET_FOLDS) + ".",
        "",
        "## BEST_B",
        f"- Policy: **{best_b['policy_label']}**; slot4 margin `{best_b['slot4_margin']}`; slot5 margin `{best_b['slot5_margin']}`; rank_safe `{('ON' if best_b['rank_safe'] else 'OFF')}`.",
        f"- Recall `{fmt(best_b['metrics']['recall'])}`; precision `{fmt(best_b['metrics']['precision'])}`; delta recall `{best_b['delta_vs_baseline']['recall']:+.15f}`; delta precision `{best_b['delta_vs_baseline']['precision']:+.15f}`.",
        "- Fold deltas: " + ", ".join(f"F{f}={best_b['delta_by_fold'][str(f)]['recall']:+.15f}" for f in evaluator.TARGET_FOLDS) + ".",
        f"- Paired changes: `{best_b['paired']['changed_queries']}` changed / `{best_b['paired']['unchanged_queries']}` unchanged; A_wrong_B_correct `{best_b['paired']['A_wrong_B_correct']}`; A_correct_B_wrong `{best_b['paired']['A_correct_B_wrong']}`.",
        f"- Relevant gained `{best_b['paired']['relevant_docs_gained']}`; lost `{best_b['paired']['relevant_docs_lost']}`; net `{best_b['paired']['net_relevant_doc_gain']}`.",
        "",
        f"## FINAL_DECISION: **{decision}**",
        f"SELECTED_POLICY: **{selected_name}**; policy frozen: **YES**.",
        "",
        "No additional model-selection or threshold experiment was opened. Private BGE worklist and submission were not created in this task.",
        "",
        f"Decision artifact: `{DECISION.relative_to(ROOT).as_posix()}`",
        f"Report artifact: `{REPORT.relative_to(ROOT).as_posix()}`",
        "",
        "Next action: BUILD_PRIVATE_BGE_WORKLIST_FROM_FROZEN_POLICY",
    ]) + "\n"
    REPORT.write_text(report, encoding="utf-8")

    print("FINAL_OFFLINE_DECISION")
    print(f"gpu_output_manifest:\n{ANCHOR_MANIFEST}")
    print(f"gpu_output_scores:\n{ANCHOR}")
    print(f"gpu_output_status:\n{anchor_manifest['status']}")
    print(f"new_anchor_rows:\n{len(anchor_rows)}/10737")
    print(f"new_anchor_unique:\n{len(anchor_keys)}/10737")
    print(f"worklist_sha256:\n{worklist_sha}")
    print("worklist_hash_match:\nYES")
    print(f"baseline_existing_current_anchors:\n{len(existing_anchor_keys)}/463")
    print("baseline_slot45_current_coverage:\n11200/11200")
    print(f"rank4_coverage:\n{rank4}/5600")
    print(f"rank5_coverage:\n{rank5}/5600")
    print("baseline_reproduced:\nYES")
    print(f"baseline_recall:\n{fmt(baseline_metric['recall'])}")
    print(f"baseline_precision:\n{fmt(baseline_metric['precision'])}")
    print("BEST_B")
    print(f"policy:\n{best_b['policy_label']}")
    print(f"slot4_margin:\n{best_b['slot4_margin']}")
    print(f"slot5_margin:\n{best_b['slot5_margin']}")
    print(f"rank_safe:\n{('ON' if best_b['rank_safe'] else 'OFF')}")
    print(f"recall:\n{fmt(best_b['metrics']['recall'])}")
    print(f"precision:\n{fmt(best_b['metrics']['precision'])}")
    print(f"delta_recall:\n{best_b['delta_vs_baseline']['recall']:+.15f}")
    print(f"delta_precision:\n{best_b['delta_vs_baseline']['precision']:+.15f}")
    print("fold_recalls:\n" + "\n".join(f"F{f}={fmt(best_b['metrics_by_fold'][str(f)]['recall'])}" for f in evaluator.TARGET_FOLDS))
    print("fold_deltas:\n" + "\n".join(f"F{f}={best_b['delta_by_fold'][str(f)]['recall']:+.15f}" for f in evaluator.TARGET_FOLDS))
    for key in ("changed_queries", "A_wrong_B_correct", "A_correct_B_wrong", "relevant_docs_gained", "relevant_docs_lost", "net_relevant_doc_gain"):
        print(f"{key}:\n{best_b['paired'][key]}")
    print(f"FINAL_DECISION:\n{decision}")
    print(f"SELECTED_POLICY:\n{selected_name}")
    print("POLICY_FROZEN:\nYES")
    print(f"decision_artifact:\n{DECISION}")
    print(f"report_artifact:\n{REPORT}")
    print("GPU_RUNS_THIS_TASK:\n0")
    print("MODAL_INFERENCE_THIS_TASK:\n0")
    print("NEXT_ACTION:\nBUILD_PRIVATE_BGE_WORKLIST_FROM_FROZEN_POLICY")


if __name__ == "__main__":
    main()
