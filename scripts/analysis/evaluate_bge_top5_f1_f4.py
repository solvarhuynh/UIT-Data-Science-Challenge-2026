"""CPU-only validation of the downloaded BGE TOP5 production artifact.

This evaluator is deliberately conservative about provenance.  It uses only
the newly downloaded BGE scores for candidate and incumbent q-docs that were
actually present in the production worklist.  The older persisted BGE cache is
not read as a score source because its model revision is unproven.

The first three baseline positions are immutable.  A candidate can replace a
baseline slot only when the dropped slot also has a current production score;
otherwise that query remains the incumbent.  This avoids inventing a score
for an unverified historical artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/task1/qwen_to_bge_minimal"
RAW = ART / "bge_top5_production_complete.jsonl"
MANIFEST = ART / "bge_top5_production_manifest.json"
WORKLIST = ART / "top5_missing_bge.jsonl"
BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
UNION = ROOT / "artifacts/task1/handoff/step4_bge_handoff/candidates/f1_f4_candidate_union.jsonl"
FOLDS = ROOT / "artifacts/task1/handoff/step4_bge_handoff/candidates/folds.json"
RUNNER = ROOT / "scripts/modal/task1_bge_subset_gpu.py"

CANONICAL = ART / "bge_top5_production_canonical.jsonl"
RESULTS = ART / "bge_top5_f1_f4_policy_results.json"
PREDICTIONS = ART / "bge_top5_f1_f4_predictions.jsonl"
SELECTED = ART / "bge_top5_selected_policy.json"
REPORT = ROOT / "reports/task1/bge_top5_f1_f4_validation.md"

EXPECTED_BASE_REV = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
EXPECTED_MODEL = "BAAI/bge-reranker-v2-m3"
EXPECTED_SELECTOR = "true_s2_bm25_within_document_v2"
EXPECTED_AGGREGATION = "MAX"
TARGET_FOLDS = (1, 2, 3, 4)

POLICIES = {
    "FT_ONLY": (1.0, 0.0, 0.0, 0.0),
    "BASE_ONLY": (0.0, 1.0, 0.0, 0.0),
    "FT_BASE_EQUAL": (0.50, 0.50, 0.0, 0.0),
    "HISTORICAL_STEP4": (0.40, 0.30, 0.20, 0.10),
    "FT_HEAVY": (0.50, 0.25, 0.15, 0.10),
    "BALANCED": (0.35, 0.35, 0.20, 0.10),
}
SLOT_MARGINS = {"slot4": (0.00, 0.05, 0.10, 0.15), "slot5": (0.00, 0.03, 0.05, 0.08, 0.10)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                yield line_no, json.loads(line)


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_folds() -> dict[str, int]:
    raw = json.loads(FOLDS.read_text(encoding="utf-8"))
    return {str(q): int(entry["fold"]) for entry in raw["folds"] for q in entry["validation_ids"]}


def norm(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    span = max(1e-12, hi - lo)
    return {key: (value - lo) / span for key, value in values.items()}


def metric(predictions: dict[str, list[str]], baseline: dict[str, dict]) -> dict:
    recalls, precisions = [], []
    for qid, top5 in predictions.items():
        gold = {str(x) for x in baseline[qid]["gold_documents"]}
        hits = len(gold & set(top5))
        recalls.append(hits / len(gold) if gold else 1.0)
        precisions.append(hits / 5.0)
    return {"recall": mean(recalls), "precision": mean(precisions), "query_count": len(predictions)}


def validate_download(manifest: dict, rows: list[dict], worklist: list[dict]) -> dict:
    raw_sha = sha256(RAW)
    if raw_sha != manifest["output_sha256"]:
        raise RuntimeError(f"raw output SHA mismatch: {raw_sha} != {manifest['output_sha256']}")
    if len(rows) != int(manifest["q_doc_count"]):
        raise RuntimeError("raw q-doc row count does not match manifest")
    work_sha = sha256(WORKLIST)
    if work_sha != manifest["worklist_sha256"]:
        raise RuntimeError(f"worklist SHA mismatch: {work_sha} != {manifest['worklist_sha256']}")

    row_keys = [(str(r["query_id"]), str(r["document_id"])) for r in rows]
    if len(set(row_keys)) != len(row_keys):
        raise RuntimeError("duplicate q-doc identities in production output")
    work_keys = [(str(r["query_id"]), str(r["document_id"])) for r in worklist]
    if len(set(work_keys)) != len(work_keys):
        raise RuntimeError("duplicate q-doc identities in worklist")
    if set(row_keys) != set(work_keys):
        raise RuntimeError("production output identities do not exactly cover worklist")

    bad = []
    for r in rows:
        if r.get("model_id") != EXPECTED_MODEL:
            bad.append("model_id")
        if r.get("base_model_revision") != EXPECTED_BASE_REV:
            bad.append("base_model_revision")
        if r.get("selector") != EXPECTED_SELECTOR:
            bad.append("selector")
        if r.get("aggregation") != EXPECTED_AGGREGATION:
            bad.append("aggregation")
        if not finite(r.get("bge_ft_score")) or not finite(r.get("bge_base_score")):
            bad.append("nonfinite_document_score")
        ft_chunks, base_chunks, selected = r.get("ft_chunk_scores"), r.get("base_chunk_scores"), r.get("selected_chunk_ids")
        if not isinstance(selected, list) or not selected or len(set(map(str, selected))) != len(selected):
            bad.append("selected_chunk_ids")
        if not isinstance(ft_chunks, list) or not isinstance(base_chunks, list) or len(ft_chunks) != len(selected) or len(base_chunks) != len(selected):
            bad.append("chunk_score_length")
        if any(not finite(x) for x in (ft_chunks or []) + (base_chunks or [])):
            bad.append("nonfinite_chunk_score")
    if bad:
        raise RuntimeError("invalid production row fields: " + ", ".join(sorted(set(bad))))
    runner_text = RUNNER.read_text(encoding="utf-8")
    if "MAX_LENGTH = 512" not in runner_text or "max_length=MAX_LENGTH" not in runner_text:
        raise RuntimeError("runner does not prove MAX_LENGTH=512")

    if CANONICAL.exists() and sha256(CANONICAL) != raw_sha:
        raise RuntimeError("existing canonical copy differs from verified raw download")
    if not CANONICAL.exists():
        shutil.copy2(RAW, CANONICAL)
    return {
        "raw_sha256": raw_sha,
        "worklist_sha256": work_sha,
        "qdoc_rows": len(rows),
        "unique_qdocs": len(row_keys),
        "unique_queries": len({q for q, _ in row_keys}),
        "max_length": 512,
        "model_id": EXPECTED_MODEL,
        "base_revision": EXPECTED_BASE_REV,
        "selector": EXPECTED_SELECTOR,
        "aggregation": EXPECTED_AGGREGATION,
        "canonical_path": str(CANONICAL.relative_to(ROOT)).replace("\\", "/"),
        "canonical_sha256": sha256(CANONICAL),
    }


def build_meta(union_rows: list[dict], worklist: list[dict]) -> dict[tuple[str, str], dict]:
    meta = {}
    for row in union_rows:
        qid = str(row.get("query_id") or row.get("question_id"))
        for hit in row.get("candidates", row.get("hits", [])):
            did = str(hit.get("doc_id") or hit.get("document_id"))
            meta[(qid, did)] = {
                "union_rank": int(hit.get("union_rank", 999)),
                "rrf_score": float(hit.get("rrf_score", 0.0) or 0.0),
                "source_support": int(hit.get("source_support", 1) or 1),
            }
    # Worklist metadata is allowed only as retrieval provenance.  Teacher
    # qwen_score/rank fields are intentionally ignored.
    for row in worklist:
        key = (str(row["query_id"]), str(row["document_id"]))
        if key not in meta:
            meta[key] = {
                "union_rank": 999,
                "rrf_score": 0.0,
                "source_support": 1,
            }
    return meta


def score_query(rows: list[dict], metadata: dict[tuple[str, str], dict], policy: str) -> dict[str, float]:
    ft = {str(r["document_id"]): float(r["bge_ft_score"]) for r in rows}
    base = {str(r["document_id"]): float(r["bge_base_score"]) for r in rows}
    rrf = {str(r["document_id"]): float(metadata[(str(r["query_id"]), str(r["document_id"]))]["rrf_score"]) for r in rows}
    support = {str(r["document_id"]): float(metadata[(str(r["query_id"]), str(r["document_id"]))]["source_support"]) / 4.0 for r in rows}
    nft, nbase, nrrf = norm(ft), norm(base), norm(rrf)
    w = POLICIES[policy]
    return {did: w[0] * nft[did] + w[1] * nbase[did] + w[2] * nrrf[did] + w[3] * support[did] for did in ft}


def evaluate_policy(policy: str, slot4_margin: float, slot5_margin: float, rank_safe: bool,
                    baseline: dict[str, dict], scored: dict[str, list[dict]], metadata: dict[tuple[str, str], dict], folds: dict[str, int]) -> tuple[dict, dict[str, list[str]]]:
    predictions = {}
    changes = Counter()
    slot_changes = Counter()
    known_anchor_queries = 0
    candidate_queries = 0

    for qid, row in baseline.items():
        base_top = [str(x) for x in row["top5"]]
        top = list(base_top)
        current = scored.get(qid, [])
        score_map = score_query(current, metadata, policy) if current else {}
        # Only outside candidates are eligible; baseline slots are retained as
        # anchors and are never re-selected as additions.
        additions = [did for did in score_map if did not in base_top[:3] and did not in base_top[3:]]
        additions.sort(key=lambda d: (-score_map[d], d))
        anchor_scored = set(score_map) & set(base_top[3:])
        if anchor_scored:
            known_anchor_queries += 1
        if additions:
            candidate_queries += 1

        for slot_index, margin in ((4, slot4_margin), (5, slot5_margin)):
            if slot_index == 4 and len(top) < 4 or slot_index == 5 and len(top) < 5:
                continue
            dropped = top[slot_index - 1]
            if dropped not in score_map:
                # No current score means no scientifically valid comparison.
                continue
            chosen = None
            for did in additions:
                if did in top[:3] or did in top[3:]:
                    continue
                candidate_score = score_map[did]
                delta = candidate_score - score_map[dropped]
                if delta < margin:
                    continue
                dm = metadata[(qid, dropped)]
                am = metadata[(qid, did)]
                if rank_safe and am["union_rank"] > dm["union_rank"] and am["source_support"] <= dm["source_support"]:
                    continue
                chosen = did
                break
            if chosen is not None:
                top[slot_index - 1] = chosen
                additions.remove(chosen)
                slot_changes[f"slot{slot_index}_replacements"] += 1
                slot_changes[f"slot{slot_index}_adds"] += 1
                slot_changes[f"slot{slot_index}_drops"] += 1

        predictions[qid] = top
        if top != base_top:
            changes["changed_queries"] += 1
            changes["added_docs"] += len(set(top) - set(base_top))
            changes["dropped_docs"] += len(set(base_top) - set(top))
    result = metric(predictions, baseline)
    by_fold = {}
    for fold in TARGET_FOLDS:
        ids = [q for q in predictions if folds[q] == fold]
        by_fold[str(fold)] = metric({q: predictions[q] for q in ids}, baseline)
    base_metric = metric({q: [str(x) for x in baseline[q]["top5"]] for q in predictions}, baseline)
    base_fold = {str(f): metric({q: [str(x) for x in baseline[q]["top5"]] for q in predictions if folds[q] == f}, baseline) for f in TARGET_FOLDS}
    delta = {"recall": result["recall"] - base_metric["recall"], "precision": result["precision"] - base_metric["precision"]}
    fold_delta = {f: {"recall": by_fold[f]["recall"] - base_fold[f]["recall"], "precision": by_fold[f]["precision"] - base_fold[f]["precision"]} for f in by_fold}
    relevant_delta = 0
    for qid, top in predictions.items():
        gold = {str(x) for x in baseline[qid]["gold_documents"]}
        relevant_delta += len(gold & set(top)) - len(gold & set(baseline[qid]["top5"]))
    changes.update({"improved_queries": 0, "harmed_queries": 0, "neutral_changed_queries": 0, "net_relevant_document_change": relevant_delta})
    for qid, top in predictions.items():
        if top == baseline[qid]["top5"]:
            continue
        gold = {str(x) for x in baseline[qid]["gold_documents"]}
        d = len(gold & set(top)) - len(gold & set(baseline[qid]["top5"]))
        changes["improved_queries" if d > 0 else "harmed_queries" if d < 0 else "neutral_changed_queries"] += 1
    gate = {
        "recall_positive": delta["recall"] > 0,
        "every_fold_recall_nonnegative": all(v["recall"] >= 0 for v in fold_delta.values()),
        "precision_guard": delta["precision"] >= -0.001,
        "complete_f1_f4": len(predictions) == 5600,
        "baseline_top1_top3_locked": True,
        "no_old_bge_score_used": True,
    }
    payload = {
        "policy": policy,
        "slot4_margin": slot4_margin,
        "slot5_margin": slot5_margin,
        "rank_safe": rank_safe,
        "metrics": result,
        "metrics_by_fold": by_fold,
        "baseline_metrics": base_metric,
        "baseline_metrics_by_fold": base_fold,
        "delta_vs_baseline": delta,
        "delta_by_fold": fold_delta,
        "changes": dict(changes),
        "slot_changes": dict(slot_changes),
        "known_current_bge_anchor_queries": known_anchor_queries,
        "queries_with_new_candidates": candidate_queries,
        "scientific_gate": {**gate, "overall": all(gate.values())},
    }
    return payload, predictions


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    worklist = [row for _, row in jsonl(WORKLIST)]
    prod = [row for _, row in jsonl(RAW)]
    download = validate_download(manifest, prod, worklist)
    baseline_rows = {str(row["query_id"]): row for _, row in jsonl(BASELINE)}
    folds = load_folds()
    if set(baseline_rows) != set(folds):
        raise RuntimeError("baseline/fold query universe mismatch")
    scientific = {q: row for q, row in baseline_rows.items() if folds[q] in TARGET_FOLDS}
    if len(scientific) != 5600:
        raise RuntimeError("F1-F4 population is not exactly 5600 queries")

    scored_all = defaultdict(list)
    for row in prod:
        qid = str(row["query_id"])
        if qid in scientific:
            scored_all[qid].append(row)
    metadata = build_meta([row for _, row in jsonl(UNION)], worklist)
    scored = {q: rows for q, rows in scored_all.items() if q in scientific}

    baseline_metric = metric({q: [str(x) for x in row["top5"]] for q, row in scientific.items()}, scientific)
    baseline_by_fold = {str(f): metric({q: [str(x) for x in scientific[q]["top5"]] for q in scientific if folds[q] == f}, scientific) for f in TARGET_FOLDS}
    all_results = [{"policy": "BASELINE", "metrics": baseline_metric, "metrics_by_fold": baseline_by_fold, "delta_vs_baseline": {"recall": 0.0, "precision": 0.0}, "scientific_gate": {"overall": False}}]
    prediction_map = {}
    for policy in POLICIES:
        for slot4 in SLOT_MARGINS["slot4"]:
            for slot5 in SLOT_MARGINS["slot5"]:
                for rank_safe in (False, True):
                    result, predictions = evaluate_policy(policy, slot4, slot5, rank_safe, scientific, scored, metadata, folds)
                    all_results.append(result)
                    prediction_map[(policy, slot4, slot5, rank_safe)] = predictions

    admissible = [r for r in all_results if r.get("scientific_gate", {}).get("overall")]
    # Selection is deterministic and prefers recall, then precision, then the
    # stricter margins/rank-safety.  It is still not a production authorization.
    winner = max(admissible, key=lambda r: (r["delta_vs_baseline"]["recall"], r["delta_vs_baseline"]["precision"], r["slot4_margin"], r["slot5_margin"], r["rank_safe"])) if admissible else None
    selected_name = "RETURN_TO_09411" if winner is None else winner["policy"]
    if winner is not None:
        selected_predictions = prediction_map[(winner["policy"], winner["slot4_margin"], winner["slot5_margin"], winner["rank_safe"])]
    else:
        selected_predictions = {q: [str(x) for x in row["top5"]] for q, row in scientific.items()}

    with PREDICTIONS.open("w", encoding="utf-8", newline="\n") as f:
        for qid in sorted(selected_predictions, key=lambda q: (folds[q], q)):
            f.write(json.dumps({"query_id": qid, "fold": folds[qid], "top5": selected_predictions[qid]}, ensure_ascii=False, sort_keys=True) + "\n")

    output = {
        "status": "BGE_TOP5_F1_F4_POLICY_EVALUATION_COMPLETE",
        "population": {"folds": list(TARGET_FOLDS), "query_count": len(scientific), "fold0_used": False, "public_labels_used": False},
        "provenance": {
            "download": download,
            "raw_manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
            "old_bge_inventory_used": False,
            "old_bge_inventory_reason": "MODEL_REVISION_UNPROVEN_FOR_PERSISTED_SCORE_ARTIFACT",
            "new_score_rows_used": len(prod),
            "new_scored_f1_f4_queries": len(scored),
            "new_scored_f1_f4_qdocs": sum(len(v) for v in scored.values()),
            "current_bge_anchor_qdocs": sum(1 for q, rows in scored.items() for r in rows if str(r["document_id"]) in set(scientific[q]["top5"][3:])),
        },
        "baseline": {"metrics": baseline_metric, "metrics_by_fold": baseline_by_fold},
        "policy_grid": all_results,
        "selection": {
            "selected_policy": selected_name,
            "winner": winner,
            "production_authorized": False,
            "reason": "No policy passed conservative F1-F4 gate" if winner is None else "Validation winner recorded; production remains separately unauthorized",
        },
        "artifacts": {
            "raw_output": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "canonical_output": str(CANONICAL.relative_to(ROOT)).replace("\\", "/"),
            "predictions": str(PREDICTIONS.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    RESULTS.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if winner is not None:
        SELECTED.write_text(json.dumps({"status": "VALIDATED_CANDIDATE_NOT_PRODUCTION_AUTHORIZATION", "selected": winner, "results": str(RESULTS.relative_to(ROOT)).replace("\\", "/")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif SELECTED.exists():
        SELECTED.unlink()

    best_text = "RETURN_TO_09411" if winner is None else f"{winner['policy']} ({winner['delta_vs_baseline']['recall']:+.9f} recall)"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# BGE TOP5 F1–F4 validation\n\n"
        f"Status: **{output['status']}**\n\n"
        "This is CPU-only replay over the downloaded BGE artifact. No Modal, GPU, Qwen inference, public labels, or Fold0 was used.\n\n"
        "## Frozen provenance\n\n"
        f"- Raw output: `{RAW.relative_to(ROOT).as_posix()}`\n- Raw SHA256: `{download['raw_sha256']}`\n"
        f"- Canonical frozen copy: `{CANONICAL.relative_to(ROOT).as_posix()}`\n- Canonical SHA256: `{download['canonical_sha256']}`\n"
        f"- Rows/q-docs: **{len(prod)}**; F1–F4 queries: **{len(scientific)}**\n"
        f"- Model: `{EXPECTED_MODEL}`; base revision: `{EXPECTED_BASE_REV}`; selector: `{EXPECTED_SELECTOR}`; aggregation: `{EXPECTED_AGGREGATION}`; max length: `512`\n"
        "- Old BGE inventory: **not used** because its persisted score provenance is marked `MODEL_REVISION_UNPROVEN_FOR_PERSISTED_SCORE_ARTIFACT`.\n\n"
        "## Baseline anchor\n\n"
        f"- Recall: `{baseline_metric['recall']:.15f}`; precision: `{baseline_metric['precision']:.15f}`\n"
        + "- Fold recall: " + ", ".join(f"F{f}={baseline_by_fold[str(f)]['recall']:.15f}" for f in TARGET_FOLDS) + "\n\n"
        "## Evaluation rule\n\n"
        "Positions 1–3 are locked. A new BGE-scored q-doc can replace slot 4/5 only when the dropped slot also has a current production score; queries without that current anchor stay unchanged. Qwen score/rank is never read. Historical Step4 weights are replayed with per-query min–max normalization of current FT/Base/RRF features.\n\n"
        "## Selection\n\n"
        f"- Candidate policy: **{best_text}**\n- Production authorized: **NO**\n- Policy result JSON: `{RESULTS.relative_to(ROOT).as_posix()}`\n- Predictions: `{PREDICTIONS.relative_to(ROOT).as_posix()}`\n\n"
        "A policy would need positive pooled recall delta, non-negative recall delta on every F1–F4 fold, and precision delta no worse than −0.001. The incumbent 0.9411 remains the safe submission anchor unless a separately authorized handoff says otherwise.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": output["status"], "baseline": baseline_metric, "selected_policy": selected_name, "winner": winner, "raw_sha256": download["raw_sha256"], "old_bge_used": False}, indent=2))


if __name__ == "__main__":
    main()
