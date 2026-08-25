"""No-GPU gate for BM25-grounded V2B Slim worklists.

All B1--B6 worklists are materialised before labels are inspected.  The
post-lock section is diagnostic only: it must never be used to rename an OOF
oracle as an inference result.  This program neither loads a reranker nor
contacts Beam.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from analyze_post_k500_bounded_union_v2 import RECOVERY, doc_features
from analyze_v2b_slim_preflight import K500_COST, build_candidates, checkpoint_chunk_counts, load_rankings


BASELINE = 0.9244452380952382
ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def distribution(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "median": median(values) if values else None,
        "p95": ordered[max(0, math.ceil(len(ordered) * .95) - 1)] if values else None,
    }


def stable_sha(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: (r["query_id"], r["doc_id"])):
        digest.update(f"{row['query_id']}\t{row['doc_id']}\n".encode())
    return digest.hexdigest()


def select_bm25_plans(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """B1--B6 from the prompt, with no gold read or consulted here."""
    bm25_rows = [r for r in rows if r["bm25_rank"] is not None]
    per_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bm25_rows:
        per_query[row["query_id"]].append(row)
    top20 = [row for qid in sorted(per_query) for row in sorted(per_query[qid], key=lambda r: (r["union_rank"], r["doc_id"]))[:20]]
    b1 = [r for r in bm25_rows if r["union_rank"] <= 50]
    b6 = [r for r in bm25_rows if r["union_rank"] <= 50 or r["knn_support"] > 0]
    return {
        "B1_union_rank_le_50_and_bm25": b1,
        "B2_top20_missing_per_query_and_bm25": top20,
        "B3_union_rank_le_100_and_bm25": [r for r in bm25_rows if r["union_rank"] <= 100],
        "B4_bm25_rank_le_20": [r for r in bm25_rows if int(r["bm25_rank"]) <= 20],
        "B5_bm25_rank_le_50": [r for r in bm25_rows if int(r["bm25_rank"]) <= 50],
        "B6_B1_or_bm25_knn_multi_source": b6,
    }


def prelock_report(rows: list[dict[str, Any]], counts: dict[tuple[str, str], int], num_queries: int) -> dict[str, Any]:
    per_query = Counter(row["query_id"] for row in rows)
    chunks = {str(cap): sum(min(cap, counts[(r["query_id"], r["doc_id"])]) for r in rows) for cap in (1, 2, 3)}
    return {
        "policy_locked_without_gold": True,
        "candidate_doc_occurrences": len(rows),
        "unique_docs": len({r["doc_id"] for r in rows}),
        "docs_per_query": distribution(list(per_query.values()) + [0] * (num_queries - len(per_query))),
        "chunks": chunks,
        "relative_cost_vs_k500_v1": {cap: chunks[cap] / K500_COST for cap in chunks},
        "source_overlap": {
            "source_support": dict(sorted(Counter(str(r["source_support"]) for r in rows).items())),
            "knn_support": dict(sorted(Counter(str(r["knn_support"]) for r in rows).items())),
            "has_bm25_and_knn": sum(r["knn_support"] > 0 for r in rows),
        },
        "worklist_sha256_query_doc": stable_sha(rows),
    }


def postlock_diagnostic(rows: list[dict[str, Any]], all_rows: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], existing_all: dict[str, set[str]], k200_docs: dict[str, set[str]]) -> dict[str, Any]:
    selected = defaultdict(set)
    for row in rows:
        selected[row["query_id"]].add(row["doc_id"])
    total = covered = 0
    k200_miss_total = k200_miss_covered = 0
    multi_total = multi_covered = 0
    # ``union_missing`` is the locked union@200 candidate pool.  The old
    # report's 205 denominator omitted this condition, so it was a wider
    # coverage cohort than the historic 120 rescue opportunity cohort.
    union_missing = {(r["query_id"], r["doc_id"]): r for r in all_rows}
    broad_missing: list[tuple[str, str]] = []
    union_rescue: list[tuple[str, str]] = []
    broad_in_baseline_top5: list[tuple[str, str]] = []
    rescue_covered = 0
    by_fold: dict[str, dict[str, int]] = {str(f): {"gold_total": 0, "gold_covered": 0, "rescued_missing_bge_total": 0, "rescued_missing_bge_captured": 0} for f in range(5)}
    for qid, base in baseline.items():
        fold = str(base["fold"])
        gold = set(map(str, base["gold_documents"]))
        total += len(gold)
        covered_here = gold & (existing_all[qid] | selected[qid])
        covered += len(covered_here)
        by_fold[fold]["gold_total"] += len(gold)
        by_fold[fold]["gold_covered"] += len(covered_here)
        for doc in gold:
            if doc not in k200_docs[qid]:
                k200_miss_total += 1
                k200_miss_covered += int(doc in existing_all[qid] or doc in selected[qid])
            if len(gold) > 1:
                multi_total += 1
                multi_covered += int(doc in existing_all[qid] or doc in selected[qid])
            if doc not in existing_all[qid]:
                key = (qid, doc)
                if doc in set(map(str, base["top5"])):
                    broad_in_baseline_top5.append(key)
                else:
                    broad_missing.append(key)
                    if key in union_missing:
                        union_rescue.append(key)
                        rescued = int(doc in selected[qid])
                        rescue_covered += rescued
                        by_fold[fold]["rescued_missing_bge_total"] += 1
                        by_fold[fold]["rescued_missing_bge_captured"] += rescued
    union_bm25 = [key for key in union_rescue if union_missing[key]["bm25_rank"] is not None]
    union_not_bm25 = [key for key in union_rescue if union_missing[key]["bm25_rank"] is None]
    broad_not_union = [key for key in broad_missing if key not in union_missing]
    return {
        "candidate_oracle_ceiling_combine_existing_k200_k500_scores": covered / total,
        "candidate_oracle_is_diagnostic_not_final_recall": True,
        "primary_rescue_diagnostic_locked_union_outside_baseline_top5": {"covered": rescue_covered, "total": len(union_rescue), "rate": rescue_covered / len(union_rescue) if union_rescue else None},
        "secondary_broad_missing_bge_coverage_outside_baseline_top5": {"covered": rescue_covered, "total": len(broad_missing), "rate": rescue_covered / len(broad_missing) if broad_missing else None},
        "cohort_reconciliation": {
            "old_205_definition": "gold absent from reusable K200/K500 BGE and absent from baseline top5; no locked-union condition",
            "gold_missing_bge_already_in_baseline_top5": len(broad_in_baseline_top5),
            "gold_missing_bge_outside_baseline_top5_broad": len(broad_missing),
            "of_broad_in_locked_union_200": len(union_rescue),
            "of_broad_not_in_locked_union_200": len(broad_not_union),
            "of_locked_union_bm25_present": len(union_bm25),
            "of_locked_union_bm25_absent": len(union_not_bm25),
            "missing_k200_bge_and_missing_k500_v1_bge": len(broad_missing),
            "samples_broad_not_in_union": [{"query_id": q, "doc_id": d} for q, d in broad_not_union[:10]],
            "samples_primary_union_cohort": [{"query_id": q, "doc_id": d} for q, d in union_rescue[:10]],
        },
        "k200_miss_gold_covered": {"covered": k200_miss_covered, "total": k200_miss_total, "rate": k200_miss_covered / k200_miss_total if k200_miss_total else None},
        "multi_gold_covered": {"covered": multi_covered, "total": multi_total, "rate": multi_covered / multi_total if multi_total else None},
        "per_fold_diagnostic_coverage": by_fold,
    }


def headroom(oracle: float) -> dict[str, Any]:
    available = oracle - BASELINE
    labels = []
    if oracle < .98:
        labels.append("ceiling_too_tight_for_0.98")
    else:
        labels.append("enough_ceiling_for_0.98")
    if oracle >= .96:
        labels.append("plausible_for_0.96")
    else:
        labels.append("diagnostic_only")
    targets = {}
    for target in (.94, .95, .96, .98):
        targets[f"{target:.2f}"] = {
            "candidate_ceiling_reaches_target": oracle >= target,
            "fraction_of_available_gain_required": (target - BASELINE) / available if available > 0 else None,
        }
    return {"baseline": BASELINE, "candidate_oracle_ceiling": oracle, "oracle_minus_baseline_headroom": available, "targets": targets, "labels": labels}


def audit_extract(report: dict[str, Any]) -> dict[str, Any]:
    audit = report["selector_audit"]
    result: dict[str, Any] = {}
    for cache in ("k200", "k500_v1"):
        bucket = audit[cache]["bm25_heavy"]
        s0, s2 = bucket["S0_token_overlap_topk_v1"], bucket["S2_bm25_within_document"]
        metrics = ("exact_best_bge_chunk_at1", "any_previously_scored_at1", "any_previously_scored_at3", "best_bge_chunk_in_selected_at3", "mean_overlap_count", "mean_jaccard")
        result[cache] = {"sample_size": s0["sample_count"], "S0": s0, "S2": s2, "S2_minus_S0": {metric: s2[metric] - s0[metric] for metric in metrics}}
    result["other_buckets_available"] = {cache: sorted(name for name in audit[cache] if name != "bm25_heavy") for cache in audit}
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RECOVERY / "bm25_grounded_selector_gate_v1")
    parser.add_argument("--evidence-report", type=Path, default=RECOVERY / "evidence_selector_recovery_v1/report.json")
    args = parser.parse_args()

    evidence_report = json.loads(args.evidence_report.read_text(encoding="utf-8"))
    audit = audit_extract(evidence_report)
    print("[START] loading locked union and reusable score caches", flush=True)
    baseline, decisions, k200, k500, lexical = load_rankings()
    print("[START] reconstructing missing-candidate pool from locked union", flush=True)
    all_rows, existing_all = build_candidates(baseline, decisions, k200, k500, lexical)
    print(f"[START] candidate pool ready: {len(all_rows):,} missing document occurrences", flush=True)
    counts = checkpoint_chunk_counts()
    if any((r["query_id"], r["doc_id"]) not in counts for r in all_rows):
        raise ValueError("missing chunk-count checkpoint for candidate pool")
    print("[START] locking B1--B6 without gold", flush=True)
    plans = select_bm25_plans(all_rows)
    locked = {name: prelock_report(rows, counts, len(baseline)) for name, rows in plans.items()}
    for name, rows in plans.items():
        atomic_json(args.output_dir / "worklists" / f"{name}.json", {"schema_version": "bm25-grounded-worklist-v1", "policy_locked_without_gold": True, "rows": sorted(rows, key=lambda r: (r["query_id"], r["union_rank"], r["doc_id"]))})
    atomic_json(args.output_dir / "report.partial.json", {
        "schema_version": "bm25-grounded-selector-gate-v1",
        "status": "POST_LOCK_DIAGNOSTIC_PENDING_NO_GPU",
        "constraints": {"no_gpu": True, "no_neural_training": True, "no_public_submission": True, "worklists_locked_before_gold": True},
        "bm25_grounded_s2_audit": audit,
        "plans": locked,
        "message": "B1--B6 worklists have been atomically locked; rerun is safe if post-lock diagnostics are interrupted.",
    })
    # Labels are first accessed below; the worklists above are already frozen.
    print("[START] post-lock diagnostic coverage and headroom", flush=True)
    k200_docs = {qid: set(doc_features(record["hits"])) for qid, record in k200.items()}
    for name, rows in plans.items():
        diagnostic = postlock_diagnostic(rows, all_rows, baseline, existing_all, k200_docs)
        locked[name]["post_lock_diagnostic"] = diagnostic
        locked[name]["headroom"] = headroom(diagnostic["candidate_oracle_ceiling_combine_existing_k200_k500_scores"])
        print(f"[PLAN] {name}: chunks@3={locked[name]['chunks']['3']} oracle={diagnostic['candidate_oracle_ceiling_combine_existing_k200_k500_scores']:.6f}", flush=True)

    k200_gain = audit["k200"]["S2_minus_S0"]
    k500_gain = audit["k500_v1"]["S2_minus_S0"]
    s2_improved = all(value > 0 for value in (k200_gain["exact_best_bge_chunk_at1"], k200_gain["any_previously_scored_at3"], k500_gain["exact_best_bge_chunk_at1"], k500_gain["any_previously_scored_at3"]))
    viable = [name for name, value in locked.items() if value["relative_cost_vs_k500_v1"]["3"] <= 1.0 and value["post_lock_diagnostic"]["candidate_oracle_ceiling_combine_existing_k200_k500_scores"] > BASELINE]
    # Fixed, non-gold choice for a diagnostic is the least-cost plan with at
    # least .96 candidate ceiling; it is not a tuned OOF selection.
    eligible = [name for name in viable if locked[name]["post_lock_diagnostic"]["candidate_oracle_ceiling_combine_existing_k200_k500_scores"] >= .96]
    chosen = min(eligible, key=lambda n: (locked[n]["chunks"]["3"], n)) if eligible else None
    if s2_improved and chosen:
        status = "BM25_SELECTOR_RECOVERED_GPU_DIAGNOSTIC_WORTH_IT"
    elif s2_improved:
        status = "BM25_SELECTOR_IMPROVED_BUT_CEILING_TOO_LOW"
    elif viable:
        status = "BM25_SELECTOR_STILL_MISMATCHED"
    else:
        status = "BM25_GROUNDED_WORKLIST_NOT_WORTH_GPU"
    report = {
        "schema_version": "bm25-grounded-selector-gate-v2-reconciled-rescue-cohort",
        "constraints": {"no_gpu": True, "no_neural_training": True, "no_public_submission": True, "worklists_locked_before_gold": True},
        "evidence_provenance": evidence_report["exact_evidence_provenance"],
        "bm25_grounded_s2_audit": audit,
        "plans": locked,
        "selection_for_possible_future_gpu_diagnostic": {"selection_is_inference_only": True, "rule": "least chunks@3 among B1--B6 with cost<=1.0x and candidate ceiling>=.96; no gold tuning", "chosen_plan": chosen},
        "status": status,
        "decision": {"s2_improved_on_bm25_heavy_known_score_pairs": s2_improved, "k200_relative_improvement_any_overlap_at3": k200_gain["any_previously_scored_at3"] / audit["k200"]["S0"]["any_previously_scored_at3"], "k500_relative_improvement_any_overlap_at3": k500_gain["any_previously_scored_at3"] / audit["k500_v1"]["S0"]["any_previously_scored_at3"], "no_gpu_run_in_this_analysis": True},
    }
    atomic_json(args.output_dir / "report.json", report)
    markdown = ["# BM25-grounded selector gate v1", "", f"Status: `{status}`", "", "Worklists B1--B6 were written before gold diagnostics. Candidate oracle is diagnostic only.", "", "| Plan | docs | chunks@3 | cost | oracle diagnostic | rescue capture |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in locked.items():
        diag = value["post_lock_diagnostic"]
        rescue = diag["primary_rescue_diagnostic_locked_union_outside_baseline_top5"]
        markdown.append(f"| {name} | {value['candidate_doc_occurrences']:,} | {value['chunks']['3']:,} | {value['relative_cost_vs_k500_v1']['3']:.3f}x | {diag['candidate_oracle_ceiling_combine_existing_k200_k500_scores']:.6f} | {rescue['covered']}/{rescue['total']} |")
    markdown += ["", "## BM25-heavy audit: S2 minus S0", "", "| Cache | best@1 | any scored@1 | any scored@3 | best in selected@3 |", "|---|---:|---:|---:|---:|"]
    for cache in ("k200", "k500_v1"):
        delta = audit[cache]["S2_minus_S0"]
        markdown.append(f"| {cache} | {delta['exact_best_bge_chunk_at1']:+.3f} | {delta['any_previously_scored_at1']:+.3f} | {delta['any_previously_scored_at3']:+.3f} | {delta['best_bge_chunk_in_selected_at3']:+.3f} |")
    (args.output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "chosen_plan": chosen, "output": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
