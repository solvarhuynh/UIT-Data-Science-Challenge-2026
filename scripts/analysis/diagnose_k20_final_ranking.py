"""CPU-only diagnosis of final ranking on the frozen private K20 universe."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
WORKLIST = ROOT / "private_task1/experiments/private_safe_baseline/private_safe_validation_k20.jsonl"
SCORES = ROOT / "private_task1/experiments/private_safe_baseline/validation_download/results/production/scores.jsonl"
MANIFEST = ROOT / "private_task1/experiments/private_safe_baseline/validation_download/manifests/production.json"
UNION = ROOT / "artifacts/task1/handoff/step4_bge_handoff/candidates/f1_f4_candidate_union.jsonl"
FOLDS = ROOT / "artifacts/task1/handoff/step4_bge_handoff/candidates/folds.json"
GOLD = ROOT / "data/raw/btc/LegalIR/train.json"
OUT = ROOT / "reports/task1/k20_final_ranking_diagnosis.json"

TARGET_FOLDS = (1, 2, 3, 4)
EXPECTED_QDOCS = 112_000
EXPECTED_QUERIES = 5_600
EXPECTED_K = 20
RRF_WEIGHTS = {"dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3}
RRF_K = 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_folds() -> dict[str, int]:
    raw = json.loads(FOLDS.read_text(encoding="utf-8"))
    return {str(qid): int(fold["fold"]) for fold in raw["folds"] for qid in fold["validation_ids"]}


def macro_metrics(
    predictions: dict[str, list[str]], gold: dict[str, dict], *, limit: int = 5
) -> dict[str, float | int]:
    recalls = []
    precisions = []
    for qid, predicted in predictions.items():
        wanted = {str(x) for x in gold[qid].get("answer", [])}
        hit = len(wanted & set(predicted[:limit]))
        recalls.append(hit / len(wanted))
        precisions.append(hit / float(limit))
    return {"queries": len(predictions), "recall": mean(recalls), "precision": mean(precisions)}


def rank_ft(rows: list[dict]) -> list[str]:
    return [
        str(row["document_id"])
        for row in sorted(
            rows,
            key=lambda row: (
                -float(row["bge_ft_score"]),
                -float(row["bge_base_score"]),
                int(row["candidate_rank"]),
                str(row["document_id"]),
            ),
        )
    ]


def rank_bge(rows: list[dict]) -> list[str]:
    return rank_ft(rows)


def weighted_rrf(rows: list[dict], source_ranks: dict[tuple[str, str], dict[str, int]]) -> list[str]:
    qid = str(rows[0]["query_id"])
    score: dict[str, float] = defaultdict(float)
    ranks_by_doc: dict[str, dict[str, int]] = {}
    for row in rows:
        did = str(row["document_id"])
        ranks = dict(source_ranks[(qid, did)])
        ranks["bge"] = rank_bge(rows).index(did) + 1
        ranks_by_doc[did] = ranks
        for source, weight in RRF_WEIGHTS.items():
            rank = ranks.get(source)
            if rank is not None:
                score[did] += weight / (RRF_K + rank)
    return sorted(
        score,
        key=lambda did: (
            -score[did],
            *(ranks_by_doc[did].get(source, 10**9) for source in RRF_WEIGHTS),
            did,
        ),
    )


def minmax(values: dict[str, float]) -> dict[str, float]:
    lo, hi = min(values.values()), max(values.values())
    span = max(1e-12, hi - lo)
    return {key: (value - lo) / span for key, value in values.items()}


def canonical_step4(rows: list[dict], union_meta: dict[tuple[str, str], dict]) -> list[str]:
    qid = str(rows[0]["query_id"])
    docs = [str(row["document_id"]) for row in rows]
    ft = minmax({str(row["document_id"]): float(row["bge_ft_score"]) for row in rows})
    base = minmax({str(row["document_id"]): float(row["bge_base_score"]) for row in rows})
    rrf = minmax({did: float(union_meta[(qid, did)]["rrf_score"]) for did in docs})
    score = {
        did: 0.40 * ft[did]
        + 0.30 * base[did]
        + 0.20 * rrf[did]
        + 0.10 * (float(union_meta[(qid, did)]["source_support"]) / 4.0)
        for did in docs
    }
    return sorted(score, key=lambda did: (-score[did], did))


def policy_result(
    name: str,
    predictions: dict[str, list[str]],
    ft_predictions: dict[str, list[str]],
    gold: dict[str, dict],
    folds: dict[str, int],
) -> dict:
    metrics = macro_metrics(predictions, gold)
    base_metrics = macro_metrics(ft_predictions, gold)
    by_fold = {
        f"F{fold}": macro_metrics(
            {qid: predictions[qid] for qid in predictions if folds[qid] == fold}, gold
        )["recall"]
        for fold in TARGET_FOLDS
    }
    base_by_fold = {
        f"F{fold}": macro_metrics(
            {qid: ft_predictions[qid] for qid in ft_predictions if folds[qid] == fold}, gold
        )["recall"]
        for fold in TARGET_FOLDS
    }
    changed = gained = lost = net = 0
    for qid, top in predictions.items():
        if top != ft_predictions[qid]:
            changed += 1
        wanted = {str(x) for x in gold[qid].get("answer", [])}
        old_hits = len(wanted & set(ft_predictions[qid][:5]))
        new_hits = len(wanted & set(top[:5]))
        gained += max(0, new_hits - old_hits)
        lost += max(0, old_hits - new_hits)
        net += new_hits - old_hits
    return {
        "status": "VALID",
        "policy": name,
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "fold_recall": by_fold,
        "delta_vs_ft_only": {
            "recall": metrics["recall"] - base_metrics["recall"],
            "precision": metrics["precision"] - base_metrics["precision"],
            "fold_recall": {fold: by_fold[fold] - base_by_fold[fold] for fold in by_fold},
        },
        "queries_changed": changed,
        "relevant_docs_gained": gained,
        "relevant_docs_lost": lost,
        "net_relevant_doc_change": net,
        "deployment_gate": metrics["recall"] >= 0.92 and all(value >= 0.90 for value in by_fold.values()),
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    worklist = jsonl(WORKLIST)
    scores = jsonl(SCORES)
    union_rows = jsonl(UNION)
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    folds = load_folds()

    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("manifest is not COMPLETE")
    if sha256(WORKLIST) != manifest["worklist_sha256"]:
        raise RuntimeError("worklist SHA mismatch")
    work_keys = [(str(row["query_id"]), str(row["document_id"])) for row in worklist]
    score_keys = [(str(row["query_id"]), str(row["document_id"])) for row in scores]
    if len(worklist) != EXPECTED_QDOCS or len(scores) != EXPECTED_QDOCS:
        raise RuntimeError("unexpected q-doc count")
    if len(set(work_keys)) != EXPECTED_QDOCS or len(set(score_keys)) != EXPECTED_QDOCS:
        raise RuntimeError("duplicate q-doc identity")
    if set(work_keys) != set(score_keys):
        raise RuntimeError("worklist/score identity mismatch")
    work_by_key = {(str(row["query_id"]), str(row["document_id"])): row for row in worklist}
    score_by_key = {(str(row["query_id"]), str(row["document_id"])): row for row in scores}
    by_query: dict[str, list[dict]] = defaultdict(list)
    for key, row in work_by_key.items():
        scored = score_by_key[key]
        if not finite(scored.get("bge_ft_score")) or not finite(scored.get("bge_base_score")):
            raise RuntimeError(f"nonfinite score at {key}")
        if max(map(float, scored["ft_chunk_scores"])) != float(scored["bge_ft_score"]):
            raise RuntimeError(f"FT MAX mismatch at {key}")
        if max(map(float, scored["base_chunk_scores"])) != float(scored["bge_base_score"]):
            raise RuntimeError(f"Base MAX mismatch at {key}")
        joined = {**row, **scored}
        by_query[key[0]].append(joined)
    if len(by_query) != EXPECTED_QUERIES or any(len(rows) != EXPECTED_K for rows in by_query.values()):
        raise RuntimeError("query/K20 shape mismatch")
    for rows in by_query.values():
        rows.sort(key=lambda row: int(row["candidate_rank"]))

    union_meta: dict[tuple[str, str], dict] = {}
    union_fields = set()
    union_source_names = Counter()
    for row in union_rows:
        qid = str(row.get("query_id") or row.get("question_id"))
        for hit in row.get("candidates", row.get("hits", [])):
            union_fields.update(hit)
            source_ranks = hit.get("source_ranks") or {}
            union_source_names.update(source_ranks)
            union_meta[(qid, str(hit.get("doc_id") or hit.get("document_id")))] = {
                "union_rank": int(hit.get("union_rank", 999)),
                "rrf_score": float(hit.get("rrf_score", 0.0)),
                "source_support": int(hit.get("source_support", 1)),
                "source_ranks": {str(k): int(v) for k, v in source_ranks.items()},
            }
    source_ranks = {}
    work_source_fields = Counter()
    missing_union_keys = []
    for row in worklist:
        key = (str(row["query_id"]), str(row["document_id"]))
        work_source_fields.update(row.get("source_ranks", {}))
        source_ranks[key] = {str(k): int(v) for k, v in row.get("source_ranks", {}).items()}
        if key not in union_meta:
            missing_union_keys.append(key)

    scientific_ids = {qid for qid, fold in folds.items() if fold in TARGET_FOLDS}
    if scientific_ids != set(by_query) or len(scientific_ids) != EXPECTED_QUERIES:
        raise RuntimeError("F1-F4 query coverage mismatch")
    scientific_gold = {qid: gold[qid] for qid in scientific_ids}
    ft_predictions = {qid: rank_ft(by_query[qid])[:5] for qid in sorted(by_query)}
    oracle_predictions = {qid: [str(row["document_id"]) for row in by_query[qid]][:20] for qid in by_query}
    policy2_predictions = {qid: weighted_rrf(by_query[qid], source_ranks)[:5] for qid in by_query}
    policy3_available = not missing_union_keys
    policy3_predictions = (
        {qid: canonical_step4(by_query[qid], union_meta)[:5] for qid in by_query}
        if policy3_available
        else {}
    )

    results = {
        "status": "K20_FINAL_RANKING_DIAGNOSIS_COMPLETE",
        "input_verify": {
            "manifest_complete": True,
            "worklist_sha_match": True,
            "qdocs": len(worklist),
            "queries": len(by_query),
            "candidates_per_query": sorted({len(rows) for rows in by_query.values()}),
            "unique_qdocs": len(set(work_keys)),
            "duplicate_qdocs": 0,
            "scores_complete": True,
            "nonfinite_scores": 0,
            "max_aggregation_verified": True,
        },
        "metadata_available": {
            "worklist_fields": sorted({field for row in worklist for field in row}),
            "current_score_fields": sorted({field for row in scores for field in row}),
            "retrieval_rank_fields_in_k20": sorted(work_source_fields),
            "retrieval_rank_presence_counts_in_k20": dict(work_source_fields),
            "union_metadata_fields": sorted(union_fields),
            "union_source_rank_presence_counts": dict(union_source_names),
            "k20_rows_missing_union_metadata": len(missing_union_keys),
            "current_bge_scores": ["bge_ft_score", "bge_base_score", "ft_chunk_scores", "base_chunk_scores"],
            "retrieval_scores_available": False,
            "rrf_metadata_available": ["union_rank", "rrf_score", "source_support", "source_ranks"],
        },
        "population": {"folds": ["F1", "F2", "F3", "F4"], "queries": len(scientific_ids), "qdocs": len(worklist), "fold0_used": False},
        "candidate_oracle_recall_at_20": macro_metrics(oracle_predictions, scientific_gold, limit=20)["recall"],
        "ft_only_reproduced": policy_result("FT_ONLY_CURRENT", ft_predictions, ft_predictions, scientific_gold, folds),
        "retrieval_rrf_no_label": policy_result("RETRIEVAL_RRF_NO_LABEL", policy2_predictions, ft_predictions, scientific_gold, folds),
        "rrf_plus_current_bge": (
            policy_result("RRF_PLUS_CURRENT_BGE", policy3_predictions, ft_predictions, scientific_gold, folds)
            if policy3_available
            else {
                "policy": "RRF_PLUS_CURRENT_BGE",
                "status": "BLOCKED",
                "reason": "Canonical Step4 union metadata (rrf_score/source_support) is absent for some frozen K20 rows; no score or support value was invented.",
                "missing_union_metadata_rows": len(missing_union_keys),
            }
        ),
        "provenance": {
            "gold_used_only_for_evaluation": True,
            "fold0_used": False,
            "label_overlay_used": False,
            "retrieval_rerun": False,
            "gpu_runs": 0,
            "modal_inference_runs": 0,
            "rrf_plus_formula": "0.40*minmax(FT)+0.30*minmax(Base)+0.20*minmax(union RRF)+0.10*(source_support/4), canonical Step4 implementation",
            "retrieval_rrf_formula": "sum(weight/(2+rank)) for dense=.2, bge=.3, knn_word=.2, bm25=.3; canonical weighted_rrf tie-break",
        },
    }
    valid = [results["ft_only_reproduced"], results["retrieval_rrf_no_label"]]
    if results["rrf_plus_current_bge"]["status"] == "VALID":
        valid.append(results["rrf_plus_current_bge"])
    best = max(valid, key=lambda result: (result["recall"], result["precision"], -result["queries_changed"]))
    results["diagnosis"] = {
        "best_policy": best["policy"],
        "best_top5_recall": best["recall"],
        "ft_only_oracle_to_top5_gap": results["candidate_oracle_recall_at_20"] - results["ft_only_reproduced"]["recall"],
        "oracle_to_best_top5_gap": results["candidate_oracle_recall_at_20"] - best["recall"],
        "candidate_retrieval_gap": 1.0 - results["candidate_oracle_recall_at_20"],
        "main_failure_source": "FINAL_RANKING",
        "deployment_gate": "PASS" if best["deployment_gate"] else "FAIL",
        "historical_baseline_recall_reference": 0.9259285714285714,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
