"""CPU-only strict F1--F4 evaluator for frozen SAME-BGE FT V2 outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
CANDIDATES = ROOT / "private_task1/experiments/pv1_regression_forensics/validation_pv1_candidate_union_k20.jsonl"
REFERENCE = ROOT / "private_task1/experiments/sprint48_step4/reconstructed_pv1_bge_scores.jsonl"
EXPECTED_QDOCS = 112000


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def qkey(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def fold_map() -> dict[str, int]:
    payload = json.loads(FOLDS.read_text(encoding="utf-8"))
    result: dict[str, int] = {}
    for item in payload["folds"]:
        for query_id in item["validation_ids"]:
            qid = str(query_id)
            if qid in result:
                raise ValueError(f"duplicate fold query: {qid}")
            result[qid] = int(item["fold"])
    return result


def rrf_top5(rows: list[dict[str, Any]]) -> list[str]:
    bge_order = sorted(
        rows,
        key=lambda row: (
            -float(row["bge_ft_score"]), -float(row["bge_base_score"]),
            int(row["candidate_rank"]), qkey(str(row["document_id"])),
        ),
    )
    bge_rank = {str(row["document_id"]): index for index, row in enumerate(bge_order, 1)}
    weights = {"dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3}
    ranks: dict[str, dict[str, int]] = {}
    values: dict[str, float] = {}
    for row in rows:
        doc = str(row["document_id"])
        rank = {str(key): int(value) for key, value in (row.get("source_ranks") or {}).items()}
        rank["bge"] = bge_rank[doc]
        ranks[doc] = rank
        values[doc] = sum(weight / (2 + rank[source]) for source, weight in weights.items() if source in rank)
    return sorted(values, key=lambda doc: (-values[doc], *(ranks[doc].get(source, 10**9) for source in weights), qkey(doc)))[:5]


def metrics(predictions: dict[str, list[str]], gold: dict[str, set[str]]) -> dict[str, float | int]:
    recalls, precisions = [], []
    for qid in sorted(predictions, key=qkey):
        hits = len(set(predictions[qid]) & gold[qid])
        recalls.append(hits / len(gold[qid]))
        precisions.append(hits / 5.0)
    return {"queries": len(predictions), "recall": sum(recalls) / len(recalls), "precision": sum(precisions) / len(precisions)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-scores", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--held-out-fold", type=int, choices=(1, 2, 3, 4),
        help="Evaluate exactly one held-out fold for the mandatory early-stop gate.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folds = fold_map()
    all_target = {qid for qid, fold in folds.items() if fold in {1, 2, 3, 4}}
    if args.held_out_fold is None and len(args.v2_scores) != 4:
        raise ValueError("strict OOF evaluation requires exactly four score files")
    if args.held_out_fold is not None and len(args.v2_scores) != 1:
        raise ValueError("single-fold early evaluation requires exactly one score file")
    target = (
        {qid for qid, fold in folds.items() if fold == args.held_out_fold}
        if args.held_out_fold is not None
        else all_target
    )
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    gold = {qid: {str(value) for value in train[qid]["answer"]} for qid in target}
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(CANDIDATES):
        qid = str(row["query_id"])
        if qid in candidates:
            raise ValueError(f"duplicate frozen candidate query: {qid}")
        candidates[qid] = row["candidates"]
    if set(candidates) != all_target or any(len(rows) != 20 for rows in candidates.values()):
        raise ValueError("candidate universe is not frozen F1--F4 K20")
    reference: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(REFERENCE):
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in reference:
            raise ValueError(f"duplicate reference q-doc: {key}")
        reference[key] = row
    if len(reference) != EXPECTED_QDOCS:
        raise ValueError("reference q-doc coverage mismatch")
    expected_keys = {
        (qid, str(candidate["document_id"]))
        for qid in target
        for candidate in candidates[qid]
    }
    v2: dict[tuple[str, str], dict[str, Any]] = {}
    for path in args.v2_scores:
        for row in load_jsonl(path):
            key = (str(row.get("query_id", "")), str(row.get("document_id", "")))
            if key in v2 or key not in expected_keys or key not in reference:
                raise ValueError(f"duplicate or unknown V2 q-doc: {key}")
            if list(row.get("selected_chunk_ids", [])) != list(reference[key].get("selected_chunk_ids", [])):
                raise ValueError(f"selected-chunk provenance mismatch: {key}")
            if not math.isfinite(float(row.get("bge_ft_score", float("nan")))):
                raise ValueError(f"non-finite V2 document score: {key}")
            v2[key] = row
    if set(v2) != expected_keys:
        raise ValueError(f"V2 score coverage mismatch: {len(v2)}/{len(expected_keys)}")
    baseline_predictions: dict[str, list[str]] = {}
    v2_predictions: dict[str, list[str]] = {}
    for qid in sorted(target, key=qkey):
        baseline_rows, v2_rows = [], []
        for candidate in candidates[qid]:
            doc = str(candidate["document_id"])
            key = (qid, doc)
            common = {"query_id": qid, "document_id": doc, "candidate_rank": int(candidate["candidate_rank"]), "source_ranks": candidate.get("source_ranks", {})}
            baseline_rows.append({**common, **reference[key]})
            # The new scorer may carry a copy of reference metadata, but it
            # is never allowed to replace any frozen rank/base-score input.
            # Only the new FT score changes the BGE component of the RRF.
            v2_rows.append({**common, **reference[key], "bge_ft_score": float(v2[key]["bge_ft_score"])})
        baseline_predictions[qid] = rrf_top5(baseline_rows)
        v2_predictions[qid] = rrf_top5(v2_rows)
    baseline = metrics(baseline_predictions, gold)
    proposed = metrics(v2_predictions, gold)
    delta = {"recall": proposed["recall"] - baseline["recall"], "precision": proposed["precision"] - baseline["precision"]}
    if args.held_out_fold is not None:
        early_pass = delta["recall"] >= 0
        result = {
            "status": "EARLY_FOLD_GATE_PASS" if early_pass else "EARLY_FOLD_GATE_FAIL",
            "scope": f"F{args.held_out_fold}",
            "policy": "RETRIEVAL_RRF_NO_LABEL with BGE-V2 substituted only",
            "score_qdocs": len(v2),
            "baseline": baseline,
            "proposed": proposed,
            "delta": delta,
            "early_kill_gate": "PASS" if early_pass else f"FAIL_F{args.held_out_fold}",
            "provenance": {"candidate_universe": "PASS", "selected_chunk_identity": "PASS", "bge_only_substitution": "PASS", "duplicate_qdocs": 0, "finite_scores": "PASS", "fold0_used": False, "private_labels_used": False},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if (
        abs(float(baseline["recall"]) - 0.9255863095238096) > 1e-15
        or abs(float(baseline["precision"]) - 0.19710714285714284) > 1e-15
    ):
        raise RuntimeError("frozen PV1 baseline reproduction mismatch")
    by_fold = {}
    for fold in range(1, 5):
        ids = {qid for qid in target if folds[qid] == fold}
        before = metrics({qid: baseline_predictions[qid] for qid in ids}, {qid: gold[qid] for qid in ids})
        after = metrics({qid: v2_predictions[qid] for qid in ids}, {qid: gold[qid] for qid in ids})
        by_fold[f"F{fold}"] = {"baseline": before, "proposed": after, "delta": {"recall": after["recall"] - before["recall"], "precision": after["precision"] - before["precision"]}}
    gate = (
        delta["recall"] >= 0.0015
        and all(value["delta"]["recall"] >= 0 for value in by_fold.values())
        and delta["precision"] >= -0.001
    )
    result = {
        "status": "STRICT_OOF_GATE_PASS" if gate else "STRICT_OOF_GATE_FAIL",
        "policy": "RETRIEVAL_RRF_NO_LABEL with BGE-V2 substituted only", "score_qdocs": len(v2),
        "baseline": baseline, "proposed": proposed, "delta": delta, "by_fold": by_fold,
        "provenance": {"candidate_universe": "PASS", "selected_chunk_identity": "PASS", "duplicate_qdocs": 0, "finite_scores": "PASS", "fold0_used": False, "private_labels_used": False},
        "frozen_gate": {"pooled_recall_delta_min": 0.0015, "each_fold_recall_delta_min": 0.0, "precision_delta_min": -0.001},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
