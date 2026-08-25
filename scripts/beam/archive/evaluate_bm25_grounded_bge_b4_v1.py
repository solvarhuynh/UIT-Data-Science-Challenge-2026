"""Strict nested-OOF evaluator for completed BM25-grounded B4 BGE scores.

Run only after downloading a *complete* Beam output.  It validates hashes and
keeps the exact public-0.93 producer top5 as the incumbent; gold is used only
inside each outer fold's four-fold policy selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / "artifacts/task1/recovery_096"
BASELINE = RECOVERY / "baseline_093_oof/predictions.jsonl"
K200_A = RECOVERY / "baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl"
K200_B = RECOVERY / "baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl"
K500 = RECOVERY / "selective_bge_k500_v1_from_beam/new_chunk_scores.jsonl"
WORKLIST = RECOVERY / "bm25_grounded_selector_gate_v2/worklists/B4_bm25_rank_le_20.json"
DEFAULT_INPUT = RECOVERY / "bm25_grounded_bge_b4_v1"
DEFAULT_OUTPUT = RECOVERY / "bm25_grounded_bge_b4_v1_evaluation"
ANCHOR = 0.9244452380952382
WORKLIST_SHA = "34d3ba1f0f9e202222c19a921f10725faace86af8869809d3c947f66b035ba32"


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2); handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row["query_id"]): row for row in rows}
    if len(result) != len(rows): raise ValueError(f"duplicate query_id: {label}")
    return result


def features(hits: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits: by_doc[str(hit["doc_id"])].append(hit)
    output = {}
    for doc, values in by_doc.items():
        scores = sorted((float(v["bge_score"]) for v in values), reverse=True)
        output[doc] = {"best_bge": scores[0], "support": float(len(values))}
    return output


def recall(row: dict[str, Any], key: str) -> float:
    return len(set(row["gold_documents"]) & set(row[key])) / len(row["gold_documents"])


def metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    recalls = [recall(row, key) for row in rows]
    precisions = [len(set(row["gold_documents"]) & set(row[key])) / 5 for row in rows]
    return {"macro_recall": sum(recalls) / len(rows), "macro_precision": sum(precisions) / len(rows), "query_count": len(rows)}


def policy(row: dict[str, Any], config: tuple[float, float, int] | None) -> tuple[list[str], dict[str, Any] | None]:
    baseline = row["baseline_top5"]
    if config is None or baseline[-1] not in row["existing_docs"]: return baseline, None
    min_score, min_margin, min_support = config
    incumbent = row["existing_docs"][baseline[-1]]
    candidates = [(doc, value) for doc, value in row["b4_docs"].items() if doc not in baseline and value["best_bge"] >= min_score and value["best_bge"] - incumbent["best_bge"] >= min_margin and value["support"] >= min_support]
    if not candidates: return baseline, None
    doc, value = min(candidates, key=lambda item: (-item[1]["best_bge"], -item[1]["support"], item[1]["bm25_rank"], item[1]["union_rank"], item[0]))
    return baseline[:4] + [doc], {"inserted_doc_id": doc, "removed_doc_id": baseline[-1], "inserted": value, "incumbent": incumbent, "bge_margin": value["best_bge"] - incumbent["best_bge"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b4-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    needed = {"report": args.b4_dir / "report.json", "scores": args.b4_dir / "new_chunk_scores.jsonl", "baseline": BASELINE, "k200a": K200_A, "k200b": K200_B, "k500": K500, "worklist": WORKLIST}
    absent = [name for name, path in needed.items() if not path.is_file()]
    if absent: raise FileNotFoundError("artifact thiếu (partial output không hợp lệ): " + ", ".join(absent))
    report = json.loads(needed["report"].read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE_NO_TOP5_EVALUATION" or report.get("worklist_sha256") != WORKLIST_SHA or not report.get("checkpoint_completeness", {}).get("complete"):
        raise ValueError("Beam B4 output is not a complete hash-locked diagnostic")
    if report.get("new_chunk_scores_sha256") != sha256(needed["scores"]): raise ValueError("new_chunk_scores SHA mismatch")
    worklist = json.loads(WORKLIST.read_text(encoding="utf-8"))["rows"]
    canonical = hashlib.sha256("".join(f"{r['query_id']}\t{r['doc_id']}\n" for r in sorted(worklist, key=lambda r: (str(r['query_id']), str(r['doc_id'])))).encode()).hexdigest()
    if canonical != WORKLIST_SHA: raise ValueError("local B4 worklist SHA mismatch")
    baseline = index(jsonl(BASELINE), "baseline")
    existing: dict[str, dict[str, Any]] = defaultdict(dict)
    for record in jsonl(K200_A) + jsonl(K200_B):
        existing[str(record["query_id"])].update(features(record["hits"]))
    for qid, row in index(jsonl(K500), "k500").items(): existing[qid].update(features(row["new_hits"]))
    score_rows = index(jsonl(needed["scores"]), "B4 scores")
    by_qdoc: dict[tuple[str, str], dict[str, Any]] = {(str(r["query_id"]), str(r["doc_id"])): r for r in worklist}
    b4: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for qid, record in score_rows.items():
        for doc, value in features(record["new_hits"]).items():
            meta = by_qdoc.get((qid, doc))
            if meta is None: raise ValueError(f"score outside locked B4 worklist: {qid}/{doc}")
            b4[qid][doc] = {**value, "bm25_rank": float(meta["bm25_rank"]), "union_rank": float(meta["union_rank"])}
    rows = [{"query_id": qid, "fold": int(base["fold"]), "gold_documents": list(map(str, base["gold_documents"])), "baseline_top5": list(map(str, base["top5"])), "existing_docs": existing[qid], "b4_docs": b4[qid]} for qid, base in baseline.items()]
    # Small, predeclared policy family.  Its thresholds are chosen only using
    # the other four folds for every held-out fold.
    configs: list[tuple[str, tuple[float, float, int] | None]] = [("KEEP_BASELINE", None), ("B4_score_ge_0_margin_ge_0_support1", (0.0, 0.0, 1)), ("B4_score_ge_0_margin_ge_0_support2", (0.0, 0.0, 2)), ("B4_score_ge_0_margin_ge_0_5_support1", (0.0, 0.5, 1))]
    selected: dict[int, tuple[str, tuple[float, float, int] | None]] = {}
    for fold in range(5):
        candidates = []
        for name, config in configs:
            train = []
            for row in rows:
                if row["fold"] == fold: continue
                top5, _ = policy(row, config); train.append({**row, "policy_top5": top5})
            metric = metrics(train, "policy_top5")
            harms = sum(recall(row, "policy_top5") < recall(row, "baseline_top5") for row in train)
            candidates.append(((metric["macro_recall"], metric["macro_precision"], -harms, name), name, config))
        _, name, config = max(candidates, key=lambda item: item[0]); selected[fold] = (name, config)
    changed, predictions = [], []
    for row in rows:
        name, config = selected[row["fold"]]; top5, detail = policy(row, config)
        row["policy_top5"] = top5
        predictions.append({"query_id": row["query_id"], "fold": row["fold"], "top5": top5, "selected_config": name})
        if detail: changed.append({"query_id": row["query_id"], "fold": row["fold"], "gold_documents": row["gold_documents"], "baseline_top5": row["baseline_top5"], "policy_top5": top5, "provenance": "B4_bm25_rank_le_20_S2", **detail})
    baseline_metric, policy_metric = metrics(rows, "baseline_top5"), metrics(rows, "policy_top5")
    per_fold = {str(fold): {"baseline": metrics([r for r in rows if r["fold"] == fold], "baseline_top5"), "policy": metrics([r for r in rows if r["fold"] == fold], "policy_top5"), "selected_config": selected[fold][0]} for fold in range(5)}
    for value in per_fold.values(): value["delta"] = value["policy"]["macro_recall"] - value["baseline"]["macro_recall"]
    better = sum(recall(row, "policy_top5") > recall(row, "baseline_top5") for row in rows); worse = sum(recall(row, "policy_top5") < recall(row, "baseline_top5") for row in rows)
    gate = {"pooled_recall_above_0_9244452380952382": policy_metric["macro_recall"] > ANCHOR, "no_negative_fold": all(v["delta"] >= 0 for v in per_fold.values()), "improvements_at_least_harms": better >= worse}
    status = "BM25_GROUNDED_BGE_SIGNAL_CONFIRMED" if all(gate.values()) else ("BM25_GROUNDED_BGE_POSITIVE_NOT_PROMOTABLE" if policy_metric["macro_recall"] > ANCHOR else "REJECT_BM25_GROUNDED_BGE_B4_ORIGINAL_BGE_NO_TRAIN_RANKING_PLATEAU_STRONGLY_CONFIRMED")
    output = {"schema_version": "bm25-grounded-bge-b4-evaluation-v1", "status": status, "validation": "PASS", "baseline": baseline_metric, "policy": policy_metric, "recall_delta": policy_metric["macro_recall"] - ANCHOR, "per_fold": per_fold, "outcomes": {"better": better, "same": len(rows) - better - worse, "worse": worse, "changed_queries": len(changed)}, "selected_config_per_outer_fold": {str(f): selected[f][0] for f in range(5)}, "gate": gate, "constraints": {"strict_nested_outer_folds": 5, "gold_not_in_inference_features": True, "no_training": True, "no_submission": True, "b4_is_diagnostic_not_final_0_98_pool": True}}
    atomic_jsonl(args.output_dir / "predictions.jsonl", predictions); atomic_jsonl(args.output_dir / "changed_docs.jsonl", changed); atomic_json(args.output_dir / "report.json", output)
    print(json.dumps({"status": status, "validation": "PASS", "recall": policy_metric["macro_recall"], "delta": output["recall_delta"], "changed_queries": len(changed)}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())