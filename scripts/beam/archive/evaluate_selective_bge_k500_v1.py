"""Fail-fast validator and strict nested-OOF evaluator for Selective BGE K500 V1.

This program is deliberately local and read-only with respect to Beam artifacts.
Run it only after downloading a completed K500 output directory from Beam.
It never trains a model and never creates a public submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / "artifacts/task1/recovery_096"
BASELINE = RECOVERY / "baseline_093_oof/predictions.jsonl"
K200_A = RECOVERY / "baseline_093_oof/sources/fold0_original_bge_chunk200_compact.jsonl"
K200_B = RECOVERY / "baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl"
DECISIONS = RECOVERY / "adaptive_k500_v1/adaptive_k500_decisions.jsonl"
RAW_K500 = ROOT / "artifacts/task1/raw_k500.jsonl"
LEXICAL_CHECKPOINTS = RECOVERY / "candidate_union_ablations/checkpointed_lexical_v1/checkpoints"
DEFAULT_K500 = RECOVERY / "selective_bge_k500_v1_from_beam"
DEFAULT_OUT = RECOVERY / "selective_bge_k500_v1_evaluation_v1"
BASELINE_RECALL = 0.9244452380952381


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            raise ValueError(f"{label}: query_id rỗng")
        if qid in result:
            raise ValueError(f"{label}: duplicate query_id={qid}")
        result[qid] = row
    return result


def recall(gold: set[str], top5: list[str]) -> float:
    return len(gold & set(top5)) / len(gold)


def metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    recalls = [recall(set(row["gold_documents"]), row[key]) for row in rows]
    precisions = [len(set(row["gold_documents"]) & set(row[key])) / 5 for row in rows]
    multi = [recall(set(row["gold_documents"]), row[key]) for row in rows if len(row["gold_documents"]) > 1]
    return {
        "macro_recall": sum(recalls) / len(recalls),
        "macro_precision": sum(precisions) / len(precisions),
        "multi_gold_recall": sum(multi) / len(multi) if multi else None,
        "query_count": len(rows),
    }


def doc_features(hits: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        grouped[str(hit["doc_id"])].append(hit)
    output: dict[str, dict[str, float]] = {}
    for doc, values in grouped.items():
        bge = sorted((float(v["bge_score"]) for v in values), reverse=True)
        output[doc] = {
            "best_bge": bge[0],
            "mean_top3_bge": sum(bge[:3]) / min(3, len(bge)),
            "support_count": float(len(values)),
            "best_dense_rank": float(min(int(v["dense_rank"]) for v in values)),
        }
    return output


def candidate_top5(row: dict[str, Any], minimum_bge: float, margin: float, min_support: int) -> tuple[list[str], dict[str, Any] | None]:
    base = row["baseline_top5"]
    if not row["expand_to_k500"]:
        return base, None
    incumbent = row["k200_docs"].get(base[-1])
    if incumbent is None:
        return base, None
    candidates = []
    for doc, feature in row["new_docs"].items():
        if doc in base:
            continue
        if feature["best_bge"] < minimum_bge or feature["support_count"] < min_support:
            continue
        if feature["best_bge"] - incumbent["best_bge"] < margin:
            continue
        candidates.append((doc, feature))
    if not candidates:
        return base, None
    doc, feature = min(candidates, key=lambda item: (-item[1]["best_bge"], -item[1]["support_count"], item[1]["best_dense_rank"], item[0]))
    return base[:4] + [doc], {
        "removed_doc_id": base[-1],
        "inserted_doc_id": doc,
        "inserted": feature,
        "removed": incumbent,
        "bge_margin": feature["best_bge"] - incumbent["best_bge"],
        "provenance": {"source": "selective_k500_rank_201_500"},
    }


def load_lexical_rankings() -> dict[str, dict[str, list[str]]]:
    """Read completed BM25/KNN rankings only; this never performs retrieval."""
    result: dict[str, dict[str, list[str]]] = {name: {} for name in ("bm25", "knn_word", "knn_char")}
    for name in result:
        for fold in range(5):
            path = LEXICAL_CHECKPOINTS / f"{name}_fold{fold}.json"
            if not path.is_file():
                raise FileNotFoundError(f"lexical provenance checkpoint thiếu: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            result[name].update({str(qid): [str(doc) for doc in docs] for qid, docs in payload["rankings"].items()})
    return result


def rrf_docs(rankings: list[list[str]], rrf_k: int = 60) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    first: dict[str, tuple[int, int]] = {}
    for source_index, docs in enumerate(rankings):
        for rank, doc in enumerate(docs, 1):
            score[doc] += 1.0 / (rrf_k + rank)
            first.setdefault(doc, (source_index, rank))
    return sorted(score, key=lambda doc: (-score[doc], first[doc], doc))


def provenance_for(qid: str, doc: str, decision: dict[str, Any], lexical: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    source_rank = {name: (docs[qid].index(doc) + 1 if doc in docs[qid] else None) for name, docs in lexical.items()}
    adaptive = [str(x) for x in decision["k200_documents"]] + [str(x) for x in decision["k500_added_documents"]]
    union = rrf_docs([adaptive, lexical["bm25"][qid], lexical["knn_word"][qid], lexical["knn_char"][qid]])[:200]
    support = [name for name, rank in source_rank.items() if rank is not None]
    return {"source": "selective_k500_rank_201_500", "k500_rank": None, "bm25_rank": source_rank["bm25"], "knn_word_rank": source_rank["knn_word"], "knn_char_rank": source_rank["knn_char"], "lexical_support_sources": support, "lexical_support_count": len(support), "bounded_union_at_200": doc in set(union), "bounded_union_rank": union.index(doc) + 1 if doc in union else None}


def validate(k500_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    needed = {
        "manifest": k500_dir / "manifest.json",
        "cost_report": k500_dir / "cost_report.json",
        "new_chunk_scores": k500_dir / "new_chunk_scores.jsonl",
        "combined_rankings": k500_dir / "combined_k500_document_rankings.jsonl",
        "baseline": BASELINE,
        "decisions": DECISIONS,
        "k200_fold0": K200_A,
        "k200_fold1to4": K200_B,
        "raw_k500": RAW_K500,
    }
    absent = [name for name, path in needed.items() if not path.is_file()]
    if absent:
        raise FileNotFoundError("artifact thiếu (không dùng partial output): " + ", ".join(absent))
    manifest = json.loads(needed["manifest"].read_text(encoding="utf-8"))
    cost = json.loads(needed["cost_report"].read_text(encoding="utf-8"))
    if manifest != cost:
        raise ValueError("manifest và cost_report khác nhau")
    if manifest.get("schema_version") != "selective-bge-k500-v1":
        raise ValueError("schema_version output K500 không đúng")
    decisions = index_unique(jsonl(DECISIONS), "adaptive decisions")
    selected = {qid for qid, row in decisions.items() if bool(row.get("expand_to_k500"))}
    baseline = index_unique(jsonl(BASELINE), "exact baseline")
    k200 = index_unique(jsonl(K200_A) + jsonl(K200_B), "BGE K200 cache")
    new_scores = index_unique(jsonl(needed["new_chunk_scores"]), "new_chunk_scores")
    combined = index_unique(jsonl(needed["combined_rankings"]), "combined_rankings")
    if set(baseline) != set(decisions) or set(k200) != set(decisions):
        raise ValueError("coverage baseline/decisions/K200 không khớp")
    if set(new_scores) != selected or set(combined) != selected:
        raise ValueError("final K500 output không phủ đúng selected queries; có thể là partial output")
    if int(manifest.get("selected_query_count", -1)) != len(selected):
        raise ValueError("selected_query_count manifest không khớp decisions")
    total_new = 0
    for qid, row in new_scores.items():
        if qid not in selected or str(row.get("query_id")) != qid or not isinstance(row.get("new_hits"), list):
            raise ValueError(f"new_chunk_scores invalid: {qid}")
        old_chunks = {str(hit["chunk_id"]) for hit in k200[qid]["hits"]}
        seen_chunks: set[str] = set()
        for hit in row["new_hits"]:
            doc, chunk = str(hit.get("doc_id", "")).strip(), str(hit.get("chunk_id", "")).strip()
            rank = int(hit.get("dense_rank", 0))
            if not doc or not chunk or chunk in seen_chunks or chunk in old_chunks or not 201 <= rank <= 500:
                raise ValueError(f"new hit invalid/overwrites K200: query={qid}, chunk={chunk}")
            if "bge_score" not in hit:
                raise ValueError(f"new hit thiếu bge_score: query={qid}, chunk={chunk}")
            seen_chunks.add(chunk)
        total_new += len(row["new_hits"])
        ranking = combined[qid].get("document_ranking")
        if not isinstance(ranking, list) or not ranking or len(ranking) != len(set(map(str, ranking))) or any(not str(x).strip() for x in ranking):
            raise ValueError(f"combined ranking invalid: {qid}")
    if int(manifest.get("new_chunks_scored", -1)) != total_new:
        raise ValueError("new_chunks_scored manifest không khớp output")
    expected_hashes = {"decisions": DECISIONS, "k200_fold0": K200_A, "k200_fold1to4": K200_B, "raw_k500": RAW_K500}
    recorded = manifest.get("input_sha256", {})
    hashes = {name: sha256(path) for name, path in expected_hashes.items()}
    mismatch = [name for name, value in hashes.items() if recorded.get(name) != value]
    if mismatch:
        raise ValueError("input/cache hash không khớp manifest: " + ", ".join(mismatch))
    return {"validation": "PASS", "selected_query_count": len(selected), "new_chunks_scored": total_new, "input_sha256_verified": hashes}, baseline, decisions, k200, new_scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k500-dir", type=Path, default=DEFAULT_K500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    validation, baseline, decisions, k200, new_scores = validate(args.k500_dir)
    lexical = load_lexical_rankings()
    if any(set(source) != set(baseline) for source in lexical.values()):
        raise ValueError("coverage lexical provenance/baseline không khớp")
    rows: list[dict[str, Any]] = []
    for qid, base in baseline.items():
        gold = [str(doc) for doc in base["gold_documents"]]
        row = {
            "query_id": qid, "fold": int(base["fold"]), "gold_documents": gold,
            "baseline_top5": [str(doc) for doc in base["top5"]],
            "expand_to_k500": bool(decisions[qid]["expand_to_k500"]),
            "miss_k200": bool(decisions[qid].get("miss_k200", False)),
            "k200_docs": doc_features(k200[qid]["hits"]),
            "new_docs": doc_features(new_scores[qid]["new_hits"]) if qid in new_scores else {},
        }
        rows.append(row)
    configs = [("KEEP_BASELINE", None)] + [
        (f"bge>={bge:g};margin>={margin:g};support>={support}", (bge, margin, support))
        for bge in (4.0, 5.0, 6.0) for margin in (0.0, 0.5) for support in (1, 2)
    ]
    selected_config: dict[int, tuple[str, tuple[float, float, int] | None]] = {}
    for fold in range(5):
        train = [row for row in rows if row["fold"] != fold]
        scored = []
        for name, config in configs:
            changed = better = worse = 0
            candidate_rows = []
            for row in train:
                top5, detail = row["baseline_top5"], None
                if config is not None:
                    top5, detail = candidate_top5(row, *config)
                candidate_rows.append({**row, "policy_top5": top5})
                if detail is not None:
                    changed += 1
                    delta = recall(set(row["gold_documents"]), top5) - recall(set(row["gold_documents"]), row["baseline_top5"])
                    better += delta > 0
                    worse += delta < 0
            score = metrics(candidate_rows, "policy_top5")
            scored.append(((score["macro_recall"], score["macro_precision"], -worse, -changed, better), name, config))
        _, name, config = max(scored, key=lambda item: item[0])
        selected_config[fold] = (name, config)
    predictions: list[dict[str, Any]] = []
    change_rows: list[dict[str, Any]] = []
    for row in rows:
        name, config = selected_config[row["fold"]]
        top5, detail = row["baseline_top5"], None
        if config is not None:
            top5, detail = candidate_top5(row, *config)
        enriched = {**row, "policy_top5": top5}
        predictions.append({"query_id": row["query_id"], "fold": row["fold"], "top5": top5, "decision": "APPLY_SELECTIVE_K500" if detail else "KEEP_BASELINE", "selected_policy": name})
        if detail:
            detail["provenance"] = provenance_for(row["query_id"], detail["inserted_doc_id"], decisions[row["query_id"]], lexical)
            matching_hits = [hit for hit in new_scores[row["query_id"]]["new_hits"] if str(hit["doc_id"]) == detail["inserted_doc_id"]]
            detail["provenance"]["k500_rank"] = min(int(hit["dense_rank"]) for hit in matching_hits)
            change_rows.append({"query_id": row["query_id"], "fold": row["fold"], "gold_documents": row["gold_documents"], "baseline_top5": row["baseline_top5"], "policy_top5": top5, **detail})
        row["policy_top5"] = enriched["policy_top5"]
    baseline_metric, policy_metric = metrics(rows, "baseline_top5"), metrics(rows, "policy_top5")
    per_fold = {}
    for fold in range(5):
        subset = [row for row in rows if row["fold"] == fold]
        per_fold[str(fold)] = {"baseline": metrics(subset, "baseline_top5"), "policy": metrics(subset, "policy_top5"), "delta": metrics(subset, "policy_top5")["macro_recall"] - metrics(subset, "baseline_top5")["macro_recall"], "selected_policy": selected_config[fold][0]}
    better = sum(recall(set(row["gold_documents"]), row["policy_top5"]) > recall(set(row["gold_documents"]), row["baseline_top5"]) for row in rows)
    worse = sum(recall(set(row["gold_documents"]), row["policy_top5"]) < recall(set(row["gold_documents"]), row["baseline_top5"]) for row in rows)
    def group(where: Any) -> dict[str, Any]:
        subset = [row for row in rows if where(row)]
        return {"query_count": len(subset), "baseline": metrics(subset, "baseline_top5"), "policy": metrics(subset, "policy_top5"), "delta": metrics(subset, "policy_top5")["macro_recall"] - metrics(subset, "baseline_top5")["macro_recall"]}
    gate = {"pooled_recall_above_anchor": policy_metric["macro_recall"] > BASELINE_RECALL, "no_negative_fold": all(item["delta"] >= 0 for item in per_fold.values()), "improvements_at_least_harms": better >= worse}
    gate["promote"] = all(gate.values())
    status = "PROMOTE_SELECTIVE_BGE_K500" if gate["promote"] else ("SELECTIVE_BGE_K500_POSITIVE_BUT_NOT_PROMOTABLE" if policy_metric["macro_recall"] > BASELINE_RECALL else "REJECT_SELECTIVE_BGE_K500")
    report = {"schema_version": "selective-bge-k500-evaluation-v1", "status": status, "validation": validation, "baseline": baseline_metric, "policy": policy_metric, "recall_delta_vs_0_9244452381": policy_metric["macro_recall"] - BASELINE_RECALL, "per_fold": per_fold, "outcomes_vs_baseline": {"better": better, "same": len(rows) - better - worse, "worse": worse, "changed_queries": len(change_rows)}, "adaptive_group": group(lambda row: row["expand_to_k500"]), "retrieval_miss_group": group(lambda row: row["miss_k200"]), "multi_gold_group": group(lambda row: len(row["gold_documents"]) > 1), "gate": gate, "constraints": {"nested_outer_folds": 5, "public_gold_as_feature": False, "neural_training": False, "public_submission_created": False, "no_blanket_top5_replacement": True}}
    atomic_jsonl(args.output_dir / "selective_k500_policy_predictions.jsonl", predictions)
    atomic_jsonl(args.output_dir / "changed_docs.jsonl", change_rows)
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps({"status": status, "validation": "PASS", "policy_recall": policy_metric["macro_recall"], "delta": report["recall_delta_vs_0_9244452381"], "changed_queries": len(change_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
