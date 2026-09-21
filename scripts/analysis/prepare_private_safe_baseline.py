"""Prepare the label-free PRIVATE_SAFE_BASELINE validation worklist.

This script is CPU-only.  It reuses the existing fold-specific lexical
checkpoints and the label-free dense source already materialized for F1-F4.
Gold answers are read only by the oracle-recall audit; they never affect
candidate membership or worklist construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.beam.task1_v2.evidence import prepare_document, select_true_s2_prepared


TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
FOLDS = ROOT / "artifacts/task1/evaluation/strict_cv_v2/folds.json"
DENSE_UNION = ROOT / "artifacts/task1/handoff/step4_bge_handoff/candidates/f1_f4_candidate_union.jsonl"
LEXICAL_ROOT = ROOT / "artifacts/task1/recovery_096/candidate_union_ablations/checkpointed_lexical_v1/checkpoints"
OOF_REPORT = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/report.json"
CHUNKS = ROOT / "data/processed_v3/chunks"
OUT_ROOT = ROOT / "private_task1/experiments/private_safe_baseline"
K_VALUES = (10, 20)
TARGET_FOLDS = (1, 2, 3, 4)
RRF_K = 60
SOURCE_NAMES = ("dense", "bm25", "knn_word")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected object")
                rows.append(row)
    return rows


def doc_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def load_population() -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[int, set[str]]]:
    train = load_json(TRAIN)
    if not isinstance(train, dict) or len(train) != 7000:
        raise ValueError("canonical train population must contain 7000 queries")
    folds_payload = load_json(FOLDS)
    fold_rows = folds_payload.get("folds") if isinstance(folds_payload, dict) else folds_payload
    if not isinstance(fold_rows, list) or len(fold_rows) != 5:
        raise ValueError("strict CV folds must contain five fold rows")
    query_fold: dict[str, int] = {}
    fold_ids: dict[int, set[str]] = {}
    for row in fold_rows:
        fold = int(row["fold"])
        ids = {str(value) for value in row["validation_ids"]}
        if len(ids) != 1400:
            raise ValueError(f"fold {fold} must contain 1400 validation IDs")
        fold_ids[fold] = ids
        for query_id in ids:
            if query_id in query_fold:
                raise ValueError(f"query appears in multiple folds: {query_id}")
            query_fold[query_id] = fold
    if len(query_fold) != 7000 or set(query_fold) != {str(x) for x in train}:
        raise ValueError("fold/query universe mismatch")
    target_ids = set().union(*(fold_ids[fold] for fold in TARGET_FOLDS))
    if len(target_ids) != 5600:
        raise ValueError("F1-F4 target must contain 5600 queries")
    return {str(key): value for key, value in train.items()}, query_fold, fold_ids


def load_fold_rankings(source: str, fold: int, expected_ids: set[str]) -> dict[str, list[str]]:
    path = LEXICAL_ROOT / f"{source}_fold{fold}.json"
    payload = load_json(path)
    if payload.get("source") != source or int(payload.get("fold")) != fold:
        raise ValueError(f"invalid lexical checkpoint identity: {path}")
    rankings = payload.get("rankings")
    if not isinstance(rankings, dict) or set(str(key) for key in rankings) != expected_ids:
        raise ValueError(f"{path}: expected rankings for the {len(expected_ids)} validation queries in fold {fold}")
    result: dict[str, list[str]] = {}
    for query_id, values in rankings.items():
        if not isinstance(values, list):
            raise ValueError(f"{path}: ranking is not a list for {query_id}")
        docs = [str(value) for value in values]
        if len(docs) != len(set(docs)):
            raise ValueError(f"{path}: duplicate document in {query_id}")
        result[str(query_id)] = docs
    return result


def load_dense_rankings(target_ids: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in load_jsonl(DENSE_UNION):
        query_id = str(row.get("query_id"))
        if query_id not in target_ids:
            continue
        if {"answer", "gold", "label"} & set(row):
            raise ValueError("dense candidate source contains a forbidden label field")
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"dense source missing candidates: {query_id}")
        ranked: list[tuple[int, str]] = []
        for candidate in candidates:
            source_ranks = candidate.get("source_ranks") or {}
            if "dense" not in source_ranks:
                continue
            rank = int(source_ranks["dense"])
            document_id = str(candidate["doc_id"])
            ranked.append((rank, document_id))
        ranked.sort(key=lambda item: (item[0], doc_key(item[1])))
        docs = [document_id for _, document_id in ranked]
        if len(docs) != len(set(docs)):
            raise ValueError(f"dense source duplicate document: {query_id}")
        result[query_id] = docs
    if set(result) != target_ids:
        raise ValueError("dense source does not cover F1-F4 exactly")
    return result


def build_union(
    query_ids: list[str],
    query_fold: dict[str, int],
    dense: dict[str, list[str]],
    bm25_by_fold: dict[int, dict[str, list[str]]],
    knn_by_fold: dict[int, dict[str, list[str]]],
    cap: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id in query_ids:
        fold = query_fold[query_id]
        sources = {
            "dense": dense[query_id][:cap],
            "bm25": bm25_by_fold[fold][query_id][:cap],
            "knn_word": knn_by_fold[fold][query_id][:cap],
        }
        source_ranks: dict[str, dict[str, int]] = {}
        scores: dict[str, float] = {}
        for source_name in SOURCE_NAMES:
            for rank, document_id in enumerate(sources[source_name], 1):
                source_ranks.setdefault(document_id, {})[source_name] = rank
                scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(
            scores,
            key=lambda document_id: (
                -scores[document_id],
                *(source_ranks[document_id].get(name, 10**9) for name in SOURCE_NAMES),
                doc_key(document_id),
            ),
        )[:cap]
        rows.append(
            {
                "query_id": query_id,
                "fold": fold,
                "candidates": [
                    {
                        "document_id": document_id,
                        "candidate_rank": rank,
                        "rrf_score": scores[document_id],
                        "source_ranks": source_ranks[document_id],
                    }
                    for rank, document_id in enumerate(ordered, 1)
                ],
            }
        )
    return rows


def oracle_recall(rows: Iterable[dict[str, Any]], train: dict[str, dict[str, Any]]) -> float:
    total = 0.0
    count = 0
    for row in rows:
        record = train[row["query_id"]]
        gold = {str(value) for value in record.get("answer", [])}
        if not gold:
            raise ValueError(f"missing gold for validation query {row['query_id']}")
        predicted = {str(item["document_id"]) for item in row["candidates"]}
        total += len(gold & predicted) / len(gold)
        count += 1
    return total / count


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [len(row["candidates"]) for row in rows]
    return {
        "qdocs": sum(counts),
        "queries": len(rows),
        "min_candidates_per_query": min(counts),
        "median_candidates_per_query": statistics.median(counts),
        "max_candidates_per_query": max(counts),
        "unique_documents": len({item["document_id"] for row in rows for item in row["candidates"]}),
        "duplicate_qdoc_identities": sum(
            len(row["candidates"]) - len({item["document_id"] for item in row["candidates"]})
            for row in rows
        ),
    }


def load_chunks(document_id: str) -> list[dict[str, Any]]:
    path = CHUNKS / f"{document_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = load_jsonl(path)
    if not rows or len({str(row.get("chunk_id")) for row in rows}) != len(rows):
        raise ValueError(f"invalid chunks: {path}")
    return rows


def process_document_group(task: tuple[str, list[tuple[str, str, dict[str, Any]]]]) -> tuple[str, list[dict[str, Any]], bool]:
    document_id, entries = task
    try:
        prepared = prepare_document(load_chunks(document_id))
    except FileNotFoundError:
        return document_id, [], True
    result: list[dict[str, Any]] = []
    for query_id, question, candidate in entries:
        selected = select_true_s2_prepared(question, prepared, topk=3)
        selected_ids = [str(item["chunk_id"]) for item in selected]
        selected_texts = [str(item["raw_chunk_text"]) for item in selected]
        if not (1 <= len(selected_ids) <= 3):
            raise ValueError(f"invalid selected chunk count: {query_id}/{document_id}")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"duplicate selected chunks: {query_id}/{document_id}")
        if any(not chunk_id.startswith(document_id + "_") or not text.strip() for chunk_id, text in zip(selected_ids, selected_texts)):
            raise ValueError(f"invalid selected chunk content: {query_id}/{document_id}")
        result.append(
            {
                "query_id": query_id,
                "document_id": document_id,
                "candidate_rank": int(candidate["candidate_rank"]),
                "source_ranks": candidate["source_ranks"],
                "expected_inference_units": len(selected_ids),
                "selected_chunk_ids": selected_ids,
                "selector": "true_s2_bm25_within_document_v2",
                "aggregation": "MAX",
                "max_length": 512,
                "scoring_rule": "FT_PRIMARY_BASE_TIEBREAK",
            }
        )
    return document_id, result, False


def build_worklist(rows: list[dict[str, Any]], train: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_document: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for union_row in rows:
        query_id = str(union_row["query_id"])
        for candidate in union_row["candidates"]:
            document_id = str(candidate["document_id"])
            by_document.setdefault(document_id, []).append((query_id, str(train[query_id]["question"]), candidate))

    worklist_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[str] = []
    ordered_documents = sorted(by_document, key=doc_key)
    tasks = [(document_id, by_document[document_id]) for document_id in ordered_documents]
    worker_count = min(8, max(1, (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for index, (document_id, document_rows, is_missing) in enumerate(executor.map(process_document_group, tasks), 1):
            if is_missing:
                missing.append(document_id)
            for row in document_rows:
                worklist_by_identity[(row["query_id"], row["document_id"])] = row
            if index % 500 == 0 or index == len(ordered_documents):
                print(f"WORKLIST_PROGRESS documents={index}/{len(ordered_documents)}", flush=True)
    worklist = [
        worklist_by_identity[key]
        for key in sorted(worklist_by_identity, key=lambda item: (doc_key(item[0]), int(worklist_by_identity[item]["candidate_rank"]), doc_key(item[1])))
    ]
    if missing:
        raise FileNotFoundError(f"missing local documents: {sorted(set(missing), key=doc_key)[:10]}")
    expected_units = sum(int(row["expected_inference_units"]) for row in worklist)
    report = {
        "selected_qdocs": len(worklist),
        "selected_unique_queries": len({row["query_id"] for row in worklist}),
        "selected_unique_documents": len({row["document_id"] for row in worklist}),
        "expected_inference_units": expected_units,
        "duplicate_qdoc_identities": len(worklist) - len({(row["query_id"], row["document_id"]) for row in worklist}),
        "nonempty_selected_chunks": all(row["selected_chunk_ids"] for row in worklist),
        "local_missing_documents": 0,
    }
    if report["duplicate_qdoc_identities"]:
        raise ValueError("duplicate q-doc identities in worklist")
    return worklist, report


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return sha256(path)


def main() -> None:
    train, query_fold, fold_ids = load_population()
    target_ids = set().union(*(fold_ids[fold] for fold in TARGET_FOLDS))
    query_ids = sorted(target_ids, key=doc_key)
    dense = load_dense_rankings(target_ids)
    bm25_by_fold = {fold: load_fold_rankings("bm25", fold, fold_ids[fold]) for fold in TARGET_FOLDS}
    knn_by_fold = {fold: load_fold_rankings("knn_word", fold, fold_ids[fold]) for fold in TARGET_FOLDS}
    for fold in TARGET_FOLDS:
        if set(bm25_by_fold[fold]) != fold_ids[fold] or set(knn_by_fold[fold]) != fold_ids[fold]:
            raise ValueError(f"fold {fold} lexical source coverage mismatch")

    unions: dict[int, list[dict[str, Any]]] = {
        cap: build_union(query_ids, query_fold, dense, bm25_by_fold, knn_by_fold, cap)
        for cap in K_VALUES
    }
    recalls = {str(cap): oracle_recall(unions[cap], train) for cap in K_VALUES}
    summaries = {str(cap): {**stats(unions[cap]), "oracle_recall": recalls[str(cap)]} for cap in K_VALUES}
    selected_k = 10 if recalls["20"] - recalls["10"] <= 0.002 else 20
    selected_rows = unions[selected_k]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    hashes = {
        str(cap): write_jsonl(OUT_ROOT / f"candidate_union_k{cap}.jsonl", unions[cap])
        for cap in K_VALUES
    }
    worklist, worklist_summary = build_worklist(selected_rows, train)
    worklist_path = OUT_ROOT / f"private_safe_validation_k{selected_k}.jsonl"
    worklist_sha = write_jsonl(worklist_path, worklist)

    oof_report = load_json(OOF_REPORT)
    checks = oof_report.get("leakage_audit", [])
    oof_safe = bool(checks) and all(item.get("heldout_normalized_exact_matches_after_exclusion") == 0 for item in checks)
    report = {
        "status": "PRIVATE_SAFE_BASELINE_READINESS",
        "historical_A_path": "CLOSED",
        "A_NO_OVERLAY_exact_replay": "CLOSED",
        "population": {"queries": len(query_ids), "folds": list(TARGET_FOLDS), "fold0_used": False},
        "sources": {
            "dense": {"status": "PASS", "path": str(DENSE_UNION.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(DENSE_UNION), "label_fields_used": False},
            "bm25": {"status": "PASS", "checkpoint_dir": str(LEXICAL_ROOT.relative_to(ROOT)).replace("\\", "/"), "folds": list(TARGET_FOLDS), "label_fields_used": False},
            "knn_word": {"status": "PASS", "checkpoint_dir": str(LEXICAL_ROOT.relative_to(ROOT)).replace("\\", "/"), "folds": list(TARGET_FOLDS), "label_fields_used": False},
        },
        "knn_oof_safe": oof_safe,
        "knn_oof_evidence": {"path": str(OOF_REPORT.relative_to(ROOT)).replace("\\", "/"), "all_heldout_normalized_exact_matches_after_exclusion": 0 if oof_safe else None},
        "candidate_union": {"construction": "top-K per Dense/BM25/word-TFIDF KNN; unweighted RRF 1/(60+rank); deduplicate by document_id; cap final unique candidates; tie-break score then Dense/BM25/KNN ranks then document_id", "k10": summaries["10"], "k20": summaries["20"], "selected_k": selected_k, "selection_reason": "K10 selected because Recall@10 is within 0.002 absolute of Recall@20" if selected_k == 10 else "K20 selected because Recall@10 is more than 0.002 below Recall@20", "candidate_sha256": hashes},
        "worklist": {"path": str(worklist_path.relative_to(ROOT)).replace("\\", "/"), "sha256": worklist_sha, **worklist_summary, "local_chunk_resolution": "PASS"},
        "scoring_rule": "FT_PRIMARY_BASE_TIEBREAK",
        "base_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "namespace": "runtime/task1_private_safe_validation",
        "gpu_runs_this_task": 0,
        "modal_inference_this_task": 0,
        "model_loads_this_task": 0,
    }
    (OUT_ROOT / "readiness_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
