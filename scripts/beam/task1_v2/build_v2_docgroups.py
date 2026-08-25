"""Build V2 document-level groups and fold0 inference groups.

Only train-fold gold labels are used to choose positives.  Fold0 groups are
materialized from candidates and questions alone; their gold labels never
enter this process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .evidence import iter_payloads, prepare_document, select_true_s2_prepared
except ImportError:  # direct ``python scripts/beam/task1_v2/...py`` execution
    from evidence import iter_payloads, prepare_document, select_true_s2_prepared


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_fold_map(path: Path) -> dict[str, int]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict) and isinstance(value.get("folds"), list):
        result: dict[str, int] = {}
        for fold_record in value["folds"]:
            fold = int(fold_record["fold"])
            for query_id in fold_record.get("validation_ids", []):
                result[str(query_id)] = fold
        if result:
            return result
    if isinstance(value, list):
        return {str(row["query_id"]): int(row["fold"]) for row in value}
    raise ValueError(f"unsupported fold map schema: {path}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in jsonl(path):
        qid = str(row["query_id"])
        unique: dict[str, dict[str, Any]] = {}
        for position, candidate in enumerate(row.get("candidates", []), 1):
            doc_id = str(candidate["doc_id"])
            if doc_id in unique:
                continue
            item = dict(candidate)
            item["doc_id"] = doc_id
            item["union_rank"] = int(item.get("union_rank", position))
            item["source_support"] = int(item.get("source_support", 0) or 0)
            item["source_ranks"] = dict(item.get("source_ranks") or {})
            unique[doc_id] = item
        result[qid] = sorted(unique.values(), key=lambda x: (x["union_rank"], x["doc_id"]))
    return result


def baseline_rows(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in jsonl(path):
        result[str(row["query_id"])] = row
    return result


def retrieval_features(candidate: dict[str, Any]) -> dict[str, Any]:
    source_ranks = dict(candidate.get("source_ranks") or {})
    return {
        "source_support": int(candidate.get("source_support", 0) or 0),
        "source_ranks": source_ranks,
        "min_source_rank": min(
            (int(value) for value in source_ranks.values() if value is not None),
            default=None,
        ),
        "has_bge_support": any(
            key.lower() in {"bge", "original_bge", "adaptive_k500"}
            for key in source_ranks
        ),
    }


def choose_train_docs(
    candidates: list[dict[str, Any]],
    gold: set[str],
    baseline_top5: list[str],
    hard_negatives: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive = [row for row in candidates if row["doc_id"] in gold]
    positive.sort(key=lambda row: (row["union_rank"], row["doc_id"]))
    baseline_rank = {str(doc_id): rank for rank, doc_id in enumerate(baseline_top5, 1)}
    negatives = [row for row in candidates if row["doc_id"] not in gold]

    def negative_key(row: dict[str, Any]) -> tuple[Any, ...]:
        doc_id = row["doc_id"]
        features = retrieval_features(row)
        if doc_id in baseline_rank:
            return (0, baseline_rank[doc_id], row["union_rank"], doc_id)
        return (
            1,
            -int(features["source_support"]),
            features["min_source_rank"] if features["min_source_rank"] is not None else 10**9,
            row["union_rank"],
            doc_id,
        )

    negatives.sort(key=negative_key)
    return positive, negatives[: max(0, int(hard_negatives))]


def group_doc(candidate: dict[str, Any], evidence: list[dict[str, Any]], label: bool | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "doc_id": str(candidate["doc_id"]),
        "union_rank": int(candidate["union_rank"]),
        "retrieval_features": retrieval_features(candidate),
        "evidence": evidence,
    }
    if label is not None:
        result["is_positive_doc"] = bool(label)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-fold", type=int, default=0)
    parser.add_argument("--hard-negatives", type=int, default=24)
    parser.add_argument("--topk-evidence", type=int, default=3)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    folds = load_fold_map(args.folds)
    candidates = candidate_rows(args.candidates)
    baseline = baseline_rows(args.baseline)
    if args.preflight:
        if set(questions) != set(folds) or set(candidates) != set(folds):
            raise ValueError("questions, folds, and candidate query sets differ")
        if any(len(rows) > 200 for rows in candidates.values()):
            raise ValueError("candidate pool exceeds authoritative @200 cap")
        print(json.dumps({"status": "PREFLIGHT_PASS", "query_count": len(folds), "gpu_launched": False}, indent=2))
        return

    eval_groups: list[dict[str, Any]] = []
    train_groups: list[dict[str, Any]] = []
    requested_docs: set[str] = set()
    positive_counts: list[int] = []
    negative_counts: list[int] = []
    candidate_gold_coverage: list[float] = []
    for qid in sorted(folds, key=lambda value: (int(value) if value.isdigit() else value)):
        if qid not in questions or qid not in candidates or qid not in baseline:
            raise ValueError(f"missing query input for {qid}")
        question = str(questions[qid]["question"])
        pool = candidates[qid]
        base_top5 = [str(doc_id) for doc_id in baseline[qid].get("top5", [])]
        if folds[qid] == args.eval_fold:
            for row in pool:
                requested_docs.add(row["doc_id"])
            eval_groups.append(
                {
                    "query_id": qid,
                    "fold": args.eval_fold,
                    "question": question,
                    "baseline_top5": base_top5,
                    "docs": [group_doc(row, [], None) for row in pool],
                }
            )
            continue

        gold = {str(doc_id) for doc_id in questions[qid].get("answer", [])}
        pool_ids = {row["doc_id"] for row in pool}
        candidate_gold_coverage.append(len(gold & pool_ids) / len(gold) if gold else 1.0)
        positive, negative = choose_train_docs(pool, gold, base_top5, args.hard_negatives)
        if not positive:
            continue
        for row in positive + negative:
            requested_docs.add(row["doc_id"])
        positive_counts.append(len(positive))
        negative_counts.append(len(negative))
        train_groups.append(
            {
                "query_id": qid,
                "fold": folds[qid],
                "question": question,
                "docs": [
                    group_doc(row, [], True) for row in positive
                ]
                + [group_doc(row, [], False) for row in negative],
            }
        )

    chunks_by_doc: dict[str, list[dict[str, Any]]] = {}
    for payload in iter_payloads(args.payloads):
        doc_id = str(payload.get("doc_id", ""))
        if doc_id in requested_docs:
            chunks_by_doc.setdefault(doc_id, []).append(payload)
    prepared_by_doc = {
        doc_id: prepare_document(chunks)
        for doc_id, chunks in chunks_by_doc.items()
    }

    stats = Counter()
    missing_docs: list[str] = []
    def attach(groups: list[dict[str, Any]]) -> None:
        for group in groups:
            question = group["question"]
            for doc in group["docs"]:
                prepared = prepared_by_doc.get(doc["doc_id"])
                if prepared is None:
                    prepared = prepare_document([])
                selected = select_true_s2_prepared(question, prepared, args.topk_evidence)
                doc["evidence"] = selected
                stats["evidence_docs"] += 1
                stats["evidence_chunks"] += len(selected)
                if selected:
                    stats["true_s2_docs"] += 1
                else:
                    stats["missing_payload_or_text_docs"] += 1
                    missing_docs.append(f"{group['query_id']}:{doc['doc_id']}")

    attach(train_groups)
    attach(eval_groups)
    output = args.output_dir
    write_jsonl(output / "train_docgroups.jsonl", train_groups)
    write_jsonl(output / "fold0_eval_docgroups.jsonl", eval_groups)
    clean = not missing_docs and len(eval_groups) == 1400 and sorted({int(group["fold"]) for group in train_groups}) == [1, 2, 3, 4]
    train_path = output / "train_docgroups.jsonl"
    eval_path = output / "fold0_eval_docgroups.jsonl"
    evidence_path = Path(__file__).resolve().with_name("evidence.py")
    report = {
        "status": "CLEAN" if clean else "DIRTY",
        "build_status": "V2_DOCGROUPS_PASS" if not missing_docs else "V2_DOCGROUPS_MISSING_EVIDENCE",
        "query_count": len(folds),
        "train_query_count": len(train_groups),
        "eval_query_count": len(eval_groups),
        "train_folds": sorted({int(group["fold"]) for group in train_groups}),
        "fold0_train_overlap": 0,
        "positive_docs_per_query": {
            "min": min(positive_counts, default=0),
            "max": max(positive_counts, default=0),
            "mean": sum(positive_counts) / len(positive_counts) if positive_counts else 0.0,
        },
        "hard_negatives_per_query": {
            "requested": args.hard_negatives,
            "min": min(negative_counts, default=0),
            "max": max(negative_counts, default=0),
            "mean": sum(negative_counts) / len(negative_counts) if negative_counts else 0.0,
        },
        "true_s2_selector_share": stats["true_s2_docs"] / stats["evidence_docs"] if stats["evidence_docs"] else 0.0,
        "missing_payload_or_raw_text": len(missing_docs),
        "missing_evidence_count": len(missing_docs),
        "missing_examples": missing_docs[:20],
        "train_candidate_gold_coverage_mean": sum(candidate_gold_coverage) / len(candidate_gold_coverage),
        "train_candidate_gold_coverage_zero_queries": sum(value == 0 for value in candidate_gold_coverage),
        "candidate_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "candidate_refs_full_sha256": sha256(args.candidates),
        "evidence_py_sha256": sha256(evidence_path),
        "selector_name": "true_s2_bm25_within_document_v2",
        "payload_path": str(args.payloads),
        "gold_used_for_train_only": True,
        "fold0_gold_used_in_materialization": False,
        "selector": "true_s2_bm25_within_document_v2",
    }
    report["train_docgroups_sha256"] = sha256(train_path)
    report["fold0_eval_docgroups_sha256"] = sha256(eval_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "docgroups_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if missing_docs:
        raise SystemExit("V2 materialization has missing payload/text documents")


if __name__ == "__main__":
    main()
