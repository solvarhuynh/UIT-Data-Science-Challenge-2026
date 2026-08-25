"""CPU materialization of the V3 residual shortlist and true-S2 evidence.

The output deliberately excludes labels.  Gold answers are consulted only for
the coverage report, never written into shortlist rows or used for selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.beam.task1_v2.evidence import iter_payloads, prepare_document, select_true_s2_prepared
from scripts.beam.task1_v3_residual.common import jsonl, load_fold_map, sha256, sort_query_ids, write_jsonl


def candidate_map(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in jsonl(path):
        unique: dict[str, dict[str, Any]] = {}
        for index, candidate in enumerate(row.get("candidates", []), 1):
            doc_id = str(candidate["doc_id"])
            if doc_id not in unique:
                item = dict(candidate)
                item["doc_id"] = doc_id
                item["union_rank"] = int(item.get("union_rank", index))
                item["source_support"] = int(item.get("source_support", 0) or 0)
                item["source_ranks"] = dict(item.get("source_ranks") or {})
                unique[doc_id] = item
        result[str(row["query_id"])] = sorted(unique.values(), key=lambda item: (item["union_rank"], item["doc_id"]))
    return result


def shortlist(pool: list[dict[str, Any]], baseline_top5: list[str]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {
        str(item["doc_id"]): item for item in pool if int(item["union_rank"]) <= 20
    }
    all_docs = {str(item["doc_id"]): item for item in pool}
    for rank, doc_id in enumerate(baseline_top5, 1):
        doc_id = str(doc_id)
        if doc_id in all_docs:
            selected.setdefault(doc_id, all_docs[doc_id])
        else:
            # A baseline doc outside candidate@200 is retained with explicit
            # missing retrieval provenance; this preserves the baseline anchor.
            selected.setdefault(doc_id, {"doc_id": doc_id, "union_rank": 10**9, "source_support": 0, "source_ranks": {}})
    return sorted(selected.values(), key=lambda item: (int(item["union_rank"]), str(item["doc_id"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    questions = json.loads(args.questions.read_text(encoding="utf-8-sig"))
    folds = load_fold_map(args.folds)
    candidates = candidate_map(args.candidates)
    baseline = {str(row["query_id"]): row for row in jsonl(args.baseline)}
    if set(questions) != set(folds) or set(candidates) != set(folds) or set(baseline) != set(folds):
        raise ValueError("V3 questions/folds/candidates/baseline query sets differ")
    if args.preflight:
        if any(len(shortlist(candidates[qid], baseline[qid]["top5"])) > 25 for qid in folds):
            raise ValueError("V3 shortlist exceeds 25 documents")
        print(json.dumps({"status": "PREFLIGHT_PASS", "query_count": len(folds), "shortlist_cap": 25, "gpu_launched": False}, indent=2))
        return

    rows: list[dict[str, Any]] = []
    requested_docs: set[str] = set()
    coverage: dict[int, list[float]] = defaultdict(list)
    shortlist_sizes: list[int] = []
    for qid in sort_query_ids(folds):
        base_top5 = [str(doc_id) for doc_id in baseline[qid]["top5"]]
        docs = shortlist(candidates[qid], base_top5)
        if len(docs) > 25 or len({item["doc_id"] for item in docs}) != len(docs):
            raise ValueError(f"invalid shortlist for {qid}")
        requested_docs.update(item["doc_id"] for item in docs)
        gold = {str(doc_id) for doc_id in questions[qid].get("answer", [])}
        coverage[folds[qid]].append(len(gold & {item["doc_id"] for item in docs}) / len(gold) if gold else 1.0)
        shortlist_sizes.append(len(docs))
        rows.append({"query_id": qid, "fold": folds[qid], "question": str(questions[qid]["question"]), "baseline_top5": base_top5, "docs": docs})

    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in iter_payloads(args.payloads):
        doc_id = str(payload.get("doc_id", ""))
        if doc_id in requested_docs:
            chunks_by_doc[doc_id].append(payload)
    prepared = {doc_id: prepare_document(chunks) for doc_id, chunks in chunks_by_doc.items()}
    missing: list[str] = []
    for row in rows:
        for doc in row["docs"]:
            prepared_doc = prepared.get(doc["doc_id"])
            if prepared_doc is None:
                prepared_doc = prepare_document(())
            selected = select_true_s2_prepared(row["question"], prepared_doc)
            if not selected:
                missing.append(f"{row['query_id']}:{doc['doc_id']}")
            doc["evidence"] = selected
    output = args.output_dir
    shortlist_path = output / "shortlist_evidence.jsonl"
    write_jsonl(shortlist_path, rows)
    report = {
        "status": "CLEAN" if not missing else "DIRTY",
        "schema": "task1-v3-residual-shortlist-v1",
        "query_count": len(rows),
        "fold_counts": {str(fold): sum(row["fold"] == fold for row in rows) for fold in range(5)},
        "shortlist": {"union_rank_max": 20, "baseline_top5_union": True, "deduplicated": True, "max_docs": max(shortlist_sizes), "mean_docs": sum(shortlist_sizes) / len(shortlist_sizes)},
        "gold_coverage_by_fold": {str(fold): sum(values) / len(values) for fold, values in coverage.items()},
        "missing_evidence_count": len(missing),
        "missing_examples": missing[:20],
        "selector_name": "true_s2_bm25_within_document_v2",
        "candidate_refs_full_sha256": sha256(args.candidates),
        "shortlist_evidence_sha256": sha256(shortlist_path),
        "evidence_py_sha256": sha256(ROOT / "scripts/beam/task1_v2/evidence.py"),
        "fold0_gold_used_for": "coverage_report_only_not_features_actions_or_policy_selection",
    }
    (output / "shortlist_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if missing:
        raise SystemExit("V3 shortlist has missing true-S2 evidence")


if __name__ == "__main__":
    main()
