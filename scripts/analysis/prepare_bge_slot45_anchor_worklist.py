"""Prepare the provenance-safe missing baseline slot-4/5 BGE worklist.

This is CPU-only.  It reads only query/document identities and question text
needed for selector resolution; ``gold_documents`` is never accessed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.beam.task1_v2.evidence import prepare_document, select_true_s2_prepared


BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
CURRENT = ROOT / "artifacts/task1/qwen_to_bge_minimal/bge_top5_production_canonical.jsonl"
OUT = ROOT / "artifacts/task1/qwen_to_bge_minimal/baseline_slot45_missing_current_bge.jsonl"
REPORT = ROOT / "artifacts/task1/qwen_to_bge_minimal/baseline_slot45_anchor_preflight.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
CHUNKS = ROOT / "data/processed_v3/chunks"
CURRENT_SHA = "a72b243e9895b2973b63bc16a48c150982346cc1ba384b77ebf82453f61c5fd7"
SELECTOR = "true_s2_bm25_within_document_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def load_chunks(doc_id: str) -> list[dict]:
    path = CHUNKS / f"{doc_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    result = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            chunk_id = str(item.get("chunk_id", ""))
            raw_text = str(item.get("raw_chunk_text", item.get("text", "")))
            if not chunk_id.startswith(doc_id + "_") or not raw_text.strip():
                raise ValueError(f"invalid chunk {path}:{line_no}")
            item["chunk_id"] = chunk_id
            item["raw_chunk_text"] = raw_text
            result.append(item)
    if not result or len({item["chunk_id"] for item in result}) != len(result):
        raise ValueError(f"invalid or duplicate chunks: {path}")
    return result


def main() -> None:
    if sha256(CURRENT) != CURRENT_SHA:
        raise RuntimeError("CURRENT_BGE_SHA_MISMATCH")

    # Do not read gold_documents.  The baseline file's fold field is sufficient
    # to exclude Fold0 from the worklist.
    baseline_order: list[tuple[str, int, int, str]] = []
    seen_queries: set[str] = set()
    for line_no, row in rows(BASELINE):
        qid = str(row["query_id"])
        fold = int(row["fold"])
        top5 = [str(value) for value in row["top5"]]
        if len(top5) != 5 or len(set(top5)) != 5:
            raise RuntimeError(f"invalid baseline top5 at line {line_no}: {qid}")
        if qid in seen_queries:
            raise RuntimeError(f"duplicate baseline query: {qid}")
        seen_queries.add(qid)
        if fold in (1, 2, 3, 4):
            baseline_order.append((qid, fold, 4, top5[3]))
            baseline_order.append((qid, fold, 5, top5[4]))
    if len(baseline_order) != 11200 or len(seen_queries) != 7000:
        raise RuntimeError(f"unexpected baseline population: rows={len(seen_queries)} anchors={len(baseline_order)}")

    current_keys: set[tuple[str, str]] = set()
    for _, row in rows(CURRENT):
        key = (str(row["query_id"]), str(row["document_id"]))
        if key in current_keys:
            raise RuntimeError(f"duplicate current BGE q-doc: {key}")
        current_keys.add(key)

    missing_rows = []
    scored = []
    for qid, _, rank, did in baseline_order:
        if (qid, did) in current_keys:
            scored.append((qid, rank, did))
            continue
        missing_rows.append({
            "query_id": qid,
            "document_id": did,
            "baseline_rank": rank,
            "source": "baseline_slot45_anchor",
        })
    if any(row["baseline_rank"] not in (4, 5) for row in missing_rows):
        raise RuntimeError("invalid missing baseline rank")

    train = json.loads(TRAIN.read_text(encoding="utf-8-sig"))
    questions = {}
    for qid, _, _, _ in baseline_order:
        if qid not in questions:
            record = train.get(qid)
            if not isinstance(record, dict) or not str(record.get("question", "")).strip():
                raise RuntimeError(f"canonical question missing: {qid}")
            questions[qid] = str(record["question"])

    missing_documents: list[str] = []
    selection_failures: list[dict] = []
    expected_units = 0
    resolved_by_identity = {}
    rows_by_document = {}
    for row in missing_rows:
        rows_by_document.setdefault(row["document_id"], []).append(row)
    for doc_id in sorted(rows_by_document):
        try:
            prepared = prepare_document(load_chunks(doc_id))
        except Exception as exc:
            missing_documents.append(doc_id)
            selection_failures.append({"document_id": doc_id, "error": str(exc)})
            continue
        for row in rows_by_document[doc_id]:
            try:
                selected = select_true_s2_prepared(questions[row["query_id"]], prepared, topk=3)
                ids = [str(item["chunk_id"]) for item in selected]
                texts = [str(item["raw_chunk_text"]) for item in selected]
                if not (1 <= len(ids) <= 3) or len(ids) != len(set(ids)):
                    raise RuntimeError("TOP3_UP_TO_AVAILABLE violation")
                if any(not chunk_id.startswith(row["document_id"] + "_") or not text.strip() for chunk_id, text in zip(ids, texts)):
                    raise RuntimeError("invalid selected chunk")
                row["expected_inference_units"] = len(ids)
                row["selected_chunk_ids"] = ids
                expected_units += len(ids)
                resolved_by_identity[(row["query_id"], row["document_id"])] = row
            except Exception as exc:
                selection_failures.append({"query_id": row["query_id"], "document_id": row["document_id"], "error": str(exc)})
    if selection_failures:
        raise RuntimeError(f"SELECTOR_RESOLUTION_FAILED:{len(selection_failures)}")
    resolved_rows = [resolved_by_identity[(row["query_id"], row["document_id"])] for row in missing_rows]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in resolved_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    work_sha = sha256(OUT)
    slot_counts = Counter(row["baseline_rank"] for row in missing_rows)
    current_slot4 = sum(1 for qid, _, rank, did in baseline_order if rank == 4 and (qid, did) in current_keys)
    current_slot5 = sum(1 for qid, _, rank, did in baseline_order if rank == 5 and (qid, did) in current_keys)
    missing_queries = {row["query_id"] for row in missing_rows}
    scored_by_query = Counter(qid for qid, _, _, did in baseline_order if (qid, did) in current_keys)
    report = {
        "status": "BGE_SLOT45_ANCHOR_WORKLIST_READY",
        "population": {"folds": [1, 2, 3, 4], "queries": 5600, "baseline_slot45_total": 11200, "fold0_used": False, "labels_used_to_form_worklist": False},
        "anchors": {"slot4_total": 5600, "slot5_total": 5600, "unique_total": len({(qid, did) for qid, _, _, did in baseline_order}), "duplicate_identity_anomalies": 11200 - len({(qid, did) for qid, _, _, did in baseline_order}), "already_current_bge": len(scored), "missing_current_bge": len(missing_rows), "already_slot4": current_slot4, "already_slot5": current_slot5, "missing_slot4": slot_counts[4], "missing_slot5": slot_counts[5], "queries_both_current": sum(1 for q in {x[0] for x in baseline_order} if scored_by_query[q] == 2), "queries_one_current": sum(1 for q in {x[0] for x in baseline_order} if scored_by_query[q] == 1), "queries_no_current": sum(1 for q in {x[0] for x in baseline_order} if scored_by_query[q] == 0)},
        "worklist": {"path": str(OUT.relative_to(ROOT)).replace("\\", "/"), "sha256": work_sha, "rows": len(missing_rows), "unique_queries": len(missing_queries), "unique_documents": len({row["document_id"] for row in missing_rows}), "expected_inference_units": expected_units, "selector": SELECTOR, "selected_chunks": "TOP3_UP_TO_AVAILABLE"},
        "local_resolution": {"required_documents": len({row["document_id"] for row in missing_rows}), "missing_documents": len(missing_documents), "unresolved_queries": 0, "selection_failures": 0},
        "current_artifact": {"path": str(CURRENT.relative_to(ROOT)).replace("\\", "/"), "sha256": CURRENT_SHA},
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
