"""Prepare the frozen label-free Private RETRIEVAL_RRF_NO_LABEL worklist.

CPU/storage preparation only.  This script never reads ``answer`` from the
Private input, never loads a neural model, and never calls Modal.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
PRIVATE_INPUT = ROOT / "private_task1/input/private-official.json"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
CONTEXTS = ROOT / "data/raw/btc/LegalIR/selected-contexts"
CHUNKS = ROOT / "data/processed_v3/chunks"
DENSE = ROOT / "private_task1/retrieval/candidates/private_dense_predictions.jsonl"
OUT_CANDIDATES = ROOT / "private_task1/retrieval/candidates/private_rrf_k20_candidates.jsonl"
OUT_WORKLIST = ROOT / "private_task1/rerank/worklists/private_rrf_k20_bge_worklist.jsonl"
OUT_MANIFEST = ROOT / "private_task1/retrieval/manifests/private_rrf_k20_manifest.json"

SOURCE_NAMES = ("dense", "bm25", "knn_word")
RRF_K = 60
CAP = 20
DENSE_DEPTH = 200
BM25_DEPTH = 200
KNN_NEIGHBORS = 20
MAX_LENGTH = 512
SELECTOR = "true_s2_bm25_within_document_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def doc_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_private_questions() -> dict[str, str]:
    payload = json.loads(PRIVATE_INPUT.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 2080:
        raise RuntimeError("Private input must contain exactly 2080 queries")
    result = {}
    for raw_id, row in payload.items():
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            raise RuntimeError(f"invalid Private query {raw_id}")
        if row.get("answer") is not None:
            raise RuntimeError(f"Private answer field is not null: {raw_id}")
        result[str(raw_id)] = row["question"]
    return result


def load_contexts():
    from udsc2026.evaluation.legal_ir_lexical import LegalContext

    contexts = []
    for path in sorted(CONTEXTS.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8-sig"))
        contexts.append(
            LegalContext(
                str(row["id"]),
                str(row["passage"]),
                str(row.get("title", row.get("name", "")) or ""),
            )
        )
    if not contexts:
        raise RuntimeError("no selected-contexts found")
    return sorted(contexts, key=lambda item: doc_key(item.doc_id))


def load_dense_rankings(query_ids: list[str], allowed_docs: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with DENSE.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or row.get("query_id"))
            if qid in result:
                raise RuntimeError(f"duplicate dense query {qid}")
            seen = set()
            docs = []
            for hit in row.get("hits", [])[:DENSE_DEPTH]:
                did = str(hit.get("doc_id") or hit.get("document_id"))
                if did not in allowed_docs:
                    raise RuntimeError(f"dense unknown document {did} at line {line_number}")
                if did not in seen:
                    seen.add(did)
                    docs.append(did)
            result[qid] = docs
    if set(result) != set(query_ids):
        raise RuntimeError("dense query coverage mismatch")
    return result


def load_train_pool(allowed_docs: set[str]):
    result = []
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    for raw_id, row in train.items():
        answer = row.get("answer")
        if not isinstance(answer, list) or not answer:
            continue
        docs = [str(value) for value in answer]
        if any(doc not in allowed_docs for doc in docs):
            raise RuntimeError(f"train KNN label references unknown document: {raw_id}")
        result.append((str(raw_id), str(row["question"]), docs))
    if not result:
        raise RuntimeError("empty labelled KNN pool")
    return result


def build_union(query_ids, dense, bm25, knn):
    rows = []
    for qid in query_ids:
        sources = {
            "dense": dense[qid][:DENSE_DEPTH],
            "bm25": bm25[qid][:BM25_DEPTH],
            "knn_word": knn[qid][:KNN_NEIGHBORS],
        }
        source_ranks: dict[str, dict[str, int]] = {}
        scores: dict[str, float] = {}
        for source in SOURCE_NAMES:
            for rank, did in enumerate(sources[source], 1):
                source_ranks.setdefault(did, {})[source] = rank
                scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(
            scores,
            key=lambda did: (
                -scores[did],
                *(source_ranks[did].get(source, 10**9) for source in SOURCE_NAMES),
                doc_key(did),
            ),
        )[:CAP]
        rows.append(
            {
                "query_id": qid,
                "candidates": [
                    {
                        "document_id": did,
                        "candidate_rank": rank,
                        "source_ranks": source_ranks[did],
                    }
                    for rank, did in enumerate(ordered, 1)
                ],
            }
        )
    return rows


def load_chunks(doc_id: str) -> list[dict[str, Any]]:
    path = CHUNKS / f"{doc_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = load_jsonl(path)
    if not rows or len({str(row.get("chunk_id")) for row in rows}) != len(rows):
        raise RuntimeError(f"invalid chunks: {path}")
    return rows


def resolve_document(task):
    from scripts.beam.task1_v2.evidence import prepare_document, select_true_s2_prepared

    doc_id, entries = task
    prepared = prepare_document(load_chunks(doc_id))
    result = []
    for qid, question, candidate in entries:
        selected = select_true_s2_prepared(question, prepared, topk=3)
        ids = [str(item["chunk_id"]) for item in selected]
        texts = [str(item["raw_chunk_text"]) for item in selected]
        if not 1 <= len(ids) <= 3 or len(ids) != len(set(ids)):
            raise RuntimeError(f"invalid selected chunks {qid}/{doc_id}")
        if any(not cid.startswith(doc_id + "_") or not text.strip() for cid, text in zip(ids, texts)):
            raise RuntimeError(f"invalid selected chunk payload {qid}/{doc_id}")
        result.append(
            {
                "query_id": qid,
                "document_id": doc_id,
                "candidate_rank": int(candidate["candidate_rank"]),
                "source_ranks": candidate["source_ranks"],
                "expected_inference_units": len(ids),
                "selected_chunk_ids": ids,
                "selector": SELECTOR,
                "aggregation": "MAX",
                "max_length": MAX_LENGTH,
            }
        )
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return sha256(path)


def main() -> None:
    questions = load_private_questions()
    contexts = load_contexts()
    allowed_docs = {item.doc_id for item in contexts}
    dense = load_dense_rankings(list(questions), allowed_docs)
    train_pool = load_train_pool(allowed_docs)

    from udsc2026.evaluation.legal_ir_lexical import build_bm25f_rankings, build_knn_rankings

    bm25 = build_bm25f_rankings(questions, contexts, title_weight=0.0, top_k=BM25_DEPTH)
    knn = build_knn_rankings(questions, train_pool, analyzer="word", ngram_range=(1, 2), neighbors=KNN_NEIGHBORS)
    candidates = build_union(list(questions), dense, bm25, knn)
    if len(candidates) != len(questions) or any(len(row["candidates"]) > CAP for row in candidates):
        raise RuntimeError("candidate union coverage/cap mismatch")
    if any(len({item["document_id"] for item in row["candidates"]}) != len(row["candidates"]) for row in candidates):
        raise RuntimeError("duplicate candidate document")
    candidate_sha = write_jsonl(OUT_CANDIDATES, candidates)

    by_doc: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    query_map = questions
    for row in candidates:
        qid = str(row["query_id"])
        for candidate in row["candidates"]:
            did = str(candidate["document_id"])
            by_doc[did].append((qid, query_map[qid], candidate))
    tasks = sorted(by_doc.items(), key=lambda pair: doc_key(pair[0]))
    worklist_rows: list[dict[str, Any]] = []
    total_tasks = len(tasks)
    for index, task in enumerate(tasks, 1):
        worklist_rows.extend(resolve_document(task))
        if index == 1 or index % 500 == 0 or index == total_tasks:
            print(f"PRIVATE_WORKLIST_PROGRESS documents={index}/{total_tasks} qdocs={len(worklist_rows)}", flush=True)
    worklist_rows.sort(key=lambda row: (doc_key(str(row["query_id"])), int(row["candidate_rank"]), doc_key(str(row["document_id"]))))
    expected_units = sum(int(row["expected_inference_units"]) for row in worklist_rows)
    if len(worklist_rows) != sum(len(row["candidates"]) for row in candidates):
        raise RuntimeError("worklist row count mismatch")
    worklist_sha = write_jsonl(OUT_WORKLIST, worklist_rows)
    manifest = {
        "status": "PRIVATE_RRF_K20_CPU_PREPARED",
        "policy": {
            "name": "RETRIEVAL_RRF_NO_LABEL",
            "weights": {"dense": 0.2, "bge": 0.3, "knn_word": 0.2, "bm25": 0.3},
            "rrf_k": 2,
            "formula": "sum(weight/(2+rank)); absent source contributes 0",
        },
        "population": {"queries": len(questions), "candidate_k": CAP, "qdocs": len(worklist_rows), "unique_documents": len(by_doc)},
        "sources": {
            "dense": {"path": str(DENSE.relative_to(ROOT)).replace("\\", "/"), "reused": True, "depth": DENSE_DEPTH},
            "bm25": {"producer": "udsc2026.evaluation.legal_ir_lexical.build_bm25f_rankings", "top_k": BM25_DEPTH, "title_weight": 0.0},
            "knn_word": {"producer": "udsc2026.evaluation.legal_ir_lexical.build_knn_rankings", "analyzer": "word", "ngram_range": [1, 2], "neighbors": KNN_NEIGHBORS, "label_pool": "train.json only"},
        },
        "contract": {"base_model": "BAAI/bge-reranker-v2-m3", "base_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "selector": SELECTOR, "aggregation": "MAX", "max_length": MAX_LENGTH, "batch_size": 16, "checkpoint_every": 256},
        "artifacts": {"candidate_path": str(OUT_CANDIDATES.relative_to(ROOT)).replace("\\", "/"), "candidate_sha256": candidate_sha, "worklist_path": str(OUT_WORKLIST.relative_to(ROOT)).replace("\\", "/"), "worklist_sha256": worklist_sha},
        "worklist": {"expected_inference_units": expected_units, "duplicate_qdocs": len(worklist_rows) - len({(r['query_id'], r['document_id']) for r in worklist_rows}), "unresolved_qdocs": 0, "unresolved_chunks": 0},
        "private_answers_read": False,
        "gpu_runs": 0,
        "modal_inference_runs": 0,
    }
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
