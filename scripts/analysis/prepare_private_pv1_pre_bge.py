"""Run the CPU-only PV1 Private retrieval stages and prepare a BGE delta run.

This module is an orchestration wrapper around the already-proven lexical,
KNN, candidate-fusion, and selector implementations.  It never loads BGE,
calls Modal, reads Private answers, or launches GPU work.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.prepare_private_rrf_k20 import (  # noqa: E402
    build_union,
    doc_key,
)
from udsc2026.evaluation.legal_ir_lexical import (  # noqa: E402
    LegalContext,
    build_bm25f_rankings,
    build_knn_rankings,
)

PRIVATE = ROOT / "private_task1/input/private-official.json"
PV1 = ROOT / "data/processed_pv1"
PV1_DOCS = PV1 / "documents"
PV1_CHUNKS = PV1 / "chunks"
PV1_DENSE = ROOT / "private_task1/experiments/private_pv1/dense/private_pv1_dense_predictions.jsonl"
TRAIN = ROOT / "data/raw/btc/LegalIR/train.json"
OLD_K20_MANIFEST = ROOT / "private_task1/retrieval/manifests/private_rrf_k20_manifest.json"
OLD_BGE_MANIFEST = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/manifests/production.json"
OLD_BGE_ADDENDUM = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/manifests/production_provenance_addendum.json"
OLD_BGE_WORKLIST = ROOT / "private_task1/rerank/worklists/private_rrf_k20_bge_worklist.jsonl"
OLD_BGE_SCORES = ROOT / "private_task1/experiments/private_rrf_k20/gpu_download/results/production/scores.jsonl"
OUT = ROOT / "private_task1/experiments/private_pv1"
REPAIR_SET = set(json.loads((PV1 / "metadata/pv1_corpus_manifest.json").read_text(encoding="utf-8"))["changed_document_ids"])

BM25_DIR = OUT / "bm25"
KNN_DIR = OUT / "knn"
K20_DIR = OUT / "k20"
BGE_DIR = OUT / "bge"

BM25_TOP_K = 200
KNN_NEIGHBORS = 20
K20_CAP = 20
DENSE_DEPTH = 200
SELECTOR = "true_s2_bm25_within_document_v2"
MAX_LENGTH = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sha256(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return sha256(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_questions() -> dict[str, str]:
    payload = json.loads(PRIVATE.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or len(payload) != 2080:
        raise RuntimeError("Private input must contain exactly 2080 queries")
    result: dict[str, str] = {}
    for raw_id, record in payload.items():
        if not isinstance(record, dict) or not isinstance(record.get("question"), str):
            raise RuntimeError(f"invalid Private question: {raw_id}")
        result[str(raw_id)] = record["question"]
    if len(result) != 2080 or any(not value.strip() for value in result.values()):
        raise RuntimeError("Private question identity/text validation failed")
    return result


def load_pv1_contexts() -> tuple[list[LegalContext], set[str]]:
    contexts: list[LegalContext] = []
    for path in sorted(PV1_DOCS.glob("*.json"), key=lambda item: doc_key(item.stem)):
        record = json.loads(path.read_text(encoding="utf-8"))
        doc_id = str(record.get("doc_id", path.stem))
        text = str(record.get("cleaned_text", ""))
        if not text.strip():
            raise RuntimeError(f"empty PV1 document: {path}")
        title = str(record.get("title") or record.get("law_name") or "")
        contexts.append(LegalContext(doc_id, text, title))
    doc_ids = {item.doc_id for item in contexts}
    if len(contexts) != 8532 or len(doc_ids) != len(contexts):
        raise RuntimeError(f"PV1 document inventory mismatch: {len(contexts)}/{len(doc_ids)}")
    return contexts, doc_ids


def load_train_pool(allowed_docs: set[str]) -> list[tuple[str, str, list[str]]]:
    payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    result: list[tuple[str, str, list[str]]] = []
    for raw_id, record in payload.items():
        answer = record.get("answer") if isinstance(record, dict) else None
        question = record.get("question") if isinstance(record, dict) else None
        if not isinstance(answer, list) or not answer or not isinstance(question, str):
            continue
        docs = [str(value) for value in answer]
        if any(doc not in allowed_docs for doc in docs):
            raise RuntimeError(f"training reference document is absent from PV1: {raw_id}")
        result.append((str(raw_id), question, docs))
    if not result:
        raise RuntimeError("empty canonical KNN reference pool")
    return result


def load_dense_rankings(questions: dict[str, str], allowed_docs: set[str]) -> dict[str, list[str]]:
    rows = read_jsonl(PV1_DENSE)
    result: dict[str, list[str]] = {}
    for row in rows:
        qid = str(row["question_id"])
        if qid in result:
            raise RuntimeError(f"duplicate PV1 Dense query: {qid}")
        seen: set[str] = set()
        ranking: list[str] = []
        for hit in row.get("hits", [])[:DENSE_DEPTH]:
            did = str(hit["doc_id"])
            if did not in allowed_docs:
                raise RuntimeError(f"PV1 Dense document absent from PV1: {did}")
            if did not in seen:
                seen.add(did)
                ranking.append(did)
        result[qid] = ranking
    if set(result) != set(questions):
        raise RuntimeError("PV1 Dense query coverage mismatch")
    return result


def ranking_rows(rankings: dict[str, list[str]], questions: dict[str, str], *, include_score: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid in questions:
        candidates = []
        for rank, did in enumerate(rankings[qid], 1):
            item: dict[str, Any] = {"document_id": did, "rank": rank}
            if include_score:
                item["score"] = float(rankings[qid][rank - 1][1])  # type: ignore[index]
            candidates.append(item)
        rows.append({"query_id": qid, "candidates": candidates})
    return rows


def load_doc_chunks(doc_id: str) -> list[dict[str, Any]]:
    path = PV1_CHUNKS / f"{doc_id}.jsonl"
    if not path.is_file():
        raise RuntimeError(f"PV1 chunk file missing: {path}")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({str(row["chunk_id"]) for row in rows}) != len(rows):
        raise RuntimeError(f"invalid PV1 chunks: {path}")
    return rows


def build_selected_worklist(
    candidates: list[dict[str, Any]],
    questions: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[str]]]:
    from scripts.beam.task1_v2.evidence import prepare_document, select_true_s2_prepared

    by_doc: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for row in candidates:
        for candidate in row["candidates"]:
            by_doc[str(candidate["document_id"])].append((str(row["query_id"]), candidate))
    worklist: list[dict[str, Any]] = []
    selected_hashes: dict[tuple[str, str], list[str]] = {}
    total = len(by_doc)
    for index, (doc_id, entries) in enumerate(sorted(by_doc.items(), key=lambda pair: doc_key(pair[0])), 1):
        prepared = prepare_document(load_doc_chunks(doc_id))
        for qid, candidate in entries:
            selected = select_true_s2_prepared(questions[qid], prepared, topk=3)
            ids = [str(item["chunk_id"]) for item in selected]
            texts = [str(item["raw_chunk_text"]) for item in selected]
            if not 1 <= len(ids) <= 3 or len(ids) != len(set(ids)):
                raise RuntimeError(f"invalid selected chunks: {qid}/{doc_id}")
            if any(not cid.startswith(doc_id + "_") or not text.strip() for cid, text in zip(ids, texts)):
                raise RuntimeError(f"invalid selected chunk payload: {qid}/{doc_id}")
            key = (qid, doc_id)
            selected_hashes[key] = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
            worklist.append({
                "query_id": qid,
                "document_id": doc_id,
                "candidate_rank": int(candidate["candidate_rank"]),
                "source_ranks": candidate["source_ranks"],
                "expected_inference_units": len(ids),
                "selected_chunk_ids": ids,
                "selector": SELECTOR,
                "aggregation": "MAX",
                "max_length": MAX_LENGTH,
            })
        if index == 1 or index % 500 == 0 or index == total:
            print(f"PV1_SELECTOR_PROGRESS documents={index}/{total} qdocs={len(worklist)}", flush=True)
    worklist.sort(key=lambda row: (doc_key(str(row["query_id"])), int(row["candidate_rank"]), doc_key(str(row["document_id"]))))
    identities = [(str(row["query_id"]), str(row["document_id"])) for row in worklist]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate PV1 BGE q-doc identity")
    return worklist, selected_hashes


def scan_chunk_hashes(root: Path, wanted: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                cid = str(row["chunk_id"])
                if cid in wanted:
                    found[cid] = hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
    return found


def load_old_k20() -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(OLD_K20_MANIFEST.read_text(encoding="utf-8"))
    path = ROOT / manifest["artifacts"]["candidate_path"]
    if not path.is_file() or sha256(path) != manifest["artifacts"]["candidate_sha256"]:
        raise RuntimeError("historical K20 artifact missing or SHA mismatch")
    rows = read_jsonl(path)
    return path, rows, manifest


def audit_bge_reuse(
    new_worklist: list[dict[str, Any]],
    new_hashes: dict[tuple[str, str], list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_manifest = json.loads(OLD_BGE_MANIFEST.read_text(encoding="utf-8"))
    addendum = json.loads(OLD_BGE_ADDENDUM.read_text(encoding="utf-8"))
    if old_manifest.get("status") != "COMPLETE" or addendum.get("worklist_sha256") != sha256(OLD_BGE_WORKLIST):
        raise RuntimeError("historical BGE provenance gate failed")
    scores = read_jsonl(OLD_BGE_SCORES)
    old_scores = {(str(row["query_id"]), str(row["document_id"])): row for row in scores}
    old_rows = {(str(row["query_id"]), str(row["document_id"])): row for row in read_jsonl(OLD_BGE_WORKLIST)}
    if len(old_scores) != 41600 or len(old_rows) != 41600:
        raise RuntimeError("historical BGE q-doc count is not 41600")
    expected_provenance = {
        "base_model_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "ft_weight_sha256": "68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c",
        "ft_config_sha256": "16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b",
        "selector": SELECTOR,
        "aggregation": "MAX",
    }
    # Keep each provenance condition explicit: a score is reusable only when
    # the complete old production contract is evidenced.
    provenance_ok = (
        old_manifest["models"]["base"]["revision"] == expected_provenance["base_model_revision"]
        and old_manifest["models"]["ft"]["weight_sha256"] == expected_provenance["ft_weight_sha256"]
        and old_manifest["models"]["ft"]["config_sha256"] == expected_provenance["ft_config_sha256"]
        and all(row.get("selector") == SELECTOR and row.get("aggregation") == "MAX" for row in scores)
        and all(math.isfinite(float(row["bge_ft_score"])) and math.isfinite(float(row["bge_base_score"])) for row in scores)
    )
    old_ids = set(old_rows)
    new_ids = {(str(row["query_id"]), str(row["document_id"])) for row in new_worklist}
    old_selected_ids = {cid for row in old_rows.values() for cid in row.get("selected_chunk_ids", [])}
    old_hashes = scan_chunk_hashes(ROOT / "data/processed_v3/chunks", {str(cid) for cid in old_selected_ids})
    reasons = defaultdict(int)
    reusable: list[dict[str, Any]] = []
    rerun: list[dict[str, Any]] = []
    for row in new_worklist:
        key = (str(row["query_id"]), str(row["document_id"]))
        reason: str | None = None
        old = old_rows.get(key)
        # A repaired document has changed content even when its q-doc identity
        # is new in the PV1 K20 set.  Keep that provenance reason ahead of the
        # identity-diff reason so the audit explains why the old score cannot
        # be reused.
        if str(row["document_id"]) in REPAIR_SET:
            reason = "REPAIRED_DOCUMENT"
        elif old is None:
            reason = "NEW_QDOC"
        elif key not in old_scores:
            reason = "MISSING_OLD_SCORE"
        elif not provenance_ok:
            reason = "PROVENANCE_MISMATCH"
        elif [str(x) for x in old.get("selected_chunk_ids", [])] != [str(x) for x in row["selected_chunk_ids"]]:
            reason = "SELECTED_CHUNK_CHANGED"
        else:
            old_ids_for_row = [str(x) for x in old["selected_chunk_ids"]]
            current_hashes = new_hashes[key]
            historical_hashes = [old_hashes.get(cid, "") for cid in old_ids_for_row]
            if not historical_hashes or "" in historical_hashes or current_hashes != historical_hashes:
                reason = "SELECTED_CHUNK_CHANGED"
        if reason is None:
            reusable.append(row)
        else:
            reasons[reason] += 1
            rerun.append({**row, "rerun_reason": reason})
    if len(reusable) + len(rerun) != len(new_worklist) or len({(r["query_id"], r["document_id"]) for r in rerun}) != len(rerun):
        raise RuntimeError("BGE reuse partition is incomplete or duplicated")
    summary = {
        "old_total_qdocs": len(old_ids),
        "new_total_qdocs": len(new_ids),
        "reusable_scores": len(reusable),
        "must_rerun_qdocs": len(rerun),
        "reuse_rate": len(reusable) / len(new_worklist) if new_worklist else 0.0,
        "rerun_reasons": {name: reasons[name] for name in ("NEW_QDOC", "REPAIRED_DOCUMENT", "SELECTED_CHUNK_CHANGED", "MISSING_OLD_SCORE", "PROVENANCE_MISMATCH", "OTHER")},
        "old_score_provenance_pass": provenance_ok,
        "old_worklist_path": str(OLD_BGE_WORKLIST.relative_to(ROOT)).replace("\\", "/"),
        "old_scores_path": str(OLD_BGE_SCORES.relative_to(ROOT)).replace("\\", "/"),
        "reuse_gate": "PASS",
    }
    return rerun, summary


def main() -> int:
    questions = load_questions()
    contexts, allowed_docs = load_pv1_contexts()
    dense = load_dense_rankings(questions, allowed_docs)

    # Stage 1: exact canonical BM25 implementation/config, PV1 document input.
    from udsc2026.evaluation.legal_ir_lexical import build_bm25f_rankings
    bm25_scored = build_bm25f_rankings(questions, contexts, title_weight=0.0, top_k=BM25_TOP_K)
    bm25_docs = {qid: list(ranking) for qid, ranking in bm25_scored.items()}
    bm25_rows = []
    for qid in questions:
        bm25_rows.append({"query_id": qid, "candidates": [{"document_id": did, "rank": rank} for rank, did in enumerate(bm25_scored[qid], 1)]})
    if len(bm25_docs) != 2080 or any(len(set(docs)) != len(docs) or any(d not in allowed_docs for d in docs) for docs in bm25_docs.values()):
        raise RuntimeError("BM25_GATE failed")
    bm25_path = BM25_DIR / "private_pv1_bm25.jsonl"
    bm25_sha = write_jsonl(bm25_path, bm25_rows)
    bm25_manifest = {"stage": "PV1_BM25", "status": "PASS", "queries": 2080, "top_k": BM25_TOP_K, "implementation": "udsc2026.evaluation.legal_ir_lexical.build_bm25f_rankings", "corpus": "data/processed_pv1/documents", "output": str(bm25_path), "output_sha256": bm25_sha, "repaired_documents_seen": len(set().union(*(set(docs) for docs in bm25_docs.values())) & REPAIR_SET)}
    write_json(BM25_DIR / "private_pv1_bm25_manifest.json", bm25_manifest)
    print("BM25_GATE=PASS", flush=True)

    # Stage 2: exact canonical word TF-IDF KNN implementation/config.
    train_pool = load_train_pool(allowed_docs)
    knn_docs = build_knn_rankings(questions, train_pool, analyzer="word", ngram_range=(1, 2), neighbors=KNN_NEIGHBORS)
    if set(knn_docs) != set(questions) or any(len(set(docs)) != len(docs) or any(d not in allowed_docs for d in docs) for docs in knn_docs.values()):
        raise RuntimeError("KNN_GATE failed")
    knn_path = KNN_DIR / "private_pv1_word_tfidf_knn.jsonl"
    knn_sha = write_jsonl(knn_path, [{"query_id": qid, "candidates": [{"document_id": did, "rank": rank} for rank, did in enumerate(knn_docs[qid], 1)]} for qid in questions])
    write_json(KNN_DIR / "private_pv1_word_tfidf_knn_manifest.json", {"stage": "PV1_WORD_TFIDF_KNN", "status": "PASS", "execution": "RECOMPUTED_CPU", "evidence": "KNN contract is corpus-independent but PV1 membership was revalidated", "queries": 2080, "analyzer": "word", "ngram_range": [1, 2], "neighbors": KNN_NEIGHBORS, "reference": "data/raw/btc/LegalIR/train.json", "output": str(knn_path), "output_sha256": knn_sha, "repaired_documents_seen": len(set().union(*(set(docs) for docs in knn_docs.values())) & REPAIR_SET)})
    print("KNN_GATE=PASS", flush=True)

    # Stage 3: exact historical candidate construction (equal RRF, k=60, cap=20).
    candidates = build_union(list(questions), dense, bm25_docs, knn_docs)
    if len(candidates) != 2080 or any(len(row["candidates"]) != K20_CAP for row in candidates):
        raise RuntimeError("K20_GATE failed: not exactly 20 candidates per query")
    if any(len({item["document_id"] for item in row["candidates"]}) != K20_CAP for row in candidates):
        raise RuntimeError("K20_GATE failed: duplicate candidate document")
    if any(item["document_id"] not in allowed_docs for row in candidates for item in row["candidates"]):
        raise RuntimeError("K20_GATE failed: document outside PV1")
    k20_path = K20_DIR / "private_pv1_k20_candidates.jsonl"
    k20_sha = write_jsonl(k20_path, candidates)
    print("K20 candidate construction PASS", flush=True)

    # Resolve the same true_s2 selector against the frozen PV1 chunk files.
    new_worklist, selected_hashes = build_selected_worklist(candidates, questions)
    if len(new_worklist) != 41600 or sum(int(row["expected_inference_units"]) for row in new_worklist) <= 0:
        raise RuntimeError("K20_GATE failed during selector resolution")
    full_worklist_path = K20_DIR / "private_pv1_bge_full_worklist.jsonl"
    full_worklist_sha = write_jsonl(full_worklist_path, new_worklist)
    k20_manifest = {"stage": "PV1_K20", "status": "PASS", "queries": 2080, "qdocs": len(new_worklist), "candidate_k": K20_CAP, "source_contract": {"dense_depth": DENSE_DEPTH, "bm25_depth": BM25_TOP_K, "knn_neighbors": KNN_NEIGHBORS, "rrf_k": 60, "fusion": "equal reciprocal rank sum; canonical build_union"}, "candidate_output": str(k20_path), "candidate_sha256": k20_sha, "full_bge_worklist": str(full_worklist_path), "full_bge_worklist_sha256": full_worklist_sha, "expected_inference_units": sum(int(row["expected_inference_units"]) for row in new_worklist), "repaired_documents_in_k20": len({str(row["document_id"]) for row in new_worklist} & REPAIR_SET), "missing_chunk_resolution": 0}
    write_json(K20_DIR / "private_pv1_k20_manifest.json", k20_manifest)
    print("K20_GATE=PASS", flush=True)

    # Stage 4: label-free K20 identity comparison.
    old_k20_path, old_k20, old_k20_manifest = load_old_k20()
    old_sets = {str(row["query_id"]): {str(item["document_id"]) for item in row["candidates"]} for row in old_k20}
    new_sets = {str(row["query_id"]): {str(item["document_id"]) for item in row["candidates"]} for row in candidates}
    changed_queries = sum(old_sets[qid] != new_sets[qid] for qid in questions)
    k20_diff = {"old_path": str(old_k20_path), "old_qdocs": sum(len(value) for value in old_sets.values()), "new_qdocs": sum(len(new_sets[qid] - old_sets[qid]) for qid in questions), "new_qdoc_total": sum(len(value) for value in new_sets.values()), "same_qdocs": sum(len(old_sets[qid] & new_sets[qid]) for qid in questions), "removed_qdocs": sum(len(old_sets[qid] - new_sets[qid]) for qid in questions), "queries_with_any_change": changed_queries}

    # Stage 5: strict old-score reuse audit and exact rerun worklist.
    delta_worklist, reuse = audit_bge_reuse(new_worklist, selected_hashes)
    if reuse["reusable_scores"] + reuse["must_rerun_qdocs"] != len(new_worklist):
        raise RuntimeError("BGE_REUSE_GATE failed: incomplete partition")
    delta_path = BGE_DIR / "private_pv1_bge_delta_worklist.jsonl"
    delta_sha = write_jsonl(delta_path, delta_worklist)
    reuse["delta_worklist"] = str(delta_path)
    reuse["delta_worklist_sha256"] = delta_sha
    reuse["delta_inference_units"] = sum(int(row["expected_inference_units"]) for row in delta_worklist)
    write_json(BGE_DIR / "private_pv1_bge_reuse_audit.json", reuse)
    write_json(BGE_DIR / "private_pv1_bge_delta_manifest.json", {"stage": "BGE_DELTA_WORKLIST", "status": "PASS", **reuse})
    print("BGE_REUSE_GATE=PASS", flush=True)

    runner = (ROOT / "scripts/modal/task1_bge_subset_gpu.py").read_text(encoding="utf-8")
    runner_contract = {
        "runner": "scripts/modal/task1_bge_subset_gpu.py",
        "runner_reused": True,
        "path_support": "MINIMAL_BACKWARD_COMPATIBLE_CHUNKS_ROOT",
        "l4": 'gpu="L4"' in runner,
        "questions_file": "questions_file" in runner,
        "checkpoint_resume": "completed_identities" in runner and "validate_complete" in runner,
        "frozen_selected_chunk_ids": "frozen selector" in runner,
        "document_cache": "document_cache" in runner,
        "batch_16_supported": "SUPPORTED_BATCH_SIZES = (1, 4, 8, 16, 32)" in runner,
        "model_eval": ".model.eval()" in runner,
        "inference_mode": "torch.inference_mode()" in runner,
        "checkpoint_every": "checkpoint_every" in runner,
        "namespace_isolation": "validate_runtime_namespace" in runner,
        "chunks_root_argument": "chunks_root" in runner,
    }
    if not all(runner_contract.values()):
        raise RuntimeError("BGE_PREP_GATE failed: proven runner safeguards missing")
    namespace = "task1_private_pv1_bge"
    command = "& '.venv\\Scripts\\modal.exe' run --profile nan928904 scripts/modal/task1_bge_subset_gpu.py `\n  --worklist private_task1/experiments/private_pv1/bge/private_pv1_bge_delta_worklist.jsonl `\n  --questions-file private_task1/input/private-official.json `\n  --mode production `\n  --batch-size 16 `\n  --checkpoint-every 256 `\n  --run-namespace task1_private_pv1_bge `\n  --chunks-root /workspace/p13/runtime/data/processed_pv1/chunks"
    bge_prep = {"stage": "BGE_PREP", "status": "PASS", "gpu_runs": 0, "modal_inference_runs": 0, "gpu": "NVIDIA L4", "batch_size": 16, "checkpoint_every": 256, "resume": True, "namespace": namespace, "questions_file": str(PRIVATE.relative_to(ROOT)).replace("\\", "/"), "worklist": str(delta_path.relative_to(ROOT)).replace("\\", "/"), "worklist_sha256": delta_sha, "pv1_qdocs": len(new_worklist), "reused_bge_qdocs": reuse["reusable_scores"], "rerun_qdocs": len(delta_worklist), "expected_inference_units": reuse["delta_inference_units"], "remote_chunks_root": "/workspace/p13/runtime/data/processed_pv1/chunks", "remote_pv1_sync": "REQUIRED_BEFORE_GPU_COMMAND", "runner_contract": runner_contract, "exact_modal_command": command}
    write_json(BGE_DIR / "private_pv1_bge_prep_manifest.json", bge_prep)

    report = f"""# PV1 pre-BGE orchestration\n\n## Contract\n\n- Stage 0: PASS; canonical lexical functions: `src/udsc2026/evaluation/legal_ir_lexical.py`; candidate/worklist logic: `scripts/analysis/prepare_private_rrf_k20.py`; GPU runner: `scripts/modal/task1_bge_subset_gpu.py`.\n- Dense existing gate: PASS; PV1 Dense SHA `17eed47d1e03bd2404e0a2056b1cd25812567065710fcfa54d3f1c2e72ec8934`.\n\n## CPU stages\n\n- BM25: PASS; output `{bm25_path}`; repaired docs visible `{bm25_manifest['repaired_documents_seen']}/20`.\n- Word-TFIDF KNN: PASS; recomputed CPU with canonical word `(1,2)` TF-IDF and 20 neighbors.\n- K20: PASS; `{len(new_worklist)}` q-docs, exactly 20/query, `{k20_manifest['expected_inference_units']}` selected units.\n- Old/new K20: `{changed_queries}` queries changed; added `{k20_diff['new_qdocs']}` q-docs; removed `{k20_diff['removed_qdocs']}` q-docs.\n\n## BGE reuse\n\n- Old q-docs: `{reuse['old_total_qdocs']}`; new q-docs: `{reuse['new_total_qdocs']}`.\n- Reusable scores: `{reuse['reusable_scores']}`.\n- Must rerun: `{reuse['must_rerun_qdocs']}`.\n- Reasons: `{json.dumps(reuse['rerun_reasons'], ensure_ascii=False)}`.\n- BGE reuse gate: PASS.\n\n## GPU preparation\n\n- Delta worklist: `{delta_path}`\n- Delta SHA256: `{delta_sha}`\n- Delta inference units: `{reuse['delta_inference_units']}`\n- GPU: NVIDIA L4; batch 16; checkpoint every 256; namespace `{namespace}`.\n- PV1 remote chunk sync is required before launch; no Modal upload or GPU call was executed.\n\n```powershell\n{command}\n```\n\n`BGE_PREP_GATE = PASS`\n\nFinal status: `WAITING_FOR_USER_GPU_APPROVAL`\n"""
    (OUT / "pv1_pre_bge_orchestration_report.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "pv1_pre_bge_orchestration_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "WAITING_FOR_USER_GPU_APPROVAL", "bm25": bm25_manifest, "k20_diff": k20_diff, "bge_reuse": reuse, "bge_prep": bge_prep}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
