"""Build the label-free, deterministic 32-shard future Qwen worklist.

This script performs no model inference and never reads a relevance field.  It
projects the canonical baseline JSONL with field-specific regular expressions
so the embedded gold_documents value is neither decoded nor accessed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = ROOT / "reports/task1/full_document_legal_field_retrieval"
FULLDOC = REPORT_ROOT / "full_document_legal_field_retrieval_candidates.jsonl"
POOL = ROOT / "artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl"
BASELINE = ROOT / "artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl"
HISTORICAL = ROOT / "artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl"
CHUNKS = ROOT / "data/processed_v3/chunks"
OUT = REPORT_ROOT / "full_doc_top200_qwen_worklists"
MANIFEST = REPORT_ROOT / "full_doc_top200_qwen_future_universe_manifest.json"
HISTORICAL_WORKLIST = ROOT / "reports/task1/workflow_b/tv2/b2a/manifests/b2a_qwen_true_s2_top3_worklist.jsonl"
PARITY_SAMPLE = REPORT_ROOT / "full_doc_top200_qwen_historical_parity_256.jsonl"
CURRENT_SAMPLE = REPORT_ROOT / "full_doc_top200_qwen_current_canary_16.jsonl"
SHARDS = 32

EXPECTED = {
    "full_document_sha256": "3519a417cbeed46491e270f11a2b63707856cad594cda6c239e37a4a78204ab0",
    "historical_sha256": "65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f",
    "full_doc_new": 598071,
    "baseline_gaps": 122,
    "overlap": 1,
    "future_qdocs": 598192,
    "future_units": 1794571,
}

QUERY_RE = re.compile(r'"query_id"\s*:\s*"([^"]+)"')
FOLD_RE = re.compile(r'"fold"\s*:\s*(\d+)')
TOP5_RE = re.compile(r'"top5"\s*:\s*\[(.*?)\]')
STRING_RE = re.compile(r'"([^"]+)"')


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def natural(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def baseline_projection() -> set[tuple[str, str]]:
    """Extract only query/fold/top5 bytes; never decode gold_documents."""
    identities: set[tuple[str, str]] = set()
    with BASELINE.open(encoding="utf-8") as handle:
        for line in handle:
            query_match, fold_match, top5_match = QUERY_RE.search(line), FOLD_RE.search(line), TOP5_RE.search(line)
            if query_match is None or fold_match is None or top5_match is None:
                raise RuntimeError("baseline projection schema mismatch")
            if int(fold_match.group(1)) not in (1, 2, 3, 4):
                continue
            query = query_match.group(1)
            docs = STRING_RE.findall(top5_match.group(1))
            if len(docs) != 5:
                raise RuntimeError(f"baseline top5 mismatch: {query}")
            identities.update((query, doc) for doc in docs)
    if len(identities) != 28000:
        raise RuntimeError(f"baseline identity mismatch: {len(identities)}")
    return identities


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    temp = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temp.open("wb") as handle:
        for row in rows:
            raw = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(raw)
            digest.update(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    return digest.hexdigest()


def main() -> None:
    if sha256_file(FULLDOC) != EXPECTED["full_document_sha256"]:
        raise RuntimeError("frozen full-document SHA256 mismatch")
    if sha256_file(HISTORICAL) != EXPECTED["historical_sha256"]:
        raise RuntimeError("frozen historical Qwen SHA256 mismatch")

    pool = {
        str(row["query_id"]): {str(item["doc_id"]) for item in row["candidates"]}
        for row in jsonl(POOL)
    }
    future: dict[tuple[str, str], dict[str, Any]] = {}
    bands: Counter[str] = Counter()
    for row in jsonl(FULLDOC):
        query, doc, rank = str(row["query_id"]), str(row["document_id"]), int(row["rank"])
        if rank <= 200 and doc not in pool[query]:
            key = (query, doc)
            if key in future:
                raise RuntimeError(f"duplicate FULLDOC identity: {key}")
            future[key] = {"query_id": query, "document_id": doc, "full_doc_rank": rank, "sources": ["FULLDOC_TOP200_NEW"]}
            bands["1-20" if rank <= 20 else "21-50" if rank <= 50 else "51-100" if rank <= 100 else "101-200"] += 1
    if len(future) != EXPECTED["full_doc_new"]:
        raise RuntimeError(f"FULLDOC_TOP200_NEW mismatch: {len(future)}")

    historical_scores = {
        (str(row["query_id"]), str(item["doc_id"])): float(item["score"])
        for row in jsonl(HISTORICAL)
        for item in row["document_scores"]
    }
    gaps = baseline_projection() - set(historical_scores)
    if len(gaps) != EXPECTED["baseline_gaps"]:
        raise RuntimeError(f"baseline gap mismatch: {len(gaps)}")
    overlap = len(set(future) & gaps)
    if overlap != EXPECTED["overlap"]:
        raise RuntimeError(f"gap overlap mismatch: {overlap}")
    for query, doc in gaps:
        key = (query, doc)
        if key in future:
            future[key]["sources"].append("BASELINE_TOP5_QWEN_GAP")
        else:
            future[key] = {"query_id": query, "document_id": doc, "full_doc_rank": None, "sources": ["BASELINE_TOP5_QWEN_GAP"]}

    docs = sorted({doc for _, doc in future}, key=natural)
    chunk_counts: dict[str, int] = {}
    chunk_manifest_digest = hashlib.sha256()
    for doc in docs:
        path = CHUNKS / f"{doc}.jsonl"
        if not path.is_file():
            raise RuntimeError(f"missing canonical chunks: {doc}")
        count = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("raw_chunk_text", row.get("text", row.get("chunk_text", "")))
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError(f"empty canonical chunk: {doc}")
                count += 1
        if count < 1:
            raise RuntimeError(f"zero canonical chunks: {doc}")
        chunk_counts[doc] = min(3, count)
        chunk_manifest_digest.update(f"{doc}\t{sha256_file(path)}\n".encode("ascii"))

    ordered = [future[key] for key in sorted(future, key=lambda key: (natural(key[0]), natural(key[1])))]
    for row in ordered:
        row["expected_inference_units"] = chunk_counts[row["document_id"]]
    total_units = sum(int(row["expected_inference_units"]) for row in ordered)
    if len(ordered) != EXPECTED["future_qdocs"] or total_units != EXPECTED["future_units"]:
        raise RuntimeError(f"future universe mismatch: {len(ordered)}/{total_units}")

    OUT.mkdir(parents=True, exist_ok=True)
    shard_manifests = []
    for shard_id in range(SHARDS):
        start = shard_id * len(ordered) // SHARDS
        end = (shard_id + 1) * len(ordered) // SHARDS
        rows = ordered[start:end]
        path = OUT / f"shard_{shard_id:02d}.jsonl"
        shard_sha = atomic_jsonl(path, rows)
        shard_manifests.append({
            "shard_id": shard_id,
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "first_identity": [rows[0]["query_id"], rows[0]["document_id"]],
            "last_identity": [rows[-1]["query_id"], rows[-1]["document_id"]],
            "q_doc_count": len(rows),
            "inference_unit_count": sum(int(row["expected_inference_units"]) for row in rows),
            "worklist_sha256": shard_sha,
        })

    universe_digest = hashlib.sha256()
    for item in shard_manifests:
        universe_digest.update(f"{item['shard_id']:02d}\t{item['worklist_sha256']}\n".encode("ascii"))

    # Frozen label-free historical parity sample: first identities in each
    # persisted chunk-count stratum.  Expected scores come only from the frozen
    # corrected-Qwen prediction artifact, never from relevance labels.
    parity_groups: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    current_key: tuple[str, str] | None = None
    current_chunks: list[str] = []
    def accept_group() -> None:
        if current_key is None or len(current_chunks) not in parity_groups:
            return
        limit = 32 if len(current_chunks) < 3 else 192
        if len(parity_groups[len(current_chunks)]) >= limit:
            return
        if current_key not in historical_scores:
            raise RuntimeError(f"historical parity score missing: {current_key}")
        parity_groups[len(current_chunks)].append({
            "query_id": current_key[0], "document_id": current_key[1],
            "selected_chunk_ids": list(current_chunks),
            "expected_document_score": historical_scores[current_key],
        })
    with HISTORICAL_WORKLIST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line); key = (str(row["query_id"]), str(row["doc_id"]))
            if current_key is not None and key != current_key:
                accept_group(); current_chunks = []
                if len(parity_groups[1]) == 32 and len(parity_groups[2]) == 32 and len(parity_groups[3]) == 192:
                    break
            current_key = key; current_chunks.append(str(row["chunk_id"]))
        else:
            accept_group()
    if [len(parity_groups[k]) for k in (1, 2, 3)] != [32, 32, 192]:
        raise RuntimeError("historical parity stratum coverage mismatch")
    parity_rows = sorted(
        parity_groups[1] + parity_groups[2] + parity_groups[3],
        key=lambda row: (natural(row["query_id"]), natural(row["document_id"])),
    )
    parity_sha = atomic_jsonl(PARITY_SAMPLE, parity_rows)

    short_docs = {"125611", "176011", "198337"}
    current_rows = [row for row in ordered if row["document_id"] in short_docs]
    if len(current_rows) != 5:
        raise RuntimeError(f"current short-document q-doc mismatch: {len(current_rows)}")
    selected_keys = {(row["query_id"], row["document_id"]) for row in current_rows}
    def band(row: dict[str, Any]) -> str:
        rank = row["full_doc_rank"]
        return "1-20" if rank is not None and rank <= 20 else "21-50" if rank is not None and rank <= 50 else "51-100" if rank is not None and rank <= 100 else "101-200"
    for wanted in ("1-20", "21-50", "51-100", "101-200"):
        if not any(band(row) == wanted for row in current_rows):
            row = next(row for row in ordered if band(row) == wanted and (row["query_id"], row["document_id"]) not in selected_keys)
            current_rows.append(row); selected_keys.add((row["query_id"], row["document_id"]))
    for row in ordered:
        if len(current_rows) >= 16:
            break
        key = (row["query_id"], row["document_id"])
        if key not in selected_keys:
            current_rows.append(row); selected_keys.add(key)
    current_rows.sort(key=lambda row: (natural(row["query_id"]), natural(row["document_id"])))
    current_sha = atomic_jsonl(CURRENT_SAMPLE, current_rows)
    manifest = {
        "schema_version": "full_doc_top200_qwen_future_universe_v1",
        "status": "FROZEN_NOT_SCORED",
        "ordering": "natural(query_id), natural(document_id)",
        "sharding": "32 deterministic contiguous slices using floor(i*N/32)",
        "q_doc_count": len(ordered),
        "inference_unit_count": total_units,
        "full_doc_top200_new": EXPECTED["full_doc_new"],
        "baseline_top5_qwen_gaps": len(gaps),
        "overlap": overlap,
        "rank_bands": dict(bands),
        "unique_documents": len(docs),
        "canonical_chunk_source_sha256": chunk_manifest_digest.hexdigest(),
        "universe_sha256": universe_digest.hexdigest(),
        "sources": {
            str(FULLDOC.relative_to(ROOT)).replace("\\", "/"): sha256_file(FULLDOC),
            str(POOL.relative_to(ROOT)).replace("\\", "/"): sha256_file(POOL),
            str(BASELINE.relative_to(ROOT)).replace("\\", "/"): sha256_file(BASELINE),
            str(HISTORICAL.relative_to(ROOT)).replace("\\", "/"): sha256_file(HISTORICAL),
        },
        "labels_accessed": False,
        "canary_manifests": {
            "historical_parity_256": {"path": str(PARITY_SAMPLE.relative_to(ROOT)).replace("\\", "/"), "sha256": parity_sha},
            "current_universe_16": {"path": str(CURRENT_SAMPLE.relative_to(ROOT)).replace("\\", "/"), "sha256": current_sha},
        },
        "shards": shard_manifests,
    }
    temp = MANIFEST.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, MANIFEST)
    print(json.dumps({"status": manifest["status"], "q_docs": len(ordered), "units": total_units, "universe_sha256": manifest["universe_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
