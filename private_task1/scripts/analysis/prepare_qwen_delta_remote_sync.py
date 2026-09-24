"""Stage and verify the exact PV1 files needed by the Qwen delta preflight."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKLIST = ROOT / "private_task1/experiments/qwen_single_rescue/validation_delta_worklist.jsonl"
SOURCE_ROOT = ROOT / "data/processed_pv1/chunks"
SYNC_ROOT = ROOT / "private_task1/experiments/qwen_single_rescue/remote_sync"
STAGE_ROOT = SYNC_ROOT / "staged_chunks"
MANIFEST = SYNC_ROOT / "manifest.json"
EXPECTED_QDOCS = 184
EXPECTED_UNITS = 552
EXPECTED_WORKLIST_SHA = "c9384fde1338e9a6b850e473876a4ff819c8f8dbfe7d90d536493a5d1d9fc428"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"non-object JSONL at {path}:{line_no}")
                rows.append(value)
    return rows


def main() -> int:
    worklist_sha = sha256(WORKLIST)
    if worklist_sha != EXPECTED_WORKLIST_SHA:
        raise RuntimeError(f"worklist SHA mismatch: {worklist_sha}")
    rows = read_jsonl(WORKLIST)
    keys = {(str(row["query_id"]), str(row["document_id"])) for row in rows}
    if len(rows) != EXPECTED_QDOCS or len(keys) != EXPECTED_QDOCS:
        raise RuntimeError(f"q-doc mismatch: {len(rows)}/{EXPECTED_QDOCS}")
    document_ids = sorted({str(row["document_id"]) for row in rows}, key=lambda value: int(value))
    selected_by_doc: dict[str, set[str]] = {doc: set() for doc in document_ids}
    for row in rows:
        did = str(row["document_id"])
        selected = [str(value) for value in row["selected_chunk_ids"]]
        if len(selected) != int(row["expected_inference_units"]):
            raise RuntimeError(f"selected-unit mismatch: {row['query_id']}/{did}")
        selected_by_doc[did].update(selected)
    selected_units = sum(len(row["selected_chunk_ids"]) for row in rows)
    if len(document_ids) != 64 or selected_units != EXPECTED_UNITS:
        raise RuntimeError(f"document/unit mismatch: {len(document_ids)}/64, {selected_units}/552")
    if STAGE_ROOT.exists():
        existing = [path for path in STAGE_ROOT.iterdir()]
        if existing:
            raise RuntimeError(f"staging directory is not empty: {STAGE_ROOT}")
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)

    files = []
    for did in document_ids:
        source = SOURCE_ROOT / f"{did}.jsonl"
        target = STAGE_ROOT / source.name
        if not source.is_file():
            raise RuntimeError(f"missing source file: {source}")
        chunks = read_jsonl(source)
        by_id = {str(chunk.get("chunk_id", "")): chunk for chunk in chunks}
        for chunk_id in selected_by_doc[did]:
            if not chunk_id.startswith(did + "_") or chunk_id not in by_id:
                raise RuntimeError(f"selected chunk unresolved: {did}/{chunk_id}")
            text = str(by_id[chunk_id].get("text", by_id[chunk_id].get("chunk_text", "")))
            if not text.strip():
                raise RuntimeError(f"selected chunk text empty: {did}/{chunk_id}")
        try:
            os.link(source, target)
            method = "hardlink"
        except OSError:
            shutil.copy2(source, target)
            method = "byte_copy"
        source_sha = sha256(source)
        staged_sha = sha256(target)
        if source_sha != staged_sha:
            raise RuntimeError(f"source/stage SHA mismatch: {did}")
        files.append({
            "document_id": did,
            "filename": source.name,
            "source_path": str(source.relative_to(ROOT)).replace("\\", "/"),
            "staging_path": str(target.relative_to(ROOT)).replace("\\", "/"),
            "bytes": source.stat().st_size,
            "sha256": source_sha,
            "selected_chunk_count": len(selected_by_doc[did]),
            "staging_method": method,
        })

    staged_names = {path.name for path in STAGE_ROOT.iterdir() if path.is_file()}
    expected_names = {f"{did}.jsonl" for did in document_ids}
    missing = sorted(expected_names - staged_names)
    extra = sorted(staged_names - expected_names)
    if len(files) != 64 or missing or extra:
        raise RuntimeError(f"staging cardinality mismatch: {len(files)}/64 missing={missing} extra={extra}")
    manifest = {
        "schema_version": "qwen_validation_delta_remote_sync_v1",
        "required_document_count": 64,
        "document_ids": document_ids,
        "required_qdoc_count": EXPECTED_QDOCS,
        "selected_chunk_units": EXPECTED_UNITS,
        "worklist_path": str(WORKLIST.relative_to(ROOT)).replace("\\", "/"),
        "worklist_sha256": worklist_sha,
        "source_root": str(SOURCE_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "staging_root": str(STAGE_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "staged_files": len(files),
        "missing": missing,
        "extra": extra,
        "source_stage_sha_gate": "PASS",
        "files": files,
    }
    SYNC_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "LOCAL_REQUIRED_DOCUMENTS": "64/64",
        "LOCAL_SELECTED_CHUNKS": "552/552",
        "STAGED_DOCUMENTS": "64/64",
        "STAGING_SHA_GATE": "PASS",
        "MISSING": 0,
        "EXTRA": 0,
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
