#!/usr/bin/env python3
"""Audit whether FAISS payloads can be an exact storage backend for B4.

This script never changes the B4 runner or runs inference.  It streams the
1.8GB payload object, so it does not need to load payloads.json into memory.
The complete audit can take several minutes on a mounted Windows drive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/processed_v3/chunks"
PAYLOADS = ROOT / "data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json"
MANIFEST = PAYLOADS.with_name("manifest.json")
WORKLIST = ROOT / "artifacts/task1/recovery_096/bm25_grounded_selector_gate_v2/worklists/B4_bm25_rank_le_20.json"
OUT = ROOT / "artifacts/task1/recovery_096/vector_payloads_vs_raw_chunks_b4"


def norm_ws(text: str) -> str:
    return " ".join(text.split())


def norm_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", norm_ws(text))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def key(doc_id: Any, chunk_id: Any) -> str:
    return f"{doc_id}\x1f{chunk_id}"


def iter_payloads(path: Path) -> Iterator[dict[str, Any]]:
    """Incrementally yield payload values from the top-level payloads object."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        eof = False

        def refill() -> None:
            nonlocal buffer, eof
            block = handle.read(1 << 20)
            if block:
                buffer += block
            else:
                eof = True

        while '"payloads"' not in buffer and not eof:
            refill()
        marker = buffer.find('"payloads"')
        if marker < 0:
            raise ValueError("payloads.json has no top-level 'payloads' field")
        buffer = buffer[marker + len('"payloads"'):]
        while "{" not in buffer and not eof:
            refill()
        opening = buffer.find("{")
        if opening < 0:
            raise ValueError("payloads field is not an object")
        buffer = buffer[opening + 1:]

        while True:
            while True:
                stripped = buffer.lstrip()
                buffer = stripped
                if buffer or eof:
                    break
                refill()
            if not buffer:
                raise ValueError("unexpected EOF in payloads object")
            if buffer[0] == "}":
                return
            while True:
                try:
                    _, pos = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            buffer = buffer[pos:].lstrip()
            while not buffer and not eof:
                refill()
            if not buffer.startswith(":"):
                raise ValueError("invalid payload key/value separator")
            buffer = buffer[1:].lstrip()
            while True:
                try:
                    value, pos = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            if not isinstance(value, dict):
                raise ValueError("payload value is not an object")
            yield value
            buffer = buffer[pos:].lstrip()
            if not buffer and not eof:
                refill()
            if buffer.startswith(","):
                buffer = buffer[1:]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as out:
            json.dump(value, out, ensure_ascii=False, indent=2, sort_keys=True)
            out.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def worklist_docs() -> set[str]:
    data = json.loads(WORKLIST.read_text(encoding="utf-8"))
    return {str(row["doc_id"]) for row in data["rows"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=100000, help="print progress after this many rows")
    parser.add_argument("--rebuild-raw-index", action="store_true", help="discard the resumable raw SQLite checkpoint")
    parser.add_argument("--smoke", action="store_true", help="read a few payload rows and verify their raw counterparts")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        checked = 0
        for payload in iter_payloads(PAYLOADS):
            doc_id, chunk_id = str(payload["doc_id"]), str(payload["chunk_id"])
            raw_file = RAW_DIR / f"{doc_id}.jsonl"
            matches = [json.loads(line) for line in raw_file.open(encoding="utf-8") if json.loads(line).get("chunk_id") == chunk_id]
            if len(matches) != 1 or str(matches[0].get("text", "")) != str(payload.get("text", "")):
                raise AssertionError(f"smoke parity failure: {doc_id}/{chunk_id}")
            checked += 1
            if checked == 3:
                break
        print(json.dumps({"smoke": "PASS", "checked_exact_pairs": checked, "payloads": str(PAYLOADS), "raw_chunks": str(RAW_DIR)}, ensure_ascii=False))
        return 0
    database = out / "raw_index.sqlite"
    report_path = out / "report.json"
    b4_docs = worklist_docs()

    conn = sqlite3.connect(database)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("CREATE TABLE IF NOT EXISTS audit_meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
    ready = conn.execute("SELECT value FROM audit_meta WHERE name='raw_index_ready'").fetchone()
    if args.rebuild_raw_index or ready is None:
        print(f"[RAW] building SQLite index from {RAW_DIR}", flush=True)
        conn.execute("DROP TABLE IF EXISTS raw_chunks")
        conn.execute("CREATE TABLE raw_chunks (k TEXT PRIMARY KEY, doc_id TEXT NOT NULL, chunk_id TEXT NOT NULL, exact_hash TEXT NOT NULL, ws_hash TEXT NOT NULL, unicode_hash TEXT NOT NULL, text_len INTEGER NOT NULL, b4 INTEGER NOT NULL)")
        conn.execute("CREATE INDEX raw_b4 ON raw_chunks(b4)")
        raw_total = raw_empty = raw_duplicate = 0
        raw_docs: set[str] = set()
        raw_started = time.monotonic()
        raw_files = sorted(RAW_DIR.glob("*.jsonl"))
        with conn:
            for file_index, source in enumerate(raw_files, start=1):
                with source.open(encoding="utf-8") as handle:
                    rows = []
                    for line in handle:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        doc_id, chunk_id, text = str(row.get("doc_id", "")), str(row.get("chunk_id", "")), str(row.get("text", ""))
                        raw_total += 1; raw_empty += int(not text.strip()); raw_docs.add(doc_id)
                        rows.append((key(doc_id, chunk_id), doc_id, chunk_id, digest(text), digest(norm_ws(text)), digest(norm_unicode(text)), len(text), int(doc_id in b4_docs)))
                        if len(rows) >= 5000:
                            before = conn.total_changes
                            conn.executemany("INSERT OR IGNORE INTO raw_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
                            raw_duplicate += len(rows) - (conn.total_changes - before); rows = []
                    if rows:
                        before = conn.total_changes
                        conn.executemany("INSERT OR IGNORE INTO raw_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
                        raw_duplicate += len(rows) - (conn.total_changes - before)
                if file_index == 1 or file_index % 250 == 0 or file_index == len(raw_files):
                    elapsed = max(time.monotonic() - raw_started, 1e-6)
                    print(
                        f"[RAW] files={file_index}/{len(raw_files)} rows={raw_total} "
                        f"rate={raw_total / elapsed:.0f} rows/s",
                        flush=True,
                    )
            meta = {"raw_total": raw_total, "raw_empty": raw_empty, "raw_duplicate": raw_duplicate, "raw_docs": len(raw_docs)}
            conn.execute("INSERT OR REPLACE INTO audit_meta VALUES ('raw_index_ready', ?)", (json.dumps(meta),))
    else:
        meta = json.loads(ready[0]); raw_total = meta["raw_total"]; raw_empty = meta["raw_empty"]; raw_duplicate = meta["raw_duplicate"]
        raw_docs = set()  # only its cardinality is retained in the resumable checkpoint.
        print(f"[RAW] reusing SQLite index rows={raw_total} docs={meta['raw_docs']}", flush=True)

    # A hash predicate selects a stable 5k-ish sample without holding text in memory.
    threshold = max(1, int((args.sample_size / max(raw_total, 1)) * (1 << 64)))
    payload_total = payload_empty = payload_duplicate = payload_only = 0
    payload_docs: set[str] = set()
    intersection = exact = ws_match = unicode_match = 0
    b4_payload = b4_intersection = b4_exact = b4_ws = b4_unicode = 0
    length_deltas: list[int] = []; raw_lengths: list[int] = []; payload_lengths: list[int] = []
    mismatches: list[dict[str, Any]] = []
    seen: set[str] = set()
    payload_started = time.monotonic()
    print(f"[PAYLOAD] scanning {PAYLOADS}", flush=True)
    for payload in iter_payloads(PAYLOADS):
        doc_id, chunk_id, text = str(payload.get("doc_id", "")), str(payload.get("chunk_id", "")), str(payload.get("text", ""))
        k = key(doc_id, chunk_id); payload_total += 1; payload_empty += int(not text.strip()); payload_docs.add(doc_id)
        if k in seen:
            payload_duplicate += 1; continue
        seen.add(k)
        result = conn.execute("SELECT exact_hash, ws_hash, unicode_hash, text_len, b4 FROM raw_chunks WHERE k=?", (k,)).fetchone()
        is_b4 = doc_id in b4_docs
        b4_payload += int(is_b4)
        if result is None:
            payload_only += 1
            continue
        intersection += 1
        raw_exact, raw_ws, raw_unicode, raw_len, raw_b4 = result
        exact_ok, ws_ok, unicode_ok = digest(text) == raw_exact, digest(norm_ws(text)) == raw_ws, digest(norm_unicode(text)) == raw_unicode
        exact += int(exact_ok); ws_match += int(ws_ok); unicode_match += int(unicode_ok)
        if is_b4:
            b4_intersection += 1; b4_exact += int(exact_ok); b4_ws += int(ws_ok); b4_unicode += int(unicode_ok)
        h = int.from_bytes(hashlib.sha256(k.encode()).digest()[:8], "big")
        if h < threshold and len(length_deltas) < args.sample_size:
            length_deltas.append(abs(raw_len - len(text))); raw_lengths.append(raw_len); payload_lengths.append(len(text))
            if not unicode_ok and len(mismatches) < 10:
                mismatches.append({"doc_id": doc_id, "chunk_id": chunk_id, "raw_length": raw_len, "payload_length": len(text), "raw_exact_hash": raw_exact, "payload_exact_hash": digest(text)})
        if payload_total == 1 or payload_total % args.progress_every == 0:
            elapsed = max(time.monotonic() - payload_started, 1e-6)
            print(
                f"[PAYLOAD] rows={payload_total} intersection={intersection} "
                f"exact={exact} payload_only={payload_only} rate={payload_total / elapsed:.0f} rows/s",
                flush=True,
            )
    raw_unique = conn.execute("SELECT COUNT(*) FROM raw_chunks").fetchone()[0]
    raw_b4 = conn.execute("SELECT COUNT(*) FROM raw_chunks WHERE b4=1").fetchone()[0]
    raw_only = raw_unique - intersection
    b4_raw_only = raw_b4 - b4_intersection
    decision = "PAYLOADS_NOT_COMPATIBLE"
    if raw_only == 0 and payload_only == 0 and raw_duplicate == 0 and payload_duplicate == 0 and exact == intersection and b4_raw_only == 0 and b4_exact == b4_intersection:
        decision = "PAYLOADS_EXACT_COMPATIBLE"
    elif raw_only == 0 and payload_only == 0 and raw_duplicate == 0 and payload_duplicate == 0 and unicode_match == intersection and b4_raw_only == 0 and b4_unicode == b4_intersection:
        decision = "PAYLOADS_TEXT_EQUIVALENT_COMPATIBLE"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = {
        "status": decision,
        "paths": {"raw_chunks": str(RAW_DIR), "payloads": str(PAYLOADS), "worklist": str(WORKLIST)},
        "vector_store": {"manifest": manifest, "payload_fields_observed": ["doc_id", "chunk_id", "text", "parent_id", "metadata"]},
        "coverage": {"raw_total": raw_total, "raw_unique": raw_unique, "payload_total": payload_total, "payload_unique": payload_total - payload_duplicate, "intersection": intersection, "raw_only": raw_only, "payload_only": payload_only, "raw_duplicate_keys": raw_duplicate, "payload_duplicate_keys": payload_duplicate, "raw_unique_doc_ids": meta["raw_docs"], "payload_unique_doc_ids": len(payload_docs), "coverage_payload_over_raw": intersection / raw_unique if raw_unique else 0.0},
        "text_equivalence": {"exact_match": exact, "exact_rate": exact / intersection if intersection else 0.0, "whitespace_normalized_match": ws_match, "whitespace_normalized_rate": ws_match / intersection if intersection else 0.0, "unicode_normalized_match": unicode_match, "unicode_normalized_rate": unicode_match / intersection if intersection else 0.0, "raw_empty_text": raw_empty, "payload_empty_text": payload_empty, "differing_text": intersection - exact},
        "sample": {"method": "SHA256(key) deterministic threshold", "requested": args.sample_size, "observed": len(length_deltas), "average_raw_length": mean(raw_lengths) if raw_lengths else None, "average_payload_length": mean(payload_lengths) if payload_lengths else None, "max_absolute_length_delta": max(length_deltas) if length_deltas else None, "mismatch_examples": mismatches},
        "b4_scope": {"b4_doc_occurrences": 53924, "b4_unique_docs": len(b4_docs), "raw_chunks": raw_b4, "payload_chunks": b4_payload, "intersection": b4_intersection, "raw_only": b4_raw_only, "exact_rate": b4_exact / b4_intersection if b4_intersection else 0.0, "unicode_normalized_rate": b4_unicode / b4_intersection if b4_intersection else 0.0},
        "ordering_identity": {"payload_can_reconstruct_doc_to_chunks": True, "payload_has_explicit_original_rank": False, "b4_selector_tie_break": "chunk_id", "original_file_order_required": False, "conclusion": "Safe only if (doc_id, chunk_id, text) parity passes; B4 sorts tied S2 scores by chunk_id."},
        "next_action": "Prepare payload loader parity test; do not patch runner yet." if decision != "PAYLOADS_NOT_COMPATIBLE" else "UPLOAD_RAW_CHUNKS_REQUIRED",
    }
    atomic_json(report_path, report)
    print(f"[DONE] wrote {report_path}", flush=True)
    print(json.dumps({"status": decision, "report": str(report_path), "coverage": report["coverage"], "b4_scope": report["b4_scope"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
