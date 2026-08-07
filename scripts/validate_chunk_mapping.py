"""Streaming validator for TV4 JSONL chunks and LegalIR hierarchical mapping."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from udsc2026.contracts import LegalChunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", default="data/processed/chunks")
    parser.add_argument("--ir-train-file")
    parser.add_argument(
        "--report", default="data/processed/metadata/chunk_mapping_report.json"
    )
    parser.add_argument(
        "--audit-report", default="data/processed/metadata/chunk_file_audit.json"
    )
    return parser.parse_args()


def iter_chunks(directory: Path) -> Iterator[tuple[Path, int, LegalChunk | None, str | None]]:
    """Yield one parsed record at a time; malformed lines are not retained."""
    for path in sorted(directory.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                try:
                    chunk = LegalChunk.model_validate(json.loads(raw))
                    yield path, line_number, chunk, None
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    yield path, line_number, None, str(exc)


def load_gold(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("LegalIR train.json root must be an object")
    contexts: list[str] = []
    for question_id, record in payload.items():
        if not isinstance(record, dict) or not isinstance(record.get("answer"), list):
            raise ValueError(f"{question_id}: expected answer: [context_id]")
        for context_id in record["answer"]:
            if not isinstance(context_id, str) or not context_id.strip():
                raise ValueError(f"{question_id}: context_id must be a non-empty string")
            contexts.append(context_id)
    return contexts


def main() -> int:
    args = parse_args()
    seen: dict[str, list[dict[str, Any]]] = {}
    mapped_keys: set[str] = set()
    errors: list[dict[str, Any]] = []
    empty_text: list[str] = []
    missing_source_or_doc_id: list[str] = []
    empty_files: list[str] = []
    no_record_files: list[str] = []
    malformed_files: dict[str, list[dict[str, Any]]] = {}
    invalid_schema_files: dict[str, list[dict[str, Any]]] = {}
    total = 0
    files = 0
    directory = Path(args.chunks_dir)

    for path in sorted(directory.glob("*.jsonl")):
        files += 1
        if path.stat().st_size == 0:
            empty_files.append(path.name)
            continue

        file_records = 0
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                file_records += 1
                total += 1
                try:
                    payload = json.loads(raw)
                except (json.JSONDecodeError, TypeError) as exc:
                    errors.append({"file": str(path), "line": line_number, "error": str(exc)})
                    malformed_files.setdefault(path.name, []).append({"line": line_number, "error": str(exc)})
                    continue

                try:
                    chunk = LegalChunk.model_validate(payload)
                except (TypeError, ValueError) as exc:
                    errors.append({"file": str(path), "line": line_number, "error": str(exc)})
                    invalid_schema_files.setdefault(path.name, []).append({"line": line_number, "error": str(exc)})
                    continue

                seen.setdefault(chunk.chunk_id, []).append({"file": str(path), "line": line_number})
                mapped_keys.add(chunk.chunk_id)
                if chunk.parent_id:
                    mapped_keys.add(chunk.parent_id)
                if not chunk.text.strip():
                    empty_text.append(chunk.chunk_id)
                if not chunk.doc_id or not chunk.source:
                    missing_source_or_doc_id.append(chunk.chunk_id)

        if file_records == 0:
            no_record_files.append(path.name)

        if total % 100_000 == 0:
            print(f"Processed {total:,} chunks", flush=True)

    duplicates = {key: locations for key, locations in seen.items() if len(locations) > 1}
    gold = load_gold(Path(args.ir_train_file)) if args.ir_train_file else []
    orphan = sorted({context for context in gold if context not in mapped_keys and not any(
        chunk_id.startswith(context) for chunk_id in seen
    )})
    report = {
        "total_records": total,
        "valid_records": sum(len(value) for value in seen.values()),
        "parse_errors": errors,
        "duplicate_chunk_id": duplicates,
        "empty_text": sorted(set(empty_text)),
        "missing_source_or_doc_id": sorted(set(missing_source_or_doc_id)),
        "ground_truth_context_count": len(gold),
        "orphan_ground_truth_context_id": orphan,
        "mapping_rule": ["chunk.parent_id == answer.context_id", "chunk.chunk_id.startswith(answer.context_id)"],
        "pass": not duplicates and not orphan,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit_report = {
        "chunks_dir": str(directory),
        "file_count": files,
        "record_count": total,
        "empty_files": empty_files,
        "no_record_files": no_record_files,
        "malformed_json_files": malformed_files,
        "invalid_schema_files": invalid_schema_files,
        "pass": not (empty_files or no_record_files or malformed_files or invalid_schema_files),
    }
    audit_output = Path(args.audit_report)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report["audit_report"] = str(audit_output)
    report["audit_summary"] = {
        "empty_files": len(audit_report["empty_files"]),
        "no_record_files": len(audit_report["no_record_files"]),
        "malformed_json_files": len(audit_report["malformed_json_files"]),
        "invalid_schema_files": len(audit_report["invalid_schema_files"]),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"duplicate chunk_id: {'FAIL' if duplicates else 'PASS'}")
    print(f"empty text: {'FAIL' if empty_text else 'PASS'}")
    print(f"missing source/doc_id: {'FAIL' if missing_source_or_doc_id else 'PASS'}")
    print(f"orphan ground-truth: {'FAIL' if orphan else 'PASS'}")
    print(f"Audit report: {audit_output}")
    print(f"Mapping report: {output}")
    return 1 if duplicates or orphan else 0


if __name__ == "__main__":
    raise SystemExit(main())
