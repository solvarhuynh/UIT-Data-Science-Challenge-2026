"""Stream raw chunk predictions into a deterministic document candidate cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collapse(row: dict, *, document_depth: int, evidence_limit: int) -> dict:
    groups: dict[str, list[dict]] = {}
    for raw_rank, raw in enumerate(row.get("hits", []), 1):
        item = raw if isinstance(raw, dict) else {"doc_id": raw}
        doc_id = str(item.get("doc_id", item.get("document_id", ""))).strip()
        if not doc_id:
            continue
        groups.setdefault(doc_id, []).append({**item, "_raw_rank": raw_rank})
    docs = []
    for doc_id, items in groups.items():
        items.sort(key=lambda item: (int(item.get("rank", item["_raw_rank"])), item["_raw_rank"]))
        best = items[0]
        evidence = []
        seen = set()
        for item in items:
            evidence_id = str(item.get("chunk_id", item.get("evidence_id", ""))).strip()
            text = str(item.get("text", item.get("evidence", ""))).strip()
            if text and (evidence_id, text) not in seen:
                evidence.append({"evidence_id": evidence_id or f"{doc_id}:chunk:{len(evidence)+1}", "text": text})
                seen.add((evidence_id, text))
            if len(evidence) == evidence_limit:
                break
        docs.append({
            "doc_id": doc_id,
            "rank": int(best.get("rank", best["_raw_rank"])),
            "dense_score": best.get("dense_score", best.get("score")),
            "law_name": best.get("law_name", ""),
            "evidence": evidence,
        })
    docs.sort(key=lambda item: (item["rank"], item["doc_id"]))
    return {
        "question_id": str(row.get("question_id", row.get("id", ""))),
        "raw_chunk_k": len(row.get("hits", [])),
        "document_depth": document_depth,
        "documents": docs[:document_depth],
        "collapse_version": "task1-document-collapse-v1",
        "evidence_limit": evidence_limit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--document-depth", type=int, default=200)
    parser.add_argument("--evidence-limit", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.document_depth <= 0:
        raise SystemExit("--document-depth must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open(encoding="utf-8-sig") as source, args.output.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            if line.strip():
                target.write(json.dumps(collapse(json.loads(line), document_depth=args.document_depth, evidence_limit=args.evidence_limit), ensure_ascii=False) + "\n")
                count += 1
    manifest = {
        "schema_version": "task1-document-candidates-v1",
        "raw_predictions": str(args.input),
        "raw_predictions_sha256": _sha256(args.input),
        "output": str(args.output),
        "document_depth": args.document_depth,
        "evidence_limit": args.evidence_limit,
        "query_count": count,
        "collapse_version": "task1-document-collapse-v1",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
