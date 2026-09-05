"""Build a label-free LegalQA ensemble from extractive evidence and Qwen.

The fixed composition rule was selected on strict held-out data, but this
command never reads reference answers. Exact known-answer rows emitted by the
generator are preserved verbatim; all other rows receive a bounded extractive
prefix followed by the Qwen answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


def _load_predictions(path: Path, *, label: str) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{label} predictions must be a non-empty JSON array")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict) or set(row) != {"id", "answer"}:
            raise ValueError(f"{label} rows must contain exactly id and answer")
        question_id = row["id"]
        answer = row["answer"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"{label} contains an invalid question ID")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{label} contains an empty answer")
        if question_id in seen:
            raise ValueError(f"{label} contains duplicate ID {question_id!r}")
        seen.add(question_id)
        rows.append({"id": question_id, "answer": answer.strip()})
    return rows


def _load_exact_overlay_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("diagnostics must be a JSON array")
    exact_ids: set[str] = set()
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("diagnostic rows must be JSON objects")
        question_id = row.get("id")
        source = row.get("source")
        if not isinstance(question_id, str) or question_id in seen:
            raise ValueError("diagnostics contain an invalid or duplicate ID")
        if not isinstance(source, str):
            raise ValueError("diagnostics contain an invalid source")
        seen.add(question_id)
        if source == "exact_known_answer":
            exact_ids.add(question_id)
    return exact_ids


def build_ensemble(
    qwen_rows: Sequence[dict[str, str]],
    extractive_rows: Sequence[dict[str, str]],
    *,
    evidence_words: int,
    exact_overlay_ids: set[str] | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Return an aligned deterministic ensemble and compact build counts."""

    if evidence_words <= 0:
        raise ValueError("evidence_words must be positive")
    qwen_ids = [row["id"] for row in qwen_rows]
    extractive_ids = [row["id"] for row in extractive_rows]
    if qwen_ids != extractive_ids:
        raise ValueError("Qwen and extractive prediction IDs/order differ")
    protected = exact_overlay_ids or set()
    unknown = protected - set(qwen_ids)
    if unknown:
        raise ValueError(f"diagnostics contain unknown IDs: {sorted(unknown)[:5]}")
    output: list[dict[str, str]] = []
    truncated = 0
    for qwen, extractive in zip(qwen_rows, extractive_rows):
        question_id = qwen["id"]
        if question_id in protected:
            answer = qwen["answer"]
        else:
            words = extractive["answer"].split()
            prefix = " ".join(words[:evidence_words])
            truncated += int(len(words) > evidence_words)
            answer = f"{prefix}\n{qwen['answer']}".strip()
        output.append({"id": question_id, "answer": answer})
    return output, {
        "question_count": len(output),
        "exact_overlays_preserved": len(protected),
        "extractive_prefixes_truncated": truncated,
        "evidence_words": evidence_words,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--extractive", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--evidence-words", type=int, default=352)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    qwen = _load_predictions(args.qwen, label="Qwen")
    extractive = _load_predictions(args.extractive, label="extractive")
    predictions, counts = build_ensemble(
        qwen,
        extractive,
        evidence_words=args.evidence_words,
        exact_overlay_ids=_load_exact_overlay_ids(args.diagnostics),
    )
    _write_json(args.output, predictions)
    manifest = args.output.with_name(f"{args.output.stem}.manifest.json")
    _write_json(
        manifest,
        {
            "schema_version": "task2-qwen-extractive-ensemble-v1",
            **counts,
            "output": str(args.output),
            "output_sha256": _sha256(args.output),
            "qwen": str(args.qwen),
            "extractive": str(args.extractive),
            "diagnostics": str(args.diagnostics) if args.diagnostics else None,
            "label_free": True,
        },
    )
    return args.output, manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        written = run(args)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Task 2 ensemble error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
