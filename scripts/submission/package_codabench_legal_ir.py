"""Package LegalIR using the array-shaped submission.json shown by BTC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_items(path: Path) -> list[dict[str, object]]:
    """Load and validate the screenshot-defined [{id, documents}] payload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("input must be a non-empty JSON array")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != {"id", "documents"}:
            raise ValueError(f"item {index} must contain exactly id and documents")
        question_id = item["id"]
        documents = item["documents"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"item {index}.id must be a non-empty string")
        if question_id in seen:
            raise ValueError(f"duplicate question id: {question_id}")
        if not isinstance(documents, list) or not all(
            isinstance(document, str) and document.strip() for document in documents
        ):
            raise ValueError(f"item {index}.documents must be an array of strings")
        if len(documents) > 5:
            raise ValueError(f"item {index}.documents exceeds BTC top-5 limit")
        if len(documents) != len(set(documents)):
            raise ValueError(f"item {index}.documents contains duplicates")
        seen.add(question_id)
        result.append({"id": question_id, "documents": documents})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.zip"))
    args = parser.parse_args()
    try:
        items = load_items(args.input)
        encoded = (json.dumps(items, ensure_ascii=False, indent=2) + "\n").encode(
            "utf-8"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(args.output, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("submission.json", encoded)
        print(f"Created {args.output}: submission.json, {len(items)} questions")
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"LegalIR Codabench packaging error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
