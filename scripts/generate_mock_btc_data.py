"""Generate a tiny, deterministic legal corpus for local ingestion tests.

The generated documents are intentionally synthetic.  They exercise Vietnamese
Unicode, common legal headings, clauses, points, JSON, and JSONL without
pretending to be an official competition dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_BASE_TEXT = """DỰ THẢO
BỘ LUẬT LAO ĐỘNG GIẢ LẬP 2026
Chương I. QUY ĐỊNH CHUNG
Mục 1. Phạm vi áp dụng
Điều 1. Phạm vi điều chỉnh
1. Bộ luật Lao động (sau đây gọi là BLLĐ) quy định quan hệ lao động.
a) Người lao động được bảo vệ quyền và lợi ích hợp pháp.
b) Người sử dụng lao động có trách nhiệm tuân thủ BLLĐ.
Điều 2. Nguyên tắc áp dụng
1. Việc hòa giải và uỷ quyền được thực hiện theo quy định của pháp luật.
a) Hồ sơ phải được lưu trữ đầy đủ.
"""


def _json_document(document_id: str, title: str, content: str) -> dict[str, Any]:
    """Build one reader-compatible JSON record."""
    return {
        "document_id": document_id,
        "title": title,
        "content": content,
        "metadata": {"dataset": "synthetic-development-only"},
    }


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one output atomically without following a pre-existing hard link."""

    if path.is_symlink():
        raise ValueError(f"mock output target must not be a symbolic link: {path}")
    if path.is_dir():
        raise ValueError(f"mock output target must not be a directory: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"mock output target must be a regular file: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        descriptor = -1
        with stream:
            stream.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def generate_mock_data(output_dir: str | Path) -> list[Path]:
    """Write TXT, JSON, and two-record JSONL examples into ``output_dir``.

    Returns:
        Paths of the three physical files written.  The JSONL file contains two
        logical documents, so ingestion produces four documents in total.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    txt_path = destination / "mock_btc_law.txt"
    json_path = destination / "mock_btc_law.json"
    jsonl_path = destination / "mock_btc_laws.jsonl"

    _atomic_write_text(txt_path, _BASE_TEXT)
    _atomic_write_text(
        json_path,
        json.dumps(
            _json_document(
                "mock-json-law",
                "Bộ luật Lao động giả lập 2026 (JSON)",
                _BASE_TEXT,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    jsonl_records = [
        _json_document(
            "mock-jsonl-law-1",
            "Bộ luật Lao động giả lập 2026 (JSONL 1)",
            _BASE_TEXT,
        ),
        _json_document(
            "mock-jsonl-law-2",
            "Bộ luật Lao động giả lập 2026 (JSONL 2)",
            _BASE_TEXT.replace("Điều 1.", "Điều 11.", 1),
        ),
    ]
    _atomic_write_text(
        jsonl_path,
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in jsonl_records
        ),
    )

    return [txt_path, json_path, jsonl_path]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a development-only mock legal corpus."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/btc/mock"),
        help="Destination directory (default: data/raw/btc/mock).",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the corpus from command-line arguments."""
    args = _parse_args()
    written = generate_mock_data(args.output_dir)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
