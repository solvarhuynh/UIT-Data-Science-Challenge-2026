"""Sinh corpus BTC giả lập an toàn để chạy thử ETL.

Corpus chỉ phục vụ phát triển, không phải dữ liệu BTC thật. Đường dẫn mặc định
được neo theo repository nên lệnh hoạt động ổn định ở mọi thư mục hiện hành.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "raw" / "btc" / "mock"

_BASE_TEXT = """DỰ THẢO
BỘ LUẬT LAO ĐỘNG GIẢ LẬP 2026
Chương I. QUY ĐỊNH CHUNG
Mục 1. Phạm vi áp dụng
Điều 1. Phạm vi điều chỉnh
1. Bộ luật Lao động (sau đây gọi là BLLĐ) quy định quan hệ lao động.
a) Người lao động được bảo vệ quyền và lợi ích hợp pháp.
b) Người sử dụng lao động có trách nhiệm tuân thủ BLLĐ.
Điều 2. Nguyên tắc áp dụng
1. Việc hoà giải và uý quyền được thực hiện theo quy định của pháp luật.
a) Hồ sơ phải được lưu trữ đầy đủ.
"""


def legal_text() -> str:
    """Return one legal sample containing the supported ETL edge cases."""

    return _BASE_TEXT


def _json_document(document_id: str, title: str, content: str) -> dict[str, Any]:
    """Build one reader-compatible JSON record."""

    return {
        "id": document_id,
        "document_id": document_id,
        "title": title,
        "content": content,
        "effective_date": "2026-01-01",
        "metadata": {"dataset": "synthetic-development-only"},
    }


def _validate_output_target(path: Path) -> None:
    """Reject output targets that cannot be replaced as regular files."""

    if path.is_symlink():
        raise ValueError(f"mock output target must not be a symbolic link: {path}")
    if path.is_dir():
        raise ValueError(f"mock output target must not be a directory: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"mock output target must be a regular file: {path}")


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one text output atomically without following an existing link."""

    _validate_output_target(path)
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


def _write_text(path: Path, text: str) -> Path:
    """Write a UTF-8 text document atomically."""

    _atomic_write_text(path, text)
    return path


def _write_json(path: Path, value: object) -> Path:
    """Write one indented JSON document atomically."""

    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(path, content)
    return path


def _write_jsonl(path: Path, text: str) -> Path:
    """Write two reader-compatible JSONL documents atomically."""

    records = [
        _json_document(
            "mock-jsonl-law-1",
            "Bộ luật Lao động giả lập 2026 (JSONL 1)",
            text,
        ),
        _json_document(
            "mock-jsonl-law-2",
            "Bộ luật Lao động giả lập 2026 (JSONL 2)",
            text.replace("Điều 1.", "Điều 11.", 1),
        ),
    ]
    content = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    _atomic_write_text(path, content)
    return path


def _write_docx_if_available(path: Path, text: str) -> Path | None:
    """Write an optional DOCX atomically when python-docx is installed."""

    try:
        from docx import Document
    except ImportError:
        return None

    _validate_output_target(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".docx",
        dir=str(path.parent),
    )
    os.close(descriptor)
    try:
        document = Document()
        for line in text.splitlines():
            document.add_paragraph(line)
        document.save(temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return path


def generate_mock_data(output_dir: str | Path | None = None) -> list[Path]:
    """Create TXT, JSON, JSONL and, when available, DOCX mock files.

    Only four known mock filenames are replaced. Unrelated files in the target
    directory are preserved, preventing accidental deletion of real BTC data.
    """

    target = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    text = legal_text()

    written = [
        _write_text(target / "mock_btc_law.txt", text),
        _write_json(
            target / "mock_btc_law.json",
            _json_document(
                "mock-json-law",
                "Bộ luật Lao động giả lập 2026 (JSON)",
                text,
            ),
        ),
        _write_jsonl(target / "mock_btc_laws.jsonl", text),
    ]
    docx_path = _write_docx_if_available(target / "mock_btc_law_docx.docx", text)
    if docx_path is not None:
        written.append(docx_path)
    return written


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục ghi mock corpus (mặc định: data/raw/btc/mock).",
    )
    return parser.parse_args()


def main() -> int:
    """Generate the development corpus from command-line arguments."""

    args = _parse_args()
    written = generate_mock_data(args.output_dir)
    print(f"Đã tạo {len(written)} file corpus giả lập:")
    for path in written:
        print(f"- {path.resolve()}")
    print("Có thể chạy ETL với raw_directory='data/raw/btc'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
