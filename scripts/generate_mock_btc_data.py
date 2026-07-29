"""Sinh corpus BTC giả lập để chạy thử ETL; không dùng dữ liệu BTC thật.

Mặc định, file được tạo ở ``data/raw/btc/mock`` tính từ repository root. Vì
đường dẫn này được xác định từ vị trí của script, lệnh chạy hoạt động đúng kể cả
khi terminal đang ở một thư mục khác.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "raw" / "btc" / "mock"


def legal_text() -> str:
    """Return one legal sample containing the supported ETL edge cases."""
    return """DỰ THẢO
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


def generate_mock_data(output_dir: Optional[Path] = None) -> List[Path]:
    """Create TXT, JSON, JSONL and, when available, DOCX mock source files.

    Only four known mock filenames are overwritten. The function never clears
    a supplied directory, so it cannot delete real BTC files by accident.
    """
    target = (output_dir or DEFAULT_OUTPUT_DIR).resolve()
    target.mkdir(parents=True, exist_ok=True)
    text = legal_text()
    written = [
        _write_text(target / "mock_btc_law.txt", text),
        _write_json(
            target / "mock_btc_law.json",
            {
                "id": "mock-json-law",
                "title": "Bộ luật Lao động giả lập 2026 (JSON)",
                "content": text,
                "effective_date": "2026-01-01",
            },
        ),
        _write_jsonl(target / "mock_btc_laws.jsonl", text),
    ]
    docx_path = _write_docx_if_available(target / "mock_btc_law_docx.docx", text)
    if docx_path is not None:
        written.append(docx_path)
    return written


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, text: str) -> Path:
    records = [
        {
            "id": "mock-jsonl-law-1",
            "title": "Bộ luật Lao động giả lập 2026 (JSONL 1)",
            "content": text,
        },
        {
            "id": "mock-jsonl-law-2",
            "title": "Bộ luật Lao động giả lập 2026 (JSONL 2)",
            "content": text.replace("Điều 1", "Điều 11", 1),
        },
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_docx_if_available(path: Path, text: str) -> Optional[Path]:
    try:
        from docx import Document
    except ImportError:
        return None
    document = Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    document.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Thư mục ghi mock corpus (mặc định: data/raw/btc/mock).",
    )
    args = parser.parse_args()
    written = generate_mock_data(args.output_dir)
    print("Đã tạo {0} file corpus giả lập:".format(len(written)))
    for path in written:
        print("- {0}".format(path.resolve()))
    print("Có thể chạy ETL với raw_directory='data/raw/btc'.")


if __name__ == "__main__":
    main()
