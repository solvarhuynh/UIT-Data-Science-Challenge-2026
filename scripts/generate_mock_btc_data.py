"""Script sinh dữ liệu giả lập (Mock Corpus) phục vụ kiểm thử ETL."""
import os
import json
import shutil
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("data/raw/btc")

def clean_directory(directory_path: Path):
    if directory_path.exists():
        for item in directory_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        directory_path.mkdir(parents=True, exist_ok=True)

def generate_mock_data(output_dir: Path = None):
    base_dir = output_dir if output_dir else DEFAULT_OUTPUT_DIR
    print(f"🧹 Đang dọn dẹp: {base_dir}...")
    clean_directory(base_dir)

    print("📝 Đang tạo dữ liệu...")
    
    # 1. FILE TXT
    file1 = base_dir / "mock_btc_law.txt"
    file1.write_text("BỘ LUẬT DÂN SỰ 2015\nDỰ THẢO\nĐiều 1. Phạm vi điều chỉnh\n", encoding="utf-8")

    # 2. FILE JSON
    file2 = base_dir / "mock_btc_law.json"
    file2.write_text(json.dumps({"content": "BỘ LUẬT LAO ĐỘNG (sau đây gọi là BLLĐ)"}, ensure_ascii=False), encoding="utf-8")

    # 3. FILE JSONL
    file3 = base_dir / "mock_btc_laws.jsonl"
    with file3.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"text": "LUẬT GIAO THÔNG\na) Người lao động"}, ensure_ascii=False) + "\n")

    return [file1, file2, file3]

def main():
    generate_mock_data()

if __name__ == "__main__":
    main()