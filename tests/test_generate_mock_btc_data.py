"""Tests for the synthetic BTC corpus generator script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_mock_btc_data import generate_mock_data
from udsc2026.ingestion.readers import extract_raw_documents

def test_mock_generator_creates_reader_compatible_corpus(tmp_path):
    output_dir = tmp_path / "btc" / "mock"
    
    # Sinh dữ liệu
    written = generate_mock_data(output_dir)
    
    # Đọc dữ liệu
    documents = extract_raw_documents(str(output_dir), str(tmp_path / "errors.json"))

    # File có tồn tại
    assert (output_dir / "mock_btc_law.txt").exists()
    assert (output_dir / "mock_btc_law.json").exists()
    assert (output_dir / "mock_btc_laws.jsonl").exists()
    assert len(written) == 3
    assert len(documents) > 0
    
    # Kiểm tra xem hệ thống có bóc tách đúng nội dung chính của các luật không
    # (Bỏ việc tìm chữ "DỰ THẢO" vì cleaner đã dọn nó rồi)
    assert any("BỘ LUẬT DÂN SỰ" in document.raw_text for document in documents)
    assert any("BỘ LUẬT LAO ĐỘNG" in document.raw_text for document in documents)
    assert any("LUẬT GIAO THÔNG" in document.raw_text for document in documents)