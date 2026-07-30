1 CẤU TRÚC ETL:

[Raw Files] (.txt, .json, .jsonl, .docx, .pdf)
│
▼ (1. Extract)
[Readers Module] -> RawDocument
│
▼ (2. Clean)
[Cleaners Module] -> CleanDocument (Chuẩn hóa Unicode, xử lý từ viết tắt, lọc nhiễu)
│
▼ (3. Parse Structure)
[Legal Structure State-Machine Parser] -> LegalStructureDocument (Luật -> Chương -> Mục -> Điều -> Khoản -> Điểm)
│
▼ (4. Chunking)
[Parent-Child Chunker] -> LegalChunk & LegalParent
│
▼ (5. Load / Deliver)
[Output Writer & Validation]
├── data/processed/chunks/.jsonl  (Child chunks - Dành cho TV2 VectorDB Indexing)
├── data/processed/parents/.jsonl (Parent Context - Dành cho TV3 QA Context)
├── data/processed/documents/*.json (Full Clean Structured Document Audit)
└── data/processed/metadata/validation_report.json (Báo cáo chất lượng dữ liệu)


---

## ⚙️ 2. Yêu Cầu Môi Trường (Prerequisites)

- **Python**: 3.10+
- **Biến môi trường**: `PYTHONPATH=src` (để Python nhận diện gói thư viện `udsc2026`).
- **Thư viện phụ thuộc**:
  - `pydantic` >= 2.0
  - `python-docx` (dành cho file `.docx`)
  - `pdfplumber` hoặc `PyMuPDF` (dành cho file `.pdf`)

---

## 🛠️ 3. Các Bước Chạy Bằng Tay (Step-by-Step Execution)

### Bước 3.1: Chuẩn bị dữ liệu thô (Raw Data Setup)

Dữ liệu thô từ Ban tổ chức phải được đặt trong thư mục:
`data/raw/btc/`

Nếu chưa có dữ liệu thật từ BTC, bạn có thể kích hoạt **Script sinh Dữ liệu giả lập (Mock Corpus Generator)** để thử nghiệm:

#### DƯỚI TERMINAL TRONG VS CODE (PowerShell / CMD):
```powershell
py -3.10 "data\raw\Mock corpus data BTC\generate_mock_btc_data.py"

Kết quả tạo ra: Dữ liệu thử nghiệm chứa các "bẫy" nâng cao (Watermark DỰ THẢO, số trang, gõ dấu kiểu cũ hoà, uý, mẫu từ viết tắt (sau đây gọi là BLLĐ), cấu trúc Chương -> Điều -> Khoản -> Điểm) tại data/raw/btc/mock/ hoặc data/raw/btc/.

Bước 3.2: Kích hoạt ETL Pipeline chính (run_ingestion_pipeline)
Lệnh kích hoạt toàn bộ luồng End-to-End từ Raw Data -> Chunks + Parents + Validation Report:

PowerShell
$env:PYTHONPATH = "src"; py -3.10 -c "from udsc2026.ingestion import run_ingestion_pipeline; print(run_ingestion_pipeline())"

Bước 3.3 (CÓ THỂ BỎ QUA): Tùy chỉnh tham số đầu vào trong Python Script (Advanced Options)
Nếu bạn muốn tạo file script riêng hoặc chạy trong Jupyter Notebook / Python REPL:

Python
import sys
from pathlib import Path

# Đảm bảo Python nạp được thư mục src
sys.path.insert(0, "src")

from udsc2026.ingestion import run_ingestion_pipeline

# Chạy pipeline với đường dẫn tùy chỉnh
results = run_ingestion_pipeline(
    raw_directory="data/raw/btc",        # Đường dẫn chứa file thô
    processed_root="data/processed",     # Thư mục đích chứa dữ liệu đầu ra
    chunk_size=512,                       # Giới hạn token tối đa cho 1 child chunk
    chunk_overlap=80                      # Độ đè ngữ cảnh giữa các phần băm
)

print(f"Hoàn tất xử lý {results['cleaned_document_count']} tài liệu.")
print(f"Chi tiết Validation Report: {results['validation_report']}")

🔍 4. Kiểm Tra & Nghiệm Thu Kết Quả Đầu Ra (Output Verification)
Sau khi chạy xong, hãy kiểm tra thư mục data/processed/:

1. data/processed/chunks/*.jsonl (File Băm Nhỏ Dành Cho TV2)
Mỗi dòng là một chuỗi JSON chuẩn mực theo contract LegalChunk:

JSON
{
  "chunk_id": "mock_btc_law_article_2_clause_1",
  "parent_id": "mock_btc_law_article_2",
  "doc_id": "mock_btc_law",
  "text": "1. Ở nước Cộng hòa xã hội chủ nghĩa Việt Nam, các quyền dân sự được công nhận, tôn trọng, bảo vệ và bảo đảm theo Hiến pháp và pháp luật.",
  "parent_text": "Điều 2. Công nhận, tôn trọng, bảo vệ và bảo đảm quyền dân sự...",
  "law_name": "BỘ LUẬT DÂN SỰ 2015",
  "chapter": "Chương I. QUY ĐỊNH CHUNG",
  "section": "Mục 1. PHẠM VI ĐIỀU CHỈNH",
  "article": "Điều 2",
  "clause": "Khoản 1",
  "point": null,
  "source": "data/raw/btc/mock_btc_law.txt",
  "metadata": {
    "expanded_terms": {},
    "char_count": 142,
    "token_count": 29
  }
}
2. data/processed/parents/*.jsonl (File Ngữ Cảnh Cha Dành Cho TV3)
Chứa toàn bộ nội dung Điều luật gốc (LegalParent), đảm bảo TV3 (QA LLM) lấy trọn ngữ cảnh lớn không bị cắt vụn.

3. data/processed/metadata/validation_report.json
Đảm bảo tất cả các chỉ số lỗi bên dưới bằng 0:

empty_chunk_count: 0 (Không có chunk rỗng)

oversized_chunk_count: 0 (Không có chunk vượt quá 512 tokens)

missing_metadata_count: 0 (Không thiếu law_name, article, source)

unicode_error_count: 0 (Không lỗi mã hóa Unicode tiếng Việt)

duplicate_chunk_id_count: 0 (Không trùng ID trong VectorDB)

orphan_chunk_count: 0 (Mọi child chunk đều nối đúng parent_id)

🚚 5. Bàn Giao Đầu Ra Cho Các Thành Viên Khác
Bàn giao cho TV1 (Integration):

TV1 gọi trực tiếp hàm run_ingestion_pipeline() từ gói udsc2026.ingestion trong file Orchestrator tổng của app.

Bàn giao cho TV2 (Dense Retrieval):

Đọc các file .jsonl tại data/processed/chunks/ để chạy script encode embedding (BKAI bi-encoder) và upsert vào Qdrant/FAISS VectorDB.

Bàn giao cho TV3 (QA & LLM):

Sử dụng parent_id hoặc trích xuất trực tiếp parent_text từ payload để nạp context cho Qwen3 LLM.
