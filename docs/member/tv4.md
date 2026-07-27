# TV4 - Advanced Data Engineer & Synthetic Dataset Specialist

## 1. Tổng quan vai trò

TV4 phụ trách Advanced Data ETL Pipeline cho hệ thống RAG Pháp luật DSC2026. Trọng tâm là biến văn bản pháp luật thô thành dữ liệu sạch, có cấu trúc phân cấp `Luật -> Chương -> Điều -> Khoản -> Điểm`, chunk đúng ngữ cảnh và metadata citation đầy đủ cho retrieval, reranking và QA.

TV4 không làm Web Frontend. Ngoài ETL, TV4 chịu trách nhiệm tạo Synthetic Benchmark Q&A gồm 100-200 cặp câu hỏi-đáp mẫu để TV5 đo điểm hệ thống.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết reader trong `src/udsc2026/ingestion/readers/` để đọc `.txt`, `.json`, `.jsonl`, `.docx` hoặc định dạng dữ liệu BTC cung cấp nếu có.
- [ ] Viết cleaner trong `src/udsc2026/ingestion/cleaners/` để chuẩn hóa Unicode, dấu tiếng Việt, khoảng trắng, xuống dòng, bullet, header/footer và ký tự lỗi OCR.
- [ ] Lọc dữ liệu rác: dòng quá ngắn vô nghĩa, số trang lặp lại, watermark, mục lục nhiễu và đoạn không thuộc nội dung pháp luật.
- [ ] Viết Regex/Rule-based parser trong `src/udsc2026/ingestion/legal_structure/` để bóc tách `law_name`, `chapter`, `section`, `article`, `clause`, `point`.
- [ ] Parser phải nhận diện các biến thể phổ biến như `Chương I`, `Mục 1`, `Điều 10.`, `1.`, `a)` và bảo toàn thứ tự xuất hiện.
- [ ] Triển khai Parent-Child Chunking trong `src/udsc2026/ingestion/chunking/`: chunk nhỏ để search, chunk lớn/parent context để QA dùng khi cần.
- [ ] Đảm bảo child chunk không mất liên kết tới parent qua `parent_id`, `chunk_id`, `doc_id` và metadata phân cấp.
- [ ] Xuất dữ liệu chuẩn vào `data/processed/chunks/*.jsonl`, mỗi dòng là một `LegalChunk`.
- [ ] Xuất parent context vào `data/processed/parents/*.jsonl` nếu parent-child được tách thành hai tập.
- [ ] Tạo validation report trong `data/processed/metadata/`: số document, số article, số chunk, chunk rỗng, chunk quá dài, metadata thiếu, Unicode lỗi.
- [ ] Viết script Synthetic Benchmark Q&A Generation sinh 100-200 cặp câu hỏi-đáp mẫu từ các điều/khoản đã parse.
- [ ] Mỗi Q&A synthetic cần có `question`, `answer`, `gold_chunk_ids`, `gold_citations`, `law_name`, `article`, `difficulty`.
- [ ] Phân nhóm benchmark theo dạng câu hỏi: hỏi định nghĩa, điều kiện, quyền/nghĩa vụ, mức phạt, thủ tục, so sánh và câu hỏi cần nhiều khoản.
- [ ] Viết test cho cleaner, parser legal structure, parent-child chunking và generator benchmark.

## 3. Quy chuẩn Clean Code & API Contract

### Clean Code bắt buộc

- Áp dụng triệt để DRY, tối ưu số dòng code và không tạo class/interface dư thừa nếu chưa có nhu cầu thật.
- Dùng type hinting đầy đủ cho mọi input/output; output ETL và benchmark bắt buộc có Pydantic model hoặc TypedDict rõ ràng.
- Mỗi hàm chỉ làm một trách nhiệm: read, clean, parse, chunk, validate hoặc generate Q&A.
- Không copy-paste regex rải rác; gom pattern pháp luật vào module cấu hình hoặc helper có tên rõ nghĩa.
- Không sửa tay dữ liệu processed nếu có thể viết rule/script tái lập.
- Không viết frontend, FastAPI router, LLM inference hoặc vector search trong phạm vi TV4.

### Data Contract

Legal chunk output:

```json
{
  "chunk_id": "doc001_article_10_clause_1_point_a",
  "parent_id": "doc001_article_10",
  "doc_id": "doc001",
  "text": "Nội dung điểm a khoản 1 điều 10...",
  "parent_text": "Toàn bộ Điều 10 hoặc phần context cha...",
  "metadata": {
    "law_name": "Bộ luật Lao động 2019",
    "chapter": "Chương II",
    "section": null,
    "article": "Điều 10",
    "clause": "Khoản 1",
    "point": "Điểm a",
    "source": "data/raw/btc/doc001.txt"
  }
}
```

Synthetic benchmark output:

```json
{
  "question_id": "syn_0001",
  "question": "Người lao động có quyền gì theo Điều 10 Bộ luật Lao động 2019?",
  "answer": "Câu trả lời tham chiếu ngắn gọn dựa trên điều/khoản nguồn.",
  "gold_chunk_ids": ["doc001_article_10_clause_1"],
  "gold_citations": ["Bộ luật Lao động 2019, Điều 10, Khoản 1"],
  "law_name": "Bộ luật Lao động 2019",
  "article": "Điều 10",
  "difficulty": "easy"
}
```

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] ETL chạy tái lập từ `data/raw/` sang `data/processed/chunks/` và `data/processed/metadata/`.
- [ ] Parser bóc tách đúng `Luật -> Chương -> Điều -> Khoản -> Điểm` trên tập mẫu.
- [ ] Unicode tiếng Việt được chuẩn hóa, không mất dấu và dữ liệu rác phổ biến được lọc.
- [ ] Parent-Child Chunking tạo được child chunk nhỏ để search và parent context đủ rộng để QA.
- [ ] TV2 có thể index output JSONL mà không cần sửa tay.
- [ ] Synthetic benchmark có 100-200 Q&A hợp lệ, có `gold_chunk_ids` và `gold_citations`.
- [ ] Có validation report chỉ ra chunk lỗi, metadata thiếu và thống kê corpus.
- [ ] Không còn bất kỳ nhiệm vụ Web Frontend trong file giao việc của TV4.
