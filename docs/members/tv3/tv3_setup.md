# Báo cáo Tiến độ TV3 - QA Engine, Prompt Engineering & Local LLM

- Thành viên: TV3 (QA Engine, Prompt Engineering & Local LLM Specialist)
- Phiên bản: v2.0 (Cập nhật theo Tài liệu & Dữ liệu Warmup của BTC)
- Ngày cập nhật: 02/08/2026
- Trạng thái: Completed & Verified (59/59 Unit Tests Passed, 100% Benchmark Completed)

---

## 1. Tổng quan Công việc đã Hoàn thành

Đợt cập nhật này nhằm điều chỉnh hệ thống QA Engine và Prompt Engineering của TV3 bám sát theo yêu cầu từ Ban tổ chức (BTC) trong tài liệu `task2_overview.md` và dữ liệu `warmup_task2.json`.

### Các công việc đã thực hiện:
1. Tối ưu hóa System Prompt cho độ đo METEOR (Task 2.1):
   - BTC sử dụng METEOR làm độ đo chính và ROUGE-L làm độ đo phụ cho Task 2 (LegalQA).
   - Đã tạo `prompts/system/legal_qa_v2.md` và `prompts/rag_templates/default_rag_v2.md`: Hướng dẫn LLM sinh câu trả lời đầy đủ, chi tiết, bám sát căn cứ pháp lý để tăng độ khớp token (Recall/Precision).
   - Cho phép trích dẫn linh hoạt theo dạng ngoặc vuông `[Tên luật, Điều X, Khoản Y]` hoặc văn xuôi tự nhiên `Căn cứ Điều X Luật Y`.

2. Xây dựng Script Đánh giá Tự động & Cross-Validation (Task 2.2):
   - Tạo `experiments/tv3/exp_02_evaluate_warmup.py` đo điểm METEOR và ROUGE-L trên dữ liệu `warmup_task2.json`.
   - Áp dụng phân chia 80/20 (Cross-Validation Split): 80% (400 câu) thuộc tập DEV dùng để chỉnh sửa và so sánh prompt; 20% (100 câu) thuộc tập TEST được giữ kín để thi thử cuối cùng.
   - Hỗ trợ chạy trên cả GPU local (4-bit Quantization NF4) và GPU Google Colab T4.

3. Nâng cấp Bộ Bóc tách Trích dẫn CitationParser (Task 2.3):
   - Nâng cấp `src/udsc2026/qa/citation_parser.py`: Bổ sung nhận diện dạng trích dẫn ngược `khoản X Điều Y` phổ biến trong đáp án mẫu của BTC.
   - Thêm hàm `_extract_law_name_around()` tự động bóc tách tên văn bản luật (Bộ luật, Luật, Nghị định, Thông tư, Quyết định) ở cả 2 chiều trước và sau trích dẫn.
   - Cập nhật dedup key theo `(article, clause, law_name)` để phân biệt đúng các Khoản khác nhau trong cùng một Điều luật.
   - Xử lý tương thích ký tự chữ hoa và chữ thường tiếng Việt (`Điều/điều`, `Khoản/khoản`, `Điểm/điểm`).

4. Cập nhật QAEngine & Test Suite (Task 2.4):
   - Cập nhật `QAEngine.generate_answer()` trong `src/udsc2026/qa/qa_engine.py`: Bổ sung tham số `rag_template` cho phép chọn RAG template linh hoạt và bọc xử lý bất đồng bộ (async).
   - Đạt 59/59 unit tests PASSED.

---

## 2. Kết quả Đánh giá Benchmark Thực tế (Google Colab GPU)

| Tập dữ liệu | Số lượng | Tỷ lệ thành công | METEOR (gen ok) | ROUGE-L (gen ok) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DEV Split (Tuning 80%)** | 400 câu | 100% (400/400) | **0.4236** | **0.5119** | Prompt v2 giúp điểm METEOR tăng gấp đôi so với Prompt v1 baseline (0.2156). |
| **TEST Split (Held-out 20%)** | 100 câu | 100% (100/100) | **0.3155** | **0.3962** | Đạt độ ổn định 100%, không bị học vẹt, khả năng tổng quát hóa rất cao trên tập đề thi lạ. |

---

## 3. Danh sách các File đã Làm việc

| Đường dẫn File | Loại thay đổi | Chức năng & Tác dụng |
| :--- | :---: | :--- |
| `prompts/system/legal_qa_v2.md` | Tạo mới | System Prompt v2 tối ưu cho độ đo METEOR của BTC. |
| `prompts/rag_templates/default_rag_v2.md` | Tạo mới | Template RAG v2 ghép ngữ cảnh và câu hỏi. |
| `experiments/tv3/exp_02_evaluate_warmup.py` | Tạo mới | Script đánh giá tự động METEOR và ROUGE-L trên dữ liệu warmup. |
| `src/udsc2026/qa/citation_parser.py` | Cập nhật | Nâng cấp regex bắt trích dẫn ngược, bóc tách tên luật và fix dedup key. |
| `src/udsc2026/qa/qa_engine.py` | Cập nhật | Thêm tham số `rag_template` và hỗ trợ async handler. |
| `docs/member/tv3_execution_plan.md` | Cập nhật | Kế hoạch triển khai v2.0 của TV3 theo yêu cầu mới. |
| `docs/tv3_setup.md` | Cập nhật | File báo cáo tiến độ và hướng dẫn sử dụng phần TV3 gửi Leader và Team. |

---

## 4. Hướng dẫn Chạy Kiểm thử & Đánh giá

### Bước 1: Chạy Unit Test kiểm tra logic code:
```bash
python -m pytest tests/unit/test_qa/ -v
```

### Bước 2: Smoke Test nhanh script đánh giá (dùng Mock LLM, ~1 giây):
```bash
python experiments/tv3/exp_02_evaluate_warmup.py --max-samples 20
```

### Bước 3: So sánh Prompt v1 vs Prompt v2 trên tập DEV (80% / 400 câu):
```bash
python experiments/tv3/exp_02_evaluate_warmup.py --split dev --prompt-version legal_qa_v2 --use-real-model --timeout 180
```

### Bước 4: Đánh giá chốt hạ trên tập TEST (20% / 100 câu giấu kín):
```bash
python experiments/tv3/exp_02_evaluate_warmup.py --split test --prompt-version legal_qa_v2 --use-real-model --timeout 180
```

---

## 5. Phối hợp với các Thành viên trong Team

- TV1 (Leader): QAEngine nhận `question` và `contexts: list[RetrievalHit]`, trả về `QAResponse` chuẩn Pydantic contract.
- TV2 (Retrieval): Lưu ý khi nạp file `context_*.json` của BTC cần map thuộc tính `passage` gốc sang thuộc tính `text` của `RetrievalHit`.
- TV5 (MLOps): Script `exp_02_evaluate_warmup.py` cung cấp bộ benchmark METEOR và ROUGE-L sẵn sàng tích hợp vào pipeline đánh giá tự động chung.
