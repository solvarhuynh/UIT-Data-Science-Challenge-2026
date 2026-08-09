# TV4 - Advanced Data Engineer & Synthetic Dataset Specialist

## 1. Vai trò

TV4 phụ trách pipeline ETL dữ liệu pháp luật: đọc nguồn BTC, làm sạch có kiểm
soát, phân tích cấu trúc, tạo parent-child chunks, audit tính toàn vẹn và sinh
synthetic benchmark. Output mới luôn được tạo ở một thư mục candidate; không ghi
đè corpus V3 đã nghiệm thu.

Quy trình và lệnh chạy chi tiết nằm trong
[`tv4_setup.md`](tv4_setup.md) và
[`docs/project/11_data_pipeline.md`](../../project/11_data_pipeline.md).

## 2. Nhiệm vụ kỹ thuật

- [ ] Duy trì readers trong `src/udsc2026/ingestion/readers/` cho định dạng BTC.
- [ ] Duy trì cleaners trong `src/udsc2026/ingestion/cleaners/`, bảo toàn nội dung
  bảng, Unicode tiếng Việt và ghi nhận mọi biến đổi có rủi ro.
- [ ] Duy trì parser cấu trúc `Luật -> Chương -> Mục -> Điều -> Khoản -> Điểm`,
  kèm trạng thái `structured`, `partial` hoặc fallback rõ ràng.
- [ ] Tạo child chunks để retrieval và parents riêng để mở rộng context sau
  reranking; mọi child phải giữ `parent_id`, `chunk_id` và `doc_id`.
- [ ] Chạy pipeline vào `data/processed_candidate`, sau đó kiểm tra disk audit,
  source coverage, metadata, gold coverage và benchmark provenance.
- [ ] Chỉ promote nguyên cây candidate sang `data/processed_v3` khi tất cả integrity
  gates đạt; không chép lẻ từng thư mục hay sửa tay dữ liệu đã xử lý.
- [ ] Sinh synthetic benchmark có giới hạn kích thước và khóa bằng corpus hash để
  TV5 đo reranker trên đúng phiên bản chunks.
- [ ] Viết regression tests cho reader, cleaner, parser, chunker, audit và generator.

## 3. Data contract

Child chunk không nhúng lại toàn bộ parent. `parent_text` phải là `null`; nội dung
cha được lưu đúng một lần trong `parents/*.jsonl` và nạp qua `parent_id`.

```json
{
  "chunk_id": "doc001_article_10_clause_1_point_a",
  "parent_id": "doc001_article_10",
  "doc_id": "doc001",
  "text": "Nội dung điểm a khoản 1 điều 10...",
  "parent_text": null,
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

Parent tương ứng:

```json
{
  "parent_id": "doc001_article_10",
  "doc_id": "doc001",
  "text": "Toàn bộ Điều 10 hoặc context cha đã được giới hạn có kiểm soát..."
}
```

Synthetic benchmark tối thiểu phải có:

```json
{
  "question_id": "syn_0001",
  "question": "Người lao động có quyền gì theo Điều 10?",
  "answer": "Câu trả lời tham chiếu dựa trên nguồn.",
  "gold_chunk_ids": ["doc001_article_10_clause_1"],
  "gold_citations": ["Bộ luật Lao động 2019, Điều 10, Khoản 1"],
  "difficulty": "easy"
}
```

Benchmark phải ghi corpus fingerprint, số source chunks, hash của từng gold chunk
và artifact SHA-256. Thay đổi chunking đồng nghĩa phải sinh lại benchmark.

## 4. Definition of Done

- [ ] Pipeline tái lập từ `data/raw/btc` sang một candidate trống.
- [ ] Không có JSONL lỗi, chunk/parent rỗng, ID trùng, orphan parent, metadata thiếu
  hoặc file đầu ra thiếu.
- [ ] Source token/bigram coverage đạt gate; mọi fallback/partial/empty-source được
  báo cáo minh bạch thay vì giả cấu trúc.
- [ ] Benchmark hợp lệ và khớp hash với chính corpus candidate.
- [ ] Preflight GPU đọc lại tree hash và phát hiện được artifact bị sửa sau audit.
- [ ] Corpus được promote nguyên cây sang `data/processed_v3`; consumers chỉ đọc V3.
- [ ] TV2 có thể index output mà không sửa tay và TV5 có thể chạy benchmark bằng
  model reranker đã duyệt.
- [ ] Test, lint, type-check và smoke test liên quan đều đạt.
