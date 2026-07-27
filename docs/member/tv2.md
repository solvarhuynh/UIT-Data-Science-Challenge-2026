# TV2: Dense Retrieval Pipeline & React Frontend Base

## 1. Tổng quan vai trò

TV2 phụ trách lớp truy hồi ngữ nghĩa của hệ thống. Đây là phần quyết định câu hỏi của người dùng được ánh xạ sang đúng đoạn văn bản pháp luật dù cách diễn đạt không trùng từ khóa. Trọng tâm là load local model `bkai-foundation-models/vietnamese-bi-encoder`, tạo embedding cho legal chunks và tích hợp VectorDB như Qdrant hoặc FAISS để search vector nhanh, ổn định. Phần bổ sung là dựng nền React frontend đủ dùng để nhóm test API sớm: ô tìm kiếm, khung chat, trạng thái loading và kết nối backend.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `EmbeddingClient` trong `src/udsc2026/infrastructure/embedding/` để load model từ `./models/bkai-bi-encoder`.
- [ ] Hỗ trợ encode một câu hỏi và encode batch chunk, có cấu hình device `cpu/cuda`, batch size và normalize embedding.
- [ ] Viết adapter VectorDB trong `src/udsc2026/infrastructure/vector_db/`, ưu tiên interface chung để đổi Qdrant/FAISS không ảnh hưởng retrieval logic.
- [ ] Tạo `DenseRetriever` trong `src/udsc2026/retrieval/dense/`, nhận query, gọi embedder, search vector store và trả `RetrievalHit`.
- [ ] Viết script index thử nghiệm trong `experiments/tv2/scripts/` để đọc `data/processed/chunks/*.jsonl` và upsert vector.
- [ ] Cấu hình collection/index: vector dimension lấy từ model, distance cosine, payload chứa metadata citation.
- [ ] Khởi tạo React base trong `frontend/`: component `SearchBox`, `ChatPanel`, `MessageBubble`, API client gọi backend.
- [ ] Tích hợp TailwindCSS, layout responsive cơ bản và trạng thái `loading/error/empty`.

## 3. API Contract & Dữ liệu giao tiếp

Dense retriever nhận:

```python
search(query: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]
```

`RetrievalHit` trả cho TV3/TV5:

```text
chunk_id, doc_id, text, score, source, law_name, article, clause, metadata
```

Vector upsert nhận:

```python
upsert(chunks: list[LegalChunk], embeddings: list[list[float]]) -> None
```

Frontend gửi request:

```json
{"question": "Điều kiện ly hôn đơn phương là gì?", "top_k": 5}
```

Frontend chỉ hiển thị response từ backend, không tự xử lý retrieval hoặc prompt.

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] Model BKAI load được từ local path, không phụ thuộc API cloud.
- [ ] Index được ít nhất một file JSONL mẫu trong `data/processed/chunks/`.
- [ ] Search trả kết quả ổn định, có score và metadata citation đầy đủ.
- [ ] VectorDB adapter có thể mock trong unit test.
- [ ] React chạy bằng `npm install` và `npm run dev` tại cổng `5173`.
- [ ] UI gửi được câu hỏi đến backend và render answer/citation placeholder.
