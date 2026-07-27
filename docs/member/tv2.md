# TV2 - Dense Retrieval Core: BKAI Bi-Encoder & VectorDB

## 1. Tổng quan vai trò

TV2 phụ trách 100% Backend Dense Retrieval. TV2 không làm Frontend, không dựng React UI, không xử lý API route. Trọng tâm là biến câu hỏi và legal chunks thành embedding, index vào VectorDB và trả về danh sách `RetrievalHit` có score và metadata đầy đủ cho pipeline của TV1.

Mục tiêu của TV2 là truy hồi ngữ nghĩa nhanh, ổn định, có thể thay đổi Qdrant/FAISS mà không làm vỡ contract chung.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `EmbeddingClient` trong `src/udsc2026/infrastructure/embedding/` để load model BKAI từ `./models/bkai-bi-encoder`.
- [ ] Hỗ trợ encode query và encode batch chunks với cấu hình `device`, `batch_size`, `normalize_embeddings`.
- [ ] Viết VectorDB adapter trong `src/udsc2026/infrastructure/vector_db/`, ưu tiên interface chung cho Qdrant và FAISS.
- [ ] Xây `DenseRetriever` trong `src/udsc2026/retrieval/dense/`, nhận query, gọi embedder, search vector store và trả `list[RetrievalHit]`.
- [ ] Viết script index trong `experiments/tv2/scripts/` để đọc `data/processed/chunks/*.jsonl` từ ETL của TV4 và upsert vector.
- [ ] Cấu hình collection/index gồm vector dimension, cosine distance, payload metadata citation và source.
- [ ] Hỗ trợ filters cơ bản theo `law_name`, `doc_id`, `article` nếu VectorDB cho phép.
- [ ] Viết unit test/mock cho embedder, vector store và dense retriever.

## 3. Quy chuẩn Code & API Contract

### Clean Code Standard

- Code ngắn gọn, tách rõ `EmbeddingClient`, `VectorStore`, `DenseRetriever`.
- Dùng Pydantic/Typing cho config, chunk input và retrieval output.
- Không viết code frontend, FastAPI router, SSE, prompt hoặc UI helper trong phạm vi TV2.
- Không copy-paste logic search giữa Qdrant và FAISS; dùng interface chung.
- Không tạo boilerplate adapter nếu chưa có nhu cầu thực; ưu tiên implementation nhỏ, test được.

### API Contract

```python
from udsc2026.contracts.retrieval import RetrievalHit

search(query: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]
```

`RetrievalHit` phải lấy từ `src/udsc2026/contracts/retrieval.py`, không định nghĩa lại. Trường bắt buộc cần bảo toàn:

```text
chunk_id, doc_id, text, score, source, law_name, article, clause, metadata
```

Vector upsert:

```python
upsert(chunks: list[LegalChunk], embeddings: list[list[float]]) -> None
```

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] BKAI bi-encoder load được từ local path, không phụ thuộc cloud API.
- [ ] Index được JSONL mẫu từ `data/processed/chunks/`.
- [ ] Search trả đúng `list[RetrievalHit]` có score và metadata citation.
- [ ] VectorDB adapter có thể mock trong test.
- [ ] Dense search có latency và top_k ổn định trên dataset mẫu.
- [ ] Không còn task React/UI trong file giao việc của TV2.
