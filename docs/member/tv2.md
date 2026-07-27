# TV2 - Dense Retrieval Specialist

## 1. Tổng quan vai trò

TV2 phụ trách Dense Retrieval Core cho hệ thống RAG Pháp luật DSC2026. Trọng tâm là load local model `bkai-foundation-models/vietnamese-bi-encoder`, sinh embedding cho query và batch document, lưu vector vào VectorDB và search top-K theo ngữ nghĩa.

TV2 không làm Web Frontend, không viết FastAPI router và không xử lý QA generation. Output của TV2 phải là danh sách `RetrievalHit` có score và metadata citation đầy đủ để TV1, TV3 và TV5 dùng tiếp.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `EmbeddingClient` trong `src/udsc2026/infrastructure/embedding/` để load model local từ `models/bkai-bi-encoder` hoặc path cấu hình tương ứng với `bkai-foundation-models/vietnamese-bi-encoder`.
- [ ] Hỗ trợ encode query đơn lẻ và batch document/chunk với cấu hình `device`, `batch_size`, `max_length`, `normalize_embeddings`.
- [ ] Chuẩn hóa output embedding về `list[float]` hoặc `numpy.ndarray` theo contract của VectorDB adapter.
- [ ] Viết `DenseRetriever` trong `src/udsc2026/retrieval/dense/`, nhận `query`, `top_k`, `filters` và trả `list[RetrievalHit]`.
- [ ] Thiết kế VectorDB Adapter trong `src/udsc2026/infrastructure/vector_db/` cho Qdrant và FAISS, ưu tiên cùng một interface nhỏ, dễ mock.
- [ ] Cấu hình collection/index gồm vector dimension, cosine distance, payload metadata và primary key `chunk_id`.
- [ ] Viết logic upsert vector từ chunk JSONL của TV4, bảo toàn `chunk_id`, `doc_id`, `law_name`, `chapter`, `article`, `clause`, `point`, `source`.
- [ ] Viết search top-K ngữ nghĩa có hỗ trợ filter cơ bản theo `law_name`, `doc_id`, `article`, `chapter` nếu backend VectorDB cho phép.
- [ ] Viết script index dữ liệu trong `scripts/` hoặc `experiments/tv2/scripts/` để đọc `data/processed/chunks/*.jsonl`, embed batch và upsert vào Qdrant/FAISS.
- [ ] Thêm config cho model path, vector DB type, collection name, batch size và device trong `configs/`.
- [ ] Viết unit test/mock cho embedder, vector store adapter, upsert và dense search.

## 3. Quy chuẩn Clean Code & API Contract

### Clean Code bắt buộc

- Áp dụng triệt để DRY, tối ưu số dòng code và không tạo class/interface dư thừa nếu chưa có nhu cầu thật.
- Dùng type hinting đầy đủ cho mọi input/output; config, chunk input và retrieval output nên dùng Pydantic hoặc typing rõ ràng.
- Mỗi hàm chỉ làm một trách nhiệm: load model, encode, upsert, search hoặc map result.
- Không copy-paste logic search giữa Qdrant và FAISS; dùng adapter chung với implementation ngắn.
- Không viết frontend, FastAPI router, prompt, QA hoặc reranking trong phạm vi TV2.
- Không định nghĩa lại schema `RetrievalHit`; luôn import từ contract chung.

### API Contract

Dense retrieval:

```python
from udsc2026.contracts.retrieval import RetrievalHit

def search(
    query: str,
    top_k: int,
    filters: dict[str, str | int | list[str]] | None = None,
) -> list[RetrievalHit]: ...
```

Embedding:

```python
def embed_query(query: str) -> list[float]: ...
def embed_documents(texts: list[str], batch_size: int) -> list[list[float]]: ...
```

Vector upsert:

```python
def upsert(
    chunks: list[LegalChunk],
    embeddings: list[list[float]],
) -> None: ...
```

`RetrievalHit` bắt buộc bảo toàn:

```text
chunk_id, doc_id, text, score, source, law_name, chapter, article, clause, point, metadata
```

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] Model BKAI bi-encoder load được từ local path, không phụ thuộc cloud API.
- [ ] Encode query và encode batch document chạy ổn định trên dataset mẫu.
- [ ] Script index đọc được JSONL từ `data/processed/chunks/` và upsert vào Qdrant hoặc FAISS.
- [ ] Search top-K trả đúng `list[RetrievalHit]` có score và metadata citation đầy đủ.
- [ ] VectorDB adapter có thể mock trong unit test.
- [ ] Dense search có latency đo được và không làm mất dấu tiếng Việt.
- [ ] Không còn bất kỳ task React/UI/API route trong file giao việc của TV2.
