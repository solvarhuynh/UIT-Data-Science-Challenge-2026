# TV2 — Chuyên trách truy hồi đầy đủ

## 1. Tổng quan vai trò

TV2 phụ trách toàn bộ Retrieval Pipeline của hệ thống RAG Pháp luật DSC2026 trong `src/udsc2026/retrieval/`, bao gồm `dense/`, `sparse/` và `hybrid/`. Mục tiêu là cung cấp một lớp truy hồi thống nhất: Dense Retrieval bằng HCMUTE embedding v2, Sparse Retrieval bằng BM25 tiếng Việt và Hybrid Fusion để trả về danh sách context tốt nhất cho reranker/QA.

TV2 cũng quản lý các adapter hạ tầng liên quan trực tiếp đến retrieval như `src/udsc2026/infrastructure/embedding/` và `src/udsc2026/infrastructure/vector_db/`. TV2 không viết Web Frontend, không viết FastAPI router, không thiết kế prompt và không sinh câu trả lời bằng LLM.

Đầu ra chuẩn của TV2 là `list[RetrievalHit]` có score, metadata citation và thông tin nguồn đầy đủ để TV1 điều phối, TV5 rerank và TV3 sinh câu trả lời.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Dùng `EmbeddingClient` để load `models/dek21-v2`, checkpoint `huyydangg/DEk21_hcmute_embedding_v2` đã được nhóm chốt.
- [ ] Hỗ trợ encode query đơn lẻ và batch document/chunk với cấu hình `device`, `batch_size`, `max_length`, `normalize_embeddings`.
- [ ] Chuẩn hóa output embedding về `list[float]` hoặc `numpy.ndarray` theo contract của VectorDB adapter.
- [ ] Viết `DenseRetriever` trong `src/udsc2026/retrieval/dense/`, nhận `query`, `top_k`, `filters` và trả `list[RetrievalHit]`.
- [ ] Thiết kế VectorDB adapter trong `src/udsc2026/infrastructure/vector_db/` cho Qdrant và FAISS, dùng interface nhỏ, dễ mock và không ràng buộc logic business vào backend cụ thể.
- [ ] Cấu hình collection/index gồm vector dimension, cosine distance, payload metadata và primary key `chunk_id`.
- [ ] Viết logic upsert vector từ chunk JSONL của TV4, bảo toàn `chunk_id`, `doc_id`, `law_name`, `chapter`, `article`, `clause`, `point`, `source`.
- [ ] Viết `BM25Retriever` trong `src/udsc2026/retrieval/sparse/`, dùng tokenization chuẩn tiếng Việt, giữ nguyên dấu, số điều/khoản/điểm và các ký hiệu pháp lý quan trọng.
- [ ] Xây index BM25 từ `data/processed_v3/chunks/*.jsonl`, hỗ trợ lưu/tải index local trong `data/vector_store/` hoặc path cấu hình.
- [ ] Viết `HybridRetriever` trong `src/udsc2026/retrieval/hybrid/` để gọi Dense + Sparse, normalize score và fusion theo cấu hình.
- [ ] Mặc định hybrid score theo công thức rõ ràng, ví dụ `hybrid_score = 0.5 * dense_score + 0.5 * sparse_score`, đồng thời hỗ trợ cấu hình `dense_weight`, `sparse_weight`, `top_k`, `candidate_k`, `min_score`.
- [ ] Merge kết quả Dense/Sparse theo `chunk_id`, không làm mất metadata citation và lưu lại `dense_score`, `sparse_score`, `hybrid_score` trong metadata hoặc schema được thống nhất.
- [ ] Viết script index dữ liệu trong `scripts/` hoặc `experiments/tv2/scripts/` để đọc `data/processed_v3/chunks/*.jsonl`, embed batch, upsert vào Qdrant/FAISS và build BM25 index.
- [ ] Thêm config cho model path, vector DB type, collection name, BM25 index path, batch size, device và trọng số hybrid trong `configs/`.
- [ ] Viết unit test/mock cho embedder, vector store adapter, BM25 tokenizer, sparse search, score normalization, hybrid fusion, upsert và dense search.

## 3. Quy chuẩn mã nguồn và contract API

### Yêu cầu mã nguồn sạch

- Áp dụng nguyên tắc tránh lặp lại, tách rõ Dense, Sparse và Hybrid nhưng dùng contract chung để tránh trùng logic map kết quả.
- Dùng type hinting đầy đủ cho mọi input/output; config, chunk input, retrieval output và score breakdown nên dùng Pydantic hoặc typing rõ ràng.
- Mỗi hàm chỉ làm một trách nhiệm: load model, tokenize, encode, upsert, search, normalize score hoặc merge result.
- Hàm nên ngắn gọn, tên rõ nghĩa, tránh side effect ẩn và tránh hard-code path/config trong logic retrieval.
- Không sao chép logic search giữa Qdrant và FAISS; dùng adapter chung với implementation ngắn.
- Không sao chép normalize/fusion giữa nhiều retriever; viết helper riêng và có test.
- Không viết frontend, FastAPI router, prompt, QA generation hoặc reranking trong phạm vi TV2.
- Không định nghĩa lại schema `RetrievalHit` hoặc `LegalChunk`; luôn import từ contract chung.

### Contract API

Dense retrieval:

```python
from udsc2026.contracts.retrieval import RetrievalHit

def search(
    query: str,
    top_k: int,
    filters: dict[str, str | int | list[str]] | None = None,
) -> list[RetrievalHit]: ...
```

Sparse retrieval:

```python
def search(
    query: str,
    top_k: int,
    filters: dict[str, str | int | list[str]] | None = None,
) -> list[RetrievalHit]: ...
```

Hybrid retrieval:

```python
def search(
    query: str,
    top_k: int,
    filters: dict[str, str | int | list[str]] | None = None,
) -> list[RetrievalHit]: ...
```

Score fusion:

```python
def fuse_scores(
    dense_hits: list[RetrievalHit],
    sparse_hits: list[RetrievalHit],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
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

Hybrid hit cần có score breakdown:

```text
chunk_id, score, dense_score, sparse_score, hybrid_score, metadata
```

## 4. Tiêu chuẩn nghiệm thu

- [ ] Model HCMUTE embedding v2 load được từ local path, không phụ thuộc cloud API.
- [ ] Encode query và encode batch document chạy ổn định trên dataset mẫu.
- [ ] Script index đọc được JSONL từ `data/processed_v3/chunks/`, upsert vào Qdrant/FAISS và build BM25 index.
- [ ] BM25 chạy độc lập với tokenization tiếng Việt và query chứa tên luật, số điều, số khoản.
- [ ] Dense search, Sparse search và Hybrid search đều trả đúng `list[RetrievalHit]` có score và metadata citation đầy đủ.
- [ ] Hybrid Search normalize score trước khi fusion, hỗ trợ cấu hình trọng số và không làm mất citation metadata.
- [ ] VectorDB adapter và BM25 retriever có thể mock trong unit test.
- [ ] Retrieval pipeline có latency đo được và không làm mất dấu tiếng Việt.
- [ ] Không còn bất kỳ task React/UI/API route/prompt/QA generation trong file giao việc của TV2.
