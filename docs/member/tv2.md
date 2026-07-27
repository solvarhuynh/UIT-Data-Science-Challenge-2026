# TV2 - Dense Retrieval Pipeline, React Frontend Basic

## Mục tiêu & Phạm vi công việc

- [ ] Xây dựng Dense Retrieval Pipeline bằng `bkai-foundation-models/vietnamese-bi-encoder`.
- [ ] Tạo embedding cho chunk văn bản pháp luật và ghi vào VectorDB local.
- [ ] Viết logic truy vấn dense top-k theo question embedding.
- [ ] Bổ sung khung React cơ bản gồm ô tìm kiếm, khung chat và trạng thái loading.
- [ ] Đảm bảo frontend gọi FastAPI qua JSON/SSE, không chứa logic RAG.

## Thư mục mã nguồn phụ trách

- [ ] `src/dsc2026_legal/retrieval/dense/`
- [ ] `src/dsc2026_legal/infrastructure/embedding/`
- [ ] `src/dsc2026_legal/infrastructure/vector_db/`
- [ ] `data/vector_store/`
- [ ] `frontend/`

## API/Interface đầu ra cần bàn giao

- [ ] `DenseRetriever.search(query: str, top_k: int, filters: dict | None) -> list[RetrievalHit]`.
- [ ] `EmbeddingClient.encode_texts(texts: list[str]) -> list[list[float]]`.
- [ ] `VectorStoreClient.upsert(chunks: list[LegalChunk]) -> None`.
- [ ] React component `SearchBox` nhận input và submit question.
- [ ] React component `ChatPanel` hiển thị answer, loading và lỗi API.

## Checklist nghiệm thu công việc

- [ ] Tải model BKAI từ `./models/bkai-bi-encoder`.
- [ ] Index thử được ít nhất một tập chunk mẫu trong `data/processed/chunks/`.
- [ ] Dense search trả về kết quả có `chunk_id`, `score`, `text`, `metadata`.
- [ ] Frontend chạy được bằng `npm install` và `npm run dev` tại cổng 5173.
- [ ] Có test hoặc script smoke test cho embedding, upsert và search.
