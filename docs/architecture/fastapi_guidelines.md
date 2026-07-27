# FastAPI Guidelines

## Mục tiêu

FastAPI là backend duy nhất của UDSC2026. Backend chỉ nhận request, điều phối retrieval/QA/tool, rồi trả JSON hoặc streaming event cho React. Không đặt logic UI, notebook hoặc model experiment trong API layer.

## Cấu trúc module

```text
src/udsc2026/
├── api/             # app, routes, dependency injection
├── contracts/       # Pydantic schemas
├── ingestion/       # ETL và chunking
├── retrieval/       # dense, sparse, hybrid, reranking
├── qa/              # prompt, LLM, citation
└── infrastructure/  # clients cho model, vector store, persistence
```

## Endpoint baseline

- `GET /health`: kiểm tra backend và dependency chính.
- `POST /api/v1/query`: trả kết quả JSON đầy đủ.
- `GET /api/v1/query/stream`: trả token/event streaming cho frontend.

## Pydantic Schema

Schema phải nằm trong `src/udsc2026/contracts/`.

```python
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    filters: dict | None = None
    stream: bool = False
```

Response cần ổn định để frontend và evaluation dùng chung:

```text
answer, citations, retrieval_hits, route, latency_ms, warnings
```

## Sync và Async

Quy tắc:

- Endpoint nhẹ như `/health` dùng sync hoặc async đều được.
- I/O như đọc vector store, gọi HTTP client, streaming nên dùng async nếu client hỗ trợ.
- Tác vụ CPU/GPU nặng như embedding và generation nên đặt sau service layer, không viết trực tiếp trong route.
- Không dùng hàng đợi background hoặc cache service ngoài trong baseline hiện tại.

## Dependency Injection

Khởi tạo model và vector store qua dependency hoặc application lifespan.

```python
def get_retriever() -> HybridRetriever:
    return app.state.retriever
```

Lợi ích:

- Dễ mock trong test.
- Không load model lặp lại mỗi request.
- Tách rõ API layer và infrastructure layer.

## Error Handling

- Validate input bằng Pydantic.
- Lỗi thiếu model trả `503 Service Unavailable`.
- Lỗi request sai trả `422` hoặc `400`.
- Lỗi không tìm thấy context không phải exception, mà là response có `warnings`.

## Streaming

Frontend React nhận stream qua SSE hoặc chunked response. Mỗi event nên có dạng:

```json
{"type": "token", "content": "..."}
```

Event cuối:

```json
{"type": "final", "answer": "...", "citations": []}
```

## Test bắt buộc

- Health check.
- Query request validation.
- Mock retriever + mock QA engine.
- Streaming trả đúng event `token` và `final`.
