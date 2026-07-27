# TV1 - Leader: Integration & Orchestration Specialist

## 1. Tổng quan vai trò

TV1 là Leader phụ trách tích hợp backend end-to-end cho hệ thống RAG Pháp luật DSC2026. Trọng tâm là xây `End-to-End Pipeline Orchestrator`, kết nối module của TV2, TV3, TV4 và TV5 vào router FastAPI, đảm bảo request đi từ API vào retrieval, reranking, QA và response cuối cùng một cách ổn định, đo được và dễ debug.

TV1 không làm Web Frontend. Vai trò chính là giữ kiến trúc backend sạch, kiểm soát API contract, Dependency Injection, async handling, caching, logging và tối ưu latency toàn pipeline.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Thiết kế FastAPI app, router, middleware, lifecycle startup/shutdown và config loading trong `src/udsc2026/api/`.
- [ ] Xây `RAGOrchestrator` điều phối luồng xử lý: nhận query -> kiểm tra cache -> gọi Dense Retrieval TV2 -> gọi Sparse/Hybrid TV3 -> gọi Reranker TV5 -> gọi QA Engine TV3 -> trả response.
- [ ] Tích hợp output ETL/chunk metadata của TV4 vào contract chung để các module retrieval, rerank và QA dùng thống nhất.
- [ ] Quản lý Dependency Injection cho `EmbeddingClient`, `VectorStore`, `SparseRetriever`, `HybridRetriever`, `Reranker`, `QAEngine`, `CacheClient` và `Logger`.
- [ ] Chuẩn hóa async handling: endpoint không block event loop, các tác vụ inference/vector search nặng phải có strategy rõ ràng như async client, threadpool hoặc worker.
- [ ] Xây endpoint `/query` nhận `QueryRequest` và trả `QueryResponse` chuẩn cho frontend hoặc script evaluation.
- [ ] Xây endpoint `/health` và `/ready` để kiểm tra model, vector DB, cache và trạng thái service.
- [ ] Xây cơ chế caching cho câu hỏi trùng lặp bằng key hash từ `question`, `filters`, `top_k`, `prompt_version`, `retriever_version` và `reranker_version`.
- [ ] Hỗ trợ cache Redis cho môi trường chạy thật và fallback in-memory cache cho local/dev.
- [ ] Tối ưu latency bằng cách đo thời gian từng stage: cache, dense retrieval, sparse retrieval, hybrid fusion, rerank, generation và total.
- [ ] Xây logging nâng cao cho request lỗi, exception, timeout, câu trả lời rỗng, retrieval score thấp, citation thiếu và feedback xấu.
- [ ] Lưu log phân tích chất lượng vào định dạng JSONL hoặc structured log để TV5 dùng lại cho benchmark/error analysis.
- [ ] Review các PR thay đổi `api/`, `contracts/`, `configs/`, orchestrator hoặc pipeline flow.

## 3. Quy chuẩn Clean Code & API Contract

### Clean Code bắt buộc

- Áp dụng triệt để DRY, tối ưu số dòng code và không tạo class/interface dư thừa nếu chưa có nhu cầu thật.
- Dùng type hinting đầy đủ cho mọi input/output; schema request/response bắt buộc dùng Pydantic.
- Mỗi hàm chỉ làm một trách nhiệm rõ ràng; router chỉ validate request và gọi service, không chứa logic retrieval, prompt, rerank hoặc cache phức tạp.
- Không copy-paste logic logging, response mapping hoặc latency tracking giữa các endpoint; tách helper nhỏ khi có lặp lại thực sự.
- Không để dead code, debug print, config hard-code hoặc exception bị nuốt im lặng.

### API Contract

Input chính:

```python
class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    top_n: int = 5
    filters: dict[str, str | int | list[str]] | None = None
    prompt_version: str = "legal_qa_v1"
    debug: bool = False
```

Pipeline nội bộ:

```python
async def answer(request: QueryRequest) -> QueryResponse: ...
```

Các module phải được gọi qua contract chung:

```python
search_dense(query: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]
search_hybrid(query: str, dense_hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]
rerank(query: str, candidates: list[RetrievalHit], top_n: int) -> list[RetrievalHit]
generate_answer(question: str, contexts: list[RetrievalHit], prompt_version: str) -> QAResponse
```

Output chính:

```text
answer, citations, retrieval_hits, latency_ms, cache_hit, prompt_version, warnings, trace_id
```

Tất cả schema dùng chung phải đặt trong `src/udsc2026/contracts/`; không định nghĩa lại `RetrievalHit`, `Citation`, `QAResponse` trong router.

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] `/query` chạy end-to-end với module thật hoặc mock chuẩn của TV2, TV3, TV4 và TV5.
- [ ] Response có answer, citations, retrieval hits, latency từng stage, `cache_hit` và `trace_id`.
- [ ] Câu hỏi lặp lại tạo cache hit, trả kết quả đúng và giảm latency rõ ràng.
- [ ] Redis cache chạy được khi có cấu hình; in-memory cache chạy được khi dev local.
- [ ] Low-score retrieval, lỗi QA, citation thiếu và exception được ghi log structured rõ ràng.
- [ ] Endpoint health/readiness phản ánh đúng trạng thái model, vector DB và cache.
- [ ] Không có business logic phức tạp nằm trong FastAPI router.
