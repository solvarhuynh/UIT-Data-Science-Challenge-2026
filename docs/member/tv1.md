# TV1 - FastAPI Core, Orchestrator, Caching & Observability

## 1. Tổng quan vai trò

TV1 phụ trách lớp FastAPI Core và điều phối end-to-end của hệ thống RAG Pháp luật DSC2026. Trọng tâm là gom các module Dense Retrieval của TV2, Hybrid/QA của TV3, dữ liệu từ TV4 và Reranker/Evaluation của TV5 thành một pipeline ổn định, đo được latency, cache được kết quả và log được lỗi chất lượng.

TV1 là người giữ API contract chung, đảm bảo backend chạy được từ request của frontend đến response cuối cùng mà không để logic nghiệp vụ rơi vào router.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Duy trì FastAPI app, router, dependency injection, config loading và error handling trong `src/udsc2026/api/`.
- [ ] Xây endpoint `/query` non-stream, nhận `QueryRequest`, gọi retrieval, rerank, QA và trả `QueryResponse` thống nhất.
- [ ] Xây endpoint streaming SSE cho frontend TV4 dùng khi cần hiển thị câu trả lời đang sinh dần.
- [ ] Viết `RAGOrchestrator` điều phối luồng: Dense TV2 -> Sparse/Hybrid TV3 -> Rerank TV5 -> QA TV3 -> Response.
- [ ] Tích hợp cache theo hash của `question`, `filters`, `top_k` và `prompt_version`; ưu tiên Redis, có fallback in-memory cho dev.
- [ ] Ghi latency từng bước: retrieval, sparse, hybrid, rerank, generation, total.
- [ ] Ghi log request, error, low-score retrieval và feedback người dùng để TV5 dùng cho evaluation.
- [ ] Review bắt buộc các PR chạm vào `api/`, `contracts/`, `configs/` hoặc thay đổi pipeline chung.

## 3. Quy chuẩn Code & API Contract

### Clean Code Standard

- Code ngắn gọn, rõ trách nhiệm; mỗi hàm chỉ xử lý một việc chính.
- Dùng Pydantic cho request/response schema; dùng type hints đầy đủ cho service và orchestrator.
- Không viết boilerplate, dead code, debug print hoặc logic trùng lặp giữa route và service.
- Không copy-paste code xử lý response/cache/log; tách helper khi lặp lại thật sự cần thiết.
- Router chỉ nhận request, validate và gọi service; business logic nằm trong orchestrator/service.

### API Contract

- Input chính: `QueryRequest` gồm `question`, `top_k`, `filters`, `stream`, `debug` nếu cần.
- Internal retrieval: `retrieve(query: str, top_k: int, filters: dict | None = None) -> list[RetrievalHit]`.
- Internal rerank: `rerank(query: str, candidates: list[RetrievalHit], top_n: int) -> list[RetrievalHit]`.
- Internal QA: `generate_answer(question: str, contexts: list[RetrievalHit]) -> QAResponse`.
- Output chính: `QueryResponse` gồm `answer`, `citations`, `retrieval_hits`, `latency_ms`, `cache_hit`, `warnings`.
- Chỉ dùng schema chung trong `src/udsc2026/contracts/`; không tạo contract riêng trong `api/`.

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] `/query` chạy được end-to-end với mock hoặc module thật của TV2, TV3, TV5.
- [ ] Response có answer, citations, latency và trạng thái cache rõ ràng.
- [ ] Lần hỏi thứ hai với cùng request có cache hit và latency giảm rõ.
- [ ] SSE endpoint stream được token/event cho frontend TV4.
- [ ] Log ghi được request lỗi, low-score query và timing từng bước.
- [ ] API route không chứa logic retrieval, rerank, prompt hoặc parsing citation.
