# TV1 - Leader: FastAPI Backend Core & Evaluation Metrics

## 1. Tổng quan vai trò

TV1 chịu trách nhiệm làm xương sống kỹ thuật của toàn hệ thống DSC2026 LegalIR & LegalQA. Trọng tâm là xây dựng FastAPI Backend Core đủ sạch để các nhóm retrieval, QA, ingestion, reranking và frontend có thể tích hợp song song mà không giẫm lên nhau. Vai trò Leader không chỉ là viết API, mà còn là người giữ chuẩn kiến trúc: quản lý dependency injection, thống nhất Pydantic contracts, thiết lập router, CORS, lifecycle load model, streaming response qua SSE và review PR trước khi merge vào `main`. Phần bổ sung của TV1 là xây module metrics cơ bản để cả nhóm đo chất lượng theo cùng một cách.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Tạo entrypoint `src/udsc2026/api/app.py` với `FastAPI()`, lifespan startup/shutdown và health check.
- [ ] Tổ chức router trong `src/udsc2026/api/routes/`, tối thiểu gồm `health.py`, `query.py`, `stream.py`.
- [ ] Cấu hình CORS cho React frontend ở `http://localhost:5173`, có biến cấu hình trong `configs/`.
- [ ] Thiết kế dependency injection để route không khởi tạo trực tiếp retriever hoặc LLM. Service được lấy qua `app.state` hoặc provider function như `get_retriever()`, `get_qa_engine()`.
- [ ] Viết SSE endpoint cho LLM streaming, trả event theo format `token`, `citation`, `final`, `error`.
- [ ] Chuẩn hóa error handling: lỗi thiếu model trả `503`, request sai trả `422`, lỗi pipeline trả JSON có `warnings`.
- [ ] Quản lý Git workflow: mỗi thành viên làm branch riêng, PR phải mô tả thay đổi contract, config và test đã chạy.
- [ ] Viết metrics trong `src/udsc2026/evaluation/metrics.py`: `mrr()`, `recall_at_k()`, `rouge_l()`, `bleu_score()`.

## 3. API Contract & Dữ liệu giao tiếp

Input chính từ frontend:

```python
QueryRequest(question: str, top_k: int = 5, filters: dict | None = None, stream: bool = False)
```

Output chuẩn:

```python
QueryResponse(answer: str, citations: list[Citation], retrieval_hits: list[RetrievalHit], latency_ms: float, warnings: list[str])
```

Streaming output:

```json
{"type": "token", "content": "..."}
{"type": "final", "answer": "...", "citations": []}
```

Metrics nhận prediction/reference hoặc ranking list và trả float hoặc dict metric để TV5 dùng batch evaluation.

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] `uvicorn udsc2026.api.app:app --reload` chạy được ở cổng `8000`.
- [ ] API route không import code từ `experiments/` hoặc frontend.
- [ ] CORS hoạt động với React dev server.
- [ ] SSE stream được ít nhất token giả lập và final response.
- [ ] Metrics có unit test cho case đúng, sai, rỗng và nhiều đáp án đúng.
- [ ] PR của các thành viên thay đổi schema phải được TV1 review trước khi merge.
