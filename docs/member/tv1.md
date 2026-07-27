# TV1 - Backend Core, System Architecture, PR Review

## Mục tiêu & Phạm vi công việc

- [ ] Thiết kế và duy trì kiến trúc tổng thể của hệ thống RAG LegalIR & LegalQA.
- [ ] Xây dựng FastAPI Backend Core, đảm bảo API trả JSON/SSE độc lập với React frontend.
- [ ] Chuẩn hóa dependency injection cho retriever, QA engine, config và logging.
- [ ] Định nghĩa nguyên tắc review PR, coding convention và checklist merge.
- [ ] Bổ sung module metrics cơ bản: MRR, Accuracy, ROUGE.

## Thư mục mã nguồn phụ trách

- [ ] `src/dsc2026_legal/api/`
- [ ] `src/dsc2026_legal/contracts/`
- [ ] `src/dsc2026_legal/evaluation/metrics.py`
- [ ] `configs/`
- [ ] `docs/10_system_design.md`

## API/Interface đầu ra cần bàn giao

- [ ] `GET /health`: kiểm tra backend và trạng thái model/vector store.
- [ ] `POST /api/v1/query`: nhận câu hỏi, trả answer, citations, retrieval metadata.
- [ ] `GET /api/v1/query/stream`: streaming token/event cho React.
- [ ] Pydantic schemas: `QueryRequest`, `QueryResponse`, `Citation`, `RetrievalHit`.
- [ ] Hàm metrics: `mean_reciprocal_rank()`, `accuracy_at_k()`, `rouge_score()`.

## Checklist nghiệm thu công việc

- [ ] Backend chạy được bằng `uvicorn dsc2026_legal.api.app:app --reload`.
- [ ] API không import trực tiếp code UI hoặc notebook thử nghiệm.
- [ ] Response schema ổn định để TV2, TV3, TV4, TV5 tích hợp.
- [ ] Có unit test cho health check, query contract và metrics.
- [ ] Mọi PR chạm vào contract/API đều được review trước khi merge.
