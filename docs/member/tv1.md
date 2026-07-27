# TV1 - Integration & Orchestration

## 1. Tổng quan vai trò

TV1 là người lắp ráp và tối ưu hóa luồng xử lý end-to-end của hệ thống RAG. Vai trò này không chỉ quản lý API mà tập trung vào việc kết nối các module lõi (Retrieval, QA) thành một pipeline hoàn chỉnh, đo lường và tối ưu hiệu năng (latency, bottlenecks). TV1 cũng chịu trách nhiệm xây dựng các cơ chế hỗ trợ như Caching và Logging để cải thiện trải nghiệm người dùng và tạo vòng lặp dữ liệu (data flywheel).

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] **Orchestration**: Trong `api/routes/query.py`, gọi tuần tự các hàm core logic từ `retrieval` (TV2) và `qa` (TV3) để xử lý một request từ đầu đến cuối, dựa trên contract chung `RetrievalHit` import từ `src/udsc2026/contracts/retrieval.py`.
- [ ] **Performance Tuning**: Tích hợp middleware hoặc decorator để đo lường latency của từng bước (retrieval, reranking, LLM generation) và xác định điểm nghẽn.
- [ ] **Caching**: Xây dựng cơ chế cache (ví dụ: dùng Redis) để lưu và trả về ngay lập tức các cặp câu hỏi-câu trả lời đã xử lý. Key cache có thể là hash của câu hỏi.
- [ ] **Logging for Feedback**:
    - Ghi nhận (log) các câu hỏi có điểm retrieval thấp từ TV2/TV5.
    - Ghi nhận các câu trả lời mà người dùng đánh giá không hài lòng (nếu frontend hỗ trợ).
    - Cung cấp log này cho TV5 để phân tích và cải thiện model/dữ liệu.
- [ ] **API Management**: Tiếp tục duy trì xương sống FastAPI: dependency injection, router, streaming (SSE), và review các PR liên quan đến `contracts/` và `api/`.

## 3. API Contract & Dữ liệu giao tiếp

- **Input**: Nhận `QueryRequest` từ client.
- **Internal Call (to TV2)**: `retriever.retrieve(query: str, top_k: int) -> list[RetrievalHit]`, với `RetrievalHit` import từ `src/udsc2026/contracts/retrieval.py`.
- **Internal Call (to TV3)**: `qa_engine.generate(query: str, contexts: list[RetrievalHit]) -> QAResponse`, dùng cùng `RetrievalHit` từ contract chung.
- **Output**: Trả về `QueryResponse` (cho non-stream) hoặc các `ServerSentEvent` (cho stream), bao gồm câu trả lời, citations, và thông tin debug (latency).

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] API endpoint `/query` hoạt động E2E, gọi đúng logic của TV2 và TV3.
- [ ] Latency của mỗi request được ghi nhận và hiển thị trong response (hoặc log).
- [ ] Một câu hỏi được hỏi 2 lần liên tiếp, lần thứ 2 phải có cache hit và tốc độ phản hồi < 50ms.
- [ ] Các truy vấn có điểm retrieval dưới ngưỡng được ghi vào file log riêng (`logs/low_score_queries.log`).
- [ ] Code tuân thủ nguyên tắc tách biệt: API route chỉ làm nhiệm vụ điều phối, không chứa logic business.
