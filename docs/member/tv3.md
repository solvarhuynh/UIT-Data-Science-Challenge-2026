# TV3 - QA & LLM: Qwen3 Local, Prompt System, BM25 và Hybrid Search

## 1. Tổng quan vai trò

TV3 chịu trách nhiệm biến kết quả truy hồi thành câu trả lời pháp lý có căn cứ. Trọng tâm là tích hợp Local LLM Qwen 3, quản lý prompt versioning trong `prompts/`, thiết kế system prompt bắt buộc citation theo Điều/Khoản và kiểm soát hallucination. Đây là vị trí ảnh hưởng trực tiếp tới chất lượng LegalQA: câu trả lời phải hữu ích, bám nguồn và tránh bịa citation. Phần bổ sung là xây Sparse Retrieval bằng BM25 và viết hàm Hybrid Search kết hợp Dense từ TV2 với Sparse để cải thiện LegalIR.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `LLMClient` trong `src/udsc2026/infrastructure/llm/` để load Qwen 3 từ `./models/qwen3-legal`, chỉ cung cấp client Python thuần độc lập với FastAPI để TV1 gọi trong End-to-End Pipeline.
- [ ] Hỗ trợ hai backend inference trong `LLMClient`: `transformers` baseline và `vLLM` nếu môi trường GPU cho phép, chỉ trả kết quả inference cho `QAEngine` và không điều phối Streaming (SSE) của TV1.
- [ ] Thiết kế `QAEngine` trong `src/udsc2026/qa/`, nhận question và list context `RetrievalHit` import từ `src/udsc2026/contracts/retrieval.py`, trả answer kèm citation theo Pydantic Schema hoặc kiểu dữ liệu đã thống nhất, không chỉnh sửa FastAPI Router, API Endpoint, Dependency Injection, Middleware, Streaming (SSE) hoặc thành phần điều phối request của TV1.
- [ ] Quản lý prompt ở `prompts/`, tối thiểu có `system/legal_qa_v1.md` và template RAG có vùng `CONTEXT`, `QUESTION`, `ANSWER`, chỉ kiểm soát prompt của TV3 và không thay đổi schema dữ liệu/citation do TV4 cung cấp.
- [ ] Prompt phải yêu cầu model chỉ trả lời dựa trên context, trích dẫn dạng `[Tên luật, Điều X, Khoản Y]`, và từ chối nếu thiếu căn cứ, trong phạm vi sinh câu trả lời của TV3 mà không thay đổi logic retrieval Dense của TV2 hoặc rerank của TV5.
- [ ] Viết parser citation để map câu trả lời về `chunk_id` hoặc metadata gốc, chỉ đọc metadata từ retrieval hits và không sửa cấu trúc chunk/citation nguồn của TV4.
- [ ] Cấu hình BM25 trong `src/udsc2026/retrieval/sparse/`, tokenization phù hợp tiếng Việt và giữ số điều/khoản, trả `RetrievalHit` từ contract chung và chỉ triển khai retrieval core để TV1 tích hợp qua lớp API riêng của TV1.
- [ ] Viết `HybridRetriever` trong `src/udsc2026/retrieval/hybrid/` với công thức baseline `0.5 * dense_score + 0.5 * sparse_score` sau normalize, nhận Dense từ TV2 và trả `RetrievalHit` từ contract chung cho pipeline mà không can thiệp vào phần rerank của TV5 hoặc điều phối API của TV1.

## 3. API Contract & Dữ liệu giao tiếp

QA engine nhận:

Mọi API trong phạm vi TV3 là hàm Python thuần thuộc Core Logic để TV1 tích hợp vào FastAPI Router hoặc Endpoint. `RetrievalHit` phải được import từ `src/udsc2026/contracts/retrieval.py`; TV3 không tạo model `RetrievalHit` riêng.
```python
from udsc2026.contracts.retrieval import RetrievalHit

generate_answer(question: str, contexts: list[RetrievalHit], stream: bool = False) -> QAResponse
```

BM25 nhận:

```python
from udsc2026.contracts.retrieval import RetrievalHit

search(query: str, top_k: int) -> list[RetrievalHit]
```

Hybrid nhận dense và sparse candidates:

```python
from udsc2026.contracts.retrieval import RetrievalHit

merge(query: str, dense_hits: list[RetrievalHit], sparse_hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]
```

Output cho API:

```text
answer, citations, used_prompt_version, retrieval_hits, confidence, warnings
```

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] Qwen 3 load từ local model path, không cần API LLM bên ngoài.
- [ ] Có prompt version rõ ràng và được ghi trong response metadata.
- [ ] Câu trả lời có citation khi context đủ căn cứ.
- [ ] Khi context thiếu, model trả lời từ chối đúng quy tắc.
- [ ] BM25 chạy độc lập và có test với query chứa số điều/tên luật.
- [ ] Hybrid Search normalize score trước khi cộng và giữ metadata để TV5 rerank.
