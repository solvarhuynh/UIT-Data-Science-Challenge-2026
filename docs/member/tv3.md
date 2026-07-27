# TV3 - QA & LLM: Qwen3 Local, Prompt System, BM25 và Hybrid Search

## 1. Tổng quan vai trò

TV3 chịu trách nhiệm biến kết quả truy hồi thành câu trả lời pháp lý có căn cứ. Trọng tâm là tích hợp Local LLM Qwen 3, quản lý prompt versioning trong `prompts/`, thiết kế system prompt bắt buộc citation theo Điều/Khoản và kiểm soát hallucination. Đây là vị trí ảnh hưởng trực tiếp tới chất lượng LegalQA: câu trả lời phải hữu ích nhưng không được bịa nguồn. Phần bổ sung là xây Sparse Retrieval bằng BM25 và viết hàm Hybrid Search kết hợp Dense từ TV2 với Sparse để cải thiện LegalIR.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `LLMClient` trong `src/udsc2026/infrastructure/llm/` để load Qwen 3 từ `./models/qwen3-legal`.
- [ ] Hỗ trợ hai backend inference: `transformers` baseline và `vLLM` nếu môi trường GPU cho phép.
- [ ] Thiết kế `QAEngine` trong `src/udsc2026/qa/`, nhận question và list context từ retrieval, trả answer kèm citation.
- [ ] Quản lý prompt ở `prompts/`, tối thiểu có `system/legal_qa_v1.md` và template RAG có vùng `CONTEXT`, `QUESTION`, `ANSWER`.
- [ ] Prompt phải yêu cầu model chỉ trả lời dựa trên context, trích dẫn dạng `[Tên luật, Điều X, Khoản Y]`, và từ chối nếu thiếu căn cứ.
- [ ] Viết parser citation để map câu trả lời về `chunk_id` hoặc metadata gốc.
- [ ] Cấu hình BM25 trong `src/udsc2026/retrieval/sparse/`, tokenization phù hợp tiếng Việt và giữ số điều/khoản.
- [ ] Viết `HybridRetriever` trong `src/udsc2026/retrieval/hybrid/` với công thức baseline `0.5 * dense_score + 0.5 * sparse_score` sau normalize.

## 3. API Contract & Dữ liệu giao tiếp

QA engine nhận:

```python
generate_answer(question: str, contexts: list[RetrievalHit], stream: bool = False) -> QAResponse
```

BM25 nhận:

```python
search(query: str, top_k: int) -> list[RetrievalHit]
```

Hybrid nhận dense và sparse candidates:

```python
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
