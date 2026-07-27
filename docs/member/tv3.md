# TV3 - QA Engine, Qwen3 Local, BM25 & Hybrid Fusion

## 1. Tổng quan vai trò

TV3 phụ trách lớp QA Engine và truy hồi từ khóa. Trọng tâm là Local Qwen3, prompt versioning, citation parsing, BM25 sparse search và Hybrid Fusion kết hợp Dense của TV2 với Sparse của TV3. TV3 tạo câu trả lời pháp lý có căn cứ, biết từ chối khi thiếu context và giữ citation đúng Điều/Khoản.

TV3 chỉ xây core logic Python cho TV1 gọi, không điều phối FastAPI và không làm frontend.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `LLMClient` trong `src/udsc2026/infrastructure/llm/` để load Qwen3 từ `./models/qwen3-legal`.
- [ ] Hỗ trợ backend inference `transformers`; thêm `vLLM` nếu môi trường GPU cho phép.
- [ ] Xây `QAEngine` trong `src/udsc2026/qa/`, nhận question và `list[RetrievalHit]`, trả `QAResponse`.
- [ ] Quản lý prompt trong `prompts/`, tối thiểu có `system/legal_qa_v1.md` và template RAG có `CONTEXT`, `QUESTION`, `ANSWER`.
- [ ] Prompt bắt buộc trả lời dựa trên context, trích dẫn dạng `[Tên luật, Điều X, Khoản Y]`, từ chối khi không đủ căn cứ.
- [ ] Viết citation parser để map citation trong answer về `chunk_id` và metadata gốc.
- [ ] Xây BM25 trong `src/udsc2026/retrieval/sparse/`, tokenization phù hợp tiếng Việt và giữ số điều/khoản.
- [ ] Xây `HybridRetriever` trong `src/udsc2026/retrieval/hybrid/`, normalize dense/sparse score và fusion theo cấu hình.

## 3. Quy chuẩn Code & API Contract

### Clean Code Standard

- Code ngắn, tách riêng LLM client, prompt builder, citation parser, BM25 và hybrid fusion.
- Dùng type hints rõ ràng cho input/output; dùng Pydantic cho `QAResponse`, citation và prompt metadata nếu cần.
- Không hard-code prompt dài trong Python; prompt phải nằm trong `prompts/` và có version.
- Không copy-paste normalize/fusion logic; viết một hàm dùng chung, có test.
- Không sửa schema chunk/citation của TV4 và không định nghĩa lại `RetrievalHit`.

### API Contract

QA engine:

```python
from udsc2026.contracts.retrieval import RetrievalHit

generate_answer(question: str, contexts: list[RetrievalHit], stream: bool = False) -> QAResponse
```

Sparse search:

```python
search(query: str, top_k: int) -> list[RetrievalHit]
```

Hybrid fusion:

```python
merge(query: str, dense_hits: list[RetrievalHit], sparse_hits: list[RetrievalHit], top_k: int) -> list[RetrievalHit]
```

Output QA cần có:

```text
answer, citations, used_prompt_version, retrieval_hits, confidence, warnings
```

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] Qwen3 load được từ local model path.
- [ ] Prompt có version và version được ghi trong response metadata.
- [ ] Câu trả lời có citation đúng khi context đủ căn cứ.
- [ ] Khi context thiếu, model từ chối theo đúng quy tắc.
- [ ] BM25 chạy độc lập và có test với query chứa tên luật, số điều, số khoản.
- [ ] Hybrid fusion normalize score trước khi cộng và giữ nguyên metadata cho TV5 rerank.
