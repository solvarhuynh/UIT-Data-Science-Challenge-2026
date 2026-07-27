# TV3 - QA & LLM Specialist

## 1. Tổng quan vai trò

TV3 phụ trách toàn bộ QA Engine và Local LLM layer cho hệ thống RAG Pháp luật DSC2026. Phạm vi chính nằm trong `src/udsc2026/qa/`, `src/udsc2026/infrastructure/llm/` và `prompts/`.

Trọng tâm của TV3 là load local LLM Qwen3, thiết kế prompt chống hallucination, quản lý prompt versioning, sinh câu trả lời pháp lý dựa trên context đã được retrieval/rerank và kiểm tra citation Điều/Khoản/Điểm chính xác.

TV3 không phụ trách BM25, Sparse Retrieval, Hybrid Search, Dense Retrieval, VectorDB, Web Frontend hoặc FastAPI orchestration. Context đầu vào của TV3 đến từ TV1/TV5 dưới dạng `list[RetrievalHit]` đã được truy hồi và rerank.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `LLMClient` trong `src/udsc2026/infrastructure/llm/` để load Qwen3 từ `models/qwen3-legal` hoặc path cấu hình.
- [ ] Hỗ trợ inference bằng `transformers`; bổ sung backend `vLLM` khi môi trường GPU cho phép.
- [ ] Thiết kế config LLM gồm `model_path`, `backend`, `device`, `dtype`, `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty` và timeout nếu cần.
- [ ] Xây `QAEngine` trong `src/udsc2026/qa/`, nhận question và contexts đã rerank, trả `QAResponse`.
- [ ] Viết `PromptBuilder` đọc template từ `prompts/`, không hard-code prompt dài trong Python.
- [ ] Quản lý prompt versioning trong `prompts/`, tối thiểu có `prompts/system/legal_qa_v1.md` và template RAG riêng cho context/question.
- [ ] Thiết kế System Prompt chống hallucination: chỉ trả lời dựa trên context được cung cấp, không suy diễn ngoài tài liệu, từ chối khi thiếu căn cứ pháp lý và nêu rõ khi context không đủ.
- [ ] Chuẩn hóa citation format, ví dụ `[Bộ luật Lao động 2019, Điều 10, Khoản 1]`, và map citation về metadata gốc trong `RetrievalHit`.
- [ ] Viết `CitationParser` trong `src/udsc2026/qa/` để trích xuất Điều/Khoản/Điểm từ câu trả lời của LLM.
- [ ] Viết logic validate citation: citation trong answer phải khớp với ít nhất một context đã cung cấp, ưu tiên so khớp theo `law_name`, `article`, `clause`, `point`, `chunk_id`.
- [ ] Trả warning rõ ràng khi citation thiếu, sai format hoặc không map được về chunk gốc.
- [ ] Viết test cho prompt builder, prompt version loading, LLM client mock, QA engine, citation parser và anti-hallucination behavior.

## 3. Quy chuẩn Clean Code & API Contract

### Clean Code bắt buộc

- Áp dụng DRY, tách rõ `LLMClient`, `PromptBuilder`, `QAEngine`, `CitationParser` và citation validator.
- Dùng type hinting đầy đủ cho mọi input/output; `QAResponse`, `Citation`, `PromptMetadata`, `LLMConfig` nên dùng Pydantic hoặc schema chung.
- Mỗi hàm chỉ làm một trách nhiệm: build prompt, render context, call LLM, parse response, extract citation hoặc validate citation.
- Không hard-code prompt dài trong Python; prompt bắt buộc nằm trong `prompts/` và có version.
- Không viết BM25, Hybrid Search, Dense Retrieval, VectorDB adapter hoặc reranking trong phạm vi TV3.
- Không định nghĩa lại `RetrievalHit` hoặc chunk schema của TV4; luôn import từ contract chung.
- Không để QA Engine tự truy vấn retrieval; retrieval là dependency đầu vào do TV1/TV5 cung cấp.
- Log/metadata phải đủ để truy vết prompt version, model path/backend, citation warnings và context đã sử dụng.

### API Contract

QA engine:

```python
from udsc2026.contracts.qa import QAResponse
from udsc2026.contracts.retrieval import RetrievalHit

def generate_answer(
    question: str,
    contexts: list[RetrievalHit],
    prompt_version: str = "legal_qa_v1",
    stream: bool = False,
) -> QAResponse: ...
```

LLM client:

```python
def generate(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str: ...
```

Citation parser:

```python
from udsc2026.contracts.qa import Citation

def parse_citations(answer: str) -> list[Citation]: ...

def validate_citations(
    citations: list[Citation],
    contexts: list[RetrievalHit],
) -> list[Citation]: ...
```

Output QA cần có:

```text
answer, citations, used_prompt_version, retrieval_hits, confidence, warnings
```

Citation cần có:

```text
law_name, article, clause, point, chunk_id, source, is_verified, warning
```

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] Qwen3 load được từ local model path bằng `transformers` hoặc `vLLM`.
- [ ] Prompt có version, lưu trong `prompts/` và version được ghi vào response metadata.
- [ ] `QAEngine` nhận context từ retrieval/reranking và không tự thực hiện BM25/Hybrid/Dense search.
- [ ] Câu trả lời chỉ dùng context được cung cấp và có citation đúng khi đủ căn cứ.
- [ ] Khi context thiếu hoặc retrieval score thấp, model từ chối trả lời theo quy tắc chống hallucination.
- [ ] `CitationParser` trích xuất được citation Điều/Khoản/Điểm từ câu trả lời của LLM.
- [ ] Citation validator phát hiện được citation thiếu, sai format hoặc không map được về chunk gốc.
- [ ] Unit test bao phủ prompt builder, citation parser, citation validation và QA engine với LLM mock.
- [ ] Không còn bất kỳ nhiệm vụ BM25, Sparse Retrieval hoặc Hybrid Search trong file giao việc của TV3.
