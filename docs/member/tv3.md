# TV3 - QA & LLM Specialist

## 1. Tổng quan vai trò

TV3 phụ trách QA Engine và LLM layer cho hệ thống RAG Pháp luật DSC2026. Trọng tâm là load local LLM `Qwen3` bằng `transformers` hoặc `vLLM`, quản lý prompt versioning trong `prompts/`, sinh câu trả lời có citation chính xác và chống hallucination.

TV3 cũng phụ trách Sparse Retrieval BM25 và Hybrid Search để kết hợp điểm Dense của TV2 với điểm Sparse. TV3 không làm Web Frontend và không điều phối FastAPI end-to-end.

## 2. Nhiệm vụ kỹ thuật chi tiết

- [ ] Viết `LLMClient` trong `src/udsc2026/infrastructure/llm/` để load Qwen3 từ `models/qwen3-legal` hoặc path cấu hình.
- [ ] Hỗ trợ inference bằng `transformers`; bổ sung backend `vLLM` khi môi trường GPU cho phép.
- [ ] Xây `QAEngine` trong `src/udsc2026/qa/`, nhận question và contexts đã rerank, trả `QAResponse`.
- [ ] Quản lý prompt trong `prompts/`, tối thiểu có `prompts/system/legal_qa_v1.md` và template RAG riêng cho context/question.
- [ ] Viết System Prompt ép model chỉ trả lời dựa trên context được cung cấp, trích dẫn chính xác và từ chối khi thiếu căn cứ pháp lý.
- [ ] Chuẩn hóa citation format, ví dụ `[Bộ luật Lao động 2019, Điều 10, Khoản 1]`, và map citation về metadata gốc.
- [ ] Viết `CitationParser` để kiểm tra citation trong answer có khớp `RetrievalHit` hay không.
- [ ] Xây Sparse Retrieval bằng BM25 trong `src/udsc2026/retrieval/sparse/`, tokenization phù hợp tiếng Việt và giữ nguyên số điều/khoản/điểm.
- [ ] Xây `HybridRetriever` trong `src/udsc2026/retrieval/hybrid/`, nhận dense hits từ TV2 và sparse hits từ BM25, normalize score rồi fusion theo cấu hình.
- [ ] Hỗ trợ cấu hình trọng số `dense_weight`, `sparse_weight`, `top_k`, `min_score`.
- [ ] Viết test cho prompt builder, citation parser, BM25 search và hybrid score fusion.

## 3. Quy chuẩn Clean Code & API Contract

### Clean Code bắt buộc

- Áp dụng triệt để DRY, tối ưu số dòng code và không tạo class/interface dư thừa nếu chưa có nhu cầu thật.
- Dùng type hinting đầy đủ cho mọi input/output; `QAResponse`, citation và prompt metadata phải có schema rõ ràng.
- Mỗi hàm chỉ làm một trách nhiệm: build prompt, call LLM, parse citation, sparse search hoặc score fusion.
- Không hard-code prompt dài trong Python; prompt bắt buộc nằm trong `prompts/` và có version.
- Không copy-paste logic normalize/fusion; viết helper nhỏ, có test.
- Không định nghĩa lại `RetrievalHit` hoặc chunk schema của TV4.

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

Sparse search:

```python
def search(query: str, top_k: int) -> list[RetrievalHit]: ...
```

Hybrid search:

```python
def merge(
    query: str,
    dense_hits: list[RetrievalHit],
    sparse_hits: list[RetrievalHit],
    top_k: int,
) -> list[RetrievalHit]: ...
```

Output QA cần có:

```text
answer, citations, used_prompt_version, retrieval_hits, confidence, warnings
```

Hybrid hit cần giữ:

```text
chunk_id, dense_score, sparse_score, hybrid_score, metadata
```

## 4. Tiêu chuẩn nghiệm thu (Definition of Done)

- [ ] Qwen3 load được từ local model path bằng `transformers` hoặc `vLLM`.
- [ ] Prompt có version, lưu trong `prompts/` và version được ghi vào response metadata.
- [ ] Câu trả lời chỉ dùng context được cung cấp và có citation đúng khi đủ căn cứ.
- [ ] Khi context thiếu hoặc retrieval score thấp, model từ chối trả lời theo quy tắc.
- [ ] BM25 chạy độc lập và có test với query chứa tên luật, số điều, số khoản.
- [ ] Hybrid Search normalize score trước khi fusion và giữ metadata citation cho TV5 rerank.
- [ ] Citation parser phát hiện được citation thiếu, sai hoặc không map được về chunk gốc.
