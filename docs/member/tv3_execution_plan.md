---
title: TV3 Detailed Execution Plan - LegalQA & LLM Infrastructure
role: TV3 - QA Engine, Prompt Engineering & Local LLM Specialist
version: 1.0.0
last_updated: 2026-07-28
status: APPROVED_BASELINE
btc_constraints:
  max_parameters: 4_000_000_000 # Must be strictly < 4B params
  license: Open-Source Only (No Commercial API / No Closed Models)
  execution_mode: Local Offline / Dockerized
models:
  baseline:
    name: thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2
    params: 1.7B
    local_path: ./models/qwen3-legal
  upgrade_candidate:
    name: Qwen/Qwen2.5-3B-Instruct
    params: 3.0B
    local_path: ./models/qwen2.5-3b-instruct
---

# Kế hoạch Triển khai Chi tiết TV3: QA Engine & Local LLM Layer

## 🎯 1. Mục tiêu & Phạm vi trách nhiệm

TV3 chịu trách nhiệm toàn bộ tầng **Sinh câu trả lời (Generation)**, **Quản lý Prompt (Prompt Registry)** và **Xác minh Trích dẫn (Citation Parsing & Anti-Hallucination)** cho hệ thống RAG Pháp luật UDSC 2026.

### Thư mục do TV3 làm chủ (Ownership):
- `src/udsc2026/qa/` - Mã nguồn QA Engine, Prompt Builder, Citation Parser.
- `src/udsc2026/infrastructure/llm/` - Adapter load và gọi mô hình LLM local (`transformers` / `vLLM`).
- `prompts/` - Lưu trữ và quản lý phiên bản (versioning) của tất cả System/RAG Prompts.
- `experiments/tv3/` - Thử nghiệm mô hình (1.7B vs 3B), đo tỷ lệ hallucination và tuning prompt.

---

## ⚡ 2. Quy định BTC & Nguyên tắc Kỹ thuật

1. **Giới hạn Mô hình**: Chỉ sử dụng LLM open-source có **kích thước < 4B tham số**.
2. **Không phụ thuộc Cloud API**: Không gọi OpenAI, Claude, Gemini hay bất kỳ API bên ngoài nào.
3. **Tuyệt đối Chống Bịa đặt (Zero-Hallucination)**: LLM chỉ trả lời dựa vào `contexts` (`list[RetrievalHit]`) được TV1/TV5 cung cấp. Nếu ngữ cảnh thiếu, bắt buộc từ chối trả lời theo quy tắc.
4. **Trích dẫn Chuẩn xác**: Mọi kết luận pháp lý phải đi kèm trích dẫn dạng `[Tên Luật, Điều X, Khoản Y]`. Citation phải được validate khớp với metadata thực tế.

---

## 🛠️ 3. Chi tiết Các Giai đoạn Thực hiện (Phases & Checklist)

### 📌 Phase 1: Tầng Hạ tầng LLM (`src/udsc2026/infrastructure/llm/`)
- [ ] **1.1. Schema Config LLM**: Tạo Pydantic model `LLMConfig` chứa các tham số:
  - `model_path`: Đường dẫn local checkpoint (`./models/qwen3-legal`).
  - `backend`: `"transformers"` hoặc `"vllm"`.
  - `device`: `"cuda"` hoặc `"cpu"`.
  - `dtype`: `"bfloat16"`, `"float16"`, hoặc `"float32"`.
  - `max_new_tokens`: Mặc định `1024`.
  - `temperature`: Mặc định `0.1` (giữ nhiệt độ thấp để giảm hallucination).
  - `top_p`: Mặc định `0.9`.
  - `repetition_penalty`: Mặc định `1.05`.
- [ ] **1.2. LLMClient Implementation**:
  - Tải model & tokenizer từ local path (sử dụng `AutoModelForCausalLM`, `AutoTokenizer`).
  - Hỗ trợ hàm `generate(prompt: str, config: LLMConfig) -> str`.
  - Hỗ trợ hàm `generate_stream(...)` cho tính năng streaming câu trả lời.
  - Xử lý mượt mà khi GPU không khả dụng (fallback CPU nhẹ nhàng).
- [ ] **1.3. Unit Test Mock Client**: Viết Mock LLM Client để TV1/TV5 có thể test integration mà không cần load weights thật.

---

### 📌 Phase 2: Thư viện Prompt & PromptBuilder (`prompts/` & `src/udsc2026/qa/`)
- [ ] **2.1. Thiết kế Thư mục `prompts/`**:
  ```text
  prompts/
  ├── README.md
  ├── system/
  │   └── legal_qa_v1.md          # System prompt chuẩn pháp lý tiếng Việt
  └── rag_templates/
      └── default_rag_v1.md       # Template ghép Context + Question
  ```
- [ ] **2.2. Xây dựng Strict System Prompt (`prompts/system/legal_qa_v1.md`)**:
  ```markdown
  Bạn là Trợ lý Pháp luật Việt Nam chuyên nghiệp và trung thực.
  Nhiệm vụ của bạn là trả lời CÂU HỎI của người dùng DỰA HOÀN TOÀN VÀO DỮ LIỆU CONTEXT ĐƯỢC CUNG CẤP.

  QUY TẮC BẮT BUỘC:
  1. CHỈ sử dụng thông tin có trong CONTEXT. KHÔNG suy diễn hoặc sử dụng kiến thức bên ngoài.
  2. Nếu CONTEXT không chứa đủ căn cứ để trả lời, bạn BẮT BUỘC trả lời: "Dựa trên dữ liệu pháp lý được cung cấp, không có đủ căn cứ để trả lời câu hỏi này."
  3. Mọi khẳng định pháp lý BẮT BUỘC phải kèm trích dẫn ở dạng: [Tên văn bản, Điều X, Khoản Y, Điểm Z].
  4. Trình bày ngắn gọn, rõ ràng, đúng trọng tâm.
  ```
- [ ] **2.3. Viết `PromptBuilder` (`src/udsc2026/qa/prompt_builder.py`)**:
  - Hàm `build_prompt(question: str, contexts: list[RetrievalHit], prompt_version: str) -> str`.
  - Tự động render danh sách `RetrievalHit` thành chuỗi văn bản đánh số rõ ràng kèm metadata cho LLM đọc.
  - Ghi nhận `prompt_version` để trace lại trong response.

---

### 📌 Phase 3: Extraction & Validation Trích dẫn (`src/udsc2026/qa/citation_parser.py`)
- [ ] **3.1. Định nghĩa Citation Contract**:
  - Pydantic Model `Citation`: `law_name`, `article`, `clause`, `point`, `chunk_id`, `is_verified: bool`, `warning: str | None`.
- [ ] **3.2. Viết `CitationParser`**:
  - Dùng Regex bóc tách các mẫu trích dẫn tiếng Việt phổ biến từ văn bản LLM sinh ra:
    - Mẫu 1: `[Bộ luật Lao động 2019, Điều 10, Khoản 1]`
    - Mẫu 2: `[Điều 15 Luật Doanh nghiệp 2020]`
    - Mẫu 3: `[Khoản 2 Điều 5]`
- [ ] **3.3. Viết Logic Validator (`validate_citations`)**:
  - Đối chiếu danh sách `Citation` vừa bóc tách với danh sách `RetrievalHit` gốc.
  - Nếu trích dẫn KHÔNG tồn tại trong các chunks được cung cấp $\rightarrow$ Đánh dấu `is_verified = False` và thêm warning: `"Cảnh báo: Trích dẫn này không có trong tài liệu truy hồi (Phát hiện Hallucination)"`.

---

### 📌 Phase 4: Core QAEngine (`src/udsc2026/qa/qa_engine.py`)
- [ ] **4.1. Tích hợp Pipeline QA**:
  - Lớp `QAEngine`:
    ```python
    class QAEngine:
        def __init__(self, llm_client: LLMClient, prompt_builder: PromptBuilder, citation_parser: CitationParser): ...
        
        async def generate_answer(
            self, 
            question: str, 
            contexts: list[RetrievalHit], 
            prompt_version: str = "legal_qa_v1"
        ) -> QAResponse: ...
    ```
- [ ] **4.2. Khâu xử lý An toàn (Safety Guard)**:
  - Nếu `contexts` trống hoặc tất cả retrieval score $< \text{threshold}$ $\rightarrow$ Trực tiếp trả về thông báo từ chối mà không cần gọi LLM (tiết kiệm GPU & giảm latency).
- [ ] **4.3. Định dạng Output `QAResponse`**:
  - Trả về đối tượng Pydantic khớp 100% với `src/udsc2026/contracts/qa.py`:
    `answer`, `citations`, `used_prompt_version`, `retrieval_hits`, `confidence`, `warnings`.

---

### 📌 Phase 5: Thử nghiệm Model & Đánh giá (`experiments/tv3/`)
- [ ] **5.1. Thử nghiệm Baseline 1.7B**:
  - Chạy `python download_models.py` tải `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2`.
  - Đo tốc độ sinh token (tokens/sec) và kiểm tra khả năng định dạng trích dẫn.
- [ ] **5.2. Thử nghiệm Nâng cấp 3B**:
  - Tải `Qwen/Qwen2.5-3B-Instruct`.
  - So sánh tỷ lệ trả lời đúng + tỷ lệ vi phạm hallucination giữa 1.7B vs 3B trên 50 câu hỏi thử nghiệm.
- [ ] **5.3. Viết Báo cáo So sánh (Model Recommendation Report)** gửi Leader (TV1) và MLOps (TV5).

---

## 🔗 4. API Contract & Schema Tham chiếu

Để đảm bảo tương thích 100% với TV1, TV2, TV4 và TV5, TV3 bắt buộc tuân thủ các signature sau:

```python
# API Contract chuẩn cho QA Engine
from udsc2026.contracts.qa import QAResponse, Citation
from udsc2026.contracts.retrieval import RetrievalHit

async def generate_answer(
    question: str,
    contexts: list[RetrievalHit],
    prompt_version: str = "legal_qa_v1",
) -> QAResponse:
    """
    Sinh câu trả lời từ question và context đã được truy hồi & rerank.
    """
    ...

def parse_and_validate_citations(
    answer_text: str,
    contexts: list[RetrievalHit],
) -> tuple[list[Citation], list[str]]:
    """
    Trích xuất và kiểm tra trích dẫn hợp lệ so với context nguồn.
    Trả về (danh sách Citation, danh sách Cảnh báo Warnings).
    """
    ...
```

---

## 🏁 5. Tiêu chuẩn Nghiệm thu (Definition of Done - DoD)

- [ ] Model LLM local load thành công từ đĩa (`./models/qwen3-legal`), không gọi bất kỳ API cloud nào.
- [ ] `QAEngine` nhận `list[RetrievalHit]` và trả về `QAResponse` chuẩn Pydantic schema.
- [ ] Không có hiện tượng tự bịa Điều/Khoản luật ngoài ngữ cảnh được cung cấp.
- [ ] `CitationParser` tự động phát hiện và gán cờ cảnh báo nếu xuất hiện trích dẫn sai.
- [ ] Quản lý prompt hoàn toàn qua thư mục `prompts/`, version được ghi vết đầy đủ trong output.
- [ ] Đạt điểm kiểm thử Unit Test trong `tests/unit/test_qa/`.
