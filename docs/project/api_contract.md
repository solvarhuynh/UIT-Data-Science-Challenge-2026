# API Contract: Retrieval (TV2) ↔ QA (TV3)

Tài liệu này cố định **giao diện (interface)** giữa module Retrieval (phụ trách bởi TV2) và module QA (phụ trách bởi TV3), để hai người có thể code song song mà không block lẫn nhau. Cả hai chỉ cần cam kết đúng chữ ký hàm và schema dữ liệu bên dưới; phần triển khai bên trong (model nào, thuật toán nào) là tự do của mỗi người.

Nguyên tắc chung (đã thống nhất trong `docs/project/khung_repo.md`):

- Toàn bộ schema dùng chung nằm ở `src/udsc2026/contracts/`. **Không ai được định nghĩa lại** `RetrievalHit` hay `QAResponse` ở module riêng của mình.
- Cả `search()` (TV2) và `generate()` (TV3) đều là **hàm Python thuần (Core Logic)**, không phụ thuộc FastAPI. TV1 chịu trách nhiệm gọi chúng trong `api/routes/`.
- Nếu một bên cần đổi contract (thêm/xóa field), phải mở PR riêng vào `src/udsc2026/contracts/`, gắn thẻ cả TV2 và TV3, và không được merge nếu chưa có sự đồng ý của bên còn lại (theo nguyên tắc merge tại `docs/project/khung_repo.md`).

## 1. Sơ đồ luồng dữ liệu

```text
question: str
     │
     ▼
┌─────────────────────┐        list[RetrievalHit]        ┌─────────────────────┐
│   TV2 - Retrieval    │ ────────────────────────────────▶│    TV3 - QA         │
│   search(query,      │                                   │   generate(question,│
│          top_k,      │                                   │            contexts)│
│          filters)    │                                   │                     │
└─────────────────────┘                                   └─────────────────────┘
                                                                     │
                                                                     ▼
                                                              QAResponse
                                                       (answer, citations, ...)
```

TV2 không cần biết TV3 dùng LLM gì; TV3 không cần biết TV2 dùng Qdrant hay FAISS. Cả hai chỉ giao tiếp qua `RetrievalHit` (input của TV3) và `QAResponse` (output của TV3, dùng chung cho TV1/TV5).

## 2. Schema dùng chung

### 2.1. `RetrievalHit` (đã có sẵn tại `src/udsc2026/contracts/retrieval.py`)

Đây là **nguồn chân lý duy nhất (single source of truth)** cho kết quả retrieval xuyên suốt pipeline (Dense, Sparse, Hybrid, Rerank, QA). TV2 và TV3 đều import từ đây, không tạo bản sao:

```python
from udsc2026.contracts.retrieval import RetrievalHit
```

```python
class RetrievalHit(BaseModel):
    """Single source of truth for retrieval results passed through the RAG pipeline."""

    chunk_id: str
    doc_id: str
    text: str
    score: Optional[float] = None
    source: Optional[str] = None
    law_name: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None
    rank: Optional[int] = None
```

Ghi chú trách nhiệm field:

| Field | Ai ghi giá trị | Ghi chú |
| :--- | :--- | :--- |
| `chunk_id`, `doc_id`, `text` | TV2 (từ output ETL của TV4) | Bắt buộc, không được rỗng |
| `law_name`, `article`, `clause`, `metadata` | TV2 (giữ nguyên metadata từ TV4) | TV3 chỉ đọc, không sửa cấu trúc |
| `score`, `dense_score` | TV2 | Điểm từ Dense Retrieval |
| `sparse_score`, `hybrid_score` | TV3 (BM25 + Hybrid) | Xem `docs/member/tv3.md` |
| `rerank_score`, `final_score`, `rank` | TV5 | TV2/TV3 không ghi vào các field này |

### 2.2. `QAResponse` (mới, cố định trong tài liệu này — cần thêm vào `src/udsc2026/contracts/qa.py`)

TV3 trả về `QAResponse` cho mọi lời gọi `generate()`. Đây là output chuẩn để
TV1 đưa vào API response và TV5 chuyển đổi sang artifact nộp Task 2.

```python
from typing import List, Optional

from pydantic import BaseModel, Field

from udsc2026.contracts.retrieval import RetrievalHit


class Citation(BaseModel):
    """Một trích dẫn cụ thể trong câu trả lời, map về nguồn gốc trong RetrievalHit."""

    chunk_id: str
    law_name: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    text_snippet: Optional[str] = None


class QAResponse(BaseModel):
    """Kết quả sinh câu trả lời của module QA (TV3), dùng chung cho TV1 và TV5."""

    answer: str
    citations: List[Citation] = Field(default_factory=list)
    used_prompt_version: str
    retrieval_hits: List[RetrievalHit] = Field(default_factory=list)
    confidence: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)
    is_refusal: bool = False
```

Quy ước:

- `answer`: câu trả lời cuối cùng, đã format, sẵn sàng hiển thị cho người dùng.
- `citations`: danh sách trích dẫn theo dạng `[Tên luật, Điều X, Khoản Y]`, map về `chunk_id` gốc trong `retrieval_hits`.
- `used_prompt_version`: bắt buộc, ví dụ `"legal_qa_v1"`, để TV1 log và TV5 dùng khi so sánh benchmark.
- `retrieval_hits`: chính là input `contexts` đã dùng để sinh câu trả lời (giữ nguyên để debug/trace), không phải danh sách mới.
- `is_refusal`: `True` khi model từ chối trả lời do thiếu căn cứ — TV1 dùng field này để quyết định hiển thị UI, TV5 dùng để loại khỏi tính điểm nếu cần.

### 2.3. Adapter nộp bài Task 2 của TV5

`QAResponse` là contract nội bộ; nó **không phải** schema gửi thẳng lên
Codabench. Tại thời điểm xác minh ngày 01/08/2026, `submission.zip` phải chứa
duy nhất `submission.json`, và JSON ở dạng object keyed by `question_id`:

```json
{
  "147194": {
    "answer": "Câu trả lời cuối cùng từ QAResponse.answer"
  }
}
```

TV3 bàn giao cặp `(question_id, QAResponse.answer)` cho TV5. TV5 chịu trách
nhiệm kiểm tra coverage, kiểu string, ID trùng/thiếu, bảo toàn Unicode và đóng
gói ZIP chính thức. Các field `citations`, `retrieval_hits`, `warnings` và log
debug chỉ dùng nội bộ, không được ghi thêm vào `submission.json`.

## 3. Chữ ký hàm bắt buộc

### 3.1. TV2 — Retrieval: `search()`

```python
from udsc2026.contracts.retrieval import RetrievalHit


def search(
    query: str,
    top_k: int,
    filters: Optional[dict] = None,
) -> list[RetrievalHit]:
    """Truy hồi top-k đoạn văn bản pháp luật liên quan nhất tới câu hỏi.

    Args:
        query: Câu hỏi gốc của người dùng (chưa qua xử lý).
        top_k: Số lượng kết quả tối đa cần trả về.
        filters: Bộ lọc metadata tuỳ chọn (ví dụ theo `law_name`).

    Returns:
        Danh sách `RetrievalHit`, sắp xếp theo `score`/`dense_score` giảm dần.
        Danh sách có thể rỗng nếu không tìm thấy kết quả phù hợp; `search()`
        không bao giờ raise exception vì không có kết quả.
    """
```

Cam kết từ TV2:

- Luôn trả `list[RetrievalHit]`, kể cả khi rỗng (không trả `None`).
- Không trả về nhiều hơn `top_k` phần tử.
- `chunk_id`/`doc_id`/`text` không được rỗng; nếu vector store lỗi, raise exception rõ ràng (không trả `RetrievalHit` rỗng giả).
- Không phụ thuộc FastAPI request/response object trong hàm này.

### 3.2. TV3 — QA: `generate_answer()`

```python
from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.contracts.qa import QAResponse


async def generate_answer(
    question: str,
    contexts: list[RetrievalHit],
    prompt_version: str = "legal_qa_v1",
    rag_template: str = "default_rag_v1",
    trace_id: str | None = None,
) -> QAResponse:
    """Sinh câu trả lời pháp lý có căn cứ từ danh sách context đã truy hồi.

    Args:
        question: Câu hỏi gốc của người dùng.
        contexts: Danh sách `RetrievalHit` làm căn cứ trả lời (đầu ra của
            `search()`/Hybrid/Rerank).
        prompt_version: Tên system prompt trong `prompts/system/`.
        rag_template: Tên user-turn template trong `prompts/rag_templates/`.
        trace_id: Mã truy vết do TV1 truyền xuống để liên kết log.

    Returns:
        `QAResponse` chứa câu trả lời, citation và metadata liên quan.

    Raises:
        ValueError: Khi `contexts` rỗng và cấu hình yêu cầu bắt buộc có căn cứ
            (trong trường hợp này TV3 vẫn nên trả `QAResponse` với
            `is_refusal=True` thay vì raise, trừ khi lỗi hệ thống thực sự).
    """
```

Cam kết từ TV3:

- Không sửa field đã có ý nghĩa cố định trong `RetrievalHit` (ví dụ không viết đè `text` gốc).
- Khi `contexts` rỗng hoặc không đủ căn cứ, trả câu từ chối với
  `confidence=0`, `warnings` có lý do và `citations=[]` thay vì raise exception,
  để TV1 xử lý nhất quán ở tầng API.
- `used_prompt_version` bắt buộc phải khớp với file thực tế trong `prompts/system/`.

## 4. Ví dụ tích hợp (do TV1 thực hiện, TV2/TV3 chỉ cần biết để hình dung)

```python
import asyncio

from udsc2026.contracts.retrieval import RetrievalHit
from udsc2026.qa.qa_engine import QAEngine
from udsc2026.retrieval.hybrid import search


async def handle_query(
    qa_engine: QAEngine,
    question: str,
    top_k: int = 5,
) -> "QAResponse":
    hits: list[RetrievalHit] = await asyncio.to_thread(search, question, top_k)
    return await qa_engine.generate_answer(
        question=question,
        contexts=hits,
        prompt_version="legal_qa_v1",
        rag_template="default_rag_v1",
    )
```

TV2 và TV3 **không cần chờ nhau code xong** để bắt đầu: mỗi bên có thể viết unit test với `RetrievalHit`/`QAResponse` giả (mock/fixture) dựa đúng theo schema ở mục 2, rồi TV1 nối hai hàm thật lại sau.

## 5. Test & mock dùng chung

Để code song song không bị block, mỗi bên nên có fixture mẫu trong `tests/fixtures/` dựa trên schema này, ví dụ:

```python
# tests/fixtures/sample_retrieval_hits.py
from udsc2026.contracts.retrieval import RetrievalHit

SAMPLE_HITS: list[RetrievalHit] = [
    RetrievalHit(
        chunk_id="luat-hon-nhan-dieu-51-khoan-1",
        doc_id="luat-hon-nhan-gia-dinh-2014",
        text="Vợ, chồng hoặc cả hai người có quyền yêu cầu Tòa án giải quyết ly hôn.",
        law_name="Luật Hôn nhân và Gia đình 2014",
        article="51",
        clause="1",
        score=0.87,
    ),
]
```

- TV3 dùng `SAMPLE_HITS` để test `generate()` mà không cần chờ TV2 dựng xong VectorDB thật.
- TV2 dùng schema `QAResponse` (mục 2.2) để test phần tích hợp giả lập ở tầng frontend/API mà không cần chờ Qwen3 load xong.
- Khi contract thay đổi, cập nhật cả tài liệu này lẫn fixture tương ứng trong cùng một PR.

## 6. Ranh giới không được vượt qua

- TV2 **không** được sửa logic trong `src/udsc2026/qa/` hoặc `prompts/`.
- TV3 **không** được sửa logic trong `src/udsc2026/infrastructure/embedding/`, `vector_db/`, hoặc `retrieval/dense/`.
- Cả hai **không** được định nghĩa FastAPI router, endpoint, dependency injection hay streaming (SSE) — đây là phạm vi của TV1.
- Mọi thay đổi field trong `RetrievalHit`/`QAResponse` phải qua PR riêng vào `contracts/`, review bởi cả TV2, TV3 và TV1 (theo `docs/project/04_git_workflow.md`).
