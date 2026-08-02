# 05. Quy chuẩn Coding Convention (Python)

Tài liệu quy định bắt buộc về code style, type hint và docstring cho toàn bộ code Python trong `src/udsc2026/`, `tests/`, và khuyến khích áp dụng cho `experiments/tvX/`. Mục tiêu là code của 5 thành viên đọc được như một người viết, giảm thời gian review và tránh lỗi runtime nhờ type checking.

Toàn bộ công cụ đã được cấu hình sẵn trong `pyproject.toml` và `.pre-commit-config.yaml`. Không tự ý đổi cấu hình các công cụ này nếu chưa thống nhất với leader TV1.

## 1. Công cụ bắt buộc

| Công cụ | Vai trò | Cấu hình |
| :--- | :--- | :--- |
| **Ruff format** | Format code tự động, không tranh cãi về style | `line-length = 88`, target Python 3.10+ |
| **Ruff isort** | Sắp xếp import | Nhóm standard library → third-party → local |
| **Ruff lint** | Phát hiện lỗi tiềm ẩn, unused import | `line-length = 88` |
| **mypy** | Kiểm tra type static | Strict config trong `pyproject.toml`, Python 3.10+ |
| **bandit** | Quét lỗ hổng bảo mật cơ bản | Bỏ qua `tests/`, `migrations/` |
| **pydocstyle** | Kiểm tra chuẩn docstring | `convention = google` |

### Cách chạy trước khi commit

```bash
pre-commit run --all-files
```

Hook này chạy quality gate của dự án bằng `.venv` cùng các kiểm tra chung
(trailing whitespace, check-yaml, check-merge-conflict...). Cách này giữ Ruff,
Mypy, Bandit và pydocstyle đúng phiên bản của dự án, đồng thời tránh hook Poetry
cô lập bị lỗi mã hóa đường dẫn trên Windows. **PR sẽ bị từ chối nếu pre-commit
và local CI không pass.**

Có thể chạy riêng từng công cụ khi debug:

```bash
python -m ruff format src tests scripts
python -m ruff check src tests scripts
python -m mypy src --no-warn-unused-configs
python -m pydocstyle src
```

## 2. Code Style (Ruff)

- **Bắt buộc chạy Ruff format** trước mỗi commit; không format code thủ công khác với output của Ruff.
- Độ dài dòng tối đa: **88 ký tự**, không tự ý nới rộng.
- Import được Ruff sắp xếp theo 3 nhóm: standard library → third-party → local (`udsc2026.*`), cách nhau bằng dòng trống.
- Không để import thừa, biến không dùng, hoặc code chết (dead code) — Ruff sẽ chặn ở CI.
- Đặt tên:
  - Module, package, biến, hàm: `snake_case` (`dense_retriever.py`, `search_query`).
  - Class: `PascalCase` (`DenseRetriever`, `RetrievalHit`, `QAEngine`).
  - Hằng số: `UPPER_SNAKE_CASE` (`DEFAULT_TOP_K = 5`).
  - Tên private/internal: prefix `_` (`_normalize_score`).
- Mỗi hàm/class nên làm đúng một việc; nếu hàm public dài quá ~50 dòng, cân nhắc tách nhỏ.
- Không commit code có `print()` debug hoặc `pdb.set_trace()`; dùng `logging`.

## 3. Type Hint bắt buộc

- **Mọi hàm/method public** (không bắt đầu bằng `_`) trong `src/udsc2026/` **bắt buộc có type hint đầy đủ** cho tham số và giá trị trả về.
- Dùng kiểu chuẩn của `typing` hoặc built-in generics (Python 3.9+: `list[str]`, `dict[str, Any]` đều chấp nhận được, ưu tiên nhất quán trong cùng file).
- Với tham số optional, dùng `Optional[X]` hoặc `X | None`, có giá trị mặc định rõ ràng.
- Dữ liệu qua lại giữa các module (contract chung) **phải** dùng Pydantic `BaseModel`, khai báo tại `src/udsc2026/contracts/`, không dùng `dict` thô hoặc `Any` để truyền dữ liệu giữa TV2 ↔ TV3 ↔ TV5.

Ví dụ hợp lệ:

```python
from typing import Optional

from udsc2026.contracts.retrieval import RetrievalHit


def search(query: str, top_k: int, filters: Optional[dict] = None) -> list[RetrievalHit]:
    """Tìm kiếm các đoạn văn bản pháp luật liên quan tới câu hỏi.

    Args:
        query: Câu hỏi hoặc truy vấn của người dùng.
        top_k: Số lượng kết quả tối đa cần trả về.
        filters: Bộ lọc metadata tuỳ chọn (ví dụ theo tên luật).

    Returns:
        Danh sách `RetrievalHit` được sắp xếp theo độ liên quan giảm dần.
    """
    ...
```

- mypy chạy ở mức `check_untyped_defs = true`; hàm thiếu type hint vẫn được kiểm tra phần thân, nên viết type hint đầy đủ ngay từ đầu để tránh lỗi CI muộn.
- Không dùng `# type: ignore` để né lỗi mypy trừ khi thực sự cần thiết (ví dụ thư viện thiếu stub như `qdrant_client`, `sentence_transformers`, `transformers` đã được khai báo chính xác trong `pyproject.toml`); nếu dùng phải kèm comment giải thích lý do.

## 4. Docstring chuẩn (Google style)

- Áp dụng **Google-style docstring** cho mọi module, class và hàm public (`pydocstyle --convention=google`).
- Docstring đặt ngay dòng đầu tiên trong thân hàm/class (kiểm tra bởi hook `check-docstring-first`).
- Cấu trúc tối thiểu cho hàm public:

```python
async def generate_answer(
    question: str,
    contexts: list[RetrievalHit],
    prompt_version: str = "legal_qa_v1",
    rag_template: str = "default_rag_v1",
    trace_id: str | None = None,
) -> "QAResponse":
    """Sinh câu trả lời pháp lý có căn cứ từ danh sách context đã truy hồi.

    Args:
        question: Câu hỏi của người dùng.
        contexts: Danh sách `RetrievalHit` làm căn cứ trả lời, đã qua retrieval/rerank.
        prompt_version: Tên system prompt trong `prompts/system/`.
        rag_template: Tên user-turn template trong `prompts/rag_templates/`.
        trace_id: Mã truy vết do TV1 truyền xuống.

    Returns:
        `QAResponse` chứa câu trả lời, citation, và metadata (prompt version, confidence).

    Raises:
        ValueError: Khi `contexts` rỗng và cấu hình không cho phép trả lời không căn cứ.
    """
```

- Module (đầu file `.py`) nên có docstring 1 dòng mô tả mục đích, ví dụ đã dùng trong `contracts/retrieval.py`:

```python
"""Shared retrieval contracts used across Dense, Hybrid, QA, and Rerank."""
```

- Class Pydantic (schema/contract) bắt buộc có docstring mô tả vai trò và phạm vi sử dụng, như `RetrievalHit` hiện tại.
- Không bắt buộc docstring cho hàm/class private (`_helper`), test (`tests/`), và code trong `migrations/` (đã loại trừ trong `pyproject.toml` qua các mã D100-D107).
- Docstring viết bằng tiếng Việt hoặc tiếng Anh đều được, nhưng phải nhất quán trong cùng một module để dễ đọc.

## 5. Checklist trước khi mở PR

- [ ] Chạy `pre-commit run --all-files`, không còn lỗi.
- [ ] Toàn bộ hàm/method public có type hint đầy đủ.
- [ ] Toàn bộ hàm/method/class public có docstring Google-style.
- [ ] Không dùng `dict`/`Any` thô cho dữ liệu đi qua ranh giới module; dùng schema trong `contracts/`.
- [ ] Không có `print()`/debug code, import thừa, hoặc biến chưa dùng.
- [ ] `mypy src/` không phát sinh lỗi mới so với `main`.
