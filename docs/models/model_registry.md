# Model Registry

## Mục tiêu

Quản lý tập trung mọi phiên bản mô hình (Retriever, Generator, Reranker) được dùng trong hệ thống, đảm bảo:

1. Toàn đội dùng đúng checkpoint đã chốt, tránh lệch kết quả giữa các máy.
2. Tuân thủ luật thi DSC2026.

## ⚠️ Điều kiện bắt buộc theo luật thi

> **Chỉ được sử dụng các mô hình mã nguồn mở (open-source) có kích thước dưới 4 tỷ (4B) tham số.**

- Mọi model thêm vào registry này **phải** khai báo rõ số tham số ở cột `Params`.
- Trước khi thêm một model mới (kể cả để thử nghiệm trong `experiments/`), người thực hiện phải xác nhận model đó là mã nguồn mở, có license cho phép sử dụng trong cuộc thi, và số tham số < 4B.
- Model không rõ nguồn gốc, closed-source, hoặc gọi qua API cloud trả phí (OpenAI, Anthropic, Gemini API...) **không được phép** dùng trong pipeline chính thức nộp bài.
- Nếu phát hiện model vi phạm điều kiện trên đang được dùng, phải gỡ khỏi `src/udsc2026/` và ghi lại trong mục "Model đã loại bỏ" bên dưới.

## Retriever

| Checkpoint | Version | Params | Local path | Input format | Output format | Trạng thái |
|---|---|---|---|---|---|---|
| `huyydangg/DEk21_hcmute_embedding_v2` | HuggingFace (resolved SHA được ghi khi tải) | ~135M | `./models/dek21-v2/` | Văn bản tiếng Việt đã word-segment bằng PyVi | Vector 768 chiều đã normalize, cosine similarity | **PROMOTED** — Task1 embedding bắt buộc |

Ghi chú:
- Encode query và encode chunk dùng chung 1 model (symmetric bi-encoder).
- Batch encode cho chunk khi index, single encode cho query khi search.
- Device cấu hình qua `configs/base.yaml` / `configs/development.yaml` (`cpu`/`cuda`).

## Reranker (Cross-Encoder)

| Checkpoint | Version | Params | Local path | Input format | Output format | Trạng thái |
|---|---|---|---|---|---|---|
| [`Qwen/Qwen3-VL-Reranker-2B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | HuggingFace (resolved SHA được ghi khi tải) | ~2B | `./models/qwen3-vl-reranker-2b/` | Query + chunk text (có thể mở rộng sang ảnh) | `list[float]` — raw relevance score | **PROMOTED** — Task1 reranker bắt buộc |

## Generator

| Checkpoint | Version | Params | Local path | Input format | Output format | Trạng thái |
|---|---|---|---|---|---|---|
| `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2` | HuggingFace (resolved SHA được ghi khi tải) | 1.7B | `./models/qwen3-legal/` | Prompt string: system prompt + bounded parent context sau rerank + câu hỏi | `str` — câu trả lời có trích dẫn Điều/Khoản, hỗ trợ streaming token (SSE) | Đang dùng (baseline generator) |

Ghi chú:
- Đây là model đã fine-tune (GRPO) trên domain pháp luật Việt Nam — vẫn tính là open-source vì checkpoint public trên HuggingFace và base model gốc (Qwen3) là open-source.
- Tổng tham số 1.7B, dưới ngưỡng 4B theo luật thi.
- Nếu thử nghiệm nâng cấp lên checkpoint Qwen3 lớn hơn, **bắt buộc kiểm tra lại params trước khi thêm vào bảng này**.

## Quy ước Version

- Với model tải từ HuggingFace: ghi rõ commit hash hoặc revision đã pin trong `download_models.py`, không dùng `main` trôi nổi để tránh đổi trọng số giữa các lần tải.
- Với model tự fine-tune thêm (LoRA adapter...): version theo `<base_model>-<mô_tả>-v<N>`, ví dụ `qwen3-1.7b-legal-lora-fewshot-v1`.

## Model đã loại bỏ / không đạt điều kiện

Các checkpoint dưới đây được giữ lại như lịch sử P8/P9, nhưng không required,
không nằm trong Task1 production và không được tải bởi profile mặc định.

| Checkpoint | Status | Task | Role | Required |
|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | `REJECTED_EXPERIMENT` | Task1 | secondary embedding | `false` |
| `Qwen/Qwen3-Reranker-0.6B` | `REJECTED_EXPERIMENT` | Task1 | experimental reranker | `false` |

| Checkpoint | Status | Lý do loại bỏ | Ngày | Người phát hiện |
|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | `REJECTED_EXPERIMENT` | P9 không được chọn làm Task1 secondary retrieval | 2026-08-11 | TV2 |
| `Qwen/Qwen3-Reranker-0.6B` | `REJECTED_EXPERIMENT` | P8 smoke không vượt BGE; không phải production component | 2026-08-11 | TV2 |

## Deferred experiments

| Checkpoint | Status | Task | Required | Activation trigger |
|---|---|---|---|---|
| `BAAI/bge-m3` | `DEFERRED_EXPERIMENT` | Task1 | `false` | CandidateDocRecall hoặc official OOF Recall plateau sau P12/P13/P15 |

## Liên quan

- Log thí nghiệm dùng các model này: xem `docs/project/06_experiment_tracking.md`.
- Kiến trúc và luồng dữ liệu: xem `docs/project/10_system_design.md`.
- Hướng dẫn Qwen3-VL-Reranker-2B: xem `docs/models/qwen3_vl_reranker_2b.md`.
- Chi tiết LoRA/QLoRA cho Qwen3: xem `docs/models/llm_optimization.md`.
