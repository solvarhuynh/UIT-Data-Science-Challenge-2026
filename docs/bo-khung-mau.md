# Kiến trúc thư mục và phân công công việc cho DSC2026

## Cấu trúc thư mục

```
udsc2026/
├── configs/
├── data/
├── docker/
├── docs/
├── experiments/
├── frontend/
├── models/
│   ├── bkai-bi-encoder/
│   └── qwen3-legal/
├── prompts/
├── scripts/
│   ├── cleanup_legacy.ps1
│   └── setup_worktrees.sh
├── src/
│   └── dsc2026_legal/
│       ├── api/
│       │   └── routes/
│       ├── contracts/
│       ├── evaluation/
│       ├── infrastructure/
│       │   ├── embedding/
│       │   ├── llm/
│       │   ├── persistence/
│       │   ├── reranker/
│       │   └── vector_db/
│       ├── ingestion/
│       │   ├── chunking/
│       │   ├── cleaners/
│       │   ├── legal_structure/
│       │   └── readers/
│       ├── qa/
│       └── retrieval/
│           ├── dense/
│           ├── hybrid/
│           ├── reranking/
│           └── sparse/
└── tests/
```

## Phân công và Quy trình làm việc

### Ranh giới công việc theo thành viên

| Thành viên | Phạm vi chính | Thư mục được ưu tiên chỉnh sửa |
| :--- | :--- | :--- |
| **TV1 — Platform/API** | FastAPI, dependency injection, tích hợp pipeline, frontend/API contract | `src/.../api/`, `frontend/`, `docker/` |
| **TV2 — Sparse Retrieval** | BM25, indexing, keyword search, retrieval baseline | `src/.../retrieval/sparse/`, `configs/retrieval/bm25.yaml`, `experiments/tv2/` |
| **TV3 — QA/Prompt** | Qwen 3, prompt versioning, context injection, citation, answer generation | `src/.../qa/`, `prompts/`, `configs/qa/`, `experiments/tv3/` |
| **TV4 — Ingestion** | Đọc dữ liệu pháp luật, làm sạch, nhận diện cấu trúc Luật, chunking | `src/.../ingestion/`, `configs/ingestion/`, `experiments/tv4/` |
| **TV5 — Dense/Hybrid/Evaluation** | BKAI embedding, Qdrant, hybrid search, reranker, đánh giá | `src/.../retrieval/dense/`, `hybrid/`, `reranking/`, `evaluation/`, `experiments/tv5/` |

### Các thư mục dùng chung

Các thư mục sau cần được kiểm soát chặt chẽ thông qua Pull Request để đảm bảo tính nhất quán:

-   `contracts/`: Mọi thay đổi về schema (cấu trúc dữ liệu giao tiếp) phải được thống nhất trước.
-   `infrastructure/`: Chỉ chứa các adapter hoặc wrapper cho dịch vụ bên ngoài (DB, LLM, ...), không chứa logic nghiệp vụ.
-   `configs/`: Thay đổi tham số phải có lý do rõ ràng và ghi rõ phiên bản môi trường.
-   `tests/`: Mỗi thay đổi trong `src/` phải có test tương ứng.
-   `data/`, `models/`, `vector_store/`: Không commit dữ liệu lớn. Chỉ commit file `.gitkeep`, metadata và tài liệu hướng dẫn cách tải lại dữ liệu.

### Chiến lược Git Worktree

Mỗi thành viên sẽ làm việc trên một `branch` và `worktree` riêng để giảm thiểu xung đột.

**Cấu trúc worktree:**
```
D:\udsc2026
├── dsc2026-main
├── dsc2026-tv1-api
├── dsc2026-tv2-retrieval
├── dsc2026-tv3-qa
├── dsc2026-tv4-ingestion
└── dsc2026-tv5-evaluation
```

**Các branch tương ứng:**
-   `feature/tv1-api`
-   `feature/tv2-retrieval`
-   `feature/tv3-qa`
-   `feature/tv4-ingestion`
-   `feature/tv5-evaluation`

### Nguyên tắc chính

> Dữ liệu thử nghiệm nằm trong `experiments/tvN/`, mã nguồn production nằm trong `src/`, và giao tiếp giữa các module chỉ thông qua `contracts/`.

Cách tiếp cận này giúp giảm xung đột Git và duy trì ranh giới rõ ràng giữa các phần: thử nghiệm, hạ tầng và logic nghiệp vụ.