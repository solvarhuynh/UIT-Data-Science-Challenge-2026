# Kiến trúc thư mục và phân công công việc cho DSC2026

## Cấu trúc thư mục hiện tại

```text
udsc2026/
├── configs/                  # YAML cấu hình hyperparameter và runtime
├── data/
│   ├── raw/                  # Dữ liệu gốc từ BTC
│   ├── processed/            # Documents/chunks/metadata sau ETL
│   └── vector_store/         # VectorDB/BM25 index local, không commit file lớn
├── docker/                   # Dockerfile, compose và cấu hình container
├── docs/
│   ├── algorithms/           # Hybrid Search, ReAct Agent
│   ├── architecture/         # FastAPI, data pipeline, system rules
│   ├── member/               # Phân công chi tiết cho TV1-TV5
│   └── models/               # Embedding và LLM optimization
├── experiments/
│   ├── tv1/                  # Thử nghiệm backend/metrics/integration
│   ├── tv2/                  # Thử nghiệm dense retrieval/frontend
│   ├── tv3/                  # Thử nghiệm QA/prompt/BM25/hybrid
│   ├── tv4/                  # Thử nghiệm ETL/chunking/UI citation
│   └── tv5/                  # Thử nghiệm rerank/evaluation/submission
├── frontend/                 # React + TailwindCSS frontend
├── models/
│   ├── bkai-bi-encoder/      # Local embedder
│   └── qwen3-legal/          # Local generator
├── prompts/                  # Versioned system prompts và RAG templates
├── scripts/                  # Setup, cleanup, indexing, evaluation, submission
├── src/
│   └── udsc2026/
│       ├── api/              # FastAPI app, routes, DI, SSE
│       ├── contracts/        # Pydantic schemas dùng chung
│       ├── evaluation/       # Metrics, benchmark, submission writer
│       ├── infrastructure/   # Adapter cho embedder, LLM, vector DB, reranker
│       ├── ingestion/        # Readers, cleaners, legal parser, chunking
│       ├── qa/               # Prompt builder, Qwen3 QA, citation
│       └── retrieval/        # Dense, sparse, hybrid, reranking
└── tests/
    └── test_cases_reference.md
```

## Phân công theo chiến lược 1 việc lớn + 1 việc nhỏ

| Thành viên | Trọng tâm | Bổ sung | File giao việc |
| :--- | :--- | :--- | :--- |
| **TV1 - Integration** | E2E Pipeline Orchestration, Performance Tuning, Caching, Logging | Metrics (MRR, Recall@K), PR Review | `docs/member/tv1.md` |
| **TV2 - Retrival & Frontend** | Dense Retrieval bằng BKAI Bi-encoder, Qdrant/FAISS VectorDB | React base, TailwindCSS, search box, chat frame | `docs/member/tv2_nghia.md` |
| **TV3 - QA & LLM** | Qwen3 local, prompt system, citation, anti-hallucination | BM25 và Hybrid Search | `docs/member/tv3.md` |
| **TV4 - Data & UI** | ETL, regex legal structure parsing, chunking metadata | Markdown rendering, Citation Viewer, responsive UI | `docs/member/tv4.md` |
| **TV5 - Rerank & DevOps** | Cross-Encoder reranking | Docker multi-stage, docker-compose, CodaLab submission | `docs/member/tv5.md` |

## Ranh giới làm việc

- `contracts/` là vùng giao tiếp chung. `RetrievalHit` chỉ được định nghĩa tại `src/udsc2026/contracts/retrieval.py`; mọi module retrieval, QA, rerank, evaluation và logging phải import từ đây thay vì tự tạo schema riêng.
- `infrastructure/` chỉ chứa adapter tới model, VectorDB hoặc storage; không đặt logic nghiệp vụ trực tiếp ở đây.
- `experiments/tvX/` là nơi thử nghiệm cá nhân. Code production chỉ được đưa vào `src/udsc2026/`.
- `data/` và `models/` không commit dữ liệu lớn hoặc model weights. Chỉ commit `.gitkeep`, metadata nhỏ và hướng dẫn tái tạo.
- `prompts/` phải version hóa rõ ràng vì thay đổi prompt ảnh hưởng trực tiếp kết quả QA.
- `frontend/` gọi API qua JSON/SSE, không tự triển khai retrieval hoặc prompt logic.

## Branch và Git Worktree đề xuất

```text
D:\udsc2026-worktrees
├── dsc2026-main
├── dsc2026-tv1-api
├── dsc2026-tv2-dense-frontend
├── dsc2026-tv3-qa-hybrid
├── dsc2026-tv4-data-ui
└── dsc2026-tv5-rerank-devops
```

Branch tương ứng:

- `feature/tv1-api`
- `feature/tv2-dense-frontend`
- `feature/tv3-qa-hybrid`
- `feature/tv4-data-ui`
- `feature/tv5-rerank-devops`

## Nguyên tắc merge

- PR phải đi kèm mô tả scope, file đã sửa, contract thay đổi và test đã chạy.
- Không merge code phá vỡ `uvicorn udsc2026.api.app:app --reload`.
- Không merge thay đổi output chunk nếu chưa thông báo TV2, TV3 và TV5.
- Không merge prompt mới nếu chưa ghi version và test câu hỏi mẫu.
- Leader TV1 review các PR chạm vào `api/`, `contracts/`, `configs/` và workflow chung.
