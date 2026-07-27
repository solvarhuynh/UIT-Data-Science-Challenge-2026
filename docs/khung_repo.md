# Kiến trúc thư mục và phân công công việc cho DSC2026

## Cấu trúc thư mục hiện tại

```text
udsc2026/
├── configs/                         # YAML/env config cho runtime, model, retrieval, cache
├── data/
│   ├── raw/                         # Dữ liệu gốc từ BTC, không commit file lớn
│   ├── processed/                   # Output ETL: chunks, parents, metadata, benchmark
│   │   ├── chunks/                  # LegalChunk JSONL cho TV2 index
│   │   ├── documents/               # Document/parent context sau khi chuẩn hóa
│   │   └── metadata/                # Validation report, corpus stats, data quality report
│   └── vector_store/                # Qdrant/FAISS/BM25 index local, không commit file lớn
├── docker/                          # Dockerfile, compose override và cấu hình container
├── docs/
│   ├── algorithms/                  # Hybrid Search, reranking, retrieval strategy
│   ├── architecture/                # FastAPI, data pipeline, system rules
│   ├── member/                      # Phân công chi tiết cho TV1-TV5
│   │   ├── tv1.md                   # Integration & Orchestration
│   │   ├── tv2.md                   # Dense Retrieval
│   │   ├── tv3.md                   # QA, LLM, BM25, Hybrid Search
│   │   ├── tv4.md                   # ETL, Parent-Child Chunking, Synthetic Benchmark
│   │   └── tv5.md                   # Reranking, Evaluation, MLOps
│   ├── models/                      # Model registry, embedding, LLM optimization
│   └── project/                     # Git workflow, coding convention, API contract
├── experiments/
│   ├── tv1/                         # Thử nghiệm orchestration, API, cache, logging
│   ├── tv2/                         # Thử nghiệm BKAI embedding, VectorDB, dense search
│   ├── tv3/                         # Thử nghiệm Qwen3, prompt, BM25, hybrid fusion
│   ├── tv4/                         # Thử nghiệm ETL, parser, chunking, synthetic Q&A
│   └── tv5/                         # Thử nghiệm rerank, evaluation, Docker, submission
├── frontend/                        # Web frontend do nhân sự riêng phụ trách
├── models/
│   ├── bkai-bi-encoder/             # Local embedder: bkai-foundation-models/vietnamese-bi-encoder
│   └── qwen3-legal/                 # Local generator checkpoint cho Qwen3
├── prompts/
│   ├── README.md                    # Prompt registry
│   ├── rag_templates/               # RAG prompt templates có version
│   └── system/                      # Versioned system prompts, ví dụ legal_qa_v1.md
├── scripts/                         # Setup, indexing, evaluation, submission, utility scripts
├── src/
│   └── udsc2026/
│       ├── api/                     # TV1: FastAPI app, routes, DI, health/readiness
│       │   └── routes/              # API route modules
│       ├── contracts/               # Pydantic schemas dùng chung
│       │   ├── retrieval.py         # RetrievalHit, LegalChunk nếu có
│       │   └── __init__.py
│       ├── evaluation/              # TV5: MRR, Recall@K, ROUGE-L, reports, submission writer
│       ├── infrastructure/
│       │   ├── embedding/           # TV2: BKAI EmbeddingClient
│       │   ├── llm/                 # TV3: Qwen3 transformers/vLLM client
│       │   ├── persistence/         # Shared storage/cache/log helpers nếu cần
│       │   ├── reranker/            # TV5: Cross-Encoder client
│       │   └── vector_db/           # TV2: Qdrant/FAISS adapters
│       ├── ingestion/
│       │   ├── readers/             # TV4: raw file readers
│       │   ├── cleaners/            # TV4: Unicode/noise normalization
│       │   ├── legal_structure/     # TV4: Luật/Chương/Điều/Khoản/Điểm parser
│       │   └── chunking/            # TV4: Parent-Child Chunking
│       ├── qa/                      # TV3: Prompt builder, QAEngine, citation parser
│       └── retrieval/
│           ├── dense/               # TV2: DenseRetriever
│           ├── sparse/              # TV3: BM25 sparse retrieval
│           ├── hybrid/              # TV3: Dense + sparse score fusion
│           └── reranking/           # TV5: CrossEncoderReranker
├── tests/                           # Unit/integration tests và test cases reference
├── docker-compose.yml               # Local stack: backend, VectorDB, Redis nếu cần
├── pyproject.toml                   # Python package/dependencies
├── requirements_dev.txt             # Dev dependencies
└── README.md                        # Tổng quan dự án và hướng dẫn chạy
```

## Phân công backend theo module

| Thành viên | Trọng tâm | Bổ sung | File giao việc |
| :--- | :--- | :--- | :--- |
| **TV1 - Integration & Orchestration Specialist** | End-to-End Pipeline Orchestrator, FastAPI router, Dependency Injection, async handling, latency tuning | Cache Redis/in-memory, structured logging, low-score/error log analysis | `docs/member/tv1.md` |
| **TV2 - Dense Retrieval Specialist** | BKAI Vietnamese Bi-Encoder, query/document embedding, DenseRetriever | Qdrant/FAISS adapter, vector upsert, semantic top-K search | `docs/member/tv2.md` |
| **TV3 - QA & LLM Specialist** | Qwen3 local inference, prompt versioning, citation accuracy, anti-hallucination | BM25 Sparse Retrieval, Hybrid Search score fusion | `docs/member/tv3.md` |
| **TV4 - Advanced Data Engineer & Synthetic Dataset Specialist** | Legal ETL, Regex/Rule parser, Unicode cleanup, Parent-Child Chunking | Synthetic Benchmark Q&A 100-200 mẫu cho evaluation | `docs/member/tv4.md` |
| **TV5 - Reranking & MLOps Specialist** | Cross-Encoder reranking, metadata-preserving final ranking | MRR, Recall@K, ROUGE-L, Docker multi-stage, `submission.csv` CodaLab | `docs/member/tv5.md` |

## Ranh giới làm việc

- Dự án của 5 thành viên hiện tại tập trung 100% backend: data processing, retrieval, reranking, QA, evaluation và deployment.
- `frontend/` vẫn tồn tại trong repo nhưng do nhân sự frontend riêng phụ trách; TV1-TV5 chỉ giữ API contract để frontend gọi.
- `contracts/` là vùng giao tiếp chung. `RetrievalHit`, `LegalChunk`, `Citation`, `QAResponse` chỉ nên định nghĩa tại `src/udsc2026/contracts/`; các module phải import lại thay vì tự tạo schema riêng.
- `api/` thuộc TV1 và chỉ chứa route, dependency injection, middleware, health/readiness; business logic nằm trong orchestrator/service.
- `ingestion/` thuộc TV4; mọi thay đổi output chunk phải báo TV2, TV3 và TV5 vì ảnh hưởng index, hybrid search và benchmark.
- `retrieval/dense/` và `infrastructure/embedding`, `infrastructure/vector_db` thuộc TV2.
- `retrieval/sparse/`, `retrieval/hybrid/`, `qa/` và `infrastructure/llm` thuộc TV3.
- `retrieval/reranking/`, `infrastructure/reranker`, `evaluation/`, Docker và submission scripts thuộc TV5.
- `experiments/tvX/` là nơi thử nghiệm cá nhân. Code production chỉ được đưa vào `src/udsc2026/` khi interface rõ ràng và có test.
- `data/`, `models/`, vector index và cache không commit file lớn. Chỉ commit `.gitkeep`, metadata nhỏ và hướng dẫn tái tạo.
- `prompts/` phải version hóa rõ ràng vì thay đổi prompt ảnh hưởng trực tiếp kết quả QA.

## Branch và Git Worktree đề xuất

```text
D:\udsc2026-worktrees
├── dsc2026-main
├── dsc2026-tv1
├── dsc2026-tv2
├── dsc2026-tv3
├── dsc2026-tv4
└── dsc2026-tv5
```

Branch tương ứng:

- `feature/tv1`
- `feature/tv2`
- `feature/tv3`
- `feature/tv4`
- `feature/tv5`

## Nguyên tắc merge

- PR phải đi kèm mô tả scope, file đã sửa, contract thay đổi và test đã chạy.
- Không merge code phá vỡ `uvicorn udsc2026.api.app:app --reload`.
- Không merge thay đổi output chunk nếu chưa thông báo TV2, TV3 và TV5.
- Không merge prompt mới nếu chưa ghi version và test câu hỏi mẫu.
- Không merge thay đổi API response nếu chưa cập nhật contract và README.
- Leader TV1 review các PR chạm vào `api/`, `contracts/`, `configs/` và workflow chung.
