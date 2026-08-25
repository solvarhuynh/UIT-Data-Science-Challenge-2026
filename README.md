# HCMUTE-SHIPCODE — UDSC 2026

Hệ thống RAG pháp luật tiếng Việt cho hai tác vụ **LegalIR** và **LegalQA** của
UIT Data Science Challenge 2026. Repository bao gồm pipeline chuẩn hóa dữ liệu,
parent–child chunking, dense/BM25/hybrid retrieval, cross-encoder reranking,
sinh câu trả lời bằng LLM local, đánh giá và đóng gói submission.

- **LegalIR:** truy hồi danh sách `document_id`; đánh giá macro Recall và macro
  Precision, hỗ trợ nhiều gold document cho một câu hỏi.
- **LegalQA:** sinh câu trả lời dựa trên context đã truy hồi; đánh giá METEOR và
  ROUGE-L, đồng thời giữ thông tin citation để kiểm tra.

## Trạng thái hiện tại

- Corpus chính thức duy nhất: `data/processed_v3`.
- Pipeline GPU đã khóa về đúng corpus V3 và kiểm tra hash trước khi index.
- Ba model inference chạy local; không phụ thuộc API LLM thương mại.
- Retrieval giữ `parent_id`; sau rerank, QA mở rộng parent theo cửa sổ và token
  budget thay vì đưa toàn bộ văn bản dài vào prompt.
- Test hợp nhất gần nhất: **851 passed, 15 skipped**.

## Mô hình chính thức

| Vai trò | Checkpoint | Local path | Cấu hình GPU |
| --- | --- | --- | --- |
| Embedding | [`huyydangg/DEk21_hcmute_embedding_v2`](https://huggingface.co/huyydangg/DEk21_hcmute_embedding_v2) | `models/dek21-v2` | CUDA, batch 128, vector 768 chiều |
| Reranker | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | `models/reranker` | CUDA FP16, batch 8, max length 1024 |
| LLM | [`thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2`](https://huggingface.co/thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2) | `models/qwen3-legal` | CUDA BF16 |

Task1 production dùng HCMUTE embedding + BGE reranker. Model được tải bằng
`download_models.py`; revision thực tế được ghi vào
`models/download_manifest.json`. Xem thêm [Model Registry](docs/models/model_registry.md).

Tải đúng model production Task1:

```powershell
python download_models.py --profile task1
```

## Kiến trúc

```mermaid
flowchart LR
    RAW[BTC raw contexts] --> ETL[Legal ETL + structure parser]
    ETL --> CHILD[Child chunks]
    ETL --> PARENT[Parent store]
    CHILD --> EMB[DEk21 embedding]
    EMB --> DENSE[FAISS / Qdrant]
    CHILD --> BM25[BM25]
    DENSE --> HYBRID[Hybrid fusion]
    BM25 --> HYBRID
    HYBRID --> RERANK[BGE cross-encoder]
    RERANK --> EXPAND[Bounded parent expansion]
    PARENT --> EXPAND
    EXPAND --> LLM[Qwen3 legal]
    LLM --> API[FastAPI / evaluation / submission]
```

Luồng chuẩn là **retrieve children → rerank → mở rộng parent có giới hạn →
LLM**. Không thay child bằng nguyên parent trước rerank.

## Corpus `processed_v3`

| Hạng mục | Giá trị đã audit |
| --- | ---: |
| Documents | 8.532 |
| Chunks | 1.270.356 |
| Parents | 184.548 |
| Synthetic benchmark nội bộ | 100 câu, 7 loại câu hỏi |
| Dung lượng | 2.905.166.051 bytes (2,706 GiB) |
| Missing source token / bigram | 0 / 0 |
| Duplicate / orphan / invalid / empty chunk | 0 |
| Oversized chunk / missing metadata | 0 / 0 |

Hash chuẩn:

```text
processed_corpus_tree_hash: 80fb33ff1133ce2583097cc5a98ddc9739240a60bfe11c979892d5412dcda647
benchmark_sha256:           80f27b47e81aa40b25ca55ab2fe7fb7edd8fc65388f41fdae841a48a104892d3
source_chunk_corpus_sha256: d2aa542f1f45ad9f310bceb43063aed3058d2e81edbb15ecf18525c3c6d6bbc7
```

Corpus có `integrity_gate_passed=true`. Báo cáo semantic vẫn ghi nhận 20 context
chính thức có `passage` rỗng ngay từ nguồn BTC; hệ thống giữ placeholder và cảnh
báo, không tự tạo nội dung. Chi tiết nằm trong:

- `data/processed_v3/metadata/processing_manifest.json`;
- `data/processed_v3/metadata/disk_audit_report.json`;
- `data/processed_v3/metadata/validation_report.json`.

Có 1.359 document được gắn cờ review cấu trúc nhưng nội dung vẫn được giữ bằng
structured/partial/fallback chunks; đây không phải số document bị loại. Benchmark
100 câu là bộ synthetic dùng để smoke và so sánh nội bộ, không phải test chính
thức. Chỉ kết luận reranker cải thiện sau khi máy GPU tạo `comparison.md`.

Dữ liệu lớn không nằm trong Git. Khi chuyển sang máy khác, copy nguyên cả năm
thư mục `documents`, `chunks`, `parents`, `benchmarks`, `metadata` bên trong
`data/processed_v3`; không trộn với corpus cũ. Xem [Data README](data/README.md).

## Cài đặt local

Khuyến nghị Python 3.11. Với PowerShell trên Windows:

```powershell
git clone https://github.com/solvarhuynh/UIT-Data-Science-Challenge-2026.git
cd UIT-Data-Science-Challenge-2026
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -c requirements_runtime.txt -e ".[llm,rerank,retrieval,models]"
python -m pip install -r requirements_dev.txt
$env:PYTHONPATH = "$PWD\src"
```

Tải model nếu cần chạy inference:

```powershell
python download_models.py
```

Máy CPU có thể chạy unit/integration test và smoke nhỏ. Full embedding,
reranking và LLM nên chạy trên GPU.

## Chạy trọn pipeline trên RTX 5060 Ti

Máy GPU Windows không nên cài Torch theo luồng CPU ở trên. Script chính thức sẽ
cài PyTorch CUDA 12.8 trước, tải model, chạy preflight, smoke rồi mới chạy toàn bộ
corpus:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/run_gpu_pipeline.ps1 `
  -ProcessedRoot data/processed_v3
```

Nếu dependency và model đã có sẵn:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/run_gpu_pipeline.ps1 `
  -ProcessedRoot data/processed_v3 `
  -SkipInstall `
  -SkipDownload
```

Theo dõi và lưu log:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/run_gpu_pipeline.ps1 `
  -ProcessedRoot data/processed_v3 2>&1 | `
  Tee-Object -FilePath outputs/tv5_gpu_pipeline.log
```

Pipeline sẽ dừng nếu corpus, benchmark, model hoặc CUDA không qua preflight.
Hướng dẫn chi tiết và xử lý OOM nằm tại
[TV5 GPU Runbook](docs/members/tv5/tv5_gpu_runbook.md).

Runbook từng cell cho Task1 trên Kaggle nằm tại
[TV2 Task1 Kaggle Runbook](docs/members/tv2/tv2_task1_kaggle.md).

Với RTX 5060 Ti 16 GB và RAM 28 GB, pipeline chính dùng dense FAISS top-50 →
reranker top-5; không build full `rank_bm25` trong cùng lượt vì tốn RAM. Lần chạy
đầu nên dành khoảng 2–4 giờ, tùy mạng và SSD.

## Chạy từng thành phần

### Index dense/FAISS

```powershell
python scripts/data_prep/index_chunks.py `
  --chunks-dir data/processed_v3/chunks `
  --config-env gpu `
  --vector-db-type faiss `
  --skip-bm25 `
  --force
```

### Sinh candidate và benchmark reranker

```powershell
python scripts/evaluation/generate_dense_candidates.py `
  --benchmark data/processed_v3/benchmarks/synthetic_qa.jsonl `
  --config-env gpu `
  --candidate-k 50 `
  --output artifacts/tv2/dense_predictions.jsonl `
  --manifest artifacts/tv2/dense_run_manifest.json

python scripts/evaluation/benchmark_reranker.py `
  --benchmark data/processed_v3/benchmarks/synthetic_qa.jsonl `
  --candidates artifacts/tv2/dense_predictions.jsonl `
  --model models/reranker `
  --device cuda `
  --batch-size 8 `
  --max-length 1024 `
  --candidate-k 50 `
  --top-n 5 `
  --fp16 `
  --output-dir artifacts/tv5/bge-reranker-v2-m3
```

Kết quả chính:
`artifacts/tv5/bge-reranker-v2-m3/evaluation/comparison.md`.

### Backend và dịch vụ phụ trợ

```powershell
docker compose up -d qdrant redis
uvicorn udsc2026.api.app:app --reload
```

Endpoint kiểm tra:

```text
GET  /health
GET  /ready
POST /api/v1/query
```

Hoặc build toàn bộ backend bằng Docker:

```powershell
docker compose up -d --build
```

### Frontend

Yêu cầu Node.js `^20.19.0` hoặc `>=22.12.0`.

```powershell
cd "frontend/giao dien"
npm install
npm run dev
```

Frontend phục vụ demo; không nằm trên đường chấm điểm chính.

## Kiểm tra chất lượng

```powershell
python -m pytest -v --basetemp .pytest_runtime -p no:cacheprovider
python -m ruff check src tests scripts
python -m mypy src scripts
```

Các kiểm tra trọng yếu gồm schema/contract, ingestion zero-loss, cache/index
versioning, parent expansion, reranker, metric LegalIR/LegalQA, API và submission.

## Cấu trúc repository

```text
.
├── configs/                         Cấu hình base, development và GPU
├── data/                            Raw/processed/index local; file lớn không commit
│   └── processed_v3/                Corpus chính thức đã audit
├── docs/
│   ├── members/                     Kế hoạch, setup và runbook của TV1–TV5
│   ├── models/                      Registry và ghi chú model
│   └── project/                     Thiết kế hệ thống, API, data pipeline, Git workflow
├── experiments/                     Thử nghiệm tách theo thành viên
├── frontend/                        Web UI demo
├── models/                          Model local; không commit weight lớn
├── outputs/                         Log chạy local/GPU
├── prompts/                         System prompt và RAG template có version
├── scripts/
│   ├── beam/                        Runner và thí nghiệm Task1 trên Beam
│   │   ├── task1_v2/                Module V2 score-first reranker
│   │   ├── task1_v3_residual/       Module V3 residual policy/ranking
│   │   │   ├── common.py            Helper chung và đường dẫn artifact V3
│   │   │   ├── train_residual_policy.py
│   │   │   ├── train_residual_policy_v3b.py
│   │   │   ├── forensic_benefit_neutral_signal.py
│   │   │   └── train_delta_recall_residual.py
│   │   ├── beam_task1_v1a_real.py
│   │   ├── beam_task1_v1a_eval_only.py
│   │   ├── beam_task1_v2_prepare_cpu.py
│   │   ├── beam_task1_v2_fold0.py
│   │   ├── beam_task1_v3_prepare_cpu.py
│   │   ├── beam_task1_v3_policy_cpu.py
│   │   ├── beam_task1_v3b_policy_cpu.py
│   │   ├── beam_task1_v3b_fold0_eval_cpu.py
│   │   └── beam_task1_delta_recall_residual_cpu.py
│   ├── ci_cd/                       Script CI/CD
│   ├── data_prep/                   Ingestion và indexing
│   ├── evaluation/                  Candidate generation, metric và reranker benchmark
│   ├── gpu/                         Preflight và pipeline RTX
│   ├── submission/                  Writer/validator submission LegalIR và LegalQA
│   ├── task1/                       Script Task1 tiện ích/đóng gói
│   └── training/                    Fine-tune/training script
├── src/udsc2026/                    Mã nguồn production
└── tests/                           Unit và integration tests
```

## Phân công

| Thành viên | Phạm vi chính |
| --- | --- |
| TV1 | FastAPI, orchestration, cache, logging và integration |
| TV2 | Embedding, dense/BM25 retrieval, FAISS/Qdrant và hybrid fusion |
| TV3 | Qwen3, QA engine, prompt, citation và anti-hallucination |
| TV4 | ETL, legal parser, parent–child chunking và benchmark data |
| TV5 | Reranking, evaluation, GPU/MLOps, submission và UI demo |

## Tài liệu chính

- [System Design](docs/project/10_system_design.md)
- [Data Pipeline](docs/project/11_data_pipeline.md)
- [API Contract](docs/project/api_contract.md)
- [Git Workflow](docs/project/git_workflow.md)
- [Docker Deployment](docs/project/12_docker_deployment.md)
- [TV1](docs/members/tv1/tv1.md)
- [TV2 Setup](docs/members/tv2/tv2_setup.md)
- [TV3 Setup](docs/members/tv3/tv3_setup.md)
- [TV4 Setup](docs/members/tv4/tv4_setup.md)
- [TV5 Setup](docs/members/tv5/tv5_setup.md)
- [TV5 GPU Runbook](docs/members/tv5/tv5_gpu_runbook.md)
- [Prompt Registry](prompts/README.md)

## Quy tắc làm việc

- Không commit raw data, processed corpus, vector store, model weights hoặc cache.
- Không đổi model/corpus chính thức mà không cập nhật manifest và benchmark.
- Không tái sử dụng index/candidate khi corpus hash hoặc model revision thay đổi.
- Mọi thay đổi contract phải kèm test và cập nhật tài liệu liên quan.
- Không force-push nhánh thành viên; hợp nhất bằng merge/fast-forward có kiểm tra.
