# TV5 — Báo cáo tiến độ Reranking, Evaluation và MLOps

> Thành viên phụ trách: Nguyên Khang
> Ngày cập nhật: 01/08/2026
> Phạm vi: Reranking, Evaluation Benchmark, Submission adapter, MLOps và
> quality gates

## 1. Tổng quan kết quả

Tui đã hoàn thành phần nền tảng để nhận candidate từ Hybrid Search, tái xếp
hạng bằng Cross-Encoder, bảo toàn căn cứ pháp luật, đánh giá kết quả trước/sau
rerank và đóng gói hệ thống để có thể kiểm tra lặp lại.

BTC đã cung cấp Warm-up và schema submission cho cả LegalIR lẫn LegalQA. Dữ
liệu đã được tách về `data/task1/warmup.json` và
`data/task2/warmup.json`. TV5 đã bổ sung evaluator theo từng task cùng
writer/validator JSON/ZIP chính thức. Workspace vẫn chưa có kho
`selected-contexts` và checkpoint/model prediction thật, nên retrieval và QA
inference thực tế còn phụ thuộc bàn giao từ TV2/TV3/TV4.

| Hạng mục | Trạng thái | Kết quả |
| --- | --- | --- |
| Reranker client và cấu hình | Hoàn thành | Lazy-load, offline-first, hỗ trợ batch/device/max length |
| Cross-Encoder reranking | Hoàn thành | Có `rerank_score`, `final_score`, `rank`, giữ score gốc |
| Hybrid → Reranker wiring | Hoàn thành | Có thể bật/tắt bằng cấu hình, kiểm tra candidate pool |
| Bảo toàn citation metadata | Hoàn thành | Giữ chunk, văn bản, điều, khoản, điểm, nguồn và parent |
| Evaluation metrics | Hoàn thành | LegalIR MRR/Recall@3; LegalQA METEOR/ROUGE-L diagnostic; metric phát triển và latency |
| Report trước/sau rerank | Hoàn thành | JSON và Markdown, kiểm tra dataset fingerprint |
| LegalIR submission | Hoàn thành | Strict JSON/ZIP writer, corpus/coverage validator, deterministic và atomic |
| LegalQA submission | Hoàn thành | Strict object JSON/ZIP writer, exact coverage, raw-answer preservation và release empty-answer gate |
| Warm-up audit | Hoàn thành | Task 1: 500 câu/37 câu multi-gold; Task 2: 500 câu cùng Unicode/whitespace diagnostics |
| Codabench contract | Đã đối chiếu 01/08/2026 | Hai task đều dùng root object keyed by question ID; scoring source vẫn ẩn |
| BM25 artifact | Hoàn thành | Build/load deterministic, ghi file atomically |
| Docker và Compose | Hoàn thành cấu hình | Chờ Docker daemon để build/run image thực tế |
| Smoke test và quality gates | Hoàn thành | Backend, frontend, security và config đều có lệnh kiểm tra |
| Warm-up local evaluation | Hoàn thành pipeline | Chờ ranking thật; 37 multi-gold cần BTC xác nhận semantics |
| Cross-Encoder inference thật | Chưa thể chạy | Chờ checkpoint/model artifact chính thức |

## 2. Những task đã hoàn thành

### 2.1. Reranker client

- Viết client Cross-Encoder có cơ chế lazy-load, không tải model khi import.
- Hỗ trợ `batch_size`, `device`, `max_length` và chế độ
  `local_files_only`.
- Tách lỗi dependency, lỗi load checkpoint và lỗi inference để dễ debug.
- Chuẩn hóa cấu hình từ YAML và biến môi trường.
- Từ chối giá trị cấu hình không hợp lệ thay vì âm thầm dùng mặc định.

### 2.2. Cross-Encoder reranking

- Nhận `query` và danh sách `RetrievalHit` từ Hybrid Search.
- Chấm điểm theo batch và sắp xếp giảm dần theo Cross-Encoder score.
- Bổ sung `rerank_score`, `final_score` và `rank`.
- Giữ nguyên `dense_score`, `sparse_score`, `hybrid_score` và object đầu vào.
- Giữ thứ tự ban đầu khi hai candidate có cùng score.
- Giới hạn output theo `top_n`.

### 2.3. Tích hợp Hybrid Search

- Tích hợp luồng Dense + BM25 → score fusion → Cross-Encoder.
- Cho phép tắt reranker để chạy baseline Hybrid Search.
- Kiểm tra điều kiện `0 < reranker.top_n <= hybrid.candidate_k`.
- Cache pipeline theo process và lazy-load model ở lần scoring đầu tiên.
- Dùng cùng đường dẫn BM25 artifact cho runtime và readiness check.

### 2.4. Bảo toàn căn cứ pháp luật

Sau khi rerank, kết quả vẫn giữ:

```text
chunk_id
doc_id
text
law_name
article
clause
point
source
parent_id
metadata
dense_score
sparse_score
hybrid_score
```

Điều này giúp tầng QA có thể sinh citation mà không phải khôi phục metadata từ
nguồn khác.

### 2.5. Evaluation benchmark

- Xây dựng schema strict cho benchmark, prediction và report.
- Load được JSON/JSONL, căn chỉnh record bằng `question_id`.
- Phát hiện ID thiếu, thừa, trùng hoặc dữ liệu không hữu hạn.
- Tính MRR, Recall@K, ROUGE-L và latency percentile.
- Tạo SHA-256 fingerprint cho dataset để ngăn so sánh nhầm hai tập dữ liệu.
- So sánh before/after và tính delta theo `after - before`.
- Kiểm tra output rerank là subset hợp lệ của candidate pool ban đầu.
- Từ chối trường hợp reranker sửa text, citation, metadata hoặc score gốc.
- Xuất report JSON và Markdown bằng thao tác ghi file atomic.
- Bổ sung evaluator Task 1 ở cấp `document_id`, tách khỏi metric chunk-level cũ.
- Tính đúng official MRR trên toàn ranking và Recall@3 dạng hit-rate.
- Có per-query gold rank/contribution, exact ID coverage và dataset fingerprint.
- Tách `official_single_gold` khỏi `warmup_any_gold`; không tự lấy `answer[0]`.
- Collapse chunk hits sang unique document IDs theo best-ranked chunk.
- Bổ sung loader/evaluator Task 2 với exact ID coverage, raw-text preservation,
  dataset/prediction fingerprint và per-query diagnostics.
- Tính METEOR exact-token và ROUGE-L token-LCS theo profile local versioned;
  report luôn ghi `official_scorer_parity=false` vì implementation scorer BTC
  chưa được công khai.

### 2.6. Submission adapter

- LegalIR: validate exact `{id, documents}`, tối thiểu ba document, duplicate,
  question coverage, corpus membership và optional full ranking.
- LegalIR: đọc/ghi JSON hoặc ZIP chỉ có `submission.json`, deterministic và
  atomic; không extract archive.
- LegalQA: chuyển prediction nội bộ `{id, answer}` sang wire object
  `question_id -> {answer}`, kiểm tra coverage, Unicode, empty-answer policy và
  ghi ZIP deterministic/atomic.
- Chặn JSON/ZIP lỗi, key trùng, constant ngoài chuẩn, UTF-8 lỗi, size limit,
  symlink/special member, zip-slip, CRC và compression không hỗ trợ.
- Giữ writer CSV schema-driven hiện có dưới nhãn legacy; CSV không phải format
  nộp Task 1 hoặc Task 2.

### 2.7. BM25 persistence

- Tạo CLI build BM25 artifact từ output chunking của TV4.
- Validate từng `LegalChunk`, ID trùng và giá trị số không hữu hạn.
- Lưu schema version và dữ liệu cần thiết dưới dạng JSON, không pickle object.
- Rebuild BM25 deterministic trong bộ nhớ khi runtime load artifact.
- Ghi file atomically để tránh artifact hỏng khi tiến trình bị ngắt.

### 2.8. MLOps và cấu hình chạy

- Tạo backend Dockerfile multi-stage.
- Loại model weights, raw data, vector index, cache và Git data khỏi build
  context.
- Tạo Compose cho backend, Qdrant và Redis.
- Chuẩn hóa `.env.example` cho model path, reranker, VectorDB, Redis, port và
  submission output.
- Bổ sung health/readiness contract và endpoint.
- Viết host smoke test không cần checkpoint thật.
- Tạo lệnh kiểm tra thống nhất cho Windows, Linux và pre-commit.

### 2.9. Tích hợp đầu ra TV4

- Hợp nhất public API Evaluation của TV5 với synthetic benchmark generator của
  TV4.
- Giữ generator mock corpus của TV4 nhưng bổ sung ghi file atomic và chống
  symlink/hard-link.
- Giữ đường dẫn output neo theo repository và bảo toàn file không liên quan.
- Smoke test kiểm tra public API Evaluation, gồm synthetic benchmark,
  LegalIR/LegalQA metric và hai submission contract.

## 3. Các file đã làm việc và tác dụng

### 3.1. Reranking

| File | Tác dụng |
| --- | --- |
| `src/udsc2026/infrastructure/reranker/client.py` | Load model và batch scoring |
| `src/udsc2026/infrastructure/reranker/config.py` | Đọc/validate cấu hình reranker |
| `src/udsc2026/infrastructure/reranker/__init__.py` | Public API của infrastructure reranker |
| `src/udsc2026/retrieval/reranking/cross_encoder.py` | Rerank `RetrievalHit` và bổ sung score/rank |
| `src/udsc2026/retrieval/reranking/pipeline.py` | Bọc Hybrid Retriever bằng reranker |
| `src/udsc2026/retrieval/reranking/__init__.py` | Public API của retrieval reranking |
| `src/udsc2026/retrieval/hybrid/hybrid_retriever.py` | Wiring Dense, BM25, fusion và reranking |
| `src/udsc2026/retrieval/hybrid/config.py` | Validate candidate pool và cấu hình Hybrid |
| `src/udsc2026/retrieval/hybrid/__init__.py` | Entry point `search()` dùng chung |
| `src/udsc2026/contracts/retrieval.py` | Contract kết quả retrieval/reranking |

### 3.2. Evaluation và submission

| File | Tác dụng |
| --- | --- |
| `src/udsc2026/evaluation/models.py` | Schema benchmark, prediction, report và comparison |
| `src/udsc2026/evaluation/metrics.py` | MRR, Recall@K, ROUGE-L và latency |
| `src/udsc2026/evaluation/loaders.py` | Load/validate JSON và JSONL |
| `src/udsc2026/evaluation/evaluator.py` | Chạy evaluation và kiểm tra before/after |
| `src/udsc2026/evaluation/reporting.py` | Xuất report JSON/Markdown |
| `src/udsc2026/evaluation/submission.py` | Schema-driven submission writer |
| `src/udsc2026/evaluation/legal_ir.py` | Metric/contract Task 1 document-level và Warm-up loader |
| `src/udsc2026/evaluation/legal_ir_submission.py` | Strict official JSON/ZIP writer-validator |
| `src/udsc2026/evaluation/legal_qa_metrics.py` | METEOR/ROUGE-L local diagnostic profile |
| `src/udsc2026/evaluation/legal_qa.py` | Loader, alignment, fingerprint và report Task 2 |
| `src/udsc2026/evaluation/legal_qa_submission.py` | Strict official Task 2 JSON/ZIP writer-validator |
| `src/udsc2026/evaluation/__init__.py` | Public API Evaluation và synthetic benchmark |
| `scripts/evaluate.py` | CLI đánh giá prediction đã tạo sẵn |
| `scripts/write_submission.py` | CLI xuất submission CSV |
| `scripts/audit_legal_ir_warmup.py` | Audit schema, labels, Unicode và multi-gold |
| `scripts/evaluate_legal_ir.py` | CLI MRR/Recall@3 document-level |
| `scripts/write_legal_ir_submission.py` | Tạo JSON/ZIP Task 1 |
| `scripts/validate_legal_ir_submission.py` | Validate artifact trước upload |
| `scripts/make_warmup_smoke_submission.py` | Oracle label-leaking có guard, chỉ smoke local |
| `scripts/audit_legal_qa_warmup.py` | Audit schema và Unicode Task 2 |
| `scripts/evaluate_legal_qa.py` | CLI METEOR/ROUGE-L diagnostic |
| `scripts/write_legal_qa_submission.py` | Chuyển prediction nội bộ sang ZIP object Task 2 |
| `scripts/validate_legal_qa_submission.py` | Standalone release validator Task 2 |
| `scripts/make_legal_qa_warmup_smoke_submission.py` | Oracle Task 2 có guard `DO_NOT_SUBMIT` |

### 3.3. Retrieval, API và MLOps

| File | Tác dụng |
| --- | --- |
| `scripts/build_bm25.py` | Build BM25 artifact từ LegalChunk |
| `src/udsc2026/retrieval/sparse/bm25_retriever.py` | Load/save và truy vấn BM25 |
| `src/udsc2026/infrastructure/vector_db/base.py` | Chuẩn hóa mapping metadata và parent ID |
| `src/udsc2026/infrastructure/vector_db/factory.py` | Khởi tạo adapter theo cấu hình |
| `src/udsc2026/api/app.py` | FastAPI health/readiness |
| `src/udsc2026/contracts/health.py` | Schema health/readiness |
| `src/udsc2026/config.py` | Load cấu hình dùng chung |
| `docker/backend.Dockerfile` | Backend image multi-stage |
| `docker-compose.yml` | Backend, Qdrant và Redis stack |
| `.dockerignore` | Loại artifact lớn/nhạy cảm khỏi image |
| `.env.example` | Danh sách biến môi trường chuẩn |
| `requirements_runtime.txt` | Khóa dependency runtime tương thích |

### 3.4. Quality gates và tài liệu

| File | Tác dụng |
| --- | --- |
| `scripts/smoke_test.py` | Smoke test contract/config/reranker/evaluation/API |
| `scripts/check-all.ps1` | Chạy toàn bộ backend, frontend và Compose checks |
| `scripts/check-tv5.ps1` | Kiểm tra nhanh phạm vi TV5 |
| `scripts/check-ci-local.sh` | CI local cho Windows Git Bash/Linux |
| `scripts/quality_gate.py` | Ruff, Mypy, pydocstyle và Bandit |
| `.pre-commit-config.yaml` | Chặn commit không đạt quality gate |
| `pyproject.toml` | Cấu hình dependency và công cụ Python |
| `pytest.ini` | Marker và chính sách warning của pytest |
| `frontend/giao dien/package.json` | Lệnh lint, type-check, test và build frontend |
| `docs/tv5_setup.md` | Hướng dẫn cài đặt và vận hành phần TV5 |
| `docs/tv5_legalir_warmup.md` | Runbook Task 1 Warm-up và checklist upload |
| `docs/project/12_docker_deployment.md` | Hướng dẫn Docker/Compose |

### 3.5. Test và fixture

Các nhóm test chính:

```text
tests/unit/test_reranking/
tests/unit/test_evaluation/
tests/unit/test_retrieval/
tests/unit/test_vector_db/
tests/unit/test_api/
tests/unit/test_embedding/
tests/fixtures/tv5/
```

Fixture TV5 gồm:

```text
dev_benchmark.jsonl
predictions_before.json
predictions_after.jsonl
qa_responses.jsonl
submission_schema.json
```

## 4. Kết quả kiểm thử gần nhất

Trạng thái Python đã xác nhận ngày 01/08/2026 sau khi tích hợp Warm-up:

| Kiểm tra | Kết quả |
| --- | --- |
| Pytest | 712 passed, 0 failed (653 + 59 native VectorDB shard) |
| Evaluation test suite | 364 passed, gồm LegalIR và LegalQA core/submission/CLI |
| LegalQA focused | 148 passed trên metric, evaluator, submission và CLI |
| Coverage | 87.97%, ngưỡng bắt buộc 70% |
| Ruff format/lint | Passed |
| Mypy | Passed trên 80 source files |
| pydocstyle | Passed |
| Bandit | Passed |
| TV5 host smoke | 9/9 passed |
| Frontend lint | Passed |
| Frontend type-check | Passed |
| Frontend production build | Passed |
| Pre-commit | Passed |
| Docker Compose static config | Passed |

FAISS 1.8 native abort trên macOS nếu chạy VectorDB ở cuối cùng một process đã
load Torch/SciPy. CI local chạy đủ toàn bộ test nhưng cô lập VectorDB thành
process thứ hai và append coverage; đây không phải skip. Không có assertion hay
test nghiệp vụ nào fail.

## 5. Cách kiểm tra lại

### Chạy toàn bộ cổng kiểm tra trên Windows

```powershell
.\scripts\check-all.ps1
```

### Chạy kiểm tra riêng TV5

```powershell
.\scripts\check-tv5.ps1
python scripts/smoke_test.py --mode host
```

### Chạy pre-commit

```powershell
python -m pre_commit run --all-files
```

### Chạy benchmark fixture trước/sau rerank

```powershell
python scripts/evaluate.py `
  --benchmark tests/fixtures/tv5/dev_benchmark.jsonl `
  --before tests/fixtures/tv5/predictions_before.json `
  --after tests/fixtures/tv5/predictions_after.jsonl `
  --output-dir artifacts/evaluation `
  --k 1 3 5
```

### Kiểm tra Compose

```powershell
docker compose config --quiet
```

## 6. Phần còn chờ để nghiệm thu thực tế

Các mục sau chưa thể tuyên bố hoàn thành thực chiến vì thiếu đầu vào bên ngoài:

1. Chạy Cross-Encoder thật trên checkpoint chính thức.
2. Benchmark trên dataset/test set BTC chính thức.
3. Đo latency thực trên GPU hoặc hạ tầng thi đấu.
4. Nhận giải thích chính thức cho 37 record Task 1 Warm-up có nhiều gold.
5. Build và chạy toàn bộ Docker stack khi Docker daemon/model/index sẵn sàng.
6. Chạy live pipeline để tự sinh prediction before/after thay vì chỉ đánh giá
   prediction file đã tạo sẵn.
7. Nhận full corpus manifest để validate document IDs; xác nhận top-K hay full
   ranking với BTC. Quy mô 10.000 × 8.500 cần writer streaming nếu full ranking
   là bắt buộc, vì giới hạn an toàn hiện tại là 128 MiB JSON.
8. Tích hợp retrieval vào HTTP query endpoint; hiện backend mới có
   `/health` và `/ready`.
9. Đối chiếu exact implementation/parameters METEOR và ROUGE-L với scorer BTC;
   điểm Task 2 hiện chỉ là diagnostic có gắn profile.
10. Nhận answer model thật từ TV3 để chạy error analysis Task 2; oracle Warm-up
    chỉ kiểm tra plumbing và tuyệt đối không được upload.

Khi TV2/TV4 bàn giao corpus và ranking thật, TV5 dùng contract hiện có để chạy
before/after, phân tích lỗi, đóng gói và bàn giao artifact; không cần đoán lại
schema submission.

## 7. Bàn giao cho thành viên khác

- **TV1/Backend:** gọi `udsc2026.retrieval.hybrid.search` khi triển khai query
  endpoint.
- **TV2/Retrieval:** cung cấp embedding model, VectorDB collection và BM25
  artifact đúng contract.
- **TV3/QA:** nhận danh sách đã rerank cùng citation metadata, tạo đúng một
  answer cho mỗi Task 2 question ID và bàn giao prediction nội bộ.
- **TV4/Data:** cung cấp `LegalChunk` và synthetic/official benchmark đúng
  `question_id`.
- **TV5:** chạy before/after benchmark, kiểm tra regression và xuất submission.

Hướng dẫn vận hành chi tiết nằm tại
[`docs/tv5_setup.md`](tv5_setup.md) và runbook Warm-up tại
[`docs/tv5_legalir_warmup.md`](tv5_legalir_warmup.md) và
[`docs/tv5_legalqa_warmup.md`](tv5_legalqa_warmup.md).
