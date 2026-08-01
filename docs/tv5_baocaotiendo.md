# TV5 — Báo cáo tiến độ Reranking, Evaluation và MLOps

> Thành viên phụ trách: Nguyên Khang
> Ngày cập nhật: 31/07/2026
> Phạm vi: Reranking, Evaluation Benchmark, Submission adapter, MLOps và
> quality gates

## 1. Tổng quan kết quả

Tui đã hoàn thành phần nền tảng để nhận candidate từ Hybrid Search, tái xếp
hạng bằng Cross-Encoder, bảo toàn căn cứ pháp luật, đánh giá kết quả trước/sau
rerank và đóng gói hệ thống để có thể kiểm tra lặp lại.

Do ban tổ chức chưa cung cấp dataset, checkpoint và schema submission chính
thức, các phần phụ thuộc dữ liệu thật được thiết kế theo contract và kiểm thử
bằng fixture deterministic

| Hạng mục | Trạng thái | Kết quả |
| --- | --- | --- |
| Reranker client và cấu hình | Hoàn thành | Lazy-load, offline-first, hỗ trợ batch/device/max length |
| Cross-Encoder reranking | Hoàn thành | Có `rerank_score`, `final_score`, `rank`, giữ score gốc |
| Hybrid → Reranker wiring | Hoàn thành | Có thể bật/tắt bằng cấu hình, kiểm tra candidate pool |
| Bảo toàn citation metadata | Hoàn thành | Giữ chunk, văn bản, điều, khoản, điểm, nguồn và parent |
| Evaluation metrics | Hoàn thành | MRR, Recall@K, ROUGE-L và latency |
| Report trước/sau rerank | Hoàn thành | JSON và Markdown, kiểm tra dataset fingerprint |
| Submission writer | Hoàn thành theo adapter | Chờ schema CodaLab chính thức để chốt mapping |
| BM25 artifact | Hoàn thành | Build/load deterministic, ghi file atomically |
| Docker và Compose | Hoàn thành cấu hình | Chờ Docker daemon để build/run image thực tế |
| Smoke test và quality gates | Hoàn thành | Backend, frontend, security và config đều có lệnh kiểm tra |
| Benchmark BTC chính thức | Chưa thể chạy | Chờ dataset/test set từ ban tổ chức |
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

### 2.6. Submission adapter

- Đọc trực tiếp dữ liệu theo contract `QAResponse`.
- Dùng schema JSON cấu hình tên cột và serializer thay vì hard-code.
- Validate field trước khi xuất.
- Ghi CSV atomically và hỗ trợ UTF-8 BOM cho Excel.
- Có fixture schema để kiểm thử trước khi CodaLab công bố schema thật.

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
- Smoke test kiểm tra đủ 28 public symbol của Evaluation, bao gồm synthetic
  benchmark API.

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
| `src/udsc2026/evaluation/__init__.py` | Public API Evaluation và synthetic benchmark |
| `scripts/evaluate.py` | CLI đánh giá prediction đã tạo sẵn |
| `scripts/write_submission.py` | CLI xuất submission CSV |

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

Trạng thái đã xác nhận sau khi hợp nhất code mới nhất từ TV4:

| Kiểm tra | Kết quả |
| --- | --- |
| Pytest | 420 passed, 5 skipped |
| Coverage | 85.63%, ngưỡng bắt buộc 70% |
| Ruff format/lint | Passed |
| Mypy | Passed trên 74 source files |
| pydocstyle | Passed |
| Bandit | Passed |
| TV5 host smoke | 9/9 passed |
| Frontend lint | Passed |
| Frontend type-check | Passed |
| Frontend production build | Passed |
| Pre-commit | Passed |
| Docker Compose static config | Passed |

Năm test bị skip đều do khác biệt nền tảng Windows:

- ba test cần symbolic link;
- hai test kiểm tra permission bit chỉ có ý nghĩa trên POSIX.

Không có test nghiệp vụ nào bị fail.

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
4. Chốt mapping submission theo schema CodaLab chính thức.
5. Build và chạy toàn bộ Docker stack khi Docker daemon/model/index sẵn sàng.
6. Chạy live pipeline để tự sinh prediction before/after thay vì chỉ đánh giá
   prediction file đã tạo sẵn.
7. Tích hợp retrieval vào HTTP query endpoint; hiện backend mới có
   `/health` và `/ready`.

Khi ban tổ chức phát hành dữ liệu, TV5 chỉ cần bổ sung adapter input, checkpoint
và schema submission; metric, report, validation và writer hiện tại có thể giữ
nguyên.

## 7. Bàn giao cho thành viên khác

- **TV1/Backend:** gọi `udsc2026.retrieval.hybrid.search` khi triển khai query
  endpoint.
- **TV2/Retrieval:** cung cấp embedding model, VectorDB collection và BM25
  artifact đúng contract.
- **TV3/QA:** nhận danh sách đã rerank cùng citation metadata để tạo câu trả lời.
- **TV4/Data:** cung cấp `LegalChunk` và synthetic/official benchmark đúng
  `question_id`.
- **TV5:** chạy before/after benchmark, kiểm tra regression và xuất submission.

Hướng dẫn vận hành chi tiết nằm tại
[`docs/tv5_setup.md`](tv5_setup.md).
