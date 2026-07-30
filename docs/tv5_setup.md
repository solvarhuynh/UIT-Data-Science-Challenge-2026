# TV5 — Reranking, Evaluation và MLOps

Tài liệu này mô tả cách chạy phần TV5 khi ban tổ chức chưa phát hành dataset
chính thức. Fixture trong `tests/fixtures/tv5/` chỉ dùng để phát triển và kiểm
thử; không phải dữ liệu cuộc thi và không được dùng để công bố điểm chính thức.

## 1. Cài đặt

Tạo môi trường Python 3.10-3.12 (chưa hỗ trợ Python 3.13+) rồi cài dependency đã
được khóa phiên bản. Dùng
`requirements_runtime.txt` làm constraint để môi trường phát triển không tự nâng
Torch/Transformers khác với image runtime:

```powershell
python -m pip install -c requirements_runtime.txt -r requirements_dev.txt
python -m pip install -c requirements_runtime.txt -e ".[rerank,retrieval]"
```

Cross-Encoder mặc định chạy offline. Đặt checkpoint tại đường dẫn cấu hình:

```text
models/reranker/
```

Hoặc override bằng biến môi trường:

```powershell
$env:RERANKER_ENABLED = "true"
$env:RERANKER_MODEL_PATH = "D:\models\reranker"
$env:RERANKER_DEVICE = "cpu"
$env:RERANKER_BATCH_SIZE = "16"
$env:RERANKER_MAX_LENGTH = "512"
$env:RERANKER_TOP_N = "10"
$env:RERANKER_LOCAL_FILES_ONLY = "true"
```

`RERANKER_LOCAL_FILES_ONLY=true` bảo đảm runtime không tự tải model. Chỉ chuyển
thành `false` khi chủ động cho phép truy cập Hugging Face Hub.

## 2. Production wiring Hybrid → Reranker

Entry point Python dùng chung của retrieval là hàm module-level
`udsc2026.retrieval.hybrid.search`:

```python
from udsc2026.retrieval.hybrid import search

hits = search(
    query="Người lao động được nghỉ phép bao nhiêu ngày?",
    top_k=5,
    filters={"law_name": "Bộ luật Lao động"},
)
```

Ở lần gọi đầu tiên trong mỗi process, hàm này:

1. đọc `configs/base.yaml` hoặc file do `UDSC2026_CONFIG_PATH` chỉ định;
2. khởi tạo dense retriever từ embedding client và VectorDB adapter;
3. load BM25 artifact từ `BM25_INDEX_PATH`, nếu không có thì dùng
   `hybrid.bm25_index_path`;
4. tạo `HybridRetriever` để fuse dense/BM25;
5. nếu reranker được bật, bọc hybrid bằng `RerankedRetriever` và
   `CrossEncoderReranker`.

Pipeline mặc định được cache trong process. Vì vậy phải đặt biến môi trường và
mount model/index **trước lần gọi `search()` đầu tiên**; sau khi đổi cấu hình,
hãy khởi động lại process. Cross-Encoder client vẫn lazy-load: bước khởi tạo
pipeline không load weights, weights chỉ được load ở lần scoring không rỗng đầu
tiên.

### Bật/tắt Cross-Encoder

Để chạy dense + BM25 hybrid mà không cần checkpoint reranker:

```powershell
$env:RERANKER_ENABLED = "false"
```

`RERANKER_ENABLED` chỉ nhận `true` hoặc `false` (không dùng `1`, `yes`). Khi là
`false`, default pipeline trả thẳng `HybridRetriever`, không tạo
`CrossEncoderClient`, và readiness không yêu cầu thư mục model reranker. Khi là
`true`, checkpoint phải có tại `RERANKER_MODEL_PATH` hoặc
`reranker.model_name_or_path`.

### Kích thước candidate pool

Hai cấu hình phải thỏa:

```text
0 < reranker.top_n <= hybrid.candidate_k
```

Với cấu hình mặc định, `candidate_k=50` và `top_n=10`. Dense và BM25 mỗi bên lấy
`candidate_k` kết quả để fuse; `RerankedRetriever` yêu cầu hybrid trả tối đa
`candidate_k` candidates, rồi Cross-Encoder trả tối đa
`min(top_k của caller, top_n)`. Pipeline từ chối cấu hình `top_n > candidate_k`
thay vì âm thầm rerank một pool quá nhỏ.

### Dùng Cross-Encoder độc lập

```python
from udsc2026.infrastructure.reranker import load_reranker_settings
from udsc2026.retrieval.reranking import CrossEncoderReranker

settings = load_reranker_settings("configs/base.yaml")
reranker = CrossEncoderReranker(settings.create_client())
reranked_hits = reranker.rerank(
    query="Người lao động được nghỉ phép bao nhiêu ngày?",
    candidates=hybrid_hits,
    top_n=settings.top_n,
)
```

Reranker:

- giữ nguyên `chunk_id`, `doc_id`, citation, metadata và score gốc;
- bổ sung `rerank_score`, `final_score`, `rank`;
- không thay đổi object đầu vào;
- giữ thứ tự ban đầu khi hai score bằng nhau;
- lazy-load model ở lần chấm điểm đầu tiên.

Nếu thiếu package, checkpoint hoặc device không hợp lệ, client báo lỗi phân
biệt rõ dependency, load model và inference.

Đây là wiring Python production trong core retrieval, chưa phải HTTP API. FastAPI
hiện chỉ đăng ký `GET /health` và `GET /ready`; chưa có `/query` hoặc
`/api/v1/query`. TV1 cần gọi `udsc2026.retrieval.hybrid.search` khi triển khai
route query end-to-end.

## 3. Benchmark phát triển

Fixture gồm 12 câu hỏi theo contract TV4:

```text
tests/fixtures/tv5/dev_benchmark.jsonl
tests/fixtures/tv5/predictions_before.json
tests/fixtures/tv5/predictions_after.jsonl
```

Chạy so sánh trước/sau rerank mà không cần model:

```powershell
python scripts/evaluate.py `
  --benchmark tests/fixtures/tv5/dev_benchmark.jsonl `
  --before tests/fixtures/tv5/predictions_before.json `
  --after tests/fixtures/tv5/predictions_after.jsonl `
  --output-dir artifacts/evaluation `
  --k 1 3 5
```

Output:

```text
artifacts/evaluation/
├── before.json
├── before.md
├── after.json
├── after.md
├── comparison.json
└── comparison.md
```

Bỏ `--after` để đánh giá một run và sinh `report.json`, `report.md`.

Metric hiện có:

- MRR: reciprocal rank của relevant chunk đầu tiên, macro-average theo câu hỏi;
- Recall@K: tỷ lệ gold chunk xuất hiện trong top K, macro-average;
- ROUGE-L: F1 theo longest common subsequence sau khi chuẩn hóa Unicode tiếng
  Việt;
- latency: total, min, max, mean, median, P95 và P99.

Loader căn chỉnh benchmark/prediction bằng `question_id`, không phụ thuộc thứ tự
record và từ chối ID thiếu, thừa hoặc trùng. JSON/JSONL được validate strict và
không chấp nhận `NaN`, `Infinity` hoặc `-Infinity`.

`evaluate_predictions()` tạo `dataset_fingerprint` SHA-256 từ toàn bộ benchmark
có thứ tự, gồm câu hỏi, reference answer, gold labels và metadata. API mức thấp
`evaluate_retrieval()` mặc định fingerprint danh sách `gold_chunk_ids`, hoặc
nhận fingerprint do caller truyền vào. JSON/Markdown report đều ghi hash này;
so sánh trước/sau bị từ chối nếu fingerprint, `sample_count`, `k_values` hoặc
coverage QA/latency không khớp. Delta trong comparison luôn được kiểm tra bằng
`after - before`, không nhận giá trị khai báo tùy ý.

Khi có `--after`, evaluator còn kiểm tra output rerank chỉ là subset của
candidate pool ban đầu, không sửa text/citation/metadata/score gốc, có
`rerank_score == final_score`, rank liên tiếp, score giảm dần và giữ thứ tự gốc
khi score bằng nhau.

## 4. Tạo BM25 index từ output ingestion

Sau khi TV4 tạo `data/processed/chunks/*.jsonl`, tạo artifact sparse retrieval:

```powershell
python scripts/build_bm25.py `
  --chunks data/processed/chunks `
  --output data/vector_store/bm25/index.json
```

Script đọc `.json`/`.jsonl` theo thứ tự ổn định, validate từng `LegalChunk`, từ
chối ID trùng hoặc dữ liệu không hữu hạn, rồi ghi index JSON atomically. Biến
`BM25_INDEX_PATH` được cả runtime và readiness dùng chung.

Artifact BM25 là JSON có `schema_version` và các `LegalChunk`, không pickle object
Python hay object `rank_bm25`. Khi runtime load, schema và từng chunk được
validate lại rồi BM25 được rebuild deterministically trong bộ nhớ. Module-level
hybrid search luôn load artifact này trước khi phục vụ truy vấn.

## 5. Submission writer

Do schema CodaLab chưa được ban tổ chức công bố, writer dùng schema JSON cấu
hình thay vì hard-code tên cột:

```powershell
python scripts/write_submission.py `
  --input tests/fixtures/tv5/qa_responses.jsonl `
  --schema tests/fixtures/tv5/submission_schema.json `
  --output artifacts/submission.csv
```

Writer đọc trực tiếp `QAResponse`, validate field/serializer, ghi CSV atomically
và hỗ trợ UTF-8 BOM cho Excel. Khi có schema chính thức, chỉ thay file schema và
adapter input; không sửa metric hoặc reranker.

## 6. Kiểm tra trước khi bàn giao

Các lệnh dưới đây là checklist cần chạy trong đúng môi trường bàn giao, không
phải tuyên bố rằng Docker image hoặc full HTTP RAG đã được xác nhận:

```powershell
python -m pytest -q
python scripts/smoke_test.py --mode host
docker compose config --quiet
```

`smoke_test.py --mode host` dùng fake scorer trong bộ nhớ; nó kiểm tra contract,
config và wiring nhẹ nhưng không tải embedding/LLM/Cross-Encoder thật.
`docker compose config --quiet` chỉ validate cấu hình Compose, không build image
và không khởi động service.

Chỉ khi Docker Engine, network và các artifact mount cần thiết sẵn sàng mới chạy:

```powershell
docker compose build backend
docker compose run --rm --no-deps backend-smoke
```

Sau khi khởi động stack, chỉ có thể probe hai endpoint hiện đã triển khai:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Không dùng `/query` làm tiêu chí pass ở thời điểm này vì route đó chưa tồn tại.
Kết quả Docker build hoặc inference model thật phải được ghi nhận riêng sau khi
đã chạy thành công trong môi trường có Docker/model/index, không suy ra từ host
smoke test.

Xem hướng dẫn container, volume, healthcheck và troubleshooting tại
[`docs/project/12_docker_deployment.md`](project/12_docker_deployment.md).

## 7. Khi có dataset chính thức

1. Viết adapter chuyển record chính thức sang `BenchmarkSample`.
2. Giữ nguyên `question_id`, `gold_chunk_ids` và reference answer từ nguồn.
3. Chạy validation, kiểm tra ID trùng/thiếu và lưu version/checksum dataset.
4. Chạy baseline trước rerank rồi mới chạy sau rerank trên cùng sample/K.
5. Chốt schema submission theo tài liệu ban tổ chức và thêm regression test.
6. Không so sánh hai report khác sample set hoặc khác `k_values`.
