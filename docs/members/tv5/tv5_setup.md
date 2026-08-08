# TV5 — Reranking, Evaluation và MLOps

Xem danh sách task đã hoàn thành, các file đã can thiệp, kết quả kiểm thử và
phạm vi bàn giao tại
[`docs/tv5_baocaotiendo.md`](tv5_baocaotiendo.md).

Hai bộ dữ liệu Warm-up canonical trong repo là `data/task1/warmup.json` cho
LegalIR và `data/task2/warmup.json` cho LegalQA. Contract, audit và lệnh vận
hành riêng của từng task nằm tại:

- [`docs/tv5_legalir_warmup.md`](tv5_legalir_warmup.md);
- [`docs/tv5_legalqa_warmup.md`](tv5_legalqa_warmup.md).

Fixture trong `tests/fixtures/tv5/` chỉ dùng để phát triển/kiểm thử, không phải
dữ liệu cuộc thi và không được dùng để công bố điểm chính thức. Workspace hiện
chưa có `selected-contexts.zip` hoặc model prediction thật.

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
$env:RERANKER_USE_FP16 = "false"
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
  --output-dir artifacts/task1/evaluation/dev `
  --k 1 3 5
```

Output:

```text
artifacts/task1/evaluation/dev/
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
  Việt; đây là metric phát triển của benchmark chung, không phải bằng chứng
  parity với scorer LegalQA ẩn;
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

### 3.1. Chạy Cross-Encoder thật cho TV5

TV5 **chưa cần train hoặc fine-tune model** ở baseline đầu tiên. Dùng trực tiếp
checkpoint pretrained `BAAI/bge-reranker-v2-m3`, đo before/after trên cùng candidate
pool, rồi chỉ cân nhắc fine-tune khi metric cho thấy baseline chưa đạt yêu cầu.

Có thể kiểm tra model và toàn bộ luồng trên fixture 12 câu trước, không cần chờ TV2:

```powershell
python scripts/evaluation/benchmark_reranker.py `
  --benchmark tests/fixtures/tv5/dev_benchmark.jsonl `
  --candidates tests/fixtures/tv5/predictions_before.json `
  --model BAAI/bge-reranker-v2-m3 `
  --allow-remote-model `
  --device cuda:0 `
  --fp16 `
  --batch-size 8 `
  --candidate-k 3 `
  --top-n 3 `
  --k 1 3 `
  --output-dir artifacts/tv5/smoke-bge-reranker-v2-m3
```

`--allow-remote-model` chỉ cần ở lần đầu nếu truyền model ID và máy chưa có model
trong Hugging Face cache. Sau khi tải xong có thể bỏ cờ này, hoặc tải checkpoint vào
`models/reranker/` và đổi `--model models/reranker`. `--fp16` chỉ dùng trên CUDA; nếu
chạy CPU thì bỏ cả `--device cuda:0` và `--fp16`.

Khi TV2 bàn giao prediction thật, file đó phải theo contract `PredictionSample`: mỗi
`question_id` có một mảng `hits`, mỗi hit tối thiểu chứa `chunk_id`, `doc_id`, `text`.
ID phải khớp chính xác với benchmark. Lệnh chạy chính:

```powershell
python scripts/evaluation/benchmark_reranker.py `
  --benchmark data/processed/benchmarks/synthetic_qa.jsonl `
  --candidates artifacts/tv2/hybrid_predictions.jsonl `
  --model BAAI/bge-reranker-v2-m3 `
  --allow-remote-model `
  --device cuda:0 `
  --fp16 `
  --batch-size 8 `
  --candidate-k 50 `
  --top-n 5 `
  --k 1 3 5 `
  --output-dir artifacts/tv5/bge-reranker-v2-m3
```

Tên `artifacts/tv2/hybrid_predictions.jsonl` là contract bàn giao đề xuất; thay bằng
đường dẫn thật TV2 cung cấp. Script không cần index và không tự chạy retrieval: nó
chỉ đọc candidate đã có, rerank, kiểm tra không làm mất/sửa provenance, rồi sinh:

```text
artifacts/tv5/bge-reranker-v2-m3/
├── predictions_before.json
├── predictions_after.jsonl
├── run_manifest.json
└── evaluation/
    ├── before.json
    ├── before.md
    ├── after.json
    ├── after.md
    ├── comparison.json
    └── comparison.md
```

`run_manifest.json` lưu SHA-256 input, model/config, số query–candidate pair,
reranker latency và delta metric. Vì vậy đây là artifact cần gửi nhóm trưởng sau mỗi
run thật, cùng với `evaluation/comparison.md`.

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

## 5. Evaluation và submission theo task

### 5.1. LegalIR chính thức

LegalIR nộp `submission.zip` chỉ chứa `submission.json`, không phải CSV. Tạo và
validate artifact bằng:

```powershell
python scripts/write_legal_ir_submission.py `
  --input artifacts/task1/predictions.json `
  --questions data/task1/warmup.json `
  --corpus-manifest artifacts/task1/corpus_document_ids.json `
  --output artifacts/task1/submission.zip

python scripts/validate_legal_ir_submission.py `
  --input artifacts/task1/submission.zip `
  --questions data/task1/warmup.json `
  --corpus-manifest artifacts/task1/corpus_document_ids.json
```

Corpus manifest chưa có trong repo; TV4 cần trích từ toàn bộ context BTC. Không
dùng tập document IDs xuất hiện trong gold Warm-up thay cho corpus thật.

Wire format trong `submission.json` là JSON object keyed theo question ID:
`{"<question_id>": {"answer": ["<document_id>", "..."]}}`. Array record dùng
trong prediction nội bộ không phải schema nộp Codabench. Mỗi câu có thể có
nhiều gold document; metric chính thức là macro Recall, còn macro Precision là
metric phụ/tiebreak. BTC không còn yêu cầu tối thiểu ba document trong mỗi
`answer`.

Không nối full corpus hoặc bù document chỉ để tăng độ dài danh sách. TV5 cần
chọn threshold/top-K trên dev để ưu tiên Recall nhưng vẫn kiểm soát Precision;
mọi document thừa đều làm giảm Precision. Thứ tự giảm dần relevance vẫn được
giữ trong JSON dù hai metric chính thức so sánh theo tập document.

### 5.2. LegalQA chính thức

Task 2 dùng dữ liệu `data/task2/warmup.json` và artifact tách riêng dưới
`artifacts/task2/`. Audit, đánh giá local, đóng gói và validate theo thứ tự:

```powershell
python scripts/audit_legal_qa_warmup.py `
  --input data/task2/warmup.json `
  --output artifacts/task2/warmup_audit.json

python scripts/evaluate_legal_qa.py `
  --references data/task2/warmup.json `
  --predictions artifacts/task2/predictions.json `
  --output artifacts/task2/evaluation/diagnostic_report.json

python scripts/write_legal_qa_submission.py `
  --input artifacts/task2/predictions.json `
  --questions data/task2/warmup.json `
  --output artifacts/task2/submission.zip

python scripts/validate_legal_qa_submission.py `
  --input artifacts/task2/submission.zip `
  --questions data/task2/warmup.json
```

Submission Task 2 là ZIP chỉ chứa `submission.json`. Wire format là JSON object
`{"<question_id>": {"answer": "<câu trả lời>"}}`; mỗi value chỉ có field
string `answer`. Prediction input local có thể được CLI biểu diễn thành các
record `id`/`answer`, nhưng writer phải serialize sang object chính thức trên.

METEOR/ROUGE-L do evaluator local tính chỉ là **diagnostic**: report luôn ghi
`evaluation_scope="local_diagnostic"` và `official_scorer_parity=false`. Chưa có
implementation/config scorer ẩn từ BTC nên không dùng các điểm này để tuyên bố
parity hoặc dự đoán điểm leaderboard. Xem toàn bộ contract và release checklist
tại [`docs/tv5_legalqa_warmup.md`](tv5_legalqa_warmup.md).

### 5.3. Legacy generic CSV adapter

Writer cũ dưới đây vẫn hữu ích cho luồng `QAResponse` tổng quát, nhưng chỉ là
legacy adapter và không phải format nộp chính thức của Task 1 hoặc Task 2:

```powershell
python scripts/write_submission.py `
  --input tests/fixtures/tv5/qa_responses.jsonl `
  --schema tests/fixtures/tv5/submission_schema.json `
  --output artifacts/legacy/submission.csv
```

Writer đọc trực tiếp `QAResponse`, validate field/serializer, ghi CSV atomically
và hỗ trợ UTF-8 BOM cho Excel. Ba contract được cô lập để legacy CSV không làm
hỏng submission ZIP của LegalIR hoặc LegalQA.

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

## 7. Quy trình khi nhận đủ corpus và phase data

Không dùng chung file prediction, report hoặc submission giữa hai task. Task 1
luôn ở `artifacts/task1/`; Task 2 luôn ở `artifacts/task2/`.

### 7.1. Task 1 — LegalIR

1. TV4 tạo `artifacts/task1/corpus_document_ids.json` từ toàn bộ context BTC và
   lưu checksum; không suy ra corpus từ gold Warm-up.
2. TV2 collapse chunk ranking sang document ranking bằng
   `legal_ir_prediction_from_hits()` và ghi `artifacts/task1/predictions.json`.
3. TV5 kiểm tra exact question coverage, duplicate, corpus membership, rồi so
   sánh baseline/rerank trên cùng sample, toàn bộ multi-gold labels và
   fingerprint.
4. Dùng macro Recall document-level để quyết định model và macro Precision làm
   metric phụ/tiebreak; metric chunk-level chỉ dùng debug retrieval nội bộ.
5. Ghi/validate `artifacts/task1/submission.zip`, xác nhận wire format object,
   rồi lưu checksum, config và report trước khi bàn giao nhóm trưởng.

### 7.2. Task 2 — LegalQA

1. TV4 bàn giao phase data immutable tương ứng với
   `data/task2/warmup.json`, question manifest và checksum.
2. TV2 bàn giao retrieval trace riêng; TV3 bàn giao đúng một answer cho mỗi
   question ID vào `artifacts/task2/predictions.json`.
3. TV5 kiểm tra exact ID coverage, duplicate/extra field và tách trace/citation
   khỏi payload submission.
4. Chạy METEOR/ROUGE-L local để chẩn đoán và so sánh run cùng fingerprint;
   không coi report này là scorer ẩn hoặc điểm leaderboard.
5. Ghi/validate `artifacts/task2/submission.zip`, xác nhận wire format object,
   rồi lưu checksum, model/config, prompt version và report trước khi bàn giao.
