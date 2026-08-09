# TV5 MLOps Runbook

Tài liệu này mô tả cách đóng gói và kiểm tra backend LegalIR/LegalQA ở môi
trường local hoặc CI. Cấu hình hiện tại ưu tiên khả năng tái lập, an toàn và
không đưa model/dữ liệu lớn vào image.

## 1. Thành phần được đóng gói

Stack mặc định trong `docker-compose.yml` gồm:

| Service | Vai trò | Health/probe |
| --- | --- | --- |
| `backend` | FastAPI runtime của dự án | Healthcheck `GET /health`; kiểm tra sâu `GET /ready` |
| `qdrant` | VectorDB | Probe ngoài gọi `GET /healthz` |
| `qdrant-ready` | Job chờ Qdrant sẵn sàng | Thoát mã `0` khi Qdrant phản hồi |
| `redis` | Hạ tầng cache, sẵn cho adapter TV1 | `redis-cli ping` |
| `backend-smoke` | Kiểm tra image không cần model | One-shot, profile `smoke` |

`backend` khởi động độc lập rồi dùng `GET /ready` để báo trạng thái dependency;
nhờ đó chế độ FAISS offline không bị phụ thuộc cứng vào Qdrant/Redis. Image
Qdrant chính thức không có `curl`/`wget`, vì vậy `qdrant-ready` dùng một
container Alpine nhỏ để gọi endpoint `/healthz` chính thức khi chạy full stack.
Readiness còn kiểm tra model directories, BM25 artifact và vector collection.

Với FAISS offline, đặt `VECTOR_DB_TYPE=faiss`, để trống `REDIS_URL` nếu không
dùng cache, rồi chạy riêng `docker compose up backend`.

Frontend không nằm trong stack này. Giao diện hiện được chạy riêng từ
`frontend/giao dien` để việc kiểm tra backend không phụ thuộc Node.js.

## 2. Chuẩn bị

Yêu cầu:

- Docker Engine hoặc Docker Desktop có Docker Compose plugin **v2.20.0+**.
  Không dùng binary Compose v1 `docker-compose`.
- Python 3.10-3.12 (chưa hỗ trợ Python 3.13+) nếu chạy smoke test trực tiếp trên
  máy.
- Model đặt trong `models/` nếu chạy inference thật. Health và smoke test không
  tải model.

Kiểm tra phiên bản Compose trước khi chạy:

```powershell
docker compose version
```

Chuẩn bị package Python cho smoke test trên host:

```powershell
python -m pip install -c requirements_runtime.txt -e .
```

Khi chạy toàn bộ LLM/retrieval/reranker thật ngoài Docker, cài đầy đủ optional
dependency:

```powershell
python -m pip install -c requirements_runtime.txt -e ".[llm,rerank,retrieval]"
```

Tạo cấu hình local:

```powershell
Copy-Item .env.example .env
```

Không commit `.env`. Các giá trị mặc định chỉ bind cổng vào `127.0.0.1`, không
public Redis hoặc Qdrant ra mạng ngoài.

### Windows và Docker Desktop

Tạo trước các thư mục bind mount để Docker không tự tạo nhầm quyền:

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
New-Item -ItemType Directory -Force .\data\processed_v3 | Out-Null
New-Item -ItemType Directory -Force .\data\vector_store | Out-Null
```

Docker Desktop phải chạy **Linux containers**. Nếu gặp `Mounts denied`,
`file sharing` hoặc container không thấy file trên ổ `D:`, cấp quyền chia sẻ
ổ/thư mục chứa repository trong Docker Desktop rồi restart Docker Desktop. Với
WSL 2, kiểm tra distro integration và chạy `wsl --update`; tránh đặt model/index
lớn trên ổ mạng hoặc thư mục đồng bộ đám mây vì bind mount có thể chậm hoặc khóa
file.

## 3. Kiểm tra nhanh trước khi chạy

Chạy smoke test offline trên máy:

```powershell
python .\scripts\ci_cd\smoke_test.py --mode host
```

Smoke test kiểm tra:

- Python runtime và package `udsc2026`;
- contract `RetrievalHit`;
- cấu trúc `configs/base.yaml`;
- độ đầy đủ của `.env.example` và các rule an toàn trong `.dockerignore`;
- public API evaluation;
- reranking bằng fake scorer trong bộ nhớ, không tải Cross-Encoder.

Kiểm tra Compose sau khi render biến môi trường:

```powershell
docker compose config --quiet
```

Lệnh trên phải hoàn tất với exit code `0`.

## 4. Build và chạy stack

Build riêng backend:

```powershell
docker compose build backend
```

Khởi động:

```powershell
docker compose up -d --wait --wait-timeout 180
docker compose ps
```

`--wait` trả exit code khác `0` nếu service không đạt trạng thái
running/healthy trong thời hạn, phù hợp hơn việc chạy `up -d` rồi kiểm tra thủ
công. Lệnh này chờ healthcheck của Compose, hiện dùng `/health` cho backend.

Kiểm tra backend:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

`/health` là **liveness**: chỉ xác nhận process API còn hoạt động, không load
model và không probe dependency. `/ready` là **readiness** theo cấu hình hiện
tại:

- luôn kiểm tra file `UDSC2026_CONFIG_PATH` (mặc định `configs/base.yaml`);
- với model path đã khai báo, yêu cầu đó là thư mục truy cập được và có ít nhất
  một file; reranker chỉ bắt buộc khi `RERANKER_ENABLED=true`;
- nếu `VECTOR_DB_TYPE=qdrant`, kiểm tra cả `/healthz` lẫn collection cấu hình và
  gửi header `api-key` khi có; nếu là `faiss`, deserialize index, kiểm tra
  checksum/schema/payload và yêu cầu ít nhất một vector;
- nếu có `BM25_INDEX_PATH`, production loader validate schema, từng `LegalChunk`
  và khả năng tokenize; kết quả FAISS/BM25 được cache theo snapshot file để
  readiness định kỳ không load lại index lớn;
- nếu có `REDIS_URL`, readiness hỗ trợ `redis`/`rediss`, AUTH, SELECT database,
  rồi gửi RESP `PING` và chỉ chấp nhận đúng `PONG`;
- mọi network probe có timeout ngắn, không load checkpoint vào RAM/VRAM;
- trả HTTP `200` với `status=ready` khi tất cả check đạt;
- trả HTTP `503` với `status=not_ready`, kèm `checks` và `missing` khi có check
  thất bại hoặc cấu hình backend/boolean không hợp lệ.

Vì healthcheck container gọi `/health`, `docker compose up --wait` vẫn có thể
thành công trong khi `/ready` trả `503` nếu thiếu model/index hoặc dependency
runtime chưa thật sự truy cập được. Đây là hành vi chủ đích để có thể khởi động
shell API và chẩn đoán mà không bị restart loop.

Xem log:

```powershell
docker compose logs --no-color --timestamps --tail 200 backend qdrant-ready qdrant redis
docker compose logs --follow backend
```

Dòng đầu phù hợp để chụp log CI; dòng thứ hai theo dõi backend đến khi nhấn
`Ctrl+C`.

Dọn container/network nhưng giữ named volume:

```powershell
docker compose down --remove-orphans
```

Chỉ khi chủ động muốn xóa sạch dữ liệu phát sinh mới chạy:

```powershell
docker compose down --volumes --remove-orphans
```

Lệnh có `--volumes` xóa Qdrant index, Redis data, model cache và submission
output trong named volumes; không thể hoàn tác bằng Compose. Các thư mục bind
mount trên host (`models/`, `data/processed_v3/`, `data/vector_store/`) không bị
xóa.

## 5. Smoke test bên trong image

Service `backend-smoke` không mở port, không cần Redis/Qdrant và không tải
model:

```powershell
docker compose run --rm --no-deps backend-smoke
```

Compose tự kích hoạt profile khi service được gọi trực tiếp. Có thể chạy rõ
profile nếu muốn:

```powershell
docker compose --profile smoke run --rm --no-deps backend-smoke
```

## 6. Model, dữ liệu và output

Docker build context loại trừ:

```text
models/
data/task1/
data/task2/
data/raw/
data/processed_v3/
data/vector_store/
frontend/
**/node_modules/
```

Backend nhận model và dữ liệu processed qua bind mount read-only:

```text
./models            -> /app/models
./data/processed_v3 -> /app/data/processed_v3
./data/vector_store -> /app/data/vector_store
```

Ba bind mount trên đều read-only trong container. Hai phase file canonical nằm
trên host tại `data/task1/warmup.json` và `data/task2/warmup.json`; Compose hiện
không mount hai thư mục này, vì vậy mặc định chạy các CLI audit, evaluation và
submission trên host. Không copy phase data vào image. Nếu cần chạy CLI trong
container, chỉ bổ sung bind mount đúng thư mục task ở chế độ read-only.

Qdrant, Redis, cache Hugging Face và output dùng named volumes riêng. Output
trong container tách thành `/app/output/task1/` và `/app/output/task2/`, tương
ứng với `artifacts/task1/` và `artifacts/task2/` trên host. Sao chép hai
submission ra host khi backend đang chạy:

```powershell
New-Item -ItemType Directory -Force .\artifacts\task1 | Out-Null
New-Item -ItemType Directory -Force .\artifacts\task2 | Out-Null
docker compose cp backend:/app/output/task1/submission.zip .\artifacts\task1\submission.zip
docker compose cp backend:/app/output/task2/submission.zip .\artifacts\task2\submission.zip
```

Cả hai task đều nộp ZIP chỉ chứa duy nhất `submission.json`, nhưng schema không
được dùng lẫn nhau:

- LegalIR: object
  `{"<question_id>": {"answer": ["<document_id>", "..."]}}`;
- LegalQA: object `{"<question_id>": {"answer": "<câu trả lời>"}}`.

Writer CSV `write_submission.py` chỉ là adapter `QAResponse` legacy, không phải
format nộp chính thức của Task 1 hay Task 2. METEOR/ROUGE-L local của Task 2 chỉ
là diagnostic và report ghi `official_scorer_parity=false`; chưa có cơ sở tuyên
bố parity với scorer ẩn. Xem runbook
[`docs/members/tv5/tv5_legalir_warmup.md`](../members/tv5/tv5_legalir_warmup.md) và
[`docs/members/tv5/tv5_legalqa_warmup.md`](../members/tv5/tv5_legalqa_warmup.md).

Image `codalab/codalab-legacy:py39` trên trang thi là runtime của scoring
program; vòng file-submission không chạy source/model của đội trong image đó và
không buộc backend của repo hạ xuống Python 3.9.

## 7. Biến môi trường chính

`.env.example` là nguồn đầy đủ và có thể copy trực tiếp. Các biến quan trọng
nhất được nhóm dưới đây.

### Build và host mount

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `PYTHON_VERSION` | `3.11.9-slim-bookworm` | Base image Python được pin |
| `APP_UID` / `APP_GID` | `10001` | User không-root trong image |
| `PYTORCH_CPU_INDEX_URL` | `https://download.pytorch.org/whl/cpu` | Official CPU wheel index cho image mặc định |
| `BACKEND_EXTRAS` | `llm,rerank,retrieval` | Extra cài vào image production |
| `BACKEND_IMAGE` | `hcmute-shipcode/udsc2026-backend:local` | Tên/tag image local |
| `BACKEND_BIND_ADDRESS` | `127.0.0.1` | Interface backend publish ra host |
| `BACKEND_PORT` | `8000` | Cổng backend trên host |
| `UDSC2026_CONFIG_PATH` | `/app/configs/base.yaml` | Config runtime trong container |
| `MODELS_DIR` | `./models` | Thư mục model mount read-only |
| `PROCESSED_DATA_DIR` | `./data/processed_v3` | Corpus V3 mount read-only |
| `VECTOR_STORE_DIR` | `./data/vector_store` | Vector index mount read-only |

### Runtime model và retrieval

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `MODEL_EMBEDDER_PATH` | `/app/models/dek21-v2` | Checkpoint DEk21 v2 |
| `MODEL_LLM_PATH` | `/app/models/qwen3-legal` | Checkpoint Qwen3 |
| `EMBEDDING_DEVICE` | `cpu` | Device embedding |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch embedding |
| `EMBEDDING_MAX_LENGTH` | `256` | Giới hạn token embedding |
| `EMBEDDING_NORMALIZE_EMBEDDINGS` | `true` | Chuẩn hóa vector trước cosine/dot search |
| `LLM_BACKEND` | `transformers` | Backend sinh câu trả lời |
| `LLM_DEVICE` | `cpu` | Device LLM |
| `LLM_DTYPE` | `float32` | Kiểu dữ liệu LLM an toàn trên CPU |
| `LLM_MAX_NEW_TOKENS` | `1024` | Số token sinh tối đa |
| `LLM_TEMPERATURE` / `LLM_TOP_P` | `0.1` / `0.9` | Tham số sampling |
| `LLM_REPETITION_PENALTY` | `1.05` | Giảm lặp nội dung |
| `LLM_STREAM` | `false` | Bật/tắt sinh token dạng stream |
| `LLM_TIMEOUT_SECONDS` | `60` | Timeout generation |
| `RERANKER_ENABLED` | `true` | Bật/tắt bước Cross-Encoder |
| `RERANKER_MODEL_PATH` | `/app/models/reranker` | Checkpoint Cross-Encoder |
| `RERANKER_DEVICE` | `cpu` | Device Cross-Encoder |
| `RERANKER_BATCH_SIZE` | `16` | Batch scoring |
| `RERANKER_MAX_LENGTH` | `512` | Giới hạn token của mỗi pair |
| `RERANKER_TOP_N` | `10` | Số candidate giữ sau rerank |
| `RERANKER_LOCAL_FILES_ONLY` | `true` | Chặn tự tải model từ mạng |
| `VECTOR_DB_TYPE` | `qdrant` | Chọn adapter `qdrant` hoặc `faiss` |
| `FAISS_INDEX_PATH` | `/app/data/vector_store/faiss` | Index FAISS trong container |
| `FAISS_COLLECTION_NAME` | `legal_chunks` | Thư mục collection dưới FAISS path |
| `BM25_INDEX_PATH` | `/app/data/vector_store/bm25/index.json` | Sparse index đã validate |

### Service và vận hành

| Biến | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `QDRANT_URL` | `http://qdrant:6333` | URL Qdrant nội bộ |
| `QDRANT_API_KEY` | rỗng | API key tùy chọn; không cần cho Qdrant local |
| `QDRANT_COLLECTION_NAME` | `legal_chunks` | Collection truy hồi |
| `QDRANT_BIND_ADDRESS` / `QDRANT_HTTP_PORT` | `127.0.0.1` / `6333` | Bind Qdrant ra host |
| `REDIS_URL` | `redis://redis:6379/0` | URL Redis nội bộ |
| `REDIS_BIND_ADDRESS` / `REDIS_PORT` | `127.0.0.1` / `6380` | Bind Redis ra host; container vẫn dùng `6379` |
| `SUBMISSION_PATH` | `/app/output/submission.csv` | Adapter CSV tổng quát (legacy, không dùng để nộp Task 1/2) |
| `LEGAL_IR_SUBMISSION_PATH` | `/app/output/task1/submission.zip` | Artifact ZIP chính thức của Task 1 |
| `LEGAL_QA_SUBMISSION_PATH` | `/app/output/task2/submission.zip` | Artifact ZIP chính thức của Task 2 |
| `QDRANT_HEALTH_RETRIES` | `30` | Số lần probe Qdrant |
| `QDRANT_HEALTH_INTERVAL_SECONDS` | `1` | Khoảng nghỉ giữa hai probe |
| `LOG_MAX_SIZE` / `LOG_MAX_FILES` | `10m` / `3` | Giới hạn log Docker |

Khi chạy Python trực tiếp ngoài Docker, đổi `QDRANT_URL` và `REDIS_URL` sang
`localhost`. Tên service `qdrant`/`redis` chỉ phân giải được trong network của
Compose. Không đặt token, password hoặc đường dẫn riêng của máy vào
`.env.example`.

## 8. Thiết kế image

`docker/backend.Dockerfile` có hai stage:

1. `wheel-builder` build wheel ứng dụng và dependency runtime.
2. `runtime` chỉ cài wheel, config, prompt, host smoke và các CLI
   audit/evaluate/write/validate chính thức của hai task.

Các dependency runtime trực tiếp được khóa trong `requirements_runtime.txt` và
được truyền vào `pip wheel --constraint`, nên cùng commit không tự trôi phiên
bản FAISS/NumPy/Transformers giữa hai lần build.

Stage build lấy riêng PyTorch CPU từ official PyTorch wheel index trước khi giải
quyết các dependency còn lại. Việc này tránh kéo các gói NVIDIA không được dùng
vào image CPU mặc định. Máy chạy GPU nên dùng Dockerfile/profile GPU riêng thay
vì đổi `LLM_DEVICE` trên image CPU.

Runtime:

- mặc định cài extra `llm,rerank,retrieval` để Qwen3, Cross-Encoder và
  VectorDB adapter hoạt động trong production;
- chạy bằng user `app` UID/GID `10001`, không chạy root;
- không dùng editable install và không chứa source tree;
- không chứa npm, test tools, model weights, raw data hoặc vector index;
- không chứa hai oracle `DO_NOT_SUBMIT`; các script rò rỉ label chỉ tồn tại ở
  môi trường phát triển local;
- dùng log unbuffered và healthcheck không cần `curl`;
- mount model/corpus read-only, tách cache/output thành volumes có thể ghi.

Nếu thêm dependency runtime, khai báo trong `pyproject.toml` rồi build lại.
Dependency chỉ dùng cho test/lint tiếp tục để trong `requirements_dev.txt`.
Có thể đặt `BACKEND_EXTRAS=` để build image core-only phục vụ chẩn đoán, nhưng
image production phải giữ `BACKEND_EXTRAS=llm,rerank,retrieval`. Redis trong
Compose hiện là hạ tầng đã sẵn sàng; chỉ thêm Redis Python client khi TV1 có
cache adapter thật, tránh cài dependency chưa được code sử dụng.

## 9. Quy trình CI đề xuất

Các bước tối thiểu:

```powershell
python .\scripts\ci_cd\smoke_test.py --mode host
docker compose config --quiet
docker compose build backend
docker compose run --rm --no-deps backend-smoke
```

Job integration có Docker daemon có thể chạy thêm:

```powershell
try {
    docker compose up -d --wait --wait-timeout 180
    if ($LASTEXITCODE -ne 0) {
        throw "Compose stack did not become healthy"
    }
    docker compose ps
    Invoke-RestMethod http://127.0.0.1:8000/health
}
finally {
    docker compose ps --all
    docker compose logs --no-color --timestamps --tail 300
    docker compose down --remove-orphans
}
```

Khối `finally` bảo đảm luôn thu thập trạng thái/log trước khi hạ stack, kể cả
khi startup hoặc assertion thất bại. Job có gắn đủ checkpoint nên assert thêm
`/ready`; job chỉ kiểm tra image shell thì không coi HTTP `503` từ `/ready` là
lỗi.

## 10. Xử lý lỗi thường gặp

### Backend unhealthy

```powershell
docker compose ps
docker compose logs --tail 200 backend
docker compose run --rm --no-deps backend-smoke
```

Nếu smoke import lỗi, kiểm tra dependency runtime đã nằm trong
`pyproject.toml`, không chỉ trong `requirements_dev.txt`.

### Qdrant probe timeout

```powershell
docker compose logs qdrant qdrant-ready
```

Kiểm tra volume permission và xung đột cổng `6333`. Có thể tăng
`QDRANT_HEALTH_RETRIES` trong `.env` trên máy chậm.

### Không ghi được model, corpus hoặc FAISS index

Đây là hành vi chủ đích: ba bind mount này read-only. Pipeline tải model, xử lý
corpus hoặc tạo index phải chạy như bước chuẩn bị riêng trên host, không thực
hiện lúc backend startup.

### Port đã được sử dụng

Đổi `BACKEND_PORT`, `QDRANT_HTTP_PORT` hoặc `REDIS_PORT` trong `.env`. URL nội
bộ container vẫn giữ cổng `8000`, `6333` và `6379`.
