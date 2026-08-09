# TV2 — Retrieval Pipeline: trạng thái, setup và vận hành
## 25/07/2026
Tài liệu này mô tả những gì TV2 đã hoàn thành, cách chạy lại và các điểm tích hợp
với TV4/TV5. TV2 chỉ phụ trách retrieval; không chứa API route, frontend, QA generation
hay cross-encoder reranking.

## 1. TV2 đã hoàn thành gì?

| Nhóm | File chính | Tác dụng |
|---|---|---|
| Contract | `src/udsc2026/contracts/chunk.py` | Định nghĩa `LegalChunk`, input chung từ ingestion TV4 vào indexing TV2. |
| Embedding | `src/udsc2026/infrastructure/embedding/client.py` | Load HCMUTE embedding v2 từ local path, encode query hoặc batch document thành vector float đã normalize. |
| Embedding config | `src/udsc2026/infrastructure/embedding/config.py` | Compatibility wrapper cho config embedding cũ. |
| Config chung | `src/udsc2026/infrastructure/config.py` | Deep-merge `base.yaml` và `development.yaml`, tránh mỗi module tự đọc YAML. |
| VectorDB interface | `infrastructure/vector_db/base.py` | Interface chung, payload mapping và chuyển payload thành `RetrievalHit`. |
| Qdrant | `qdrant_adapter.py` | Lưu/search vector qua Qdrant, giữ payload citation. |
| FAISS | `faiss_adapter.py` | Lưu vector `IndexFlatIP` local, kèm JSON side-store metadata. |
| Factory | `vector_db/factory.py` | Chọn Qdrant hoặc FAISS từ config. |
| Dense | `retrieval/dense/dense_retriever.py` | Encode query rồi gọi VectorDB, gán `dense_score` và rank. |
| Sparse | `retrieval/sparse/tokenizer.py`, `bm25_retriever.py` | Tokenize tiếng Việt, giữ dấu/cụm Điều-Khoản-Điểm, build/search/save BM25 độc lập. |
| Hybrid | `retrieval/hybrid/score_fusion.py`, `hybrid_retriever.py` | Normalize, merge theo `chunk_id`, weighted fusion và giữ citation. |
| Index script | `scripts/data_prep/index_chunks.py` | Orchestrate JSONL → embedding → VectorDB → BM25. |
| Tests | `tests/retrieval/` | Mock model/Qdrant, test FAISS thật, BM25, Dense, Hybrid và fusion. |

Kết quả verify sau lần tích hợp ngày 09/08/2026: toàn repository `850 passed,
15 skipped`; nhóm test tập trung TV2/ingestion `143 passed, 1 skipped`.

## 2. TV2 có cần train model không?

TV2 hiện dùng HCMUTE embedding v2 đã pretrained và chỉ làm inference để tạo embedding.
Khi có dữ liệu TV4, quy trình cần chạy là:

1. TV4 sinh JSONL chunk hợp lệ.
2. TV2 validate JSONL bằng `LegalChunk`.
3. Chạy script indexing để tạo embedding, FAISS/Qdrant index và BM25 index.
4. Dense/Sparse/Hybrid dùng các index đó để trả `list[RetrievalHit]` cho TV1/TV5.

Chỉ cần fine-tune nếu benchmark cho thấy embedding pretrained chưa đủ tốt và team quyết
định làm một task ML riêng. Fine-tune không thuộc bước vận hành mặc định của TV2.

## 3. Phụ thuộc chéo với TV4 — đã đối chiếu chưa?

Đã đối chiếu output hiện tại của TV4 với `LegalChunk`:

- TV4 đã có ingestion thật trong `src/udsc2026/ingestion/`.
- Đã kiểm tra 4 file JSONL và 20 record TV4 trước sample dev.
- Không phát hiện field thiếu hoặc field thừa so với `LegalChunk`.
- Validation report của TV4 không có missing metadata, duplicate chunk ID hay orphan chunk.
- Các field mở rộng `parent_id`, `parent_text`, `section`, `effective_date` cũng đã được
  giữ trong contract/payload mapping.

Vì vậy hiện tại TV2 và TV4 đã khớp. Tuy nhiên, khi TV4 đổi schema JSONL, hai thành viên
phải đối chiếu lại trước khi merge; không tự sửa một bên.

## 4. Cấu hình

`configs/base.yaml` chứa cấu hình chung cho embedding, VectorDB, BM25 và hybrid.
`configs/development.yaml` chỉ override môi trường dev: FAISS và CPU.

```yaml
embedding:
  model_path: ./models/dek21-v2
  model_id: huyydangg/DEk21_hcmute_embedding_v2
  device: cpu
  batch_size: 32
  max_length: 256
  normalize_embeddings: true
```

Model `DEk21_hcmute_embedding_v2` được export bằng `sentence-transformers 5.4.1`
và `transformers 5.0.0`. Các thành viên phải cài đúng runtime contract này;
không dùng bản `sentence-transformers 3.x`/`transformers 4.x` cũ vì model sẽ
tham chiếu module `sentence_transformers.base` không tồn tại trong bản cũ.

FAISS dùng NumPy 1.x trong môi trường hiện tại, nên `requirements_dev.txt` pin
`numpy==1.26.4` và `scipy<1.14`.

Từ thư mục gốc repository, dùng Python 3.10–3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -c requirements_runtime.txt -e ".[llm,rerank,retrieval]"
```

Lệnh trên là môi trường CPU/dev. Trên máy thuê RTX, không cài `.[gpu]` thủ công
trước Torch CUDA; dùng `scripts/gpu/run_gpu_pipeline.ps1` theo
[`tv5_gpu_runbook.md`](../tv5/tv5_gpu_runbook.md). Script đó cài Torch CUDA 12.8
trước rồi mới cài các dependency GPU còn lại.

Không commit `.venv/`. Nếu chưa dùng editable install, đặt tạm package path:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Tải và kiểm tra model embedding local:

```powershell
python download_models.py --only embedding
python scripts/data_prep/verify_embedding.py
```

Chạy smoke index trên CPU/dev:

```powershell
python scripts/data_prep/index_chunks.py `
  --chunks-dir data/processed_v3/chunks `
  --config-env development `
  --vector-db-type faiss `
  --batch-size 32 `
  --max-chunks 1000 `
  --skip-bm25 `
  --force
```

Để chạy toàn bộ corpus trên RTX, dùng pipeline trong runbook TV5; không thay
`--config-env development` thành `gpu` trên máy local không có CUDA.

Script tạo:

- `data/vector_store/faiss/legal_chunks/index.faiss`
- `data/vector_store/faiss/legal_chunks/payloads.json`
- `data/vector_store/bm25/legal_chunks.pkl`
- manifest đi kèm dense index trong `data/vector_store/faiss/`

Chạy test:

```powershell
python -m pytest tests/retrieval/ -v
```

| Tasks | Phạm vi | Verify sau khi hoàn tất |
|---|---|---|
| 0 | `LegalChunk` contract | `python -c "from udsc2026.contracts import LegalChunk, RetrievalHit; print('ok')"` |
| 1 | `EmbeddingClient` HCMUTE Embedding v2 local | `python -c "from udsc2026.infrastructure.embedding import EmbeddingClient; print('ok')"` |
| 2 | Qdrant/FAISS adapter | `python -c "from udsc2026.infrastructure.vector_db.factory import get_vector_db_adapter; print('ok')"` |
| 3 | `DenseRetriever` | `python -c "from udsc2026.retrieval.dense.dense_retriever import DenseRetriever; print('ok')"` |
| 4 | `BM25Retriever` + tokenizer | `python -c "from udsc2026.retrieval.sparse.tokenizer import tokenize_vi; print(tokenize_vi('Điều 10 Bộ luật Lao động'))"` |
| 5 | `HybridRetriever` + fusion | `python -c "from udsc2026.retrieval.hybrid import fuse_scores; print(fuse_scores([], []))"` |
| 6 | Indexing script | Chạy script index với JSONL TV4 và kiểm tra `data/vector_store/`. |
| 7 | Hợp nhất config | Kiểm tra đủ `embedding`, `vector_db`, `sparse`, `hybrid`. |
| 8 | Unit tests | `python -m pytest tests/unit/test_retrieval -v` phải pass 100%. |
| 9 | Đối chiếu DoD | Sửa mọi mục FAIL trước khi bàn giao. |

Sau mỗi verify pass, commit đúng nhóm prompt. Không chạy prompt sau nếu verify bước hiện
tại còn fail.

## 5. Cấu trúc và trách nhiệm

- `src/udsc2026/contracts/`: `LegalChunk` từ TV4 và `RetrievalHit` dùng chung.
- `src/udsc2026/infrastructure/embedding/`: load HCMUTE embedding v2 local và encode query/chunks.
- `src/udsc2026/infrastructure/vector_db/`: interface backend-neutral, Qdrant và FAISS.
- `src/udsc2026/retrieval/dense/`: query embedding và dense search.
- `src/udsc2026/retrieval/sparse/`: tokenizer tiếng Việt và BM25 độc lập.
- `src/udsc2026/retrieval/hybrid/`: normalize, merge và weighted score fusion; không rerank.
- `configs/base.yaml`: model path, vector DB và hybrid settings.
- `data/vector_store/`: FAISS/BM25 index local; không commit dữ liệu index lớn.

TV2 cung cấp các API `search(query, top_k, filters=None)` cho dense, sparse và hybrid.
TV1 có thể gọi hybrid để lấy context; TV5 nhận `list[RetrievalHit]` để rerank.

## Tóm tắt vị trí file

### Cấu trúc các file TV2 đã làm việc

```text
udsc2026/
├── pyproject.toml              # ĐÃ SỬA: thêm [build-system] + [project]
├── requirements_dev.txt        # SẼ ĐƯỢC AGENT BỔ SUNG dần theo bảng trên (Prompt 1/2/4/7)
├── .venv/                      # môi trường ảo, KHÔNG commit vào git
├── configs/
│   ├── base.yaml                # SẼ CÓ THÊM section embedding/vector_db/sparse/hybrid
│   └── development.yaml
├── data/
│   ├── processed_v3/chunks/     # corpus canonical đã qua preflight
│   └── vector_store/            # output của Prompt 6: faiss/, bm25/ (qdrant chạy ngoài Docker)
├── scripts/
│   └── data_prep/index_chunks.py
├── src/udsc2026/
│   ├── contracts/                 # Prompt 0: chunk.py (LegalChunk) + retrieval.py (RetrievalHit, có sẵn)
│   ├── infrastructure/
│   │   ├── config.py              # Prompt 7
│   │   ├── embedding/             # Prompt 1
│   │   └── vector_db/             # Prompt 2
│   └── retrieval/
│       ├── dense/                 # Prompt 3
│       ├── sparse/                # Prompt 4
│       └── hybrid/                # Prompt 5
├── tests/unit/test_retrieval/     # Prompt 8
└── docs
    └── tv2_setup.md               # chính là file bạn đang đọc
```

# 06/08/2026

## Hướng dẫn test dữ liệu chính thức

```powershell
New-Item -ItemType Directory -Path D:\udsc2026\.pytest_runtime -Force
.\.venv\Scripts\python.exe -m pytest tests/retrieval/ -v
```

Các test logic nhỏ vẫn dùng mock để cô lập BM25/top-k/batch error. Khi kiểm tra
integration với dữ liệu chính thức, dùng Docker/Qdrant theo lệnh sau:

```powershell
docker compose up -d qdrant qdrant-ready
docker compose run --rm backend-smoke
docker compose logs --tail 200 qdrant qdrant-ready
```

Lệnh `backend-smoke` dùng image backend và các volume đã khai báo trong
`docker-compose.yml`; không dùng dữ liệu mock thay cho corpus chính thức.

## Chạy toàn bộ tasks TV2 trên Corpus thật (Data từ TV4)

Đây là luồng end-to-end của TV2 sau khi TV4 bàn giao corpus. Nếu TV4 vừa tái
tạo chunks, bước index phải dùng `--force` để không bị manifest cache bỏ qua.

### A. Nhận corpus canonical từ TV4

TV2 không chạy lại ingestion vào `data/processed_v3`. Thư mục này là corpus đã
được audit và phải được xem như immutable trong lúc build index. Trước khi chạy,
xác nhận bản copy từ Drive còn nguyên hash:

```powershell
python scripts/gpu/preflight.py --processed-root data/processed_v3
```

Nếu TV4 thật sự cần tạo corpus mới, phải ghi vào một thư mục trống khác rồi audit
trước khi thay thế bản canonical; không ghi đè trong lúc TV2 đang index:

```powershell
python scripts/data_prep/run_ingestion.py `
  --raw-directory data/raw/btc `
  --processed-root data/processed_candidate
```

Chỉ khi preflight đạt exit code `0` mới bắt đầu index TV2.

### B. Kiểm tra và index — bước TV2

1. Preflight trước khi chạy corpus:

```powershell
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
python -c "from udsc2026.contracts import LegalChunk, RetrievalHit; from udsc2026.retrieval.sparse.tokenizer import tokenize_vi; print('TV2 imports: OK', tokenize_vi('Điều 10'))"
python -m pytest tests/retrieval tests/unit/test_retrieval -q -o addopts=''
```

Chunks thật đã có sẵn tại `data/processed_v3/chunks/`. Với dữ liệu chính thức,
chạy toàn bộ chuỗi qua Docker/Qdrant theo đúng thứ tự:

Nếu vừa thay đổi source trong `src/` hoặc dependency, build lại image trước khi
chạy index. Container chỉ mount `scripts/` và dữ liệu runtime, không mount
source package `src/`.

```powershell
docker compose build --no-cache backend
```

2. Đầu tiên chạy validate để kiểm tra mapping giữa chunk và ground-truth, đồng thời
   tự tạo report audit về các file JSONL rỗng, không có record hoặc sai schema.
   Lệnh này không sửa dữ liệu và không ảnh hưởng tới các bước sau; nó chỉ giúp bạn
   biết còn những file nào cần TV4 sửa trước khi index.

```powershell
docker compose up -d qdrant qdrant-ready
docker compose run --rm --volume "${PWD}/scripts:/app/scripts:ro" --volume "${PWD}/data/processed_v3:/app/data/processed_v3:ro" --volume "${PWD}/data/processed_v3/metadata:/app/tv2_metadata" --volume "${PWD}/data/raw/btc:/app/data/raw/btc:ro" backend python /app/scripts/validate_chunk_mapping.py --chunks-dir /app/data/processed_v3/chunks --ir-train-file /app/data/raw/btc/LegalIR/train.json --report /app/tv2_metadata/chunk_mapping_report.json --audit-report /app/tv2_metadata/chunk_file_audit.json
```

Lệnh này dùng để khởi động Qdrant và chạy validate trên corpus TV4. Nó sẽ kiểm tra mapping giữa chunk và ground-truth, đồng thời tạo report audit về các file JSONL rỗng, không có record hoặc sai schema. Ý nghĩa của lệnh này là xác nhận dữ liệu chunk đã sẵn sàng cho bước index hay chưa, nhưng không sửa dữ liệu và không ảnh hưởng tới các bước sau.

Lệnh này sẽ sinh hai report:
- `data/processed_v3/metadata/chunk_mapping_report.json`: kết quả mapping chunk ↔ ground-truth.
- `data/processed_v3/metadata/chunk_file_audit.json`: danh sách file rỗng, không có record,
  JSON lỗi hoặc record không đúng schema.

3. Chỉ khi exit code 0 mới chạy bước index để build vector index và BM25 index.

```powershell
docker compose run --rm `
  --volume "${PWD}/scripts:/app/scripts:ro" `
  --volume "${PWD}/data/processed_v3:/app/data/processed_v3:ro" `
  --volume "${PWD}/data/processed_v3/metadata:/app/data/processed_v3/metadata" `
  --volume "${PWD}/data/vector_store:/app/data/vector_store" `
  backend python /app/scripts/data_prep/index_chunks.py `
  --chunks-dir /app/data/processed_v3/chunks `
  --vector-db-type qdrant `
  --batch-size 128 `
  --force
```

Lệnh này dùng để build index dense/sparse từ các chunk đã qua validate. Ý nghĩa của lệnh này là tạo vector index và BM25 index cho corpus, để các bước retrieval sau này có thể dùng được. Tác dụng chính là chuẩn bị dữ liệu truy vấn cho TV2.

`--batch-size 128` là mức khởi đầu an toàn cho CPU. Nếu container còn dư RAM và
không có lỗi out-of-memory, tăng lần lượt lên `256` rồi `512`; không chạy nhiều
process embedding song song vì mỗi process sẽ nạp một bản model vào RAM.

4. Sau khi manifest xác nhận, chạy benchmark sparse:

```powershell
docker compose run --rm `
  --volume "${PWD}/scripts:/app/scripts:ro" `
  --volume "${PWD}/data/processed_v3:/app/data/processed_v3:ro" `
  --volume "${PWD}/data/raw/btc:/app/data/raw/btc:ro" `
  --volume "${PWD}/data/reports:/app/data/reports" `
  backend python /app/scripts/benchmark_retrieval_internal.py `
  --chunks-dir /app/data/processed_v3/chunks `
  --ir-train-file /app/data/raw/btc/LegalIR/train.json `
  --mode sparse `
  --top-k 5

```

`benchmark_retrieval_internal.py` hiện chỉ triển khai `sparse`; không chạy
`--mode dense` hoặc `--mode hybrid` vì script sẽ chủ động báo lỗi. Hai benchmark
này chỉ được thêm vào hướng dẫn khi đã có implementation dùng configured vector
index và hybrid retriever.

Nếu thiếu model/backend, giữ nguyên lỗi môi trường và báo cụ thể; không nới lỏng validation.

TV2 được xem là hoàn tất khi preflight, validate, indexing, sparse benchmark
và dense/hybrid benchmark đã được triển khai đều pass. Artifact bàn giao gồm
dense index/collection, BM25 index, manifest, payload mapping,
`index_errors.json` và các report benchmark.

## Bức tranh bàn giao: Output của TV2 và Trách nhiệm của các thành viên tuyến sau

Phần này bám theo Pipeline chuẩn trong `docs/udsc2026_plan.md`:

`BTC files → TV4 ingestion/chunking → LegalChunk JSONL → TV2 indexes → TV5 rerank/evaluation → TV3 QA → TV1 API/submission orchestration`.

### 1. Danh sách output chi tiết của TV2

TV2 bàn giao các artifact sau:

- **Corpus đầu vào đã được kiểm tra:** các file `chunks/*.jsonl` chứa
  `LegalChunk` hợp lệ từ TV4. Mỗi chunk giữ `chunk_id`, `parent_id`, `doc_id`,
  `text`, `source` và các trường cấu trúc pháp lý cần cho truy hồi/citation.

- **Dense index:** index FAISS hoặc collection Qdrant được build từ toàn bộ
  corpus đã nghiệm thu. Artifact dense phải có payload mapping để từ kết quả
  vector truy ngược được chunk gốc và metadata của chunk.

- **Sparse index:** BM25 index chứa các chunk và tokenization phục vụ tìm kiếm
  theo từ khóa, số điều/khoản/điểm, số hiệu văn bản và thuật ngữ pháp lý.

- **`manifest.json`:** manifest đi kèm index, ghi số lượng chunk, loại backend,
  collection/index path, `corpus_hash`, `model_hash` hoặc model identity,
  cấu hình liên quan và commit/config nếu có. Manifest dùng để xác nhận index
  đang tương ứng với đúng corpus và model, tránh dùng index cũ hoặc stale.

- **`payloads.json` và payload mapping:** với FAISS, `payloads.json` lưu payload
  theo vị trí vector, gồm `chunk_id`, `doc_id`, `text`, `source`, các trường
  `law_name/article/clause` và `metadata`; với Qdrant, cấu trúc tương đương nằm
  trong payload của point. Dữ liệu này phải bảo toàn citation metadata, không
  chỉ lưu vector và score.

- **`list[RetrievalHit]`:** các retriever dense, sparse và hybrid trả về danh
  sách đối tượng đúng contract `RetrievalHit`. Mỗi hit mang `chunk_id`, `doc_id`,
  `text`, score phù hợp với backend, rank khi tầng đó gán rank, cùng citation
  metadata như `source`, `law_name`, `article`, `clause` và `metadata` nguyên
  vẹn. Đây là interface runtime để các thành viên tuyến sau dùng, không tự
  đọc trực tiếp FAISS/Qdrant/BM25.

### 2. Phân luồng bàn giao: ai nhận output và để làm gì?

#### Đối với TV5 — Reranking & MLOps

TV5 nhận `list[RetrievalHit]` từ TV2 làm candidate set cho Cross-encoder
Reranker:

- TV5 chạy Cross-encoder trên `text` của các candidate và query tương ứng.
- Reranker gán `rerank_score` hoặc `final_score`, sau đó sắp xếp lại thứ tự
  candidate và giới hạn về top-k cần bàn giao.
- Mục tiêu là cải thiện chất lượng xếp hạng, đặc biệt Precision/Recall ở nhóm
  kết quả đầu bảng so với thứ tự dense/sparse/hybrid ban đầu.
- TV5 tuyệt đối không được làm mất `chunk_id`, `doc_id`, `text`, `source`,
  `law_name`, `article`, `clause` hoặc metadata citation do TV2 truyền sang.
  Điểm rerank chỉ bổ sung/chỉnh thứ tự; không được thay thế hoặc tự suy đoán
  citation metadata.
- TV5 dùng các hit đã bảo toàn citation để tạo evaluation report, validate
  top-5 và bàn giao ordered `RetrievalHit` cho TV3/TV1.

#### Đối với TV3 — QA & LLM Engine

TV3 nhận các `RetrievalHit` đã được TV5 rerank; nếu pipeline baseline bypass
TV5 thì TV3 nhận trực tiếp hit từ TV2:

- TV3 bóc tách trường `text` của các hit để tạo Context đưa vào prompt cho
  Qwen3 hoặc LLM được cấu hình.
- TV3 dùng thứ tự, score và số lượng hit để chọn context trong giới hạn token,
  nhưng không được làm thay đổi sai lệch identity của nguồn.
- TV3 dùng `law_name`, `article`, `clause`, `source`, `chunk_id` và metadata để
  tạo câu trả lời pháp lý có trích dẫn chính xác.
- Citation parser phải ánh xạ citation về đúng `RetrievalHit`; nếu không xác
  minh được thì trả warning/unverified theo contract QA, không tự bịa nguồn.
- QA output của TV3 là `QAResponse` cùng answer, citations, warnings và thông
  tin prompt/model version để TV1 tiếp tục orchestration.

#### Đối với TV1 — API Orchestration & Submission

TV1 là lớp điều phối và đóng gói, không truy cập trực tiếp database retrieval:

- TV1 gọi hàm `search`/retrieval interface của TV2 thông qua dependency
  injection hoặc orchestrator; TV1 không gọi trực tiếp FAISS, Qdrant hay BM25.
- Với LegalIR, TV1 lấy các ID từ output `RetrievalHit`, ưu tiên identity đã
  được pipeline xác nhận như `doc_id` hoặc `parent_id`/mapping tương ứng.
- TV1 lọc các ID trùng nhưng giữ thứ tự xếp hạng, cắt tối đa đúng 5 ID theo
  ràng buộc của BTC, rồi format thành `submission.json` cuối cùng.
- TV1 phải bảo toàn đầy đủ question ID từ `public_official.json`, không làm mất,
  lặp hoặc tự điền đáp án từ `train.json`.
- TV1 giao artifact prediction cho TV5 validate exact schema/scorer và sau đó
  Leader đóng gói/nộp `submission.zip` cho BTC.

### Nguyên tắc bàn giao chung

- TV2 giao artifact có manifest/hash và contract `RetrievalHit`, không giao
  một index không truy được payload về chunk gốc.
- TV5, TV3 và TV1 dùng output qua contract; không tự đoán field mới hoặc đọc
  backend trực tiếp để sửa thiếu metadata.
- Mọi thay đổi corpus, index, model hoặc mapping phải có report/hash tương ứng
  để truy nguyên kết quả benchmark và submission.
