# UDSC2026 — Engineering Execution Plan

## Phạm vi và nguyên tắc

Phạm vi gồm Task 1 — LegalIR và Task 2 — LegalQA. Pipeline chuẩn:

`BTC files → TV4 ingestion/chunking → LegalChunk JSONL → TV2 indexes → TV5 rerank/evaluation → TV3 QA → TV1 API/submission orchestration`.

```mermaid

%% =========================
%% GIAI ĐOẠN 1
%% =========================
subgraph A["📂 DỮ LIỆU GỐC (RAW DATA)"]
    A1[(Dataset / Documents)]
end

%% =========================
%% GIAI ĐOẠN 2
%% =========================
subgraph B["🧹 XỬ LÝ & CHIA NHỎ DỮ LIỆU"]
    B1["TV4<br/>Read Data"]
    B2["TV4<br/>Parse Documents"]
    B3["TV4<br/>Clean & Normalize"]
    B4["TV4<br/>Chunking"]
end

%% =========================
%% GIAI ĐOẠN 3
%% =========================
subgraph C["🔎 CHỈ MỤC & TRUY XUẤT"]
    C1["TV2<br/>Build Index (FAISS / Qdrant)"]
    C2["TV2<br/>BM25 Retrieval"]
    C3["TV2<br/>Dense Retrieval"]
    C4["TV2<br/>Hybrid Retrieval"]
end

%% =========================
%% GIAI ĐOẠN 4
%% =========================
subgraph D["⭐ ĐÁNH GIÁ & RERANK"]
    D1["TV5<br/>Reranking"]
    D2["TV5<br/>Scorer Evaluation"]
end

%% =========================
%% GIAI ĐOẠN 5
%% =========================
subgraph E["🤖 SINH CÂU TRẢ LỜI"]
    E1["TV3<br/>QA Engine"]
    E2["TV3<br/>LLM Generation"]
end

%% =========================
%% GIAI ĐOẠN 6
%% =========================
subgraph F["🌐 ĐÓNG GÓI & TÍCH HỢP"]
    F1["TV1<br/>API Orchestration"]
    F2["TV1<br/>Pipeline Integration"]
end

%% =========================
%% GIAI ĐOẠN 7
%% =========================
subgraph G["🏁 NỘP BÀI"]
    G1["Leader<br/>Submit to Codabench"]
end

%% =========================
%% DATA FLOW
%% =========================
A1 --> B1
B1 --> B2
B2 --> B3
B3 --> B4

B4 --> C1
C1 --> C2
C2 --> C3
C3 --> C4

C4 --> D1
D1 --> D2

D2 --> E1
E1 --> E2

E2 --> F1
F1 --> F2

F2 --> G1
```

Raw data, model weights và index lớn không commit. Mọi module giao tiếp qua `src/udsc2026/contracts/`.

## PHẦN 1 — Phân tích tài nguyên dữ liệu và hiện trạng repository

### 1.1. Dữ liệu BTC

BTC gồm hai bài toán có dữ liệu liên quan với nhau nhưng cách sử dụng khác nhau: LegalIR dùng để tìm đúng các đoạn tài liệu làm căn cứ, còn LegalQA dùng để tạo câu trả lời có trích dẫn. Tất cả đường dẫn dưới đây đều được tính từ thư mục gốc của project.

#### `train.json` — dữ liệu mẫu có đáp án chuẩn

- **Nội dung:** Có một file cho mỗi bài toán: `data/raw/btc/LegalIR/train.json` và `data/raw/btc/LegalQA/train.json`. Đây là tập câu hỏi dùng để kiểm tra pipeline trong quá trình phát triển. Mỗi bản ghi có một mã câu hỏi và nội dung câu hỏi. Với LegalIR, trường `answer` là danh sách các `context_id` được BTC xác định là đoạn tài liệu liên quan. Với LegalQA, trường `answer` là câu trả lời tham chiếu; câu trả lời này thường được dùng cùng thông tin trích dẫn để đánh giá chất lượng hệ thống.
- **Tác dụng:** Đây là dữ liệu có ground truth, không phải dữ liệu để hệ thống trả lời khi nộp bài. Team dùng nó để benchmark retrieval, QA, citation và toàn bộ luồng end-to-end trước khi chạy trên `public_official.json`. Kết quả trên file này giúp phát hiện lỗi đọc dữ liệu, lỗi ánh xạ ID, thiếu context hoặc thay đổi mô hình làm điểm số giảm.
- **Người phụ trách:** TV4 tiếp nhận file, kiểm tra schema, chuẩn hóa encoding và tạo fixture/manifest để các thành viên dùng chung; TV2 dùng phần LegalIR để build và benchmark index; TV3 dùng phần LegalQA để phát triển prompt, sinh câu trả lời và citation; TV5 chạy benchmark, đối chiếu kết quả với scorer BTC và ghi nhận score report. TV1 bảo đảm pipeline có thể đọc đúng file mà không cần sửa tay.

#### `public_official.json` — danh sách câu hỏi chính thức để chạy inference

- **Nội dung:** Có một file cho mỗi bài toán: `data/raw/btc/LegalIR/public_official.json` và `data/raw/btc/LegalQA/public_official.json`. Cấu trúc gần giống `train.json`: mỗi câu hỏi có một ID và nội dung câu hỏi, nhưng không có đáp án công khai (`answer` là `null` hoặc không có giá trị). Vì vậy đây là danh sách yêu cầu mà hệ thống phải xử lý, chứ không phải bộ dữ liệu để học đáp án trực tiếp.
- **Tác dụng:** Đây là input inference để tạo file submission. Hệ thống phải xử lý đầy đủ mọi ID, giữ nguyên số lượng và thứ tự logic cần thiết, sau đó tạo output đúng schema BTC: LegalIR trả về các `context_id` được chọn; LegalQA trả về câu trả lời và phần trích dẫn theo quy ước của dự án. Không được dùng đáp án của `train.json` để điền thủ công vào file này.
- **Người phụ trách:** TV1 chịu trách nhiệm điều phối luồng từ input đến output, kiểm tra không mất hoặc trùng ID và đóng gói submission; TV2 thực hiện retrieval cho LegalIR; TV3 thực hiện answer generation và citation cho LegalQA; TV5 kiểm tra schema, giới hạn số context và các điều kiện có thể khiến submission bị loại. TV4 chỉ hỗ trợ xác nhận schema/manifest, không chịu trách nhiệm sinh prediction.

#### `selected-contexts.zip` — kho tài liệu làm căn cứ truy hồi

- **Nội dung:** Có một gói cho mỗi bài toán tại `data/raw/btc/LegalIR/selected-contexts.zip` và `data/raw/btc/LegalQA/selected-contexts.zip`. Bên trong là thư mục `selected-contexts/` với các file như `context_<id>.json`; mỗi file chứa nội dung một đoạn tài liệu cùng mã định danh của nó và các metadata đi kèm nếu BTC cung cấp. Có thể hiểu đây là “kho văn bản nguồn” để hệ thống tìm đoạn phù hợp với câu hỏi.
- **Tác dụng:** Đây là corpus đầu vào cho bước ingest, chunk/normalize và build BM25, FAISS hoặc Qdrant. Mã `context_id` trong kho phải được bảo toàn để kết quả retrieval có thể đối chiếu với `answer` của LegalIR và dùng làm căn cứ trích dẫn cho LegalQA. File ZIP không được đưa thẳng vào VectorDB; nó phải được giải nén, kiểm tra, lập manifest/hash và chuyển thành các `LegalChunk` hợp lệ trước khi index.
- **Người phụ trách:** TV4 là người trực tiếp giải nén, validate schema, loại/ghi nhận file lỗi, chuẩn hóa nội dung, ánh xạ `context_id` sang `LegalChunk.chunk_id` và bàn giao corpus đã kiểm kê. TV2 nhận corpus đã chuẩn hóa để build và version hóa các index. TV5 kiểm tra mapping bằng benchmark; TV1 chỉ sử dụng artifact đã được TV4 nghiệm thu và không tự sửa corpus trong lúc chạy submission.

#### `scoring.py` — chương trình chấm điểm của BTC

- **Nội dung:** Có hai scorer tương ứng với hai bài toán: `data/raw/btc/LegalIR/scoring.py` và `data/raw/btc/LegalQA/scoring.py`; thư mục phụ trợ như `data/raw/btc/LegalQA/rouge_score/` cũng phải được giữ nguyên nếu scorer yêu cầu. Đây là chương trình nhận dữ liệu tham chiếu và file kết quả của hệ thống, đọc các trường đáp án rồi tính điểm theo luật của BTC. LegalIR tập trung vào độ chính xác và độ bao phủ của các `context_id` (Precision/Recall); LegalQA dùng các thước đo chất lượng câu trả lời như ROUGE-L và METEOR, cùng các dependency NLP cần thiết.
- **Tác dụng:** Đây là barem chính để kiểm tra submission trước khi gửi. Team phải dùng nó để golden-test evaluator nội bộ, phát hiện khác biệt giữa cách tính điểm local và BTC, đồng thời kiểm tra các ràng buộc định dạng — đặc biệt LegalIR chỉ được trả tối đa 5 context; prediction vượt giới hạn có thể bị tính 0. Scorer không dùng để train model và cũng không thay thế bước retrieval/QA.
- **Người phụ trách:** TV5 là owner của scorer: thiết lập môi trường chạy, kiểm tra dependency, viết test đối chiếu và phát hành score report cho từng iteration. TV1 tích hợp scorer vào luồng validate/đóng gói submission và bảo đảm input/output đúng format. TV2 chịu trách nhiệm tối ưu retrieval theo điểm LegalIR; TV3 tối ưu câu trả lời, citation và điểm LegalQA. Không thành viên nào được tự ý thay đổi logic trong scorer BTC để làm đẹp kết quả.

### 1.2. Repository hiện tại

Repo đã có nền tảng: readers/cleaners/legal parser/parent-child chunking; `LegalChunk`; dense/BM25/hybrid; FAISS/Qdrant; reranker; QA/prompt/citation; evaluator/submission; FastAPI/orchestrator/cache; scripts và tests.

Còn thiếu khi đưa dữ liệu BTC vào:

1. Corpus BTC chưa ở `data/raw/btc`, chưa có manifest/hash/schema version.
2. ZIP chưa được ingest; index hiện tại không chứng minh được build từ corpus BTC.
3. Chưa có adapter chính thức cho JSON object của train/public và context JSON.
4. Chưa nghiệm thu mapping `context_id → LegalChunk.chunk_id`, `doc_id`, source, article/clause.
5. Phải build lại FAISS/Qdrant và BM25 từ JSONL BTC; phải version hóa index theo corpus/model hash.
6. Evaluator nội bộ phải golden-test với scorer BTC, đặc biệt top-5, ROUGE-L và METEOR.
7. Chưa xác nhận model runtime, dependency cache và Docker trên corpus BTC.

**Kết luận:** Repo không chạy ngay để tạo submission BTC hợp lệ. Unit tests/skeleton có thể chạy, nhưng inference bị chặn bởi corpus chưa ingest, index chưa build, mapping ID chưa nghiệm thu và evaluator chưa đối chiếu scorer BTC. Không cần train bắt buộc; ingest → index → benchmark trước, fine-tune chỉ mở khi baseline chứng minh cần.

## PHẦN 2 — Timeline và công việc bắt đầu ngay

### Day-1 Alignment — blocker chung cho cả 5 thành viên

Không thành viên nào bắt đầu feature riêng trước khi hoàn tất alignment sau:

- **Contract:** khóa `LegalChunk`, `LegalParent`, `RetrievalHit`, `QAResponse`, request/response API và submission schema. Mọi thay đổi contract phải có version/changelog và test tương thích.
- **Local data layout:** thống nhất `data/raw/btc/{legalir,legalqa}`, `data/processed/{chunks,parents,metadata}`, `data/vector_store/{faiss,qdrant,bm25}`, `data/submissions/` và `data/reports/`.
- **Environment:** thống nhất `.env.example`, model paths, `VECTOR_DB_TYPE`, index paths, Qwen3 endpoint/model, reranker model, cache URI, seed, timeout và `BTC_DATA_ROOT`. Secret không commit.
- **Mock data:** TV4 tạo bộ mock đúng schema gồm context files, IR train/public, QA train/public và vài record có citation. TV1, TV2, TV3, TV5 dùng cùng fixture này để chạy contract/integration test mà không chờ corpus thật.
- **Baseline command:** thống nhất một command có thể chạy từ clean checkout: mock ingest → mock index → retrieve → rerank → QA mock → validate submission.

**Artifact bắt buộc sau alignment:** `docs/data_contract.md`, `.env.example` cập nhật, `data/raw/btc/mock/`, fixtures trong `tests/fixtures/`, script baseline và biên bản schema approval do Leader xác nhận.

### Timeline 06/08/2026 — 18/09/2026

| Giai đoạn | Mục tiêu | Việc chính | Output bắt buộc |
|---|---|---|---|
| Phase 1 — Ngày 1–3 | Barebone baseline | Cả team chạy mock end-to-end; TV4 import một subset BTC; TV2 build index tối thiểu; TV3 dùng MockLLM/Qwen3 nhẹ; TV5 validate; TV1 đóng gói pipeline. | Submission JSON đầu tiên, log runtime, validation report, commit/tag baseline. Leader nộp thử ngay khi file hợp lệ. |
| Phase 2 — Các tuần tiếp theo | Iterative optimization | TV4 tăng chất lượng corpus/chunk; TV2 so sánh BM25/dense/hybrid; TV3 tune prompt/Qwen3/citation; TV5 rerank và scorer; TV1 ổn định API/cache. | Mỗi iteration có config, corpus/index/model hash, score report và quyết định giữ/bỏ. Submit tối đa 10 lần/ngày theo quota Codabench. |
| Phase 3 — Tuần cuối | Freeze & polish | Đóng băng schema, code path, model, index và prompt; chỉ sửa bug, latency, format, packaging và rollback. | Release candidate, backup checkpoint/index, checksum, final validation, submission archive. |

### Vòng lặp mỗi iteration

`Build artifact → chạy local smoke → validate schema → chạy scorer BTC/local metrics → Leader submit → ghi log điểm/quota → chọn một thay đổi có kiểm soát → rebuild và so sánh`. Không thay đổi đồng thời data, model, prompt và reranker vì sẽ không truy được nguyên nhân điểm tăng/giảm.

## PHẦN 3 — Phân công chi tiết

### TV1 — Integration & Orchestration

**Khu vực tác nghiệp (Workspace):** `src/udsc2026/api/app.py`, `orchestrator.py`, `cache.py`, `contracts/api.py`, `configs/`, scripts inference/submission.

**Thiếu:** Luồng public JSON → retrieval/rerank/QA → BTC output; DI backend thật; cache key/version theo index; readiness check; error mapping.

**Thực hiện:** Thêm orchestrator cho LegalIR/LegalQA; inject retriever/reranker/QA/cache qua factory; cấu hình path bằng env; reject index thiếu/stale; timeouts/concurrency; writer giữ đủ public IDs và đúng schema.

**Kết quả (Đầu ra):** API pipeline; `predictions/legalir.json`, `predictions/legalqa.json`; health/readiness report.

**Tiêu chí nghiệm thu (Definition of Done):** Chạy từ public input đến submission không sửa tay; mọi ID đúng một lần; IR có 1–5 context ID; QA answer là string; DI/cache/readiness tests pass.

### TV2 — Full Retrieval

**Khu vực tác nghiệp (Workspace):** `src/udsc2026/retrieval/`, `src/udsc2026/infrastructure/vector_db/`, embedding, `scripts/data_prep/index_chunks.py`, `configs/`, `data/vector_store/`.

**Thiếu:** Index thật từ BTC; kiểm soát dimension/model version; dense/BM25/hybrid baseline; payload mapping context ID.

**Thực hiện:** Nhận `LegalChunk` JSONL; build FAISS/Qdrant và BM25; kiểm tra duplicate/missing ID; lưu metadata; benchmark top-k; trả `list[RetrievalHit]` với score/rank/citation; version hóa index.

**Kết quả (Đầu ra):** `chunks/*.jsonl`, FAISS `index.faiss` + `payloads.json`, BM25 index, manifest, `RetrievalHit`.

**Tiêu chí nghiệm thu (Definition of Done):** Build reproducible; index count đúng; mọi hit ánh xạ được chunk; không dimension mismatch; retrieval tests và LegalIR smoke test pass; submission không quá top-5.

### TV3 — QA & LLM

**Khu vực tác nghiệp (Workspace):** `src/udsc2026/qa/`, `src/udsc2026/infrastructure/llm/`, `contracts/qa.py`, `prompts/`.

**Thiếu:** Qwen3 runtime/config; prompt BTC; citation validation; fallback/error policy.

**Thực hiện:** Kết nối Qwen3; prompt version có question/context/citation rules; giới hạn token; parse citation về `RetrievalHit`; đánh dấu unverified; retry/timeout; deterministic decoding; ablation prompt/model.

**Kết quả (Đầu ra):** `QAResponse`, prompt templates, citation report, LegalQA prediction và metric log.

**Tiêu chí nghiệm thu (Definition of Done):** Mọi public ID có answer; không exception/None; citation mapping đúng hoặc warning; QA tests pass; scorer BTC chạy được và baseline lưu lại.

### TV4 — Advanced Data Engineer

**Khu vực tác nghiệp (Workspace):** `src/udsc2026/ingestion/`, `contracts/chunk.py`, `data/raw/btc/`, `data/processed/{chunks,parents,metadata}/`, synthetic generator.

**Thiếu:** BTC extractor; giải nén/manifest/hash; context parser; ID mapping; validation và benchmark không rò rỉ.

**Thực hiện:** Import command nhận dataset root; giải nén an toàn; đọc/validate JSON; normalize Unicode; parse legal structure; parent-child chunk; giữ `context_id`; ghi JSONL atomic; sinh manifest, error/duplicate/orphan report; synthetic benchmark.

**Kết quả (Đầu ra):** Corpus chuẩn hóa; chunks/parents JSONL; `manifest.json`; validation report; synthetic benchmark.

**Tiêu chí nghiệm thu (Definition of Done):** Import idempotent; lỗi không silently drop; `LegalChunk` pass; không duplicate chunk; ground-truth IDs tồn tại; không orphan; manifest có hash/schema version.

### TV5 — Reranking & MLOps

**Khu vực tác nghiệp (Workspace):** `src/udsc2026/retrieval/reranking/`, reranker infrastructure, `src/udsc2026/evaluation/`, `scripts/evaluate*.py`, validators, `docker/`, compose.

**Thiếu:** Cross-encoder trên BTC; candidate/final-k tuning; scorer adapter; Dockerized evaluation; comparison report.

**Thực hiện:** Rerank candidate TV2; tune k; đo LegalIR Precision/Recall/MRR nội bộ và LegalQA ROUGE-L/METEOR/citation; chạy scorer BTC isolated; validate output; đóng gói dependency/model; lưu config/score.

**Kết quả (Đầu ra):** Reranked `RetrievalHit`; evaluation reports; validators; Docker image/compose profile; baseline-vs-rerank report.

**Tiêu chí nghiệm thu (Definition of Done):** Không mất ID/citation; top-5 hợp lệ; scorer reproducible; report có input hash/config/model version; Docker clean start; quality gate pass.

## PHẦN 4 — Trình tự bàn giao và điều kiện chốt bản build thi đấu

### Trình tự bàn giao

1. **TV4 → TV2/TV5/TV3:** đọc BTC files, giải nén context, clean/parse/chunk; giao `chunks/*.jsonl`, `parents/*.jsonl`, manifest và validation report. TV2 chỉ nhận artifact có schema pass.
2. **TV2 → TV5/TV3/TV1:** build FAISS/Qdrant/BM25 từ chunks; giao index, payload mapping, manifest và `list[RetrievalHit]` contract. TV1 không gọi backend trực tiếp.
3. **TV5 → TV3/TV1:** rerank candidate và giao ordered `RetrievalHit` có `rerank_score/final_score`, cùng benchmark report. Không được làm mất `chunk_id`, source hoặc citation metadata.
4. **TV3 → TV1/TV5:** sinh `QAResponse`, answer, citations, warnings và prompt/model version; giao QA prediction cùng metric report.
5. **TV1 → TV5/Leader:** chạy orchestration trên toàn bộ public input, cache/readiness, ghi `legalir.json`, `legalqa.json` và runtime log.
6. **TV5 → Leader:** validate exact schema, chạy scorer BTC, đóng gói `submission.zip`, kèm checksum, score report, config và rollback manifest.
7. **Leader:** kiểm tra tên file/đường dẫn cuối, ghi quota và timestamp, nộp Codabench; lưu submission ID và điểm vào experiment log.

### Điều kiện đóng băng và handover

- **Điều kiện 1 — Format:** output `.json` pass 100% validator do TV5 duy trì; đủ ID, đúng key/schema, LegalIR tối đa 5 context ID, LegalQA answer là string.
- **Điều kiện 2 — Runtime:** pipeline end-to-end local chạy từ input sạch, không sửa tay, không crash; runtime và memory nằm trong giới hạn triển khai đã cấu hình.
- **Điều kiện 3 — Reproducibility:** có corpus hash, commit, prompt version, model version, index manifest, random seed và scorer report.
- **Điều kiện 4 — Rollback:** có backup `index.faiss`/payload hoặc Qdrant snapshot, BM25 index và model checkpoint/config; backup được test load trước khi freeze.
- **Điều kiện 5 — Release:** quality gate, unit/integration/smoke tests và BTC scorer pass; Leader duyệt release candidate.

**Leader action:** TV1/TV5 chỉ giao artifact đã đóng gói. Leader là người duy nhất cầm `submission.zip`, kiểm tra lần cuối và submit Codabench. Kết quả submit không được ghi đè artifact trước đó; mỗi lần submit có thư mục/version riêng.

## PHẦN 5 — Ma trận phụ thuộc và critical path

| Đầu ra | Người tạo | Người dùng |
|---|---|---|
| Manifest, chunks/parents JSONL | TV4 | TV2, TV5, TV3 |
| FAISS/Qdrant, BM25, `RetrievalHit` | TV2 | TV5, TV3, TV1 |
| Reranked hits và score report | TV5 | TV3, TV1 |
| `QAResponse`, prompt/citation report | TV3 | TV1, TV5 |
| API, cache, submission files | TV1 | BTC scorer/frontend |
| Scorer adapter, Docker, quality gate | TV5 | Tất cả |

### Blockers và thứ tự ưu tiên

1. **TV4/P0:** giải nén BTC, khóa schema, tạo manifest và `LegalChunk/LegalParent`; không index khi validation còn duplicate/orphan.
2. **TV2/P0:** build FAISS/Qdrant/BM25 từ corpus TV4; xác nhận context ID truy hồi được.
3. **TV5/P0:** nối cross-encoder, tạo baseline/rerank comparison và khóa scorer adapter.
4. **TV3/P0:** chạy Qwen3 trên hits đã rerank; khóa prompt, citation parser và QA baseline.
5. **TV1/P0:** tích hợp qua DI/orchestrator, bật cache, sinh public submissions/readiness.
6. **TV5 + TV1/P0:** Docker end-to-end cả hai task; exact-schema validation; chạy scorer BTC; chỉ phát hành khi quality gate pass.

TV1 có thể dựng API skeleton song song; TV3 dùng MockLLM; TV5 dựng scorer adapter bằng synthetic data. Đây chưa phải integrated baseline cho tới khi data và index gates hoàn tất.

### Gate phát hành

`Data gate (TV4) → Index gate (TV2) → Retrieval/rerank gate (TV5) → QA gate (TV3) → API/submission gate (TV1) → BTC scorer gate (TV5)`.

Mỗi gate phải lưu commit, config, model version, corpus hash và score. Gate fail thì dừng tuning/phát hành.
