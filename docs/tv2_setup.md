# TV2 — Retrieval Pipeline: trạng thái, setup và vận hành

Tài liệu này mô tả những gì TV2 đã hoàn thành, cách chạy lại và các điểm tích hợp
với TV4/TV5. TV2 chỉ phụ trách retrieval; không chứa API route, frontend, QA generation
hay cross-encoder reranking.

## 1. TV2 đã hoàn thành gì?

| Nhóm | File chính | Tác dụng |
|---|---|---|
| Contract | `src/udsc2026/contracts/chunk.py` | Định nghĩa `LegalChunk`, input chung từ ingestion TV4 vào indexing TV2. |
| Embedding | `src/udsc2026/infrastructure/embedding/bkai_client.py` | Load BKAI bi-encoder từ local path, encode query hoặc batch document thành vector float đã normalize. |
| Embedding config | `src/udsc2026/infrastructure/embedding/config.py` | Compatibility wrapper cho config embedding cũ. |
| Config chung | `src/udsc2026/infrastructure/config.py` | Deep-merge `base.yaml` và `development.yaml`, tránh mỗi module tự đọc YAML. |
| VectorDB interface | `infrastructure/vector_db/base.py` | Interface chung, payload mapping và chuyển payload thành `RetrievalHit`. |
| Qdrant | `qdrant_adapter.py` | Lưu/search vector qua Qdrant, giữ payload citation. |
| FAISS | `faiss_adapter.py` | Lưu vector `IndexFlatIP` local, kèm JSON side-store metadata. |
| Factory | `vector_db/factory.py` | Chọn Qdrant hoặc FAISS từ config. |
| Dense | `retrieval/dense/dense_retriever.py` | Encode query rồi gọi VectorDB, gán `dense_score` và rank. |
| Sparse | `retrieval/sparse/tokenizer.py`, `bm25_retriever.py` | Tokenize tiếng Việt, giữ dấu/cụm Điều-Khoản-Điểm, build/search/save BM25 độc lập. |
| Hybrid | `retrieval/hybrid/score_fusion.py`, `hybrid_retriever.py` | Normalize, merge theo `chunk_id`, weighted fusion và giữ citation. |
| Index script | `scripts/index_chunks.py` | Orchestrate JSONL → embedding → VectorDB → BM25. |
| Tests | `tests/retrieval/` | Mock model/Qdrant, test FAISS thật, BM25, Dense, Hybrid và fusion. |

Kết quả verify gần nhất: `16 passed`. Index thực tế đã xử lý 23 chunk, embed 23/23,
upsert 23/23 và không skip lỗi schema.

## 2. TV2 có cần

TV2 hiện dùng BKAI bi-encoder đã pretrained và chỉ làm inference để tạo embedding.
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
  model_path: ./models/bkai-bi-encoder
  device: cpu
  batch_size: 32
  max_length: 256
  normalize_embeddings: true
```

FAISS dùng NumPy 1.x trong môi trường hiện tại, nên `requirements_dev.txt` pin:
`numpy==1.26.4`, `scipy<1.1Từ thư mục gốc repository, dùng Python 3.10-3.12 (chưa hỗ trợ Python 3.13+):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements_dev.txt
python -m pip install --upgrade pip -c requirements_runtime.txt
python -m pip install -c requirements_runtime.txt -e .
python -m pip install -c requirements_runtime.txt -r requirements_dev.txt
```

Không commit `.venv/`. Nếu chưa dùng editable install, đặt tạm package path:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Kiểm tra model local:

```powershell
python scripts/verify_embedding.py
python -m pip install -c requirements_runtime.txt sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements_runtime.txt qdrant-client faiss-cpu
python -m pip install -c requirements_runtime.txt rank-bm25 pyvi
python -m pip install -c requirements_runtime.txt pyyaml
```

Index dữ liệu TV4 hoặc sample dev:

```powershell
python scripts/index_chunks.py `
  --chunks-dir data/processed/chunks `
  --vector-db-type faiss
```

Script tạo:

- `data/vector_store/faiss/legal_chunks/index.faiss`
- `data/vector_store/faiss/legal_chunks/payloads.json`
- `data/vector_store/bm25/legal_chunks.pkl`
- `data/processed/metadata/index_errors.json`

Chạy test:

```powershell
python -m pytest tests/retrieval/ -v --basetemp "$PWD\.pytest_tmp"

| Tasks | Phạm vi | Verify sau khi hoàn tất |
|---|---|---|
| 0 | `LegalChunk` contract | `python -c "from udsc2026.contracts import LegalChunk, RetrievalHit; print('ok')"` |
| 1 | `EmbeddingClient` BKAI local | `python -c "from udsc2026.infrastructure.embedding.bkai_client import EmbeddingClient; print('ok')"` |
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

## 3. Cấu trúc và trách nhiệm

- `src/udsc2026/contracts/`: `LegalChunk` từ TV4 và `RetrievalHit` dùng chung.
- `src/udsc2026/infrastructure/embedding/`: load BKAI bi-encoder local và encode query/chunks.
- `src/udsc2026/infrastructure/vector_db/`: interface backend-neutral, Qdrant và FAISS.
- `src/udsc2026/retrieval/dense/`: query embedding và dense search.
- `src/udsc2026/retrieval/sparse/`: tokenizer tiếng Việt và BM25 độc lập.
- `src/udsc2026/retrieval/hybrid/`: normalize, merge và weighted score fusion; không rerank.
- `configs/base.yaml`: model path, vector DB và hybrid settings.
- `data/vector_store/`: FAISS/BM25 index local; không commit dữ liệu index lớn.

TV2 cung cấp các API `search(query, top_k, filters=None)` cho dense, sparse và hybrid.
TV1 có thể gọi hybrid để lấy context; TV5 nhận `list[RetrievalHit]` để rerank.

## Tóm tắt vị trí file

# Cấu trúc các file tv2 đã làm việc

udsc2026/
├── pyproject.toml              # ĐÃ SỬA: thêm [build-system] + [project]
├── requirements_dev.txt        # SẼ ĐƯỢC AGENT BỔ SUNG dần theo bảng trên (Prompt 1/2/4/7)
├── .venv/                      # môi trường ảo, KHÔNG commit vào git
├── configs/
│   ├── base.yaml                # SẼ CÓ THÊM section embedding/vector_db/sparse/hybrid
│   └── development.yaml
├── data/
│   ├── processed/chunks/        # sample_dev.jsonl do Prompt 6 tạo (dev only, xoá khi TV4 có data thật)
│   └── vector_store/            # output của Prompt 6: faiss/, bm25/ (qdrant chạy ngoài Docker)
├── scripts/
│   └── index_chunks.py          # Prompt 6
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

## 4. Quy tắc tích hợp

`LegalChunk` phải khớp JSONL output của TV4. ## 4. Quy tắc tích hợp

`LegalChunk` phải khớp JSONL output của TV4. Nếu TV4 đổi schema, cập nhật contract sau khi
thống nhất với TV4. Không định nghĩa lại `RetrievalHit` trong retriever.

`final_score` của TV2 là điểm hybrid dùng để xếp candidate; TV5 có thể bổ sung
`rerank_score` theo pipeline của mình. Không đưa cross-encoder, QA generation, API route hoặc
frontend vào TV2.

## 5. Chạy thử sau khi có index

```powershell
python scripts/index_chunks.py --vector-db-type faiss --chunks-dir data/processed/chunks
python -c "from udsc2026.retrieval.hybrid import search; print(search('Điều 10 Bộ luật Lao động quy định gì?', 5))"
python -m pytest tests/unit/test_retrieval -v
/ -v --basetemp "$PWD\.pytest_tmp"
```

Nếu dùng Qdrant, khởi động Qdrant local rồi chạy:

```powershell
python scripts/index_chunks.py --chunks-dir data/processed/chunks --vector-db-type qdrant
```

Các API module-level cho truy vấn nhanh:

```python
from udsc2026.retrieval.dense import search as dense_search
from udsc2026.retrieval.sparse import search as sparse_search
from udsc2026.retrieval.hybrid import search as hybrid_search

dense_hits = dense_search("Điều 10 Bộ luật Lao động quy định gì?", 5)
sparse_hits = sparse_search("Điều 10 Bộ luật Lao động quy định gì?", 5)
hybrid_hits = hybrid_search("Điều 10 Bộ luật Lao động quy định gì?", 5)
```

## 6. Phạm vi TV2 và các thành viên khác

- TV1 gọi retrieval qua contract, không cần biết Qdrant/FAISS chi tiết.
- TV4 cung cấp chunk JSONL và metadata pháp luật.
- TV5 nhận candidate `RetrievalHit` để rerank/evaluate; TV5 cũng phụ trách Web UI/UX
  theo phân công hiện tại.
- TV3 phụ trách QA/LLM/prompt/citation generation.

TV2 không train model trong bước indexing và không phụ trách frontend, API route hoặc QA.

## 7. Git workflow và nhóm commit đề xuất

Không commit thẳng vào `main`. Các lệnh dưới đây chỉ là nhóm lệnh để chạy trên nhánh
`tv2`; kiểm tra `git status` trước mỗi nhóm và không add các file index/cache sinh tự động.

### Nhóm 1 — dependency, config và embedding

```powershell
git switch tv2
git add requirements_dev.txt configs/base.yaml configs/development.yaml `
  src/udsc2026/infrastructure/config.py `
  src/udsc2026/infrastructure/embedding/bkai_client.py `
  src/udsc2026/infrastructure/embedding/config.py scripts/verify_embedding.py
git commit -m "chore(tv2): align retrieval dependencies and configuration"
git pull origin tv2 --rebase
git push origin tv2
```

### Nhóm 2 — VectorDB, retrieval và indexing

```powershell
git add src/udsc2026/infrastructure/vector_db `
  src/udsc2026/retrieval scripts/index_chunks.py
git commit -m "feat(tv2): complete dense sparse hybrid retrieval pipeline"
git pull origin tv2 --rebase
git push origin tv2
```

### Nhóm 3 — tests và contract

```powershell
git add src/udsc2026/contracts/chunk.py tests/retrieval
git commit -m "test(tv2): cover retrieval contracts and adapters"
git pull origin tv2 --rebase
git push origin tv2
```

### Nhóm 4 — tài liệu và phân công

```powershell
git add docs/tv2_setup.md README.md
git commit -m "docs(tv2): document retrieval setup and team scope"
git pull origin tv2 --rebase
git push origin tv2
```

Không add `.pytest_tmp*`, `.pytest_cache`, `data/vector_store/` hoặc model weights vào
commit. Nếu các file này hiện trong VS Code Changes, bỏ qua hoặc thêm vào `.gitignore`.
