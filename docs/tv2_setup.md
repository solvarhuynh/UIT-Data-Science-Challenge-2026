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
`numpy==1.26.4`, `scipy<1.14`, `faiss-cpu==1.8.0`.

## 5. Cài đặt và chạy lại toàn bộ phần TV2

Từ thư mục gốc repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements_dev.txt
$env:PYTHONPATH = "$PWD\src"
```

Kiểm tra model local:

```powershell
python scripts/verify_embedding.py
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
