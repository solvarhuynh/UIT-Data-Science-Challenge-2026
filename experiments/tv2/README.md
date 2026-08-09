# TV2 retrieval experiment

Đây là smoke test thủ công cho pipeline retrieval của TV2; logic ổn định nằm trong `src/` và `scripts/`.

## Đầu vào

- JSONL tại `data/processed/chunks/*.jsonl`.
- Mỗi dòng phải là một `LegalChunk` theo schema TV4/TV2: `chunk_id`, `doc_id`, `text` và metadata citation.
- Model HCMUTE Embedding v2 local tại `models/hcmute-embedding-v2/` nếu chạy dense/hybrid.

## Chạy

Từ thư mục gốc repository:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/index_chunks.py --chunks-dir data/processed/chunks --vector-db-type faiss
```

Lệnh trên đọc JSONL, validate `LegalChunk`, tạo embedding theo batch, ghi FAISS vào
`data/vector_store/faiss/` và BM25 vào `data/vector_store/bm25/`. Dòng JSONL lỗi được ghi
ở `data/processed/metadata/index_errors.json` và không làm dừng toàn bộ indexing.

Sau khi index, có thể smoke-test BM25 bằng module Python:

```powershell
python -c "from udsc2026.retrieval.sparse.bm25_retriever import BM25Retriever; b=BM25Retriever('data/vector_store/bm25/legal_chunks.pkl'); b.load('data/vector_store/bm25/legal_chunks.pkl'); print(b.search('Điều 10 hợp đồng lao động', 3))"
```

## Đầu ra cần kiểm tra

- `data/vector_store/faiss/`: index vector và side-store metadata.
- `data/vector_store/bm25/legal_chunks.pkl`: sparse index và danh sách `LegalChunk`.
- Kết quả truy vấn là `list[RetrievalHit]`, có score và citation metadata; `text` phải giữ nguyên dấu tiếng Việt.

Đây là bước kiểm tra thủ công khi có dữ liệu mới, không phải unit test chính thức. Unit test
chạy bằng một lệnh chung ở README gốc.

