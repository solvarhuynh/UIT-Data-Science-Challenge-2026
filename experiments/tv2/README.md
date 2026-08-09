# TV2 retrieval experiments

Thư mục này dành cho smoke test và thử nghiệm retrieval của TV2. Mã production
nằm trong `src/udsc2026/retrieval/`, `src/udsc2026/infrastructure/embedding/` và
`scripts/data_prep/index_chunks.py`.

## Đầu vào chính thức

- Chunks: `data/processed_v3/chunks/*.jsonl`.
- Benchmark: `data/processed_v3/benchmarks/synthetic_qa.jsonl`.
- Embedding model: `models/dek21-v2`.
- Vector store local: `data/vector_store/`.
- Artifact/log thử nghiệm: `artifacts/tv2/`.

Không dùng `data/processed` hoặc model path cũ
`models/hcmute-embedding-v2`. Không ghi report vận hành vào
`data/processed_v3/metadata`, vì thao tác đó sẽ làm thay đổi tree hash của corpus
đã audit.

## Dense FAISS smoke

Từ root repository:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts/data_prep/index_chunks.py `
  --chunks-dir data/processed_v3/chunks `
  --config-env development `
  --vector-db-type faiss `
  --max-chunks 10000 `
  --batch-size 32 `
  --skip-bm25 `
  --error-report artifacts/tv2/smoke_index_errors.json `
  --force
```

Lệnh này giới hạn 10.000 chunks để kiểm schema, embedding, batch và FAISS trước
khi chạy full corpus. Manifest nằm cùng collection trong
`data/vector_store/faiss/`; lỗi record nằm ngoài corpus tại `artifacts/tv2/`.

## Candidate benchmark

```powershell
python scripts/evaluation/generate_dense_candidates.py `
  --benchmark data/processed_v3/benchmarks/synthetic_qa.jsonl `
  --config-env development `
  --candidate-k 50 `
  --limit 10 `
  --benchmark-subset artifacts/tv2/smoke_benchmark.jsonl `
  --output artifacts/tv2/smoke_dense_predictions.jsonl `
  --manifest artifacts/tv2/smoke_dense_manifest.json
```

Kết quả phải giữ nguyên `chunk_id`, `parent_id`, `doc_id`, text và citation
metadata. Cache/index chỉ được tái sử dụng khi corpus hash, model identity,
collection và `max_chunks` đều khớp manifest.

## BM25 nhỏ

Full BM25 bằng `rank_bm25` không phù hợp với RAM 28 GB. Chỉ chạy corpus giới hạn
để kiểm thử:

```powershell
python scripts/data_prep/index_chunks.py `
  --chunks-dir data/processed_v3/chunks `
  --config-env development `
  --vector-db-type faiss `
  --max-chunks 5000 `
  --bm25-index-path data/vector_store/bm25/tv2_smoke.json `
  --error-report artifacts/tv2/bm25_smoke_index_errors.json `
  --force
```

BM25 được lưu bằng JSON có version, không dùng pickle. Với full RTX pipeline,
thực hiện theo [TV5 GPU Runbook](../../docs/members/tv5/tv5_gpu_runbook.md):
dense FAISS top-50 rồi chuyển candidate cho TV5 rerank top-5.

