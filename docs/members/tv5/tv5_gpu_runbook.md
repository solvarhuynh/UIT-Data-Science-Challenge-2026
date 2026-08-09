# TV5 — chạy retrieval + reranker trên RTX 5060 Ti 16 GB

Máy mục tiêu: Windows, i7-12700K, RAM 28 GB, RTX 5060 Ti 16 GB. Pipeline dùng:

- embedding: `huyydangg/DEk21_hcmute_embedding_v2`;
- reranker: `BAAI/bge-reranker-v2-m3`;
- LLM: `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2`.

## 1. Bộ dữ liệu chuẩn

Chỉ dùng `data/processed_v3`. Không index `data/processed`, `processed_v2` hay thư
mục `processed_v3_failed_*`.

Số liệu V3 đã audit ngày 09/08/2026:

| Hạng mục | Giá trị |
| --- | ---: |
| Documents | 8.532 |
| Chunks | 1.275.250 |
| Parents | 185.339 |
| Dung lượng tổng | khoảng 2,70 GiB |
| Missing token/bigram trong child | 0 / 0 |
| Duplicate/orphan/invalid/empty chunk | 0 |
| Synthetic benchmark | 100 câu, đủ 7 nhóm |

Các hash chuẩn nằm trong `metadata/processing_manifest.json`:

```text
source_corpus_hash: cda01fcb55da1e5656f1190eb35f07d77314ed72220d13aa6a1f421737d202ed
processed_corpus_tree_hash: 5420eb3f9e26b018351a83faf7a5139fa790d2e2c05d006578cb2192bcd52765
synthetic_benchmark.benchmark_sha256: 01d59551a69f6e249e6dc0644ff9b541c1558c426e33e72dad690cb6d20a7fd0
synthetic_benchmark.source_chunk_corpus_sha256: 46ab8928bf548b16d167464f70da6db9bfd722d07193d9d3f911bf57fa583180
```

V3 có `integrity_gate_passed=true`. `semantic_completeness_gate_passed=false`
chỉ vì BTC cung cấp 20 context có `passage` rỗng; 6 context nằm trong gold
LegalIR và 9 câu có toàn bộ gold rỗng. Không tự bịa nội dung cho các context này.

## 2. Chuẩn bị máy thuê

```powershell
git clone https://github.com/solvarhuynh/UIT-Data-Science-Challenge-2026.git
cd UIT-Data-Science-Challenge-2026
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

Đảm bảo ổ chứa repo còn ít nhất 40 GiB trống; preflight sẽ dừng nếu thấp hơn.

Dữ liệu lớn không đi theo Git. Copy nguyên thư mục sau từ máy local/Drive:

```text
data/processed_v3/documents/
data/processed_v3/chunks/
data/processed_v3/parents/
data/processed_v3/benchmarks/
data/processed_v3/metadata/
```

Không cần copy cả hai cây raw `selected-contexts`: LegalIR và LegalQA là 8.532
cặp byte-identical và V3 đã deduplicate có audit.

## 3. Chạy tự động

Lần đầu, để script cài CUDA dependencies, tải ba model, chạy preflight, smoke và
full pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/run_gpu_pipeline.ps1 `
  -ProcessedRoot data/processed_v3
```

Nếu dependencies và model đã có sẵn:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/run_gpu_pipeline.ps1 `
  -ProcessedRoot data/processed_v3 `
  -SkipInstall `
  -SkipDownload
```

Muốn ghi log để theo dõi:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/run_gpu_pipeline.ps1 `
  -ProcessedRoot data/processed_v3 2>&1 | `
  Tee-Object -FilePath outputs/tv5_gpu_pipeline.log
```

Script sẽ lần lượt:

1. cài PyTorch CUDA 12.8 và dependencies GPU;
2. tải ba model, ghi download manifest;
3. preflight CUDA/model/disk và xác minh V3 manifest, audit, counts, benchmark SHA;
4. smoke LLM, index 10.000 chunks và rerank 10 câu;
5. build dense FAISS cho toàn bộ 1.275.250 chunks;
6. sinh dense top-50 cho 100 câu benchmark;
7. rerank bằng BGE, giữ top-5 và xuất báo cáo before/after.

## 4. Kết quả phải kiểm tra

File chính:

```text
artifacts/tv5/bge-reranker-v2-m3/evaluation/comparison.md
```

File truy vết:

```text
models/download_manifest.json
data/processed_v3/metadata/processing_manifest.json
data/processed_v3/metadata/disk_audit_report.json
artifacts/tv2/dense_run_manifest.json
artifacts/tv5/bge-reranker-v2-m3/run_manifest.json
```

Chỉ nhận kết quả khi:

- preflight không có `failures`, riêng data phải có `ok=true`;
- data integrity pass và benchmark SHA khớp manifest;
- dense manifest ghi đủ 1.275.250 chunks;
- đủ 100 câu trong dense predictions;
- reranker chỉ sắp xếp/cắt top-5, không sửa text/citation/metadata;
- Recall/MRR before và after dùng cùng benchmark, cùng candidate pool.

## 5. Thời gian và xử lý lỗi

Với RTX 5060 Ti 16 GB, nên dành khoảng **2–4 giờ** cho lần đầu gồm tải model,
smoke, full embedding, candidate generation và reranking. Tải model/mạng và tốc
độ SSD có thể làm thay đổi đáng kể.

- CUDA false: kiểm tra PyTorch CUDA 12.8 rồi chạy lại `scripts/gpu/preflight.py`.
- OOM embedding: hạ `embedding.batch_size` trong `configs/gpu.yaml` từ 128 xuống 64.
- OOM reranker: đổi cả hai `--batch-size "8"` thành `"4"` trong
  `scripts/gpu/run_gpu_pipeline.ps1`; giữ `max_length=1024` nếu còn đủ VRAM.
- Data preflight fail: pipeline sẽ dừng và không có cờ bỏ qua; copy lại trọn
  `data/processed_v3`, rồi kiểm tra manifest/hash trước khi chạy lại.
- Không build BM25 full bằng `rank_bm25` trên RAM 28 GB; pipeline hiện dùng dense
  top-50 → reranker.
- Parent context đã bật; QA mở parent sau rerank theo cửa sổ/budget, không nạp mù
  toàn bộ parent rất dài vào prompt.
