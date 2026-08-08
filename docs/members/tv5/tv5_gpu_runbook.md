# TV5 — chạy retrieval + reranker trên RTX 5060 Ti 16 GB

Máy mục tiêu: Windows, Intel i7-12700K, RAM 28 GB, RTX 5060 Ti 16 GB và
khoảng 400 GB SSD còn trống. Cấu hình này đủ cho embedding 0.1B, reranker
0.6B và Qwen3 Legal khoảng 2B ở BF16/FP16, nhưng không nên giữ BM25 của gần
1 triệu chunks hoàn toàn trong RAM.

## 1. Chuẩn bị repo và dữ liệu

```powershell
git clone https://github.com/solvarhuynh/UIT-Data-Science-Challenge-2026.git
cd UIT-Data-Science-Challenge-2026
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

`data/processed` không đi theo Git. Sau khi clone, tải/copy bộ processed của
nhóm vào đúng vị trí sau:

```text
data/processed/chunks/*.jsonl
data/processed/parents/*.jsonl
data/processed/benchmarks/synthetic_qa.jsonl
data/processed/metadata/*.json
```

Không cần copy thêm bản `LegalQA/selected-contexts` nếu nó trùng với LegalIR.

## 2. Chạy tự động

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/run_gpu_pipeline.ps1
```

Script sẽ:

1. cài PyTorch CUDA 12.8 cho RTX 50-series và nhóm dependency `gpu`;
2. tải và ghi SHA của ba model vào `models/download_manifest.json`;
3. kiểm tra CUDA, VRAM, dung lượng, packages, model và dữ liệu;
4. load Qwen3 và sinh thử một câu ngắn bằng native chat template;
5. khôi phục fallback chunks cho các tài liệu mà parser cũ đã để file rỗng;
6. build thử 10.000 chunks và rerank 10 câu;
7. thay smoke index bằng dense index đầy đủ;
8. sinh 50 candidates/câu;
9. chạy `BAAI/bge-reranker-v2-m3`, giữ top 5 và xuất báo cáo before/after.

Nếu dependency và model đã có sẵn:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/gpu/run_gpu_pipeline.ps1 `
  -SkipInstall `
  -SkipDownload
```

Nếu hôm đó chỉ benchmark retrieval/reranker và chưa muốn load LLM, thêm
`-SkipLLMSmoke`; model LLM vẫn được tải mặc định để cả bộ checkpoint có manifest.

## 3. Kết quả phải kiểm tra

File quan trọng nhất:

```text
artifacts/tv5/bge-reranker-v2-m3/evaluation/comparison.md
```

Các file truy vết:

```text
models/download_manifest.json
data/vector_store/faiss/legal_chunks_dek21_v2_768/manifest.json
artifacts/tv2/dense_run_manifest.json
artifacts/tv5/bge-reranker-v2-m3/run_manifest.json
```

Chỉ chấp nhận kết quả full khi:

- preflight không có `failures`;
- `chunk_count` trong dense manifest khớp số chunks đầu vào hợp lệ;
- `empty_chunk_repair.json` không có `invalid_documents`;
- đủ 100 câu trong `dense_predictions.jsonl`;
- reranker không tạo/mất candidate ngoài việc cắt top 5;
- Recall/MRR sau rerank được so trên cùng benchmark và cùng candidate pool.

## 4. Ước lượng thời gian

Với dữ liệu processed hiện tại khoảng 20 GB và gần 1 triệu chunks:

- cài môi trường + tải ba model: khoảng 20–60 phút, phụ thuộc mạng;
- recovery dữ liệu + smoke pipeline: khoảng 10–25 phút;
- full embedding + FAISS persist: khoảng 60–150 phút;
- sinh candidates 100 câu: khoảng 1–5 phút;
- rerank khoảng 5.000 cặp: khoảng 2–15 phút.

Tổng thực tế nên dành khoảng **2–4 giờ**; mốc lập kế hoạch hợp lý là khoảng
**3 giờ**. Đây là ước lượng; tốc độ PyVi,
SSD, phiên bản PyTorch hỗ trợ RTX 50-series và mạng tải model có thể làm thay
đổi thời gian.

## 5. Xử lý lỗi nhanh

- `torch.cuda.is_available() == false`: cài lại PyTorch CUDA tương thích RTX
  50-series rồi chạy lại `scripts/gpu/preflight.py`.
- Thiếu `data/processed`: tải/copy data trước; clone Git không mang data lớn.
- `CUDA out of memory` ở embedding: đổi `batch_size` trong `configs/gpu.yaml`
  từ 128 xuống 64.
- OOM ở reranker: đổi batch 8 xuống 4; giữ `max_length=1024` trước, chỉ hạ
  xuống 512 nếu vẫn thiếu VRAM.
- Không chạy BM25 full bằng `rank_bm25`; pipeline ngày mai chủ động dùng
  dense top-50 → reranker.
