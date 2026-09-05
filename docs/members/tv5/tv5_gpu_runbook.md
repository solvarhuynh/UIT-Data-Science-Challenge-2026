# TV5 — chạy retrieval + reranker trên GPU

Runbook hỗ trợ máy RTX Windows, Beam và Kaggle GPU. Khi Beam đã được cấu hình,
ưu tiên quy trình RTX 4090 có volume/resume tại
[`tv5_beam_task1_p13.md`](tv5_beam_task1_p13.md). Kaggle là phương án dự phòng tại
[`tv5_task1_kaggle_p13.md`](tv5_task1_kaggle_p13.md). Không chạy full fine-tune
trên CPU local. Pipeline dùng:

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
| Chunks | 1.270.356 |
| Parents | 184.548 |
| Dung lượng tổng | 2.905.166.051 bytes (2,706 GiB) |
| Missing token/bigram trong child | 0 / 0 |
| Duplicate/orphan/invalid/empty chunk | 0 |
| Synthetic benchmark | 100 câu, đủ 7 nhóm |

Các hash chuẩn nằm trong `metadata/processing_manifest.json`:

```text
source_corpus_hash: cda01fcb55da1e5656f1190eb35f07d77314ed72220d13aa6a1f421737d202ed
processed_corpus_tree_hash: 80fb33ff1133ce2583097cc5a98ddc9739240a60bfe11c979892d5412dcda647
synthetic_benchmark.benchmark_sha256: 80f27b47e81aa40b25ca55ab2fe7fb7edd8fc65388f41fdae841a48a104892d3
synthetic_benchmark.source_chunk_corpus_sha256: d2aa542f1f45ad9f310bceb43063aed3058d2e81edbb15ecf18525c3c6d6bbc7
git_commit: 00d45381dd85e8cae0152c5fe0cf4dde8278dc51
```

V3 có `integrity_gate_passed=true`. `semantic_completeness_gate_passed=false`
chỉ vì BTC cung cấp 20 context có `passage` rỗng; 6 context nằm trong gold
LegalIR và 9 câu có toàn bộ gold rỗng. Không tự bịa nội dung cho các context này.

## 2. Chuẩn bị máy thuê

```powershell
git clone https://github.com/solvarhuynh/UIT-Data-Science-Challenge-2026.git
cd UIT-Data-Science-Challenge-2026
git pull --ff-only origin main
$VerifiedCommit = "00d45381dd85e8cae0152c5fe0cf4dde8278dc51"
git merge-base --is-ancestor $VerifiedCommit HEAD
if ($LASTEXITCODE -ne 0) { throw "HEAD chưa chứa code V3 đã được audit: $VerifiedCommit" }
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
5. build dense FAISS cho toàn bộ 1.270.356 chunks;
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
- dense manifest ghi đủ 1.270.356 chunks;
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

## 6. Thử nghiệm stacking cũ — không nộp lại

`artifacts/task1/stacked_final/submission.zip` đã cho điểm leaderboard thấp hơn
mốc 0.9309. Đây là thử nghiệm bị loại, không phải bản nâng cấp và **không được
nộp lại**. Bản 0.9309 vẫn được giữ tại `artifacts/task1/submission.zip` để làm
baseline an toàn.

Stacking cuối chạy CPU, không chạy lại embedding hay reranker. Nó dùng ranking,
điểm dense/BGE và embedding câu hỏi đã lưu; hai fold được huấn luyện/chấm chéo và
loại trọn nhóm câu validation khỏi nguồn nhãn lân cận. Lệnh chuẩn từ repo root:

```powershell
python scripts/submission/build_legal_ir_stacked_ensemble.py `
  --train data/raw/btc/LegalIR/train.json `
  --questions data/raw/btc/LegalIR/public-official.json `
  --components artifacts/task1/public_ensemble_v2_components.json `
  --fold-a-questions artifacts/task1/cv_strict/fold_a_questions.json `
  --fold-a-components artifacts/task1/cv_strict/fold_a_components.json `
  --fold-a-scores artifacts/task1/train500-bge-reranker-200-m512/predictions_after.jsonl `
  --fold-b-questions artifacts/task1/cv_strict/fold_b_questions.json `
  --fold-b-components artifacts/task1/cv_strict/fold_b_components.json `
  --fold-b-scores artifacts/task1/train500b-bge-reranker-200-m512/predictions_after.jsonl `
  --train-embeddings artifacts/task1/train_question_embeddings.npy `
  --train-embedding-ids artifacts/task1/train_question_ids.json `
  --public-embeddings artifacts/task1/public_question_embeddings.npy `
  --public-embedding-ids artifacts/task1/public_question_ids_embedding_order.json `
  --public-scores artifacts/task1/public-bge-reranker-200-m512/predictions_after.jsonl `
  --corpus-manifest artifacts/task1/corpus_document_ids.json `
  --overlay-labels data/task1/warmup.json `
  --positive-weight 80 `
  --output artifacts/task1/stacked_final/predictions.json `
  --report artifacts/task1/stacked_final/report.json

Đóng gói và kiểm tra độc lập:

```powershell
python scripts/submission/write_legal_ir_submission.py `
  --input artifacts/task1/stacked_final/predictions.json `
  --questions data/raw/btc/LegalIR/public-official.json `
  --corpus-manifest artifacts/task1/corpus_document_ids.json `
  --output artifacts/task1/stacked_final/submission.zip

python scripts/submission/validate_legal_ir_submission.py `
  --input artifacts/task1/stacked_final/submission.zip `
  --questions data/raw/btc/LegalIR/public-official.json `
  --corpus-manifest artifacts/task1/corpus_document_ids.json
```

Chỉ upload khi validator báo đúng 1.000 câu, mỗi câu đúng 5 document duy nhất.
`report.json` phải ghi hash toàn bộ input, commit, cấu hình model, CV core và CV
sau exact overlay. Script không tự upload lên Codabench.

## 7. Pipeline nâng cấp Task 1 bằng full OOF reranker

Không chọn submission mới bằng hai fold 500 câu nữa. Pipeline chính dùng đủ
7.000 câu train, năm fold group-disjoint, hard-negative mining và chỉ cho phép
promote khi OOF hoàn chỉnh tốt hơn pretrained BGE.

### 7.1. Local CPU: folds và candidate cache

```powershell
python scripts/evaluation/build_strict_legal_ir_cv.py `
  --input data/raw/btc/LegalIR/train.json `
  --output-dir artifacts/task1/evaluation/strict_cv_v2 `
  --folds 5 --seed 2026

python scripts/evaluation/generate_precomputed_legal_ir_candidates.py `
  --chunk-k 1000 --document-depth 500 --evidence-limit 1 --batch-size 256 `
  --output artifacts/task1/training/full_dense500_candidates.jsonl `
  --manifest artifacts/task1/training/full_dense500_candidates_manifest.json

python scripts/evaluation/check_task1_p13_prerequisites.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --candidates artifacts/task1/training/full_dense500_candidates.jsonl `
  --candidates-only `
  --output artifacts/task1/evaluation/p12_p13_candidates_gate.json

python scripts/evaluation/generate_precomputed_legal_ir_candidates.py `
  --train data/raw/btc/LegalIR/public-official.json --allow-unlabeled `
  --embeddings artifacts/task1/public_question_embeddings.npy `
  --embedding-ids artifacts/task1/public_question_ids_embedding_order.json `
  --chunk-k 1000 --document-depth 500 --evidence-limit 1 --batch-size 256 `
  --output artifacts/task1/training/public_dense500_candidates.jsonl `
  --manifest artifacts/task1/training/public_dense500_candidates_manifest.json
```

Kết quả chuẩn hiện tại: 7.000/7.000 query, không thiếu/thừa/trùng; candidate
macro Recall là 0.985574 tại depth 200, 0.987538 tại depth 300 và 0.988038 tại
depth 500. Depth 200 là cấu hình chính vì chỉ kém depth 500 khoảng 0.00246 nhưng
giảm mạnh thời gian chấm GPU.

### 7.2. Local CPU: mine hard negatives

```powershell
python scripts/training/mine_task1_negatives.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --rankings hcmute=artifacts/task1/training/full_dense500_candidates.jsonl `
  --processed-root data/processed_v3 `
  --output-dir artifacts/task1/training/negatives `
  --evidence-limit 1

python scripts/evaluation/check_task1_p13_prerequisites.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --candidates artifacts/task1/training/full_dense500_candidates.jsonl `
  --negatives-dir artifacts/task1/training/negatives `
  --output artifacts/task1/evaluation/p12_p13_prerequisites.json
```

### 7.3. GPU: smoke rồi full five-fold OOF

Nếu không còn RTX 5060 Ti, không chạy các lệnh PowerShell dưới đây trên CPU.
Dùng runbook Kaggle riêng tại
[`tv5_task1_kaggle_p13.md`](tv5_task1_kaggle_p13.md). Gói upload duy nhất là
`artifacts/task1/task1_p13_kaggle_upload.tar.zst`; quy trình Kaggle chạy tuần tự
từng fold và bắt buộc tải checkpoint về sau mỗi fold.

```powershell
python scripts/training/finetune_task1_bge_reranker.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --negatives-dir artifacts/task1/training/negatives `
  --candidates artifacts/task1/training/full_dense500_candidates.jsonl `
  --base-model models/reranker `
  --output-dir artifacts/task1/models/bge_reranker_finetune/gpu_smoke `
  --candidate-depth 200 --evidence-limit 1 `
  --folds-to-run 0 --max-training-pairs 800 --max-validation-queries 20 `
  --epochs 2 --batch-size 4 --inference-batch-size 16 `
  --gradient-accumulation 4 `
  --device cuda --diagnostic-only
```

Sau smoke, chạy đủ năm fold với `--max-training-pairs 8000` (sampler cân bằng
theo query, không cắt theo ID), bỏ `--max-validation-queries`, đổi output sang
`.../full_oof` và giữ nguyên các cấu hình khác. Nếu inference OOM thì hạ riêng
`--inference-batch-size` từ 16 xuống 8; không tự đổi learning rate/epoch theo
outer validation. Chỉ đi tiếp sang public inference khi `final_decision.json` ghi
`promotable=true`, đủ năm fold/7.000 OOF rows và fine-tuned Recall@5 cao hơn
pretrained. Không thể bảo đảm hạng 2 trước khi Codabench chấm chính thức.

```powershell
python scripts/training/finetune_task1_bge_reranker.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --negatives-dir artifacts/task1/training/negatives `
  --candidates artifacts/task1/training/full_dense500_candidates.jsonl `
  --base-model models/reranker `
  --output-dir artifacts/task1/models/bge_reranker_finetune/full_oof `
  --candidate-depth 200 --evidence-limit 1 `
  --max-training-pairs 8000 --epochs 2 `
  --batch-size 4 --inference-batch-size 16 --gradient-accumulation 4 `
  --folds-to-run 0 1 2 3 4 --device cuda
```

Bundle đã kiểm tra để chuyển máy là
`artifacts/task1/task1_p13_gpu_bundle.tar.zst` (416.816.801 bytes), SHA-256:
`d6b0c2ee1e3ac9c04785771ec6b46ab6e18d26c7c30e7a7dfa6a4d99b92893b8`.
Giải nén tại repo root bằng `tar -xaf artifacts/task1/task1_p13_gpu_bundle.tar.zst`.

### 7.4. Public inference chỉ sau khi OOF được promote

Script dưới đây tự chặn nếu quyết định chưa đủ năm fold hoặc không cải thiện.
Nó cache ranking từng checkpoint, ensemble năm fold bằng RRF, giữ dense rank làm
neo và bảo toàn exact-label overlay đã có ở baseline 0.9309.

```powershell
python scripts/submission/build_legal_ir_finetuned_fold_ensemble.py `
  --questions data/raw/btc/LegalIR/public-official.json `
  --overlay-labels data/raw/btc/LegalIR/train.json `
  --candidates artifacts/task1/training/public_dense500_candidates.jsonl `
  --checkpoint-root artifacts/task1/models/bge_reranker_finetune/full_oof `
  --candidate-depth 200 --evidence-limit 1 `
  --batch-size 16 --max-length 512 --device cuda `
  --output artifacts/task1/finetuned_oof_public/predictions.json `
  --report artifacts/task1/finetuned_oof_public/report.json

python scripts/submission/write_legal_ir_submission.py `
  --input artifacts/task1/finetuned_oof_public/predictions.json `
  --questions data/raw/btc/LegalIR/public-official.json `
  --corpus-manifest artifacts/task1/corpus_document_ids.json `
  --output artifacts/task1/finetuned_oof_public/submission.zip

python scripts/submission/validate_legal_ir_submission.py `
  --input artifacts/task1/finetuned_oof_public/submission.zip `
  --questions data/raw/btc/LegalIR/public-official.json `
  --corpus-manifest artifacts/task1/corpus_document_ids.json
```

Không ghi đè `artifacts/task1/submission.zip`; đó là bản leaderboard 0.9309 để
rollback. File mới chỉ được nộp sau khi validator qua và report/hash đầy đủ.
