# TV2 — Hướng dẫn vận hành Nhiệm vụ 1 LegalIR

## Dữ liệu, model và quy tắc chung

- Corpus chuẩn: `data/processed_v3`; chỉ đọc trong benchmark, mining và train.
- Labels train: `data/raw/btc/LegalIR/train.json`.
- Dense model: `huyydangg/DEk21_hcmute_embedding_v2` tại `models/dek21-v2`.
- Reranker: `BAAI/bge-reranker-v2-m3` tại `models/reranker`.
- Mọi output experiment đặt dưới `artifacts/task1/`; không ghi đè model base
  hoặc corpus chuẩn.
- Official prediction phải có từ 1 đến 5 document ID khác nhau cho mỗi query.

## Tạo index dense

**File chính**

```text
scripts/data_prep/index_chunks.py
configs/gpu.yaml
```

Chỉ build lại index khi corpus hoặc manifest model không còn khớp.

```powershell
python scripts/data_prep/index_chunks.py --chunks-dir data/processed_v3/chunks --vector-db-type faiss
```

## Tạo raw candidates và document candidates

**File chính**

```text
scripts/evaluation/generate_dense_candidates.py
scripts/evaluation/collapse_task1_candidates.py
```

Raw retrieval dùng K=500; sau collapse giữ top 200 documents và tối đa 2
evidence chunks/document.

```powershell
python scripts/evaluation/generate_dense_candidates.py `
  --legal-ir-train data/raw/btc/LegalIR/train.json `
  --output artifacts/task1/raw_k500.jsonl `
  --manifest artifacts/task1/raw_k500_manifest.json `
  --candidate-k 500 `
  --query-batch-size 8

python scripts/evaluation/collapse_task1_candidates.py `
  --input artifacts/task1/raw_k500.jsonl `
  --output artifacts/task1/candidates/train7000_document_candidates.jsonl `
  --manifest artifacts/task1/candidates/train7000_document_candidates_manifest.json `
  --document-depth 200 `
  --evidence-limit 2
```

## Chia dữ liệu đánh giá không leakage

**File chính**

```text
scripts/evaluation/build_strict_legal_ir_cv.py
artifacts/task1/evaluation/strict_cv_v2/folds.json
```

Artifact hiện tại có 5 folds: mỗi fold gồm 1,400 query validation và 5,600
query train.

## Tạo hard/semi-hard negative samples

**File chính**

```text
scripts/training/mine_task1_negatives.py
scripts/evaluation/check_task1_p13_prerequisites.py
```

```powershell
python scripts/training/mine_task1_negatives.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --rankings hcmute=artifacts/task1/candidates/train7000_document_candidates.jsonl `
  --output-dir artifacts/task1/training/negatives

python scripts/evaluation/check_task1_p13_prerequisites.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl `
  --negatives-dir artifacts/task1/training/negatives
```

## Fine-tune BGE reranker bằng hard negatives

**File chính**

```text
scripts/training/finetune_task1_bge_reranker.py
models/reranker
```

Dry-run kiểm tra dữ liệu/split mà không load model:

```powershell
python scripts/training/finetune_task1_bge_reranker.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --negatives-dir artifacts/task1/training/negatives `
  --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl `
  --base-model models/reranker `
  --output-dir artifacts/task1/models/bge_reranker_finetune/dry_run `
  --dry-run --device cpu
```

Đánh giá đầy đủ dùng 5 folds, 2 epochs/fold, candidate depth 200, batch 4,
gradient accumulation 4 và max length 512. Chỉ dùng kết quả đầy đủ để quyết
định có thay stack cuối hay không.

## Bổ sung synthetic legal queries nếu cần

Chưa thực hiện. Chỉ bổ sung query từ legal corpus nếu fine-tuned reranker không
còn cải thiện đủ; không tạo synthetic label cho public questions.

## Chạy pipeline cuối và tạo submission

**File chính**

```text
scripts/task1/run_legal_ir_pipeline.py
scripts/submission/write_legal_ir_submission.py
scripts/submission/validate_legal_ir_submission.py
```

Sau khi chọn stack cuối, chạy public inference rồi tạo và validate
`submission.zip`. Kiểm tra coverage câu hỏi, document IDs distinct/hợp lệ và
ZIP chỉ chứa `submission.json` trước khi nộp.

## Kiểm tra nhẹ trên máy local

```powershell
python -m compileall -q src scripts
git diff --check
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier unit
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier data
```

## Tài liệu liên quan

- [`tv2_task1_kaggle.md`](tv2_task1_kaggle.md): hướng dẫn vận hành GPU tổng quát.
- [`tv2_task1_kaggle.md`](tv2_task1_kaggle.md): runbook theo từng cell cho Kaggle/GPU, candidate, negatives, reranker và submission.
