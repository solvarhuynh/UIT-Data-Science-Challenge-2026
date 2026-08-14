# TV2 Nhiệm vụ 1 — Hướng dẫn chạy GPU/Kaggle

Tài liệu này là runbook duy nhất cho Nhiệm vụ 1 trên Kaggle/GPU: chuẩn bị môi
trường, tạo candidate, tạo hard/semi-hard negatives, fine-tune BGE reranker,
đánh giá 5 folds và chỉ sau đó mới chạy public inference/submission.

Không dùng run partial hoặc smoke để kết luận production. Không ghi đè
`models/dek21-v2`, `models/reranker` hoặc production config khi chưa có artifact
đánh giá đầy đủ.

## Mục tiêu và giới hạn

- Stack Nhiệm vụ 1 hiện dùng HCMUTE embedding và BGE reranker.
- Không tải Qwen hoặc BGE-M3 cho pipeline mặc định.
- Không chạy public inference hoặc tạo submission trước khi review đầy đủ kết
  quả đánh giá 5 folds.
- Full training GPU/Kaggle only; local chỉ dùng compile, unit, dry-run hoặc
  mock.

## CELL 0 — Kiểm tra GPU và dung lượng

```python
!nvidia-smi
!df -h /kaggle/working /kaggle/input
!python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); assert torch.cuda.is_available(), 'Cần CUDA cho GPU run'"
```

## CELL 1 — Lấy source và cố định commit

```bash
%cd /kaggle/working
!git clone <REPO_URL> udsc2026
%cd /kaggle/working/udsc2026
!git checkout <PINNED_COMMIT>
```

## CELL 2 — Cài đặt và kiểm tra CLI

```bash
!python -m pip install -e '.[gpu]' --no-deps
!python download_models.py --help
!python scripts/training/mine_task1_negatives.py --help
!python scripts/training/finetune_task1_bge_reranker.py --help
```

Giữ nguyên CUDA PyTorch của Kaggle; không cài bản PyTorch CPU.

## CELL 3 — Gắn dữ liệu đầu vào

Cần có:

```text
data/raw/btc/LegalIR/train.json
data/processed_v3/
models/dek21-v2/
models/reranker/
data/vector_store/faiss/
```

Không dùng Qwen, BGE-M3 hoặc public data trong giai đoạn P12/P13.

## CELL 4 — Kiểm tra path bắt buộc

```python
from pathlib import Path

required = [
    Path("data/raw/btc/LegalIR/train.json"),
    Path("data/processed_v3"),
    Path("models/dek21-v2"),
    Path("models/reranker"),
]
missing = [str(path) for path in required if not path.exists()]
assert not missing, missing
```

## CELL 5 — Preflight và hash

```bash
!python scripts/gpu/preflight.py --processed-root data/processed_v3
```

Ghi lại commit, hash train/corpus/model, thông tin CUDA/GPU, dung lượng đĩa và
VRAM trước khi chạy heavy job.

## CELL 6 — Tạo hoặc kiểm tra 5 folds

```bash
!python scripts/evaluation/build_strict_legal_ir_cv.py \
  --output-dir artifacts/task1/evaluation/strict_cv_v2
```

Chỉ reuse folds hiện có khi manifest hash còn khớp.

## CELL 7 — Smoke pipeline nhỏ

```bash
!python scripts/task1/run_legal_ir_pipeline.py \
  --config configs/task1_baseline.yaml \
  --questions data/raw/btc/LegalIR/train.json \
  --max-questions 20 --dry-run
```

Smoke chỉ kiểm tra luồng script/config; không dùng để kết luận chất lượng.

## CELL 8 — Tạo raw dense candidates K=500

Dense retrieval dùng HCMUTE. Không embed lại corpus và không rebuild FAISS nếu
cache/index hợp manifest còn dùng được.

```bash
!python scripts/evaluation/generate_dense_candidates.py \
  --legal-ir-train data/raw/btc/LegalIR/train.json \
  --output artifacts/task1/raw_k500.jsonl \
  --manifest artifacts/task1/raw_k500_manifest.json \
  --candidate-k 500 \
  --query-batch-size 8
```

Script ghi streaming từng batch và có resume theo query đã hoàn thành.

## CELL 9 — Collapse chunk thành document candidates

Raw chunk K=500 được collapse thành top 200 documents, mỗi document tối đa 2
evidence chunks.

```bash
!python scripts/evaluation/collapse_task1_candidates.py \
  --input artifacts/task1/raw_k500.jsonl \
  --output artifacts/task1/candidates/train7000_document_candidates.jsonl \
  --manifest artifacts/task1/candidates/train7000_document_candidates_manifest.json \
  --document-depth 200 \
  --evidence-limit 2
```

Output này là candidate chính cho mining negatives và reranker fine-tune.

## CELL 10 — Kiểm tra candidate gate

```bash
!python scripts/evaluation/check_task1_p13_prerequisites.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl \
  --candidates-only \
  --output artifacts/task1/evaluation/p12_p13_candidates_gate.json
```

Gate phải complete trước khi tạo negatives; output partial chỉ dùng chẩn đoán.

## CELL 11 — Tạo hard/semi-hard negatives

```bash
!python scripts/training/mine_task1_negatives.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --rankings hcmute=artifacts/task1/candidates/train7000_document_candidates.jsonl \
  --processed-root data/processed_v3 \
  --output-dir artifacts/task1/training/negatives
```

Kiểm tra mọi positive có canonical evidence và giữ lại manifest.

## CELL 12 — Kiểm tra điều kiện trước fine-tune

```bash
!python scripts/evaluation/check_task1_p13_prerequisites.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl \
  --negatives-dir artifacts/task1/training/negatives \
  --output artifacts/task1/evaluation/p12_p13_prerequisites.json
```

Yêu cầu đủ 7,000 query, 5 folds, không leakage, có positive và có
hard/semi-hard negatives.

## CELL 13 — Smoke GPU một fold với BGE

```bash
!python scripts/training/finetune_task1_bge_reranker.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --negatives-dir artifacts/task1/training/negatives \
  --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl \
  --base-model models/reranker \
  --output-dir artifacts/task1/models/bge_reranker_finetune/gpu_smoke \
  --folds-to-run 0 \
  --max-training-pairs 800 \
  --max-validation-queries 20 \
  --epochs 2 \
  --device cuda \
  --diagnostic-only
```

Smoke phải chạy được forward/backward/AMP/checkpoint/reload/rerank và luôn là
`diagnostic_only=true`, `promotable=false`.

## CELL 14 — Dry-run 5 folds

```bash
!python scripts/training/finetune_task1_bge_reranker.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --negatives-dir artifacts/task1/training/negatives \
  --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl \
  --output-dir artifacts/task1/models/bge_reranker_finetune/dry_run \
  --dry-run \
  --folds-to-run 0 1 2 3 4
```

## CELL 15 — Train fold 0

Chạy fixed epochs. Outer validation chỉ được score một lần sau khi train xong.

```bash
!python scripts/training/finetune_task1_bge_reranker.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --negatives-dir artifacts/task1/training/negatives \
  --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl \
  --base-model models/reranker \
  --output-dir artifacts/task1/models/bge_reranker_finetune \
  --folds-to-run 0 \
  --epochs 2 \
  --device cuda
```

## CELL 16 — Train folds 1–4 và resume

```bash
!python scripts/training/finetune_task1_bge_reranker.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --negatives-dir artifacts/task1/training/negatives \
  --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl \
  --base-model models/reranker \
  --output-dir artifacts/task1/models/bge_reranker_finetune \
  --folds-to-run 1 2 3 4 \
  --epochs 2 \
  --device cuda \
  --resume
```

Chỉ resume khi input hash và hyperparameter khớp.

## CELL 17 — Tổng hợp đánh giá 5 folds

Yêu cầu đủ 5 fold manifests, 7,000 OOF rows, candidate sets cố định và không
chọn checkpoint theo outer fold. Kiểm tra `comparison.json`, `comparison.md` và
`final_decision.json`.

Không promote run partial/diagnostic và không đổi config hoặc `models/reranker`
tự động.

## CELL 18 — Public inference và submission

Chỉ chạy sau khi stack cuối đã được chọn.

```bash
!python scripts/task1/run_legal_ir_pipeline.py \
  --config configs/task1_baseline.yaml \
  --questions <PUBLIC_QUESTIONS> \
  --stage all \
  --resume

!python scripts/submission/write_legal_ir_submission.py \
  --input <PREDICTIONS> \
  --questions <PUBLIC_QUESTIONS> \
  --corpus-manifest <CORPUS_MANIFEST> \
  --output artifacts/task1/submission.zip

!python scripts/submission/validate_legal_ir_submission.py \
  --input artifacts/task1/submission.zip \
  --questions <PUBLIC_QUESTIONS> \
  --corpus-manifest <CORPUS_MANIFEST>
```

Trước khi nộp, kiểm tra đủ public question IDs, mỗi answer có tối đa 5 document
IDs khác nhau, tất cả IDs tồn tại trong corpus và ZIP chỉ chứa `submission.json`.

## CELL 19 — Lưu artifact

Lưu checkpoint, OOF, manifest, log, hash và quyết định dưới `artifacts/task1/`.
Không xóa pretrained BGE và không commit model weights.

## CELL 20 — Kết thúc giai đoạn

Dừng sau đánh giá đầy đủ 5 folds. Synthetic queries chỉ chạy nếu reranker
plateau; production promotion chỉ làm khi có `final_decision.json`.

## Kiểm tra local trước khi đẩy code

```powershell
python -m compileall -q src scripts
git diff --check
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier unit
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier data
```
