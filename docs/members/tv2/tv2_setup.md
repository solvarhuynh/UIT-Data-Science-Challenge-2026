# TV2 — Hướng dẫn vận hành Task 1 LegalIR

Hướng dẫn này dựa trên repository, source, test, manifest và official scorer
hiện tại. TV2 phụ trách retrieval, indexing, tạo candidate, gộp document và bàn
giao cho TV5. Phần reranking và kiểm tra submission của TV5 cũng được mô tả để
một người có thể vận hành toàn bộ Task 1.

## 1. Contract Task 1

Canonical corpus: `data/processed_v3` (read-only for experiments). Training
labels: `data/raw/btc/LegalIR/train.json`. Experiment outputs belong under
`artifacts/tv2/` or `artifacts/task1/`. Final predictions use exact question
coverage and distinct document IDs. Internal rankings may be deep, but official
predictions must contain 1–5 documents per question.

## 2. Official scorer

The source of truth is [`scoring.py`](../../Scoring-Program-Task-LegalIR/scoring.py).
One through five predictions score normally; empty or more than five predictions
score Recall 0 and Precision 0; multi-gold Recall is intersection divided by
gold-set size; macro Recall is primary and macro Precision is secondary. Local
parity is in [`legal_ir.py`](../../../src/udsc2026/evaluation/legal_ir.py) with
golden tests in `tests/unit/test_evaluation/test_legal_ir.py`.

## 3. Bố cục dữ liệu

```text
data/processed_v3/{chunks,parents,documents,metadata}
```

Current audit counts: 1,270,356 chunks, 184,548 parents, and 8,532 documents.
Strict CV is `artifacts/task1/evaluation/strict_cv_v2/`: 5 folds, 7,000
questions, with normalized duplicate groups kept together.

## 4. Baseline hiện đã xác nhận

[`build_legal_ir_ensemble.py`](../../../scripts/submission/build_legal_ir_ensemble.py)
loads cached DEk21 dense and BGE rankings, word TF-IDF KNN, passage BM25, RRF,
and optional exact-label overlay. The validated dense manifest uses collection
`legal_chunks_dek21_v2_768`, dimension 768, corpus hash
`d92782fe2a66a865745721756c1f790f0c4b5073f31c29004f29efb5e0941cc1`, and model
`huyydangg/DEk21_hcmute_embedding_v2`, revision
`99a2963b2f51fa7a570a3e7f550d7993b9de90a8`.

```text
Current validated baseline
Query → DEk21 child dense candidates
      → cached BGE reranking when available
      → word KNN + passage BM25 + RRF ensemble
      → top 5 distinct document IDs
```

P2 cached smoke (`train500_dense200`, 500 questions) measured
CandidateDocRecall@200 `0.965`, mean unique docs at depth 200 `61.258`, and
11 retrieval misses. This is a cached diagnostic, not a public score.

## 5. Kiến trúc Top-1 mục tiêu

The following components are experimental until full GPU OOF validation:

```text
                         ┌─ child dense HCMUTE
                         ├─ parent/article dense
Query ───────────────────┼─ document BM25/BM25F
                         ├─ query KNN word+char
                         └─ optional Qwen3 dense
                                  │
                                  ▼
                         document candidate union
                             100–200 docs
                                  │
                       top 2 evidence chunks/doc
                                  │
                                  ▼
                   BGE/Qwen3 document reranker
                                  │
                        document-level RRF
                                  │
                              TOP 5
```

P3 lexical, P4 parent/multi-granularity, and P5 document-candidate code are
implemented, but are not current production claims without a matching GPU
benchmark manifest. Current production reranker remains BGE.

## 6. Kiểm tra nhẹ trên CPU local

### P12 — mine hard/semi-hard negatives từ cache

P12 không train và không load model. Mỗi `fold_F.jsonl` chỉ chứa records dùng
để train fold F, nên loại toàn bộ query thuộc validation fold F. Default dùng
cached HCMUTE; thêm parent/BM25F/citation/reranker cache bằng nhiều flag
`--rankings source=path`.

```powershell
python scripts/training/mine_task1_negatives.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --rankings hcmute=artifacts/task1/train500_dense200_predictions.jsonl `
  --output-dir artifacts/task1/training/negatives
```

Outputs: `fold_0.jsonl` đến `fold_4.jsonl`, `mining_manifest.json` và
`statistics.md`. Near-duplicate/ambiguous candidates are excluded; all
negative records keep source/rank/score provenance.

Để chạy ablation `easy-random` của P13, tạo lại một P12 artifact riêng (không
ghi đè artifact mặc định) với optional easy band:

```powershell
python scripts/training/mine_task1_negatives.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --rankings hcmute=artifacts/task1/train500_dense200_predictions.jsonl `
  --max-easy-per-query 5 `
  --output-dir artifacts/task1/training/negatives_with_easy
```

### P13 — local schema smoke, không load model

P13 chỉ fine-tune BGE trên Kaggle GPU. Local dry-run kiểm tra hash, split,
leakage và document-evidence formatting; cache `train500` chỉ là partial OOF
nên flag này không bao giờ tạo promotion decision.

```powershell
python scripts/training/finetune_task1_bge_reranker.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --negatives-dir artifacts/task1/training/negatives `
  --candidates artifacts/task1/train500_dense200_predictions.jsonl `
  --output-dir artifacts/task1/models/bge_reranker_finetune_dryrun `
  --allow-partial-oof --dry-run
```

These commands are light/medium CPU checks and do not rebuild the full index:

```powershell
python -m compileall -q src scripts
git diff --check
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier unit
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier data
```

Runner smoke, maximum 20 questions:

```powershell
python scripts/task1/run_legal_ir_pipeline.py `
  --config configs/task1_baseline.yaml `
  --questions data/raw/btc/LegalIR/train.json `
  --max-questions 20 --dry-run
```

## 7. Pipeline GPU

The actual GPU profile is [`configs/gpu.yaml`](../../../configs/gpu.yaml):
DEk21 v2, CUDA, FAISS collection `legal_chunks_dek21_v2_768`, and local
`BAAI/bge-reranker-v2-m3`. This is medium/heavy and requires approved model and
index inputs; it must not silently download or rebuild locally.

```powershell
python scripts/task1/run_legal_ir_pipeline.py `
  --config configs/task1_baseline.yaml `
  --questions data/raw/btc/LegalIR/train.json `
  --processed-root data/processed_v3 `
  --contexts-dir data/raw/btc/LegalIR/selected-contexts `
  --vector-store-root data/vector_store `
  --artifacts-root artifacts/task1/pipeline_gpu `
  --stage all --resume
```

Full project pre-push check (medium/heavy):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-all.ps1 -SkipFrontend -SkipCompose
```

### P13 — Kaggle GPU strict OOF

Mount/copy the local base checkpoint as `models/reranker`; P13 never overwrites
it and never downloads weights. Before the full command, P12 and the cached
candidate ranking must cover all 7,000 strict-CV IDs. Do not pass
`--allow-partial-oof` for a promotable result.

```powershell
python scripts/training/finetune_task1_bge_reranker.py `
  --train data/raw/btc/LegalIR/train.json `
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json `
  --negatives-dir artifacts/task1/training/negatives_full `
  --candidates artifacts/task1/train_dense200_predictions.jsonl `
  --base-model models/reranker `
  --ablation semi-hard-plus-hard `
  --candidate-depth 200 --evidence-limit 2 `
  --learning-rate 2e-5 --epochs 2 `
  --batch-size 4 --gradient-accumulation 4 `
  --max-length 512 --warmup-ratio 0.1 `
  --seed 2026 --device cuda `
  --output-dir artifacts/task1/models/bge_reranker_finetune
```

Run `baseline`, `semi-hard`, `semi-hard-plus-hard`, and `same-law-boosted` into
separate output directories for the bounded A/C/D/E comparison. `easy-random`
requires the separate `negatives_with_easy` artifact above. A checkpoint can
only be considered when `final_decision.json` reports `PROMOTE_CANDIDATE`; the
runner does not edit `configs/task1_top1.yaml` automatically.

## 8. Artifact được tạo

The runner creates `run_manifest.json` and `corpus_document_ids.json` under its
artifact root.

| Artifact | Producer/input | Purpose | Rebuild/delete | Submission effect |
|---|---|---|---|---|
| `artifacts/tv2/data_audit/*` | TV2 audit over processed corpus | integrity | rebuildable | none |
| `artifacts/task1/*dense*` | DEk21 + indexed chunks | cached candidates | hash-dependent | indirect |
| `artifacts/task1/*reranker*` | TV5 BGE over candidates | rerank diagnostics | rebuildable | indirect |
| `artifacts/task1/cv_strict/*` | strict CV builder + train labels | OOF folds | rebuildable | none |
| `artifacts/task1/evaluation/*` | P1–P4 analyzers/smokes | diagnostics | rebuildable | none |
| `submission.zip` | submission writer | package containing only submission.json | regenerate | direct |

## 9. Submission

The runner's final stage calls the existing writer and validator. Direct commands
are:

```powershell
python scripts/submission/write_legal_ir_submission.py `
  --input artifacts/task1/pipeline_gpu/ensemble/predictions.json `
  --questions data/raw/btc/LegalIR/train.json `
  --corpus-manifest artifacts/task1/pipeline_gpu/corpus_document_ids.json `
  --output artifacts/task1/pipeline_gpu/submission.zip

python scripts/submission/validate_legal_ir_submission.py `
  --input artifacts/task1/pipeline_gpu/submission.zip `
  --questions data/raw/btc/LegalIR/train.json `
  --corpus-manifest artifacts/task1/pipeline_gpu/corpus_document_ids.json
```

Validation is light; full package writing is medium.

## 10. Lỗi thường gặp

- Public `LABEL_NOT_AVAILABLE` rows are unlabeled transport data, not gold.
- Six documents score zero under the official scorer.
- Compare corpus/model hashes and collection names before reusing an index.
- Strict CV excludes validation IDs and normalized duplicate questions from
  supervised neighbor training.
- The Windows gate creates a fresh pytest runtime to avoid stale ACL failures.

## 11. Tái lập và liên kết

Promoted runs require git commit, config, input/corpus hashes, model revision,
 representation versions, candidate depth, reranker, RRF weights, timestamp, and
 output hashes. See [`model_registry.md`](../../models/model_registry.md),
 [`tv5_gpu_runbook.md`](../tv5/tv5_gpu_runbook.md), and
 [`tv5_legalir_warmup.md`](../tv5/tv5_legalir_warmup.md). The cell-by-cell
 The temporary P12/P13 GPU source of truth is
 [`tv2_task1_kaggle_p12_p13.md`](tv2_task1_kaggle_p12_p13.md). The older guide
 is not a final submission guide for this stage.

Temporary P12/P13 Kaggle guide riêng cho TV2:
[`tv2_task1_kaggle_p12_p13.md`](tv2_task1_kaggle_p12_p13.md).
