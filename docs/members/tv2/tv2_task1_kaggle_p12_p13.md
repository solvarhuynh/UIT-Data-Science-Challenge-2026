# TV2 Task1 — Temporary Kaggle GPU runbook (P12/P13 only)

Temporary GPU stage only. It stops after strict P13 OOF: no public inference,
submission package, P14/P15, production config changes, or overwrite of
`models/reranker`. Statuses are `CODE_READY` → `KAGGLE_GPU_SMOKE_PENDING` →
`P12_FULL_PENDING` → `P13_OOF_PENDING` → `P13_OOF_COMPLETE`.

## CELL 0 — GPU/filesystem

```python
!nvidia-smi
!df -h /kaggle/working /kaggle/input
```

## CELL 1 — Clone and pin

```bash
%cd /kaggle/working
!git clone <REPO_URL> udsc2026
%cd /kaggle/working/udsc2026
!git checkout <PINNED_COMMIT>
```

## CELL 2 — Install and help

```bash
!python -m pip install -e '.[gpu]' --no-deps
!python scripts/training/finetune_task1_bge_reranker.py --help
!python scripts/training/mine_task1_negatives.py --help
```

Preserve CUDA PyTorch; do not install CPU torch or flash-attn.

## CELL 3 — Add Data

Mount the original LegalIR train file, `data/processed_v3`, HCMUTE model,
`models/reranker`, and cached HCMUTE dense predictions. No Qwen/public data.

## CELL 4 — Assert paths

```python
from pathlib import Path
required = [Path('data/raw/btc/LegalIR/train.json'), Path('data/processed_v3'), Path('models/reranker')]
assert all(p.exists() for p in required), required
```

## CELL 5 — Preflight/hashes

```bash
!python scripts/gpu/preflight.py --processed-root data/processed_v3
```

Record commit, train/corpus/model hashes, CUDA/GPU, disk, and free VRAM.

## CELL 6 — Strict five folds

```bash
!python scripts/evaluation/build_strict_legal_ir_cv.py --output-dir artifacts/task1/evaluation/strict_cv_v2
```

Reuse existing folds only when manifest hashes match.

## CELL 7 — 20-query retrieval smoke

Run HCMUTE retrieval on exactly 20 organizer queries; record raw chunk K,
unique document depth, ordering, latency, and VRAM.

## CELL 8 — K=200 versus K=500 probe

Generate/reuse the same 500-query raw caches, then stream-collapse them:

```bash
!python scripts/evaluation/collapse_task1_candidates.py --input artifacts/task1/raw_k200.jsonl --output artifacts/task1/candidates/train7000_document_candidates_k200.jsonl --manifest artifacts/task1/candidates/train7000_k200_manifest.json --document-depth 200 --evidence-limit 2
!python scripts/evaluation/collapse_task1_candidates.py --input artifacts/task1/raw_k500.jsonl --output artifacts/task1/candidates/train7000_document_candidates_k500.jsonl --manifest artifacts/task1/candidates/train7000_k500_manifest.json --document-depth 500 --evidence-limit 2
```

Compare CandidateDocRecall and gold-present-but-dropped. Write
`artifacts/task1/evaluation/p12_p13_candidate_depth_decision.json`; do not
automatically try K=1000.

## CELL 9 — Full raw candidates

Produce one streamed HCMUTE row for all 7,000 strict-CV queries, or reuse a
matching raw cache. No corpus re-embedding or FAISS rebuild in this stage.

## CELL 10 — Collapse and candidate-only gate

```bash
!python scripts/evaluation/collapse_task1_candidates.py --input artifacts/task1/raw_k500.jsonl --output artifacts/task1/candidates/train7000_document_candidates.jsonl --manifest artifacts/task1/candidates/train7000_document_candidates_manifest.json --document-depth 200 --evidence-limit 2
!python scripts/evaluation/check_task1_p13_prerequisites.py --train data/raw/btc/LegalIR/train.json --folds artifacts/task1/evaluation/strict_cv_v2/folds.json --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl --candidates-only --output artifacts/task1/evaluation/p12_p13_candidates_gate.json
```

The gate must be complete before P12; partial output is diagnostic only.

## CELL 11 — Full P12 mining

```bash
!python scripts/training/mine_task1_negatives.py --train data/raw/btc/LegalIR/train.json --folds artifacts/task1/evaluation/strict_cv_v2/folds.json --rankings hcmute=artifacts/task1/candidates/train7000_document_candidates.jsonl --processed-root data/processed_v3 --output-dir artifacts/task1/training/negatives
```

Check every positive has canonical evidence and retain the manifest.

## CELL 12 — Full prerequisite gate

```bash
!python scripts/evaluation/check_task1_p13_prerequisites.py --train data/raw/btc/LegalIR/train.json --folds artifacts/task1/evaluation/strict_cv_v2/folds.json --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl --negatives-dir artifacts/task1/training/negatives --output artifacts/task1/evaluation/p12_p13_prerequisites.json
```

Require 7,000 queries, five folds, no leakage, positives, and hard/semi-hard negatives.

## CELL 13 — Real BGE one-fold GPU smoke

```bash
!python scripts/training/finetune_task1_bge_reranker.py --train data/raw/btc/LegalIR/train.json --folds artifacts/task1/evaluation/strict_cv_v2/folds.json --negatives-dir artifacts/task1/training/negatives --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl --base-model models/reranker --output-dir artifacts/task1/models/bge_reranker_finetune/gpu_smoke --folds-to-run 0 --max-training-pairs 800 --max-validation-queries 20 --epochs 2 --device cuda --diagnostic-only
```

Must exercise forward/backward/AMP/checkpoint/reload/rerank and remain
`diagnostic_only=true`, `promotable=false`.

## CELL 14 — P13 dry-run

```bash
!python scripts/training/finetune_task1_bge_reranker.py --train data/raw/btc/LegalIR/train.json --folds artifacts/task1/evaluation/strict_cv_v2/folds.json --negatives-dir artifacts/task1/training/negatives --candidates artifacts/task1/candidates/train7000_document_candidates.jsonl --output-dir artifacts/task1/models/bge_reranker_finetune/dry_run --dry-run --folds-to-run 0 1 2 3 4
```

## CELL 15 — Train fold 0

Use fixed epochs only. Outer validation is scored once after training.

## CELL 16 — Train folds 1–4

```bash
!python scripts/training/finetune_task1_bge_reranker.py ... --folds-to-run 1 2 3 4 --resume
```

Resume only when input hashes and hyperparameters match.

## CELL 17 — Aggregate strict OOF

Require five complete fold manifests, 7,000 OOF rows, fixed candidate sets,
and no outer-fold checkpoint selection.

## CELL 18 — Decision artifact

Inspect `comparison.json`, `comparison.md`, and `final_decision.json`. Never
promote a partial/diagnostic run; do not change configs or `models/reranker`.

## CELL 19 — Save artifacts

Save fold checkpoints, OOF, manifests, logs, hashes, and candidate-depth
decision under `artifacts/task1/`. Do not save a submission.

## CELL 20 — Stop

Stop after P13 OOF. P14, P15, public inference, `submission.zip`, and
production promotion are a later approved stage.
