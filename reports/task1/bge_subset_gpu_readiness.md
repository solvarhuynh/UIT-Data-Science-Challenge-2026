# BGE TOP5 subset GPU readiness

Date: 2026-09-18

Status: runner prepared and locally validated; no Modal invocation and no GPU
execution occurred.

## Step4 model contract

The authoritative handoff is
`artifacts/task1/handoff/step4_bge_handoff/scripts/modal_step4_ensemble.py`.
It scores two independent Cross-Encoders on the same selected evidence:

- Model A, fine-tuned BGE-M3: `/data/bge_ft/bge_m3_finetuned`.
- Model B, base BGE-M3: `/data/models/reranker`.

Therefore the TOP5 Step4-compatible experiment requires
`REQUIRED_GPU_MODELS = FT_AND_BASE`. Base-only scoring would not reproduce the
Step4 decision signal.

The handoff source proves `max_length=512`, `CrossEncoder.predict` with the
default single-label activation, and batch scoring. It does not record a
fine-tuned Hugging Face revision. The immutable FT identity is therefore the
exact artifact hash pair, not a fabricated revision.

`BGE_FT_PROVENANCE = VERIFIED_BY_ARTIFACT_SHA256`.

The local FT preflight artifact is
`outputs/task1/bge_ft_model/bge_m3_finetuned` and passed without loading the
model:

- Weight SHA256:
  `68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c`
- Config SHA256:
  `16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b`
- Artifact ID:
  `sha256:68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c`
- Config contract: `xlm-roberta`, one label, valid BGE sequence classifier.

The Base model is `BAAI/bge-reranker-v2-m3`, expected revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. The local Base snapshot has exact
revision tree metadata at
`models/reranker/.cache/huggingface/trees/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e.json`.
The GPU runner independently requires the same revision metadata in the
mounted model Volume; it does not guess from a directory name.

## Composite and calibration contract

The handoff proves:

`S = 0.40*N_FT + 0.30*N_Base + 0.20*N_RRF + 0.10*N_Support`

where neural scores and RRF are min-max normalized per query and
`N_Support = source_support / 4.0`.

The raw Step4 report uses rank-5 margin `0.08` and rank-4 margin `0.15`.
The calibrated 0.9411 gate additionally requires `delta_ensemble >= 0.40`
and rank safety: added rank must not be worse unless added support is higher.
Positions 1–3 remain locked to the 0.9391 anchor.

## Local subset versus historical Step4

The local scorer and the historical public Step4 runner are not numerically
equivalent. This is an explicit gate, not a silent substitution:

- Question source: local subset uses canonical F1–F4 `train.json`; Step4 uses
  the public question source. The local source is appropriate for this TOP5
  worklist population, but it is not the public Step4 source.
- Document source: local subset uses canonical `data/processed_v3/chunks`;
  Step4 uses precomputed public evidence or its public context fallback.
- Selection: local uses canonical `true_s2_bm25_within_document_v2` top-three;
  Step4 uses its evidence-cache/ad-hoc context selection.
- Maximum length: both use 512 — match.
- Activation: both use the default single-label CrossEncoder activation —
  match; no manual sigmoid is applied.
- Aggregation: local subset uses `MAX`; Step4 uses bounded LogSumExp —
  mismatch.
- Normalization: local subset emits raw per-model document scores; Step4
  applies per-query min-max normalization before fusion — mismatch.
- Dtype: local CPU reference is float32; the Step4 source gives no explicit
  reduced-precision override, but exact GPU dtype parity is not proven.
- Model ordering: the prepared runner keeps FT and Base scores separate and
  never averages them.

Consequently:

- `step4_contract_reconstructable = NO` because the historical runner's
  evidence/LogSumExp/fusion contract is not the local MAX subset contract.
  FT artifact provenance itself is now verified by SHA256.
- `local_vs_step4_contract = FAIL`.

The new GPU runner follows the explicitly requested subset contract (canonical
selector, top-three, MAX aggregation) and does not claim to recreate the
historical Step4 calibrated submission.

## Prepared runner

New file:

`scripts/modal/task1_bge_subset_gpu.py`

It provides `--worklist`, `--mode canary|production`, `--limit`,
`--batch-size`, and `--checkpoint-every`. It uploads the worklist once as a
compressed argument, loads questions once, prepares each document once by
direct path, keeps one model instance per model, batches pairs, validates
finite scores, and emits separate `bge_ft_score` and `bge_base_score` fields.

The supported batch sizes are `1, 4, 8, 16, 32`; no batch benchmark was run.
The recommended first canary setting is batch `16`, with checkpoint commits
every `64` q-docs.

Checkpoint identity includes mode, selected worklist SHA, q-doc/unit counts,
selector, aggregation, max length, contract SHA, Base model ID/revision, FT
model ID/artifact ID, FT weight SHA, and FT config SHA. A stale model,
revision, FT artifact, worklist, or contract is rejected before scoring. The
checkpoint test passed for durable resume and stale FT identity rejection.

The namespace is separate from all Qwen and Step4 artifacts:

- `runtime/task1_bge_subset_top5/checkpoints/`
- `runtime/task1_bge_subset_top5/results/`
- `runtime/task1_bge_subset_top5/manifests/`

## CPU reference

The deterministic first-100 prefix was already scored by the local CPU BGE
scorer and is retained as the bounded reference:

`artifacts/task1/qwen_to_bge_minimal/bge_subset_benchmark_100.jsonl`

Metrics:

`artifacts/task1/qwen_to_bge_minimal/bge_subset_benchmark_100_metrics.json`

It covers 100 q-docs / 300 selected chunk pairs with the same local selector
and MAX aggregation. A 256-row CPU rerun was intentionally not performed;
the GPU canary is bounded to 256 q-docs / at most 768 pairs, while parity
comparison can use the deterministic 100-row prefix. The CPU reference is
Base-only; FT numerical parity requires the GPU canary.

## Commands — prepared, not run

The following canary command is the only prepared GPU command. It must not be
run until `/data/bge_ft/bge_m3_finetuned/model.safetensors` and
`/data/bge_ft/bge_m3_finetuned/config.json` match the two pinned SHA256 values
above and the preflight can resolve both model contracts:

```powershell
& '.venv\Scripts\modal.exe' run --profile nghiadethuong3107 scripts/modal/task1_bge_subset_gpu.py `
  --worklist artifacts/task1/qwen_to_bge_minimal/top5_missing_bge.jsonl `
  --mode canary `
  --limit 256 `
  --batch-size 16 `
  --checkpoint-every 64
```

Production remains explicitly blocked:

```powershell
& '.venv\Scripts\modal.exe' run --profile nghiadethuong3107 scripts/modal/task1_bge_subset_gpu.py `
  --worklist artifacts/task1/qwen_to_bge_minimal/top5_missing_bge.jsonl `
  --mode production `
  --batch-size 16 `
  --checkpoint-every 256
```

`DO_NOT_RUN_PRODUCTION_UNTIL_CANARY_PASS`.

Validation performed: Python compile passed; exact FT file-only preflight
passed; mutated expected FT SHA, wrong Base revision, and stale FT checkpoint
tests all failed closed as required. GPU runs: `0`; Modal runs: `0`.
