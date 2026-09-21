# Task1 active runtime normalization

Date: 2026-09-18

## Active defaults before

Five active model-selection locations still selected Qwen:

1. `configs/base.yaml` -> `./models/qwen3-vl-reranker-2b`.
2. `RerankerSettings.model_name_or_path` -> the Qwen path.
3. `scripts/evaluation/benchmark_reranker.py --model` default -> the Qwen path.
4. `download_models.py` task1 profiles selected the Qwen `reranker` key.
5. `scripts/gpu/run_gpu_pipeline.ps1` passed the Qwen path explicitly for both
   bounded and production benchmark calls.

The benchmark default output directory also used a Qwen name and was
normalized alongside its model default.

## Active defaults after

| Location | Final active value |
|---|---|
| `configs/base.yaml` reranker path | `./models/reranker` |
| `RerankerSettings` default | `./models/reranker` |
| benchmark CLI model default | `models/reranker` |
| benchmark CLI output default | `artifacts/tv5/bge-reranker-v2-m3` |
| `configs/task1_top1.yaml` | `./models/reranker`, `BAAI/bge-reranker-v2-m3` |
| `configs/task1_baseline.yaml` | BGE `BAAI/bge-reranker-v2-m3`; runtime default path `./models/reranker` |
| `configs/gpu.yaml` | `./models/reranker` |
| active downloader profiles | `embedding + bge_reranker` |
| GPU pipeline benchmark calls | `models/reranker` |

The explicit historical profile is now `task1-qwen`; it is not selected by any
active Task1 profile.

## Files modified

- `configs/base.yaml`
- `src/udsc2026/infrastructure/reranker/config.py`
- `scripts/evaluation/benchmark_reranker.py`
- `download_models.py`
- `scripts/gpu/run_gpu_pipeline.ps1`
- `docs/models/qwen3_vl_reranker_2b.md` (only the historical download command,
  changed to explicit `task1-qwen`)
- `reports/task1/bge_active_runtime_normalization.md`

The config loader retains the earlier minimal alias normalization:
Task1's `model_path` is mapped to the shared runtime
`model_name_or_path`; the Qwen default is no longer used.

## Qwen artifacts intentionally preserved

No Qwen model code or historical artifacts were deleted or renamed. In
particular, these remain untouched:

- `scripts/modal/task1_b2a_qwen3vl2b.py`
- `scripts/modal/task1_full_doc_top200_qwen3vl2b.py`
- `scripts/modal/task1_full_doc_top200_qwen3vl2b_optimized.py`
- `artifacts/task1/full_doc_qwen_optimized_outputs/`
- `artifacts/task1/full_doc_qwen_optimized_merged/`
- `reports/task1/full_document_legal_field_retrieval/`

The generic Qwen branch in
`src/udsc2026/infrastructure/reranker/client.py`, the Qwen downloader
`ModelSpec`, the explicit `task1-qwen` profile, and the Qwen model-card
documentation remain for historical/auxiliary reproduction. The Qwen LLM
default under `src/udsc2026/infrastructure/llm/config.py` is unrelated to the
reranker and was not changed.

## Regression smoke

All smoke runs used the existing local BGE model, CPU only, no labels, and at
most five queries / 25 candidate pairs per smoke:

| Smoke | Resolved path | Resolved family | Finite scores | Result |
|---|---|---|---:|---|
| Explicit `configs/task1_top1.yaml` | `./models/reranker` | BGE | YES | PASS |
| Benchmark CLI without `--model` | `models/reranker` | BGE | YES | PASS |
| Generic `configs/base.yaml` default | `./models/reranker` | BGE | YES | PASS |

Observed score ranges:

- explicit config: `0.21502485871315002` .. `0.9999357461929321`
- generic default: `0.356961727142334` .. `0.9867396354675293`
- default CLI: `0.8818067312240601` .. `0.9999357461929321`

Hard assertions passed:

- model id: `BAAI/bge-reranker-v2-m3`
- resolved path: `models/reranker`
- unexpected Qwen runtime: NO
- finite scores: YES
- bounded output schema: PASS

## Diff and blockers

- Active Qwen defaults before: 5
- Active Qwen defaults after: 0
- Historical Qwen files/artifacts modified: 0
- Full inference runs: 0
- Modal runs: 0
- Python compile: PASS
- PowerShell parse: PASS
- Optional pytest regression suite: not run because `pytest` is not installed
- Blocker for active BGE runtime: none
