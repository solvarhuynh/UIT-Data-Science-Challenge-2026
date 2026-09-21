# BGE runtime restore audit

Audit date: 2026-09-18

Scope: textual source/configuration files under `scripts/`, `src/`, `configs/`,
`tests/`, `reports/task1/`, and `artifacts/task1/`. Model weights, Hugging Face
cache blobs, JSONL/SQLite score stores, and generated binary artifacts were not
loaded or modified. Inventory categories below are intentionally overlapping:
a runtime file can also be configuration or inference code.

## `download_models.py`

- Path: `download_models.py`
- Minimal change: added a separate `bge_reranker` `ModelSpec`.
- Hugging Face model id: `BAAI/bge-reranker-v2-m3`
- Local destination: `models/reranker`
- Added explicit profile: `task1-bge` (HCMUTE embedding + BGE).
- Historical `task1`/`task1-baseline`/`task1-top1` profile behavior and the
  Qwen `reranker` entry were preserved.
- No downloader was executed, so the existing `models/download_manifest.json`
  was not overwritten.

Read-only Hub resolution succeeded:

```text
model id: BAAI/bge-reranker-v2-m3
resolved revision: 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
```

## Active BGE files

These are the executable/configuration paths that can participate in the BGE
Task1 path. The generic `CrossEncoderClient` uses BGE through its standard
Sentence-Transformers branch; it does not contain a BGE model-id literal.

| Path | Category | Uses BGE? | Runtime critical? | Current model/path | Qwen contamination suspected? | Action now |
|---|---|---:|---:|---|---:|---|
| `download_models.py` | CONFIGURATION | Yes | Yes | BGE -> `models/reranker` | No | Modified minimally |
| `configs/task1_top1.yaml` | CONFIGURATION | Yes | Yes | BGE -> `./models/reranker` | No | Read-only |
| `configs/task1_baseline.yaml` | CONFIGURATION | Yes | Yes | BGE model id | No | Read-only |
| `configs/gpu.yaml` | CONFIGURATION | Path only | Yes | `./models/reranker` | No | Read-only |
| `src/udsc2026/infrastructure/reranker/client.py` | RUNTIME_CRITICAL | Generic BGE path | Yes | Sentence-Transformers branch | No; Qwen branch is explicit | Read-only |
| `src/udsc2026/infrastructure/reranker/config.py` | CONFIGURATION | Generic path | Yes | Default remains Qwen | Yes | Deferred |
| `scripts/gpu/preflight.py` | RUNTIME_CRITICAL | BGE local-path check | Yes | `models/reranker` | No | Read-only |
| `scripts/evaluation/benchmark_reranker.py` | RUNTIME_CRITICAL | Accepts BGE | Yes when invoked with BGE | CLI default remains Qwen | Yes | Deferred |
| `src/udsc2026/evaluation/legal_ir_document_candidates.py` | RUNTIME_CRITICAL | BGE-compatible adapter | Yes | `RerankerClient` | No | Read-only |
| `src/udsc2026/evaluation/legal_ir_chunk_aggregation.py` | RUNTIME_CRITICAL | BGE score source | Yes for BGE artifacts | BGE/Qwen-neutral schema | No | Read-only |
| `src/udsc2026/evaluation/legal_ir_recovery.py` | RUNTIME_CRITICAL | BGE feature weight | Task1 recovery path | Mixed historical signals | Deferred | Read-only |
| `scripts/evaluation/aggregate_task1_chunks.py` | INFERENCE | BGE rank source | Task1 artifact path | `--rank-source bge` | No | Read-only |
| `scripts/beam/beam_task1_v3_public_adaptive_k200.py` | INFERENCE | BGE scoring | Historical Beam path | `models/reranker` | Historical model contract | Read-only |
| `scripts/training/finetune_task1_bge_reranker_v3.py` | TRAINING | BGE base model | No runtime load here | `models/reranker` | No | Read-only |
| `scripts/submission/build_legal_ir_ensemble.py` | SUBMISSION | BGE cached rankings | Submission path | Cached BGE input | No | Read-only |
| `scripts/submission/supervised_legal_qa_third_parent.py` | SUBMISSION | Dense/BGE pools | Submission path | Cached BGE input | No | Read-only |

Additional source inventory results: 23 source/config files under `scripts/`,
`src/`, and `configs/` matched `bge`, `bge-reranker-v2-m3`, or
`models/reranker`; three test files matched the same terms. The test files were
not changed or run as a model smoke test.

The executable inventory outside the table is:

- Training (2): `scripts/training/build_task1_chunk_training_v3.py`,
  `scripts/training/finetune_task1_bge_reranker_v3.py`.
- Inference/analysis (16):
  `scripts/analysis/task1_direct_document_neural_signal_top5.py`,
  `scripts/analysis/task1_evaluate_full_document_legal_field_retrieval.py`,
  `scripts/analysis/task1_pairwise_document_preference_top5.py`,
  `scripts/analysis/workflow_a/step2_k77_oof_policy_realizability.py`,
  `scripts/analysis/workflow_c/workflow_c_c0a_oracle_opportunity_anatomy.py`,
  `scripts/beam/beam_task1_v1a_real.py`,
  `scripts/beam/beam_task1_v2_fold0.py`,
  `scripts/beam/beam_task1_v3_frozen_features.py`,
  `scripts/beam/beam_task1_v3_prepare_cpu.py`,
  `scripts/beam/beam_task1_v3_public_adaptive_k200.py`,
  `scripts/beam/beam_task1_v3_public_prepare_cpu.py`,
  `scripts/beam/task1_v2/build_v2_docgroups.py`,
  `scripts/beam/task1_v3_residual/score_frozen_features.py`,
  `scripts/evaluation/ablate_legal_ir_lexical.py`,
  `scripts/evaluation/aggregate_task1_chunks.py`,
  `scripts/evaluation/tune_legal_ir_document_aggregation.py`.
- Submission (2): `scripts/submission/build_legal_ir_ensemble.py`,
  `scripts/submission/supervised_legal_qa_third_parent.py`.

Category counts for the scoped inventory (categories can overlap): runtime
critical 6, configuration 5, training 2, inference/analysis 16, submission 2,
historical metadata/report matches 70, and active Qwen-boundary suspects 3.

## Minimal changes made

Only one file was modified:

```text
download_models.py
  added ModelSpec("bge_reranker", "BAAI/bge-reranker-v2-m3", "models/reranker")
  added the explicit task1-bge profile
```

No Qwen entry, Qwen artifact, scientific result, scorer, prompt, or historical
report was changed.

## Smoke test

Commands run:

```text
python -m py_compile download_models.py
python -c "import download_models as d; ..."
python -c "import sys; sys.path.insert(0, 'src'); from udsc2026.infrastructure.reranker.client import CrossEncoderClient; ..."
python -c "from huggingface_hub import HfApi; HfApi().model_info('BAAI/bge-reranker-v2-m3')"
```

Results:

- downloader syntax: PASS
- downloader BGE spec/profile: PASS
- generic BGE runtime import and lazy-client construction: PASS
- Hub model-id resolution: PASS
- local destination `models/reranker`: PRESENT and loadable by the end of the
  audit (a complete local weight file became available during the audit; this
  task did not invoke the downloader)
- model loaded: PASS on CPU, from `models/reranker`
- tiny inference: PASS, two query-document pairs
- finite score: YES; scores were `0.29098397493362427` and
  `0.19312091171741486`

The standard runtime import path was not changed. This was only a two-pair CPU
smoke test; no full inference or evaluation was run.

## Deferred cleanup

These are detected Qwen/BGE boundary risks and were deliberately not changed:

- `configs/base.yaml` defaults to `./models/qwen3-vl-reranker-2b`.
- `src/udsc2026/infrastructure/reranker/config.py` defaults to the Qwen path.
- `scripts/evaluation/benchmark_reranker.py` defaults to the Qwen path.
- `models/download_manifest.json` is the existing Qwen-inclusive manifest and
  was preserved.

The scoped read-only scan found 70 metadata/report files under
`reports/task1/` and `artifacts/task1/` containing BGE-related terms. Those
historical artifacts were not rewritten, deleted, or treated as active runtime
configuration.
