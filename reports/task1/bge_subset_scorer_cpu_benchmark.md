# CPU-only BGE subset scorer benchmark

Date: 2026-09-18

Status: bounded smoke and bounded benchmark passed. The full worklist was not
scored.

## Scope and safety gates

Worklist:

`artifacts/task1/qwen_to_bge_minimal/top5_missing_bge.jsonl`

The worklist contains 25,902 q-doc pairs, 5,594 queries, and 6,069 unique
documents. This run scored only a 10-row smoke and a 100-row benchmark. No
GPU, Modal, Qwen inference, labels, Fold0, submission, or full 25,902-pair
run was used.

## Scoring contract

- Model: `BAAI/bge-reranker-v2-m3`
- Local model path: `models/reranker`
- Local runtime branch: Sentence-Transformers `CrossEncoder` on CPU
- Expected Hub revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- Local revision metadata: proven by
  `models/reranker/.cache/huggingface/trees/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e.json`
- Local config: `xlm-roberta`, one label inferred from `id2label`, float32
- Pair input: canonical question and raw selected chunk text
- Maximum length: 512 with truncation
- Selector: `true_s2_bm25_within_document_v2`, up to three chunks
- Document aggregation: `MAX`
- Activation: CrossEncoder's default single-label sigmoid; no manual sigmoid
- Dual Step4 FT+Base score: unavailable locally (`NO`)

The local `config.json` names the upstream encoder as `BAAI/bge-m3`, while the
local README and the active Task1 contract identify the reranker as
`BAAI/bge-reranker-v2-m3`. The architecture, one-label head, BGE README, and
revision tree metadata all agree with the BGE reranker contract.

## Results

### Smoke: 10 worklist rows

- Result: `PASS`
- Chunk pairs scored: 30
- Unique documents: 10
- Preparation: 6.7407 s
- Model load: 22.8912 s
- Inference: 8.8575 s
- Total wall: 38.5104 s
- Worklist throughput: 0.2597 rows/s
- Chunk throughput: 3.3870 pairs/s
- Peak observed RSS: 1,973.3 MiB

### Benchmark: 100 worklist rows

- Result: `PASS`
- Chunk pairs scored: 300
- Unique documents: 99
- Preparation: 28.6061 s
- Model load: 21.6448 s
- Inference: 78.5046 s
- Total wall: 128.7815 s
- Worklist throughput: 0.7765 rows/s
- Chunk throughput: 3.8214 pairs/s
- Seconds per worklist row: 1.2878
- Peak observed RSS: 2,084.8 MiB
- BGE document-score range: 0.0015913 to 0.9355426

Both bounded runs resolved every identity, produced finite scores, produced no
duplicate q-doc identities, and emitted the schema:

`query_id`, `document_id`, `bge_score`, `model_id`, `model_revision`,
`aggregation`, `selector`, `selected_chunk_ids`, `chunk_scores`.

The output contains no `teacher_qwen_score` field.

## Full-worklist estimate

The 25,902-row worklist implies 77,706 chunk pairs when the observed three
selected chunks per row are maintained. Estimates use the 100-row benchmark;
model startup is counted once and output-writing overhead is not separately
measured.

- LOW / optimistic: scale preparation by 6,069 unique documents and inference
  by 77,706 chunks — 6.14 h.
- CENTRAL: same inference scaling, but scale preparation by worklist rows —
  7.71 h.
- HIGH / conservative: scale observed total wall time linearly by 25,902 / 100
  — 9.27 h.

Even the optimistic estimate is above six hours. Recommendation:
`LOCAL_CPU_IMPRACTICAL` for the complete 25,902-pair worklist without another
bounded optimization or a GPU path.

The estimate is not a parity result. Exact numerical parity against the
historical persisted BGE artifact remains `SKIP_PARITY_UNPROVEN` because that
artifact does not establish the exact immutable model-weight provenance for
every persisted score. No labels were used.

## Files and commands

Scorer:

`scripts/evaluation/score_task1_bge_subset.py`

The local config compatibility check accepts the valid one-label BGE config
where `num_labels` is omitted but `id2label` contains one label. It still
fails hard for Qwen paths and non-BGE model configs.

Smoke output and metrics:

`artifacts/task1/qwen_to_bge_minimal/bge_subset_smoke.jsonl`

`artifacts/task1/qwen_to_bge_minimal/bge_subset_smoke_metrics.json`

Benchmark output and metrics:

`artifacts/task1/qwen_to_bge_minimal/bge_subset_benchmark_100.jsonl`

`artifacts/task1/qwen_to_bge_minimal/bge_subset_benchmark_100_metrics.json`

Validation: `python -m py_compile scripts/evaluation/score_task1_bge_subset.py`
passed.
