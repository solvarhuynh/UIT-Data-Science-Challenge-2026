# Task1 BGE bounded smoke

Result: `TASK1_REAL_RUNTIME_USES_BGE`

## Canonical entrypoint

The repository's actual reranking entrypoint is:

```text
scripts/evaluation/benchmark_reranker.py
```

It was invoked through its existing `build_parser()`/`run()` entrypoint with a
temporary bounded fixture containing five unlabeled public questions and their
existing cached candidate hits:

```text
--model D:\udsc2026\models\reranker
--device cpu
--batch-size 2
--max-length 128
--candidate-k 5
--top-n 3
```

The runner `scripts/task1/run_legal_ir_pipeline.py` is an orchestration and
manifest runner; its neural stages only reuse cached artifacts or report
missing stages. Therefore it is not the actual bounded scorer entrypoint.

## Config resolution

- Explicit config: `configs/task1_top1.yaml`
- Config resolution is explicit; `src/udsc2026/config.py` does not merge
  `configs/base.yaml` into an explicitly supplied path.
- Final reranker path: `./models/reranker`
- Absolute path used: `D:\udsc2026\models\reranker`
- Final model id declared by the Task1 config: `BAAI/bge-reranker-v2-m3`
- `configs/base.yaml` Qwen value does not override the explicit Task1 config.
- The `RerankerSettings` Qwen default is not used because the Task1 config
  supplies a model path after alias normalization.

## Minimal blocker fix

| File | Old | New | Why |
|---|---|---|---|
| `src/udsc2026/infrastructure/reranker/config.py` | Strict settings received Task1's `name` and `model_path` as unknown fields, so the explicit config could not instantiate. | Remove the domain-only `name` key and map `model_path` to `model_name_or_path` only when the latter is absent. | Honor `configs/task1_top1.yaml` without changing global Qwen defaults. |

No change was made to `configs/base.yaml`, the Qwen default field, the Qwen
branch, scorer behavior, labels, or historical artifacts.

## Local model provenance

- `models/reranker/config.json` architecture:
  `XLMRobertaForSequenceClassification`
- model type: `xlm-roberta`
- local `_name_or_path`: `BAAI/bge-m3`
- local README explicitly documents `BAAI/bge-reranker-v2-m3` and
  `FlagReranker` usage.
- cached Hugging Face tree metadata:
  `models/reranker/.cache/huggingface/trees/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e.json`
- local revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- local `model.safetensors` SHA256:
  `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`

The local revision is therefore proven, not `LOCAL_REVISION_UNPROVEN`.

## Runtime result

- queries: 5
- candidate pairs scored: 25 (5 candidates/query)
- runtime branch: Sentence-Transformers `CrossEncoder` / BGE
- Qwen code path entered: NO
- score minimum: `0.8818067312240601`
- score maximum: `0.9999357461929321`
- all scores finite: YES
- output schema: PASS; five `PredictionSample` records, three reranked hits/query
- labels used: NO
- full inference: NO
- GPU/Modal: NO

## Files modified

- `src/udsc2026/infrastructure/reranker/config.py`
- `reports/task1/bge_task1_bounded_smoke.md`

