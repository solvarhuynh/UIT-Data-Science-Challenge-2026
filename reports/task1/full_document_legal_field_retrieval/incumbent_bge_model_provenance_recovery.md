# Incumbent BGE model provenance recovery

## Status

`BGE_MODEL_PROVENANCE_IRRECOVERABLE`

## Expected incumbent model contract

| Field | Evidence | Classification |
|---|---|---|
| Architecture | `XLMRobertaForSequenceClassification`, one label | `RECOVERED_EXACTLY` |
| Model directory | `models/reranker` | `RECOVERED_EXACTLY` |
| Config hash | `13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a` | `RECOVERED_EXACTLY` |
| Weight hash | `model.safetensors`: `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286` | `RECOVERED_EXACTLY` |
| Tokenizer hashes | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` pinned by runner | `RECOVERED_EXACTLY` |
| Model ID | not recorded | `UNKNOWN` |
| Remote revision/commit | not recorded | `UNKNOWN` |
| Framework semantics | Transformers pair input, CUDA fp16, max length 512, batch 64 | `RECOVERED_EXACTLY` |

The required minimal parity file set is `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`.

## Local search

- **repository:** no hash-matching `models/reranker` files; the current directory is empty.
- **git history:** the runner was added in commit `e1e933d`; model bytes were not committed with it.
- **Git LFS:** no local LFS-tracked matching object/pointer found.
- **HuggingFace/Windows caches:** only the DEk21 embedding snapshot and unrelated Qwen models are present; none matches the required XLM-R hashes.
- **other artifacts/checkpoints:** manifests reference `models/reranker`, but contain no base-model bytes and no immutable external origin.

## Remote-origin evidence

`LOCAL_ONLY_UNKNOWN_ORIGIN`. The surviving runner is deliberately local-files-only and records expected file SHA256 values, but neither a public model identifier nor a revision. Remote recovery is not eligible because an arbitrary same-family checkpoint cannot establish byte identity.

## Persisted-score bypass

- **FULLDOC_TOP10 new q-docs:** 6,523
- **existing exact incumbent BGE scores:** 21
- **still require actual inference:** 6,502
- **broader matching historical cache:** not found.

## Producer provenance

- **historical scorer source SHA:** not recorded in the persisted F1–F4 BGE source manifest.
- **current reference source:** `scripts/beam/beam_task1_v3_public_adaptive_k200.py`, revision traceable to `e1e933d`.
- **status:** `SOURCE_UNKNOWN` for historical artifact production; source contract is only partially recoverable.

## Primary blocker

Exact incumbent model and tokenizer bytes cannot be recovered because the only surviving contract pins hashes but records neither local bytes nor an immutable remote origin.

## Safe next action

`CLOSE_HISTORICAL_BGE_SCORER_PATH_AND_DESIGN_ONE_PROSPECTIVE_COMMON_SCORER_EXPERIMENT`

Scientifically clean future options are limited to: (1) rescore both incumbent and new candidates prospectively with one fully reproducible scorer, or (2) separately authorize a deterministic retrieval-only final-top5 experiment. Neither was executed here.
