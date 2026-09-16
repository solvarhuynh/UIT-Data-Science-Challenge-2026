# Incumbent BGE scorer parity preflight

## Status

`BLOCKED_BGE_MODEL_PROVENANCE`

## Incumbent BGE contract

- **reference:** `scripts/beam/beam_task1_v3_public_adaptive_k200.py`
- **model:** local `models/reranker`; XLM-R sequence classification, one logit
- **required model files/hashes:** the reference script pins `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`.
- **query source:** canonical LegalIR questions.
- **chunk source:** canonical `data/vector_store/faiss/legal_chunks_dek21_v2_768/payloads.json`.
- **input formatting:** `(query, chunk text)` tokenizer pair input.
- **max length / batches:** 512 / 64 in the surviving reference runner.
- **chunk scoring:** single classification logit under CUDA float16 autocast.
- **document aggregation:** `MAX(bge_score)` over candidate document chunks.

## Model verification

`models/reranker` contains none of the pinned local model files. The surviving reference explicitly rejects any byte-nonidentical checkpoint. Therefore neither the model weights nor tokenizer bytes can be verified against the incumbent contract.

The active environment also reports `torch 2.13.0+cpu` and `cuda=False`, but device feasibility is secondary: byte-locked model provenance fails first.

## Parity set

Not constructed and not scored. Labels were not used. Building a parity set without the exact model cannot demonstrate incumbent score reproduction.

## FULLDOC_TOP10 workload

- **new q-docs:** 6,523
- **existing BGE reusable:** 21
- **need new BGE scoring:** 6,502
- **total query-chunk pairs:** not materialized, because the model provenance gate failed before candidate scoring.
- **CPU/CUDA estimates:** not valid without the exact model and its reference execution path.

## New scoring artifact

Not created. No new candidate received a BGE score.

## Safe next action

`DO_NOT_INTEGRATE_UNCOMPARABLE_SCORES`
