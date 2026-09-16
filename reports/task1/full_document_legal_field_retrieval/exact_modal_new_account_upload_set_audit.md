# AUDIT_EXACT_MODAL_NEW_ACCOUNT_UPLOAD_SET

Status: `BLOCKED_DEPENDENCY_AMBIGUOUS`

The local Modal CLI and Python package are absent, so current CLI syntax/account inspection cannot be verified. No Volume creation, upload, Modal launch, GPU, or inference was performed.

## Runtime findings

- Fresh Volume required by code: `udsc-p13`, mounted at `/workspace/p13`.
- `create_if_missing=False`; the Volume must already exist in the active account.
- `PROFILE = "hanky77k1"` is declared in the historical module but unused by execution.
- Qwen is downloaded at runtime by `download_locked_snapshot()` from public Hugging Face revision `4bd860ac4f15ad1897a214615cccc700f8f71818`; no HF secret is referenced. Cache is `/tmp/hf-cache`; model weights are not Volume uploads.
- `train.json` is required at `/workspace/p13/runtime/data/raw/btc/LegalIR/train.json`.
- Canonical chunks are required at `/workspace/p13/runtime/data/processed_v3/chunks/`.
- Production worklists, universe manifest, historical parity sample, and current canary are image-embedded under `/opt/...` by `add_local_dir/add_local_file`; they are not Volume uploads.
- Historical `run_state.json` is read by the new runner's contract check and must be uploaded at `/workspace/p13/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json`.

## Upload accounting

Local source `data/raw/btc/LegalIR/train.json`: 1,526,741 bytes; SHA256 `c39cde9e74977e350f1456e7d487aafe67d2bcbaa4fa26fcabd557fe635635b`.

Local source run state: `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json`; 1,722 bytes; SHA256 `e949e84ac1902e16c1f2295d50014c613ce2466b166ea6819259039c89852f4d`.

- Canary parity/current sample chunk set: 214 documents, 89,960,215 bytes (0.083782 GiB), plus train and run_state.
- Full production worklist chunk set: 8,261 documents, 1,802,587,393 bytes (1.678790 GiB), plus train and run_state.
- Additional production data beyond the canary set: 8,047 documents, 1,712,627,178 bytes (1.595008 GiB).
- All 8,532 canonical chunks are 1,809,161,876 bytes; this is safer than a partial set only if the production worklists are changed to encounter all documents.

## Important source consistency

The production runner hard-codes the embedded manifest cardinality `598192` q-docs and `1794571` units. The earlier new-only forensic universe was `598071`; these are distinct universes. The existing production worklists/manifest must be treated as authoritative for the future production runner and must not be silently replaced.

## Exact mappings

`D:/udsc2026/data/raw/btc/LegalIR/train.json` → `/runtime/data/raw/btc/LegalIR/train.json` → `/workspace/p13/runtime/data/raw/btc/LegalIR/train.json`

`D:/udsc2026/data/processed_v3/chunks/` → `/runtime/data/processed_v3/chunks/` → `/workspace/p13/runtime/data/processed_v3/chunks/`

`D:/udsc2026/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json` → `/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json` → `/workspace/p13/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/run_state.json`

## Image-embedded, no Volume upload

`full_doc_top200_qwen_worklists/*`, `full_doc_top200_qwen_future_universe_manifest.json`, `full_doc_top200_qwen_historical_parity_256.jsonl`, and `full_doc_top200_qwen_current_canary_16.jsonl` are added to `/opt/...` by the image definition.

## Required verification before any canary

Use the installed/current Modal CLI help to verify account/profile, `volume create`, `volume put`, and `volume ls` syntax. That verification could not be performed because neither `modal` nor the Python `modal` module is installed locally. Do not execute guessed commands.
