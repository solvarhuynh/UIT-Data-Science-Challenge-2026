# BGE TOP5 Modal storage readiness — `nan928904`

Date: 2026-09-18

Status: `NETWORK/STORAGE READY`. No Modal app run, GPU allocation, model
inference, canary, or production execution occurred.

## Staging

The exact worklist was resolved through the same `<document_id>.jsonl`
canonical mapping used by the CPU scorer and GPU runner.

- Worklist pairs: 25,902
- Queries: 5,594
- Unique documents: 6,069
- Required chunk files: 6,069
- Missing documents: 0
- Extra chunk files: 0
- Staging bytes: 1,569,659,698
- Link strategy: NTFS hardlinks for the chunk subset; no symlinks
- Worklist SHA256: `e822790b92a8448fe7b97625c2da10de6b3cf7255c395f494c2bf13acc7b076b`
- Train SHA256: `c39cde9e74977e350f1456e7d487aafe67d2bcba4fa26fcabd557fe635635b7`

Staging root:

`artifacts/task1/modal_stage_bge_top5/`

Manifest:

`artifacts/task1/modal_stage_bge_top5/staging_manifest.json`

The manifest and local staging verification both report `READY_FOR_UPLOAD`,
with train hash, byte count, exact document coverage, and zero missing/extra
files passing.

## Modal volumes and paths

Profile used for every Modal CLI operation: `nan928904`.

Created exactly these previously-absent volumes without resetting anything:

- Data: `udsc-p13`
- Models: `udsc-task1-modal`

Data upload used `volume put` from the staged `runtime` directory to remote
`/runtime`. Remote root contains `runtime` only, and verified paths are:

- `/runtime/data/raw/btc/LegalIR/train.json`
- `/runtime/data/processed_v3/chunks/` with 6,069 listed JSONL files

Model upload used the exact local directories and verified remote roots:

- Base: `/models/reranker`
- FT: `/bge_ft/bge_m3_finetuned`

The model volume root contains `models` and `bge_ft`; `/data` was not created
inside the volume. This is correct because the runner mounts the volume at
`/data`.

Verified remote model files:

- `/models/reranker/config.json` — 795 B
- `/models/reranker/model.safetensors` — 2.1 GiB
- `/models/reranker/.cache/huggingface/trees/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e.json` — 2.1 KiB
- `/bge_ft/bge_m3_finetuned/config.json` — 841 B
- `/bge_ft/bge_m3_finetuned/model.safetensors` — 2.1 GiB

Base revision resolves locally to
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, and the corresponding hidden
revision metadata path is present remotely.

The Modal CLI exposes rounded remote sizes and no server-side SHA operation was
available for the 2.1 GiB weight. Therefore:

`REMOTE_WEIGHT_HASH_RECHECK_NOT_AVAILABLE`

The remote FT weight was uploaded from the locally verified exact file:

`68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c`

The local FT config hash is:

`16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b`

## Runner static gate

`scripts/modal/task1_bge_subset_gpu.py` was checked without running Modal:

- `VOLUME_NAME = udsc-p13`
- `MODEL_VOLUME_NAME = udsc-task1-modal`
- train and chunk paths match the uploaded `/runtime` tree
- Base path `/data/models/reranker`
- FT path `/data/bge_ft/bge_m3_finetuned`
- Base revision gate is exact
- FT SHA256 gates are pinned
- batch 16 is the recommended initial setting
- canary limit is exactly 256 q-docs
- checkpoint interval is 64 q-docs
- GPU declaration is A10

No scoring semantics or checkpoint contract was changed in this task.

## Commands used

Only the allowed Modal storage operations were used: `volume create`,
`volume put`, and `volume ls`, all with `--profile nan928904`. No `modal run`
or app/function execution was called.

The canary command is prepared but intentionally not executed:

```powershell
& '.venv\Scripts\modal.exe' run --profile nan928904 scripts/modal/task1_bge_subset_gpu.py `
  --worklist artifacts/task1/qwen_to_bge_minimal/top5_missing_bge.jsonl `
  --mode canary `
  --limit 256 `
  --batch-size 16 `
  --checkpoint-every 64
```

Production remains unauthorized until the canary passes.

