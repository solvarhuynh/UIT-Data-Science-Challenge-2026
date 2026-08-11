# TV2 Task1 LegalIR — Kaggle Runbook

> **DEFERRED for the current stage:** use
> [`tv2_task1_kaggle_p12_p13.md`](tv2_task1_kaggle_p12_p13.md) for the active
> P12/P13 GPU run. The public-inference and submission sections below are not
> authorized in this stage and must not be run.

Notebook-style runbook cho Task1 trên Kaggle. [`tv2_setup.md`](tv2_setup.md)
là hướng dẫn GPU/general; file này là quy trình từng cell, có checkpoint và
prerequisite gate để không chạy P13 khi candidate coverage chưa đủ.

## Trạng thái và nguyên tắc

Task1 production hiện tại:

| Role | Repository | Local path | Status |
|---|---|---|---|
| Embedding | `huyydangg/DEk21_hcmute_embedding_v2` | `models/dek21-v2` | PROMOTED |
| Reranker | `BAAI/bge-reranker-v2-m3` | `models/reranker` | PROMOTED |


Không rebuild embedding/FAISS nếu manifest còn hợp lệ. Không ghi vào `/kaggle/input`; mọi index mới, manifest cập nhật, checkpoint và report phải đi vào `/kaggle/working`.

## Datasets cần Add Data

Kaggle path names không được hard-code vì mỗi Dataset slug có thể khác nhau.
Thay các biến placeholder sau khi đọc cell 0:

| Biến | Nội dung cần chuẩn bị |
|---|---|
| `<KAGGLE_PROCESSED_V3>` | `data/processed_v3/documents`, `chunks`, `parents`, `benchmarks`, `metadata` |
| `<KAGGLE_LEGALIR_RAW>` | `data/raw/btc/LegalIR/train.json`, `public-official.json`, `selected-contexts/` |
| `<KAGGLE_VECTOR_STORE>` | `data/vector_store/faiss/`, `data/vector_store/bm25/` nếu dùng lại |
| `<KAGGLE_BASELINE_MODELS>` | `models/dek21-v2/`, `models/reranker/` |
| `<KAGGLE_CACHED_ARTIFACTS>` | dense candidates, component rankings, OOF/prediction caches |

Ưu tiên Add Data vector store và cached artifacts đã tạo sẵn. Không commit
weights vào Git. `scripts/package_kaggle.py` thuộc TV3/Task2; nếu cần source
bundle offline cho Task1, dùng packer riêng:

```powershell
python scripts/package_kaggle_task1.py --output artifacts/task1/kaggle/task1_code_bundle.zip
```

Bundle chỉ chứa code/config/docs, không chứa data, artifacts hoặc model weights;
cell 1 vẫn ưu tiên clone đúng commit.

## CELL 0 — Check GPU

```python
!find /kaggle/input -maxdepth 3 -type d | sort | head -200
!nvidia-smi
!python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA is required for a planned GPU run'"
```

Nếu CUDA fail, chỉ chạy các cell kiểm tra file/hash; dừng trước embedding,
reranking, indexing và P13. Không silent CPU fallback.

## CELL 1 — Clone repo và pin commit

Trên local, lấy commit trước khi mở Kaggle:

```powershell
git branch --show-current
git rev-parse HEAD
```

Ghi commit đó vào `PINNED_COMMIT`; không ghi GitHub token vào notebook.

```python
import os
import subprocess
from pathlib import Path

REPO_URL = os.environ.get("UDSC_REPO_URL", "<REPO_URL>")
PINNED_COMMIT = "<PINNED_COMMIT>"
REPO_ROOT = Path("/kaggle/working/udsc2026")
if not REPO_ROOT.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO_ROOT)], check=True)
subprocess.run(["git", "checkout", PINNED_COMMIT], cwd=REPO_ROOT, check=True)
print(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip())
```

Nếu Internet OFF, Add Data một code bundle vào `<KAGGLE_CODE_BUNDLE>` rồi copy
vào `/kaggle/working/udsc2026`; ghi checksum bundle nếu không có `.git`.

## CELL 2 — Install dependencies

Giữ CUDA-enabled PyTorch của Kaggle. Không cài Torch CPU và không cài
`flash-attn` bắt buộc:

```python
%cd /kaggle/working/udsc2026
!python -m pip install -q --no-deps -e .
!python -m pip install -q \
  "accelerate>=1.1,<2" "faiss-cpu>=1.8,<2" "huggingface-hub>=1.3,<2" \
  "numpy>=1.26,<2" "pyvi>=0.1,<1" "PyYAML>=6" \
  "sentence-transformers==5.4.1" "transformers==5.0.0" "tqdm>=4.66,<5"
!python -c "import faiss, torch, transformers; print(faiss.__version__, torch.__version__, transformers.__version__)"
```

Nếu pip định thay CUDA Torch bằng bản CPU, dừng và sửa environment.

## CELL 3 — Create working layout

`/kaggle/input` read-only; symlink dataset chỉ đọc, output ghi ở working:

```python
from pathlib import Path

WORK = Path("/kaggle/working/udsc2026")
DATA, MODELS, ARTIFACTS = WORK / "data", WORK / "models", WORK / "artifacts"
for path in (DATA, MODELS, ARTIFACTS):
    path.mkdir(parents=True, exist_ok=True)

def link_readonly(source: str, destination: Path) -> None:
    source_path = Path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() and not destination.is_symlink():
        destination.symlink_to(source_path, target_is_directory=source_path.is_dir())

link_readonly("<KAGGLE_PROCESSED_V3>", DATA / "processed_v3")
link_readonly("<KAGGLE_LEGALIR_RAW>", DATA / "raw" / "btc")
link_readonly("<KAGGLE_VECTOR_STORE>", DATA / "vector_store")
link_readonly("<KAGGLE_BASELINE_MODELS>", MODELS)
if Path("<KAGGLE_CACHED_ARTIFACTS>").exists():
    link_readonly("<KAGGLE_CACHED_ARTIFACTS>", ARTIFACTS / "imported")
```

## CELL 4 — Verify data structure

```python
required = [
    DATA / "raw/btc/LegalIR/train.json",
    DATA / "raw/btc/LegalIR/public-official.json",
    DATA / "raw/btc/LegalIR/selected-contexts",
    DATA / "processed_v3/documents",
    DATA / "processed_v3/parents",
    DATA / "processed_v3/chunks",
    DATA / "processed_v3/metadata",
]
missing = [str(path) for path in required if not path.exists()]
assert not missing, f"missing required data: {missing}"
assert list((DATA / "processed_v3/documents").glob("*.json"))
assert list((DATA / "processed_v3/parents").glob("*.jsonl"))
assert list((DATA / "processed_v3/chunks").glob("*.jsonl"))
print("data structure OK")
```

## CELL 5 — Verify hashes và preflight

```python
%cd /kaggle/working/udsc2026
!python scripts/gpu/preflight.py --processed-root data/processed_v3
```

Preflight phải pass package, CUDA, corpus audit và hai model required. Đối
chiếu index manifest trước reuse:

```python
import json
index_manifests = sorted((DATA / "vector_store/faiss").rglob("manifest.json"))
processing = json.loads((DATA / "processed_v3/metadata/processing_manifest.json").read_text())
print("processed corpus hash:", processing.get("processed_corpus_tree_hash"))
for path in index_manifests:
    manifest = json.loads(path.read_text())
    print(path, {key: manifest.get(key) for key in (
        "corpus_hash", "model_id", "model_revision", "dimension",
        "representation_version", "collection_name")})
```

Reuse chỉ khi corpus hash, HCMUTE model revision/hash, dimension 768, collection
name và representation version khớp. Nếu mismatch: STOP, không search stale
index; rebuild vào working bằng config/index path riêng.

## CELL 6 — Baseline smoke 20 questions

```python
!python scripts/task1/run_legal_ir_pipeline.py \
  --config configs/task1_baseline.yaml \
  --questions data/raw/btc/LegalIR/train.json \
  --processed-root data/processed_v3 \
  --contexts-dir data/raw/btc/LegalIR/selected-contexts \
  --vector-store-root data/vector_store \
  --artifacts-root artifacts/task1/kaggle_smoke20 \
  --stage all --max-questions 20 --resume
```

Class: medium. Kiểm tra `run_manifest.json`, stage status, output IDs và model
paths. Nếu smoke fail, dừng trước full candidate generation.

## CELL 7 — Optional selected-model smoke

Current selected stack là HCMUTE + BGE. Không tải Qwen3 hoặc BGE-M3 cho cell
này. Nếu có smoke artifacts hợp lệ, dùng CLI reranker thật:

```python
!python scripts/evaluation/benchmark_reranker.py \
  --benchmark <SMOKE_BENCHMARK_JSONL> \
  --candidates <SMOKE_CANDIDATES_JSONL> \
  --model models/reranker --device cuda --batch-size 4 --max-length 512 \
  --candidate-k 20 --top-n 5 --fp16 \
  --output-dir artifacts/task1/kaggle_reranker_smoke
```

Nếu thiếu smoke candidate/benchmark, ghi `SKIPPED`; không rebuild corpus chỉ
cho cell này.

## CELL 8 — Prepare full 7,000-query P13 coverage

P13 dry-run trước đó chỉ phủ khoảng 500/7.000 query. Không được đi thẳng vào
fine-tuning.

### 8.1 Strict five-fold definitions

Reuse `artifacts/task1/evaluation/strict_cv_v2/folds.json` nếu schema là
`legal-ir-strict-cv-v2`, input hash đúng `train.json`, có 5 fold và 7.000 IDs.
Chỉ tạo lại khi manifest không khớp:

```python
!python scripts/evaluation/build_strict_legal_ir_cv.py \
  --input data/raw/btc/LegalIR/train.json \
  --output-dir artifacts/task1/evaluation/strict_cv_v2 \
  --folds 5 --seed 2026
```

Class: light CPU. Output: `folds.json`, `fold_stats.json`, `manifest.json`.

### 8.2 Generate/reuse full HCMUTE dense candidates

Ưu tiên reuse FAISS đã pass hash gate. Nếu chưa có cache đầy đủ, CLI thực tế
cho organizer train là:

```python
!python scripts/evaluation/generate_dense_candidates.py \
  --legal-ir-train data/raw/btc/LegalIR/train.json \
  --config-env gpu --candidate-k 200 --query-batch-size 64 \
  --output artifacts/task1/kaggle_train7000_dense200_predictions.jsonl \
  --manifest artifacts/task1/kaggle_train7000_dense200_manifest.json
```

Class: heavy GPU, input HCMUTE + FAISS, output JSONL và manifest. `200` là
fallback dense đã validated; nếu config/experiment mới chọn depth khác, dùng
giá trị đó và ghi lại. Reuse output khi manifest hash còn khớp; không rerun full
embedding chỉ vì session chết.

### 8.3 Report exact candidate coverage

```python
!python scripts/evaluation/check_task1_p13_prerequisites.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --candidates artifacts/task1/kaggle_train7000_dense200_predictions.jsonl \
  --negatives-dir artifacts/task1/training/negatives_full \
  --output artifacts/task1/p13_prerequisite_before_mining.json
```

Candidate report phải có `total_queries=7000`, `candidate_query_ids=7000`,
`queries_missing_candidates=[]` và `duplicate_candidate_query_ids=[]`.

### 8.4 Mine P12 negatives on the full cache

```python
!python scripts/training/mine_task1_negatives.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --rankings hcmute=artifacts/task1/kaggle_train7000_dense200_predictions.jsonl \
  --output-dir artifacts/task1/training/negatives_full
```

Class: medium CPU sau khi cache tồn tại. Output: năm fold JSONL,
`mining_manifest.json`, `statistics.md`; không load model/train. Chỉ ablation B
mới chạy thêm output directory với `--max-easy-per-query 5`.

### 8.5 Required prerequisite result

```python
!python scripts/evaluation/check_task1_p13_prerequisites.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --candidates artifacts/task1/kaggle_train7000_dense200_predictions.jsonl \
  --negatives-dir artifacts/task1/training/negatives_full \
  --output artifacts/task1/p13_prerequisite_complete.json
```

P13 chỉ được chạy nếu report có:

```text
coverage_status = complete
total_queries = 7000
candidate_query_ids = 7000
queries_missing_candidates = []
queries_missing_negatives = []
fold_leakage_query_ids = []
```

Nếu incomplete: STOP. Không train, không promote.

### 8.6 P13 dry-run rồi mới train

```python
!python scripts/training/finetune_task1_bge_reranker.py --help
!python scripts/training/finetune_task1_bge_reranker.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --negatives-dir artifacts/task1/training/negatives_full \
  --candidates artifacts/task1/kaggle_train7000_dense200_predictions.jsonl \
  --base-model models/reranker --output-dir artifacts/task1/p13_dryrun \
  --ablation semi-hard-plus-hard --candidate-depth 200 \
  --evidence-limit 2 --device cuda --dry-run
```

Dry-run phải ghi `expected_query_count=7000`, `evaluated_query_count=0` và
không tạo checkpoint. Chỉ sau đó mới chạy full P13:

```python
!python scripts/training/finetune_task1_bge_reranker.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --negatives-dir artifacts/task1/training/negatives_full \
  --candidates artifacts/task1/kaggle_train7000_dense200_predictions.jsonl \
  --base-model models/reranker \
  --ablation semi-hard-plus-hard --candidate-depth 200 --evidence-limit 2 \
  --learning-rate 2e-5 --epochs 2 --batch-size 4 \
  --gradient-accumulation 4 --max-length 512 --warmup-ratio 0.1 \
  --seed 2026 --device cuda \
  --output-dir artifacts/task1/models/bge_reranker_finetune
```

Class: heavy GPU/Kaggle. Checkpoint chỉ ghi dưới output mới, không overwrite
`models/reranker`. Chạy A/C/D/E vào các output directory riêng; ablation
`easy-random` cần P12 artifact riêng có `--max-easy-per-query`.

## CELL 9 — Build missing parent/Qwen/BGE-M3 indexes only when selected

Baseline hiện không cần index mới. P4/P9/P11 là experimental/deferred. Nếu
future experiment được chọn, copy config với FAISS/BM25 path riêng dưới
`/kaggle/working`, rồi dùng CLI hiện có:

```python
!python scripts/data_prep/index_chunks.py \
  --chunks-dir data/processed_v3/chunks --config-env gpu \
  --vector-db-type faiss --skip-bm25 \
  --error-report artifacts/task1/isolated_index_errors.json --force
```

Class: heavy GPU/disk. Không dùng `--force` trên Add Data path. Chỉ build full
1.27M child index nếu OOF chứng minh cần; mọi index experimental phải có path,
dimension và manifest riêng.

## CELL 10 — Public 1,000-question run

```python
from udsc2026.evaluation.legal_ir import load_legal_ir_question_ids
public_ids = load_legal_ir_question_ids("data/raw/btc/LegalIR/public-official.json")
assert len(public_ids) == 1000 and len(public_ids) == len(set(public_ids))
print("public IDs:", len(public_ids))
```

Dùng operational `task1_baseline.yaml`; `task1_top1.yaml` là model-selection
record, còn baseline config chứa candidate/ensemble settings cho runner:

```python
!python scripts/task1/run_legal_ir_pipeline.py \
  --config configs/task1_baseline.yaml \
  --questions data/raw/btc/LegalIR/public-official.json \
  --processed-root data/processed_v3 \
  --contexts-dir data/raw/btc/LegalIR/selected-contexts \
  --vector-store-root data/vector_store \
  --artifacts-root artifacts/task1/public1000 --stage all --resume
```

Class: medium/heavy. Reuse valid caches; không truncate top 5 trước reranking.

## CELL 11 — Build `submission.zip`

```python
!python scripts/submission/write_legal_ir_submission.py \
  --input artifacts/task1/public1000/ensemble/predictions.json \
  --questions data/raw/btc/LegalIR/public-official.json \
  --corpus-manifest artifacts/task1/public1000/corpus_document_ids.json \
  --output /kaggle/working/final_task1/submission.zip
```

## CELL 12 — Validate submission

```python
!python scripts/submission/validate_legal_ir_submission.py \
  --input /kaggle/working/final_task1/submission.zip \
  --questions data/raw/btc/LegalIR/public-official.json \
  --corpus-manifest artifacts/task1/public1000/corpus_document_ids.json
```

Expected: `status=valid`, `question_count=1000`, answers 1–5 distinct docs,
and every doc ID exists in the corpus manifest.

## CELL 13 — Inspect ZIP

```python
import zipfile
submission = "/kaggle/working/final_task1/submission.zip"
with zipfile.ZipFile(submission) as archive:
    print(archive.namelist())
    assert archive.namelist() == ["submission.json"]
```

## CELL 14 — SHA256 và save artifacts

```python
from pathlib import Path
import shutil

final_dir = Path("/kaggle/working/final_task1")
for source in (Path("artifacts/task1/public1000"),
               Path("artifacts/task1/p13_prerequisite_complete.json")):
    destination = final_dir / source.name
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    elif source.is_file():
        shutil.copy2(source, destination)
```

```python
!sha256sum /kaggle/working/final_task1/submission.zip
!find /kaggle/working/final_task1 -maxdepth 3 -type f | sort
```

Lưu submission, pipeline/candidate/P12/P13 manifests, model revisions, config,
commit và hashes. Submission ZIP không chứa model weights.

## Session chết hoặc bị interrupt

Reattach cached candidate JSONL và manifest, validated FAISS/BM25, P12 fold
JSONL và completed public artifacts. Recreate working symlinks, chạy preflight
và hash checks, rồi dùng `--resume`. Chỉ reuse checkpoint khi training manifest,
dataset hash, base revision và commit khớp. Không rerun full embedding khi cache
manifest còn hợp lệ.

Với GPU 16 GB, bắt đầu batch conservative. Khi OOM, giảm embedding/reranker
batch; không truncate candidate pool xuống top 5 trước reranking. BGE dùng
FP16/BF16 theo hardware, nhưng full run phải fail rõ nếu CUDA mất.

## Local pre-push check

```powershell
git status
git diff --check
python -m compileall -q src scripts
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier unit
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier data
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-all.ps1 -SkipFrontend -SkipCompose
git branch --show-current
git rev-parse HEAD
git add .
git commit -m "docs: add Task1 Kaggle runbook"
git push
```

Local checks không yêu cầu full embedding/model test. Ghi `git rev-parse HEAD`
vào Kaggle manifest và dùng đúng commit đó cho run.
