# TV5 — chạy Task 1 P13 trên Kaggle khi không còn máy RTX

Máy local không có CUDA chỉ dùng để chuẩn bị dữ liệu, kiểm tra và tải kết quả.
Không chạy full fine-tune năm fold trên CPU. GPU miễn phí của Kaggle là đường chạy
chính; kết quả leaderboard 0.9309 vẫn được giữ nguyên và không bị ghi đè.

Kaggle tính quota GPU theo tài khoản và hiển thị quota còn lại trong Notebook
Settings/Session Management. Chạy smoke trước, sau đó chạy tuần tự từng fold và
lưu checkpoint ngay khi mỗi fold hoàn tất.

## 1. File cần upload

Tạo một **private Kaggle Dataset** từ file sau:

```text
artifacts/task1/task1_p13_kaggle_upload.tar.zst
```

Kích thước: `1.763.448.793` bytes (1,642 GiB). SHA-256:
`21dda96ecab20686ef81b85d1f6813d8c6fa0d80f892e43c61604a3f90469921`.
So hash sau khi upload; sai một ký tự thì không giải nén/chạy tiếp.

Gói này đã chứa:

- code P13 hiện tại trong `src/` và `scripts/`;
- LegalIR train/public;
- `BAAI/bge-reranker-v2-m3` ở `models/reranker`;
- strict CV, candidate depth 500, hard negatives năm fold;
- corpus document manifest dùng để kiểm tra submission.

Không public Dataset vì nó chứa dữ liệu cuộc thi. Với Kaggle image hiện tại dùng
PyTorch CUDA 12.8, chọn **GPU T4 x2**: T4 (`sm_75`) được wheel hiện tại hỗ trợ.
Không chọn P100 (`sm_60`) nếu PyTorch cảnh báo build chỉ hỗ trợ `sm_70` trở lên;
`CUDA available=True` trong trường hợp đó vẫn chưa đủ để chạy model. Bật Internet
để cài đúng dependency, rồi Add Input Dataset vừa tạo. Pipeline hiện dùng một T4;
GPU thứ hai để trống nhưng đây vẫn là đường tương thích và ít rủi ro hơn thay cả
PyTorch trong phiên Kaggle.

## 2. Cell giải nén và kiểm tra GPU

Kaggle image có thể không cài executable `zstd`, vì vậy dùng Python
`zstandard` để giải nén streaming thay vì gọi `tar -xaf` trực tiếp.

```python
%pip install -q zstandard
```

```python
from pathlib import Path
import os
import tarfile
import zstandard


def extract_tar_zst(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Đang giải nén {archive.name} ({archive.stat().st_size / 1e9:.2f} GB)...", flush=True)
    with archive.open("rb") as source:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(source) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as tar:
                tar.extractall(destination, filter="data")
    print(f"Đã giải nén xong {archive.name}", flush=True)

archives = list(Path("/kaggle/input").rglob("task1_p13_kaggle_upload.tar.zst"))
assert len(archives) == 1, archives
os.environ["P13_ARCHIVE"] = str(archives[0])
worktree = Path("/kaggle/working/udsc2026")
if not (worktree / "pyproject.toml").is_file():
    extract_tar_zst(archives[0], worktree)

bundle = worktree / "artifacts/task1/task1_p13_gpu_bundle.tar.zst"
assert bundle.is_file(), bundle
if not (worktree / "artifacts/task1/training/negatives/fold_4.jsonl").is_file():
    extract_tar_zst(bundle, worktree)

os.chdir(worktree)
print(archives[0])
print("Repo:", worktree)
print("Giải nén hoàn tất")
```

```bash
%cd /kaggle/working/udsc2026
!nvidia-smi
!df -h /kaggle/working
```

`/kaggle/input` là read-only; toàn bộ checkpoint phải ghi vào
`/kaggle/working/udsc2026/artifacts/...`.

## 3. Cell cài môi trường, không thay Torch CUDA có sẵn

```bash
%cd /kaggle/working/udsc2026
!python -m pip install -q "accelerate>=1.1,<2" "transformers==5.0.0" "sentence-transformers==5.4.1"
!python -m pip install -q -e . --no-deps
```

```python
import torch
from pathlib import Path

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) >= (7, 0), (
    "PyTorch wheel hiện tại không hỗ trợ GPU dưới sm_70; đổi sang GPU T4 x2"
)

required = [
    Path("data/raw/btc/LegalIR/train.json"),
    Path("data/raw/btc/LegalIR/public-official.json"),
    Path("models/reranker/model.safetensors"),
    Path("artifacts/task1/evaluation/strict_cv_v2/folds.json"),
    Path("artifacts/task1/training/full_dense500_candidates.jsonl"),
    Path("artifacts/task1/training/public_dense500_candidates.jsonl"),
    Path("artifacts/task1/training/negatives/fold_4.jsonl"),
]
assert all(path.is_file() for path in required), [p for p in required if not p.is_file()]
```

## 4. Cell smoke thật trên GPU

Smoke phải chạy forward, backward, save và reload checkpoint. Dùng depth 100 để
đo tốc độ trước; output này chỉ là diagnostic và tuyệt đối không dùng để nộp.

```bash
%cd /kaggle/working/udsc2026
!python scripts/training/finetune_task1_bge_reranker.py \
  --train data/raw/btc/LegalIR/train.json \
  --folds artifacts/task1/evaluation/strict_cv_v2/folds.json \
  --negatives-dir artifacts/task1/training/negatives \
  --candidates artifacts/task1/training/full_dense500_candidates.jsonl \
  --base-model models/reranker \
  --output-dir artifacts/task1/models/bge_reranker_finetune/kaggle_smoke \
  --candidate-depth 100 --evidence-limit 1 \
  --folds-to-run 0 --max-training-pairs 800 --max-validation-queries 20 \
  --epochs 2 --batch-size 4 --inference-batch-size 16 \
  --gradient-accumulation 4 --max-length 512 \
  --device cuda --diagnostic-only
```

Nếu OOM, chỉ hạ `--inference-batch-size 16` xuống 8, rồi hạ `--batch-size 4`
xuống 2. Không hạ `max-length` hoặc đổi learning rate dựa theo outer validation.

## 5. Chạy full OOF theo từng fold

Chọn `DEPTH=100` nếu cần giữ trong quota; candidate recall đo trước đã đạt
0.979252 tại depth 100. Nếu smoke cho thấy còn nhiều thời gian GPU thì chọn 200
(candidate recall 0.985574). Một khi fold 0 đã chạy, giữ nguyên depth và toàn bộ
hyperparameter cho bốn fold còn lại.

Cell dưới đây chạy **một fold**. Đổi `FOLD` lần lượt từ 0 đến 4. Fold 0 để
`RESUME=False`; từ fold 1 trở đi phải giữ lại cùng thư mục output và đặt
`RESUME=True`.

```python
from pathlib import Path
import subprocess

FOLD = 0
RESUME = False
DEPTH = 100
output = Path("artifacts/task1/models/bge_reranker_finetune/full_oof")

cmd = [
    "python", "scripts/training/finetune_task1_bge_reranker.py",
    "--train", "data/raw/btc/LegalIR/train.json",
    "--folds", "artifacts/task1/evaluation/strict_cv_v2/folds.json",
    "--negatives-dir", "artifacts/task1/training/negatives",
    "--candidates", "artifacts/task1/training/full_dense500_candidates.jsonl",
    "--base-model", "models/reranker",
    "--output-dir", str(output),
    "--candidate-depth", str(DEPTH), "--evidence-limit", "1",
    "--max-training-pairs", "8000", "--epochs", "2",
    "--batch-size", "4", "--inference-batch-size", "16",
    "--gradient-accumulation", "4", "--max-length", "512",
    "--folds-to-run", str(FOLD), "--device", "cuda",
]
if RESUME:
    cmd.append("--resume")
subprocess.run(cmd, check=True)
```

Sau mỗi fold, kiểm tra `fold_N/metrics.json`, rồi đóng gói output để tải về máy
local trước khi chạy fold kế tiếp:

```python
import subprocess

fold = FOLD
archive = f"/kaggle/working/task1_p13_after_fold_{fold}.tar.zst"
subprocess.run([
    "tar", "-caf", archive,
    "artifacts/task1/models/bge_reranker_finetune/full_oof",
], check=True)
print("Download ngay:", archive)
```

Nếu phiên Kaggle bị ngắt, giải nén archive mới nhất trở lại repo, đặt đúng `FOLD`
tiếp theo và `RESUME=True`. Không chạy lại fold đã có `status=COMPLETE`.

## 6. Gate trước khi tạo submission

Sau fold 4:

```python
import json
from pathlib import Path

decision_path = Path("artifacts/task1/models/bge_reranker_finetune/full_oof/final_decision.json")
decision = json.loads(decision_path.read_text(encoding="utf-8"))
print(json.dumps(decision, ensure_ascii=False, indent=2))
assert decision["complete_oof"] is True
assert decision["promotable"] is True
assert decision["status"] == "PROMOTE_CANDIDATE"
```

Nếu một assert fail thì dừng, giữ submission 0.9309 và không nộp checkpoint mới.
Leaderboard không thể được bảo đảm trước khi Codabench chấm.

## 7. Public inference và đóng gói, chỉ khi gate pass

Giữ `--candidate-depth` đúng bằng `DEPTH` đã dùng cho cả năm fold:

```bash
!python scripts/submission/build_legal_ir_finetuned_fold_ensemble.py \
  --questions data/raw/btc/LegalIR/public-official.json \
  --overlay-labels data/raw/btc/LegalIR/train.json \
  --candidates artifacts/task1/training/public_dense500_candidates.jsonl \
  --checkpoint-root artifacts/task1/models/bge_reranker_finetune/full_oof \
  --candidate-depth 100 --evidence-limit 1 \
  --batch-size 16 --max-length 512 --device cuda \
  --output artifacts/task1/finetuned_oof_public/predictions.json \
  --report artifacts/task1/finetuned_oof_public/report.json

!python scripts/submission/write_legal_ir_submission.py \
  --input artifacts/task1/finetuned_oof_public/predictions.json \
  --questions data/raw/btc/LegalIR/public-official.json \
  --corpus-manifest artifacts/task1/corpus_document_ids.json \
  --output artifacts/task1/finetuned_oof_public/submission.zip

!python scripts/submission/validate_legal_ir_submission.py \
  --input artifacts/task1/finetuned_oof_public/submission.zip \
  --questions data/raw/btc/LegalIR/public-official.json \
  --corpus-manifest artifacts/task1/corpus_document_ids.json
```

Tải về cả `submission.zip`, `report.json`, `final_decision.json` và archive năm
fold. Không ghi đè `artifacts/task1/submission.zip` — đó là bản rollback 0.9309.
