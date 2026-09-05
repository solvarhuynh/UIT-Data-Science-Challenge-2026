# TV5 — chạy Task 2 P14 trên Kaggle T4 x2

Đây là đường chạy độc lập với Beam. Không sử dụng Beam Volume, scheduler,
`nvproxy` hoặc credit Beam. Pipeline giữ nguyên dữ liệu/gate P14 nhưng dùng FP16
phù hợp T4 và DataParallel để huấn luyện Qwen trên cả hai GPU. Profile Kaggle
dùng giới hạn `2048/1200/768` token để các batch dài không vượt VRAM 16 GB;
profile Beam/RTX 4090 vẫn giữ cấu hình dài hơn.

## 1. Tạo hai file để đưa lên Kaggle Dataset

Tại máy local, từ thư mục gốc repo:

```powershell
python scripts/package_kaggle_task2.py
```

Đưa đúng hai file sau vào cùng một Kaggle Dataset:

```text
artifacts/task2/task2_p14_beam_input.tar.zst
artifacts/task2/kaggle/task2_p14_code_bundle.zip
```

Tên `beam_input` chỉ là tên lịch sử. File này không chứa code Beam và dùng được
trên Kaggle.

## 2. Cấu hình notebook

- Có thể import thẳng notebook đã chuẩn bị:
  `notebooks/task2_p14_kaggle_t4x2.ipynb`.
- Accelerator: `GPU T4 x2`.
- Internet: bật để tải model Qwen từ Hugging Face.
- Không chọn P100 vì PyTorch hiện tại của Kaggle không còn chứa kernel `sm_60`.

## 3. Cell duy nhất để setup và chạy

```python
from pathlib import Path
import os
import shutil
import subprocess
import sys
import zipfile

input_root = Path("/kaggle/input")
working = Path("/kaggle/working/udsc2026")
working.mkdir(parents=True, exist_ok=True)

code_bundles = list(input_root.rglob("task2_p14_code_bundle.zip"))
code_dirs = [
    path
    for path in input_root.rglob("task2_p14_code_bundle")
    if path.is_dir()
]
archives = list(input_root.rglob("task2_p14_beam_input.tar.zst"))
assert len(code_bundles) + len(code_dirs) == 1, code_bundles + code_dirs
assert len(archives) == 1, archives

if code_bundles:
    with zipfile.ZipFile(code_bundles[0]) as bundle:
        bundle.extractall(working)
else:
    shutil.copytree(code_dirs[0], working, dirs_exist_ok=True)

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "accelerate>=1.1,<2",
        "huggingface-hub>=1.3,<2",
        "nltk>=3.8,<4",
        "peft>=0.17,<1",
        "rouge-score>=0.1,<1",
        "safetensors>=0.5,<1",
        "scikit-learn>=1.5,<2",
        "sentencepiece>=0.2,<1",
        "transformers==5.0.0",
        "zstandard==0.23.0",
    ],
    check=True,
)

# Task 2 dùng LoRA FP16 thường, không dùng TorchAO. Xóa bản TorchAO cũ được
# Kaggle cài sẵn để PEFT không chọn nhầm dispatcher không tương thích.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    check=False,
)

os.chdir(working)
subprocess.run(
    [
        sys.executable,
        "scripts/cloud/kaggle_task2_p14.py",
        "--archive",
        str(archives[0]),
        "--workspace",
        str(working),
        "--output-dir",
        "/kaggle/working/task2_p14",
        "--stage",
        "all",
        "--epochs",
        "2",
    ],
    check=True,
)
```

Log đúng phải có cả hai dòng:

```text
KAGGLE_CUDA_COMPUTE_PASS
QWEN_DATA_PARALLEL enabled devices=2 batch_size=2
```

Pipeline tự bỏ qua stage đã hoàn tất nếu chạy lại trong cùng notebook session.
Nếu strict METEOR dưới `0.60`, trạng thái là `HELDOUT_REJECTED` và script không
tạo submission public. Đây là gate bảo vệ, không phải crash.

## 4. File kết quả

Kết quả tổng hợp:

```text
/kaggle/working/task2_p14/task2_p14_kaggle_result.zip
```

Chỉ nộp file bên trong dưới đây khi summary báo `PUBLIC_CANDIDATE_READY`:

```text
/kaggle/working/task2_p14/submission.zip
```

Trước khi đóng notebook, chọn **Save Version** để checkpoint/output được Kaggle
lưu lại. Không tải `task2_p14_kaggle_result.zip` lên Codabench; đó là gói backup,
không phải submission.
