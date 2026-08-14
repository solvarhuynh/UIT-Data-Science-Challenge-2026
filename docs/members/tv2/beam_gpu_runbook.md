# Beam Cloud GPU: hướng dẫn chạy job dài

Các bước về Beam có thể dùng lại cho job GPU khác.

> Quy ước quan trọng: Beam SDK trong project được vận hành từ Ubuntu/WSL.
> PowerShell chỉ dùng để mở WSL hoặc kiểm tra file Windows. Mỗi khối lệnh đều

## 1. Beam dùng để làm gì?

Beam cung cấp máy compute từ xa có GPU. Beam không thay thế code, dữ liệu hay
pipeline của repository.

Dùng Beam khi máy local không có CUDA/VRAM phù hợp, khi training hoặc inference
kéo dài nhiều giờ, hoặc khi cần Volume bền vững để giữ checkpoint, log và output
giữa các lần chạy. Với Task1, local chỉ kiểm tra compile, unit test, dataset và
dry-run; Beam thực hiện transformer training nặng.

## 2. Bắt đầu từ Windows và vào repository

Repository trên Windows:

```text
D:\udsc2026
```

Đường dẫn tương ứng trong Ubuntu/WSL:

```text
/mnt/d/udsc2026
```

**Chạy ở: Windows PowerShell**

```powershell
wsl
```

Expected output là một shell Ubuntu, ví dụ `solvarh@TrungNghia:~$`. Từ đây,
các command Beam chính phải chạy trong Ubuntu.

**Chạy ở: Ubuntu / WSL**

```bash
cd /mnt/d/udsc2026
pwd
git status --short
```

Expected output của `pwd` là `/mnt/d/udsc2026`. Nếu `git status` có thay đổi,
không tự động xóa hoặc reset; đó có thể là công việc local chưa commit.

## 3. Tạo virtualenv và cài Beam SDK

Không dùng `.venv` Windows để chạy Beam. Tạo một virtualenv riêng trong WSL để
tránh trộn Python, binary và package giữa Windows với Linux.

**Chạy ở: Ubuntu / WSL**

```bash
cd /mnt/d/udsc2026
python3 --version
python3 -m venv ~/beam-venv
source ~/beam-venv/bin/activate
python --version
python -m pip install --upgrade pip
```

Expected output sau `source` có tiền tố `(beam-venv)` và Python nằm trong
`~/beam-venv/bin/python`.

Tên package Beam không được khai báo trong `pyproject.toml` của repository.
Vì vậy không ghi cứng tên package từ tài liệu cũ. Cài package theo Dashboard/
SDK version hiện tại của team:

**Chạy ở: Ubuntu / WSL**

```bash
<VERIFY WITH CURRENT BEAM CLI> python -m pip install <CURRENT_BEAM_SDK_PACKAGE>
```

Sau khi cài, kiểm tra import đúng với source hiện tại:

**Chạy ở: Ubuntu / WSL**

```bash
python -c "from beam import function, Image, Volume; print('Beam SDK import: OK')"
python -c "from beam import Sandbox, Image, PythonVersion, Volume; print('Beam Sandbox import: OK')"
```

Nếu import fail, dừng ở đây và dùng đúng package/version do Beam cung cấp.

## 4. Tạo token và kiểm tra kết nối

Trong Beam Dashboard, tạo hoặc lấy token cho tài khoản/team được cấp quyền.
Không commit token vào Git, không ghi token vào file launcher, và không dán
token vào issue hoặc log.

**Thực hiện trên: Beam Dashboard**

1. Đăng nhập workspace của team.
2. Mở phần API keys/tokens theo giao diện hiện tại.
3. Tạo token có quyền chạy Function/Sandbox và truy cập Volume cần dùng.
4. Sao chép token một lần vào nơi an toàn.

Cách đặt biến môi trường và lệnh login phụ thuộc Beam CLI/SDK đang cài; repo
không chứa command auth làm source of truth. Dùng cú pháp được CLI hiện tại
hiển thị:

**Chạy ở: Ubuntu / WSL**

```bash
<VERIFY WITH CURRENT BEAM CLI> beam auth/login <TOKEN_OR_CURRENT_AUTH_FLAGS>
<VERIFY WITH CURRENT BEAM CLI> beam whoami/status
```

Nếu SDK yêu cầu environment variable, đặt token chỉ trong shell hiện tại theo
hướng dẫn của phiên bản đó, không ghi vào repository:

**Chạy ở: Ubuntu / WSL**

```bash
export <CURRENT_BEAM_TOKEN_VARIABLE>=<TOKEN>
```

Kiểm tra CLI:

**Chạy ở: Ubuntu / WSL**

```bash
command -v beam
beam --help
python -c "import beam; print('Beam Python module: OK')"
```

Expected: `beam --help` in ra nhóm lệnh của SDK hiện tại và Python import
thành công. Nếu `beam` không tồn tại nhưng import thành công, dùng launcher
Python của repository; không tự đổi sang CLI khác.

## 5. Hiểu serverless GPU selection

Trong source hiện tại, job production được khai báo bằng decorator sau:

```python
@function(
    name="udsc-p13-full-5fold",
    cpu=4,
    memory="64Gi",
    gpu="RTX5090",
    timeout=-1,
    retries=0,
    headless=True,
)
```

Ý nghĩa:

- `gpu="RTX5090"`: yêu cầu loại GPU mà job đã được kiểm tra.
- `cpu=4`, `memory="64Gi"`: tài nguyên CPU/RAM của Function production.
- `timeout=-1`: job dài theo chính sách Beam hiện hành.
- `retries=0`: Beam không tự chạy lại toàn bộ training sau lỗi.
- `headless=True`: job tiếp tục khi terminal local đóng sau lúc submit.

Smoke hạ tầng dùng `RTX5090`, 2 CPU và 8 GiB. Đây chỉ là kiểm tra GPU/Volume,
không chứng minh full training sẽ chạy hết.

## 6. Tạo và kiểm tra Beam Volume

Job hiện tại dùng Volume:

```text
name: udsc-p13
mount: /workspace/p13
```

**Thực hiện trên: Beam Dashboard**

Tạo hoặc chọn Volume `udsc-p13`, bảo đảm tài khoản có quyền đọc/ghi. Cú pháp
CLI tạo Volume không được lưu trong repository, nên không tự dùng command cũ:

```text
<VERIFY WITH CURRENT BEAM CLI> create/select Volume udsc-p13
```

Nếu Volume đã tồn tại, không tạo Volume mới chỉ vì một lần reconnect. Volume là
nơi giữ state; `/tmp` trong container chỉ là disk tạm và có thể mất.

## 7. Chuẩn bị và upload bundle

Launcher production tìm đúng ba archive ở root Volume:

```text
/workspace/p13/udsc_p13_code.zip
/workspace/p13/udsc_p13_data.zip
/workspace/p13/udsc_p13_reranker.zip
```

Bundle code phải giải nén ra runtime có tối thiểu:

```text
runtime/pyproject.toml
runtime/scripts/training/finetune_task1_bge_reranker.py
```

Bundle data phải có train, candidate, strict folds và năm file negatives mà
launcher kiểm tra. Bundle reranker phải giải nén được thành:

```text
runtime/models/reranker/
```

**Chạy ở: Ubuntu / WSL**

Kiểm tra file local trước khi upload:

```bash
cd /mnt/d/udsc2026
ls -lh udsc_p13_code.zip udsc_p13_data.zip udsc_p13_reranker.zip
unzip -l udsc_p13_code.zip | head
unzip -l udsc_p13_data.zip | head
unzip -l udsc_p13_reranker.zip | head
```

Cách upload phụ thuộc Beam CLI/SDK hiện tại. Nếu chưa xác minh được command,
dùng giao diện Volume hoặc command do CLI in ra:

**Thực hiện trên: Beam Dashboard**

Upload ba archive vào root của Volume `udsc-p13`, không upload vào một thư mục
con khác nếu không sửa launcher.

```text
<VERIFY WITH CURRENT BEAM CLI> upload udsc_p13_code.zip to Volume udsc-p13:/
<VERIFY WITH CURRENT BEAM CLI> upload udsc_p13_data.zip to Volume udsc-p13:/
<VERIFY WITH CURRENT BEAM CLI> upload udsc_p13_reranker.zip to Volume udsc-p13:/
```

Model weights và archive lớn không commit vào Git. `.beamignore` chỉ điều khiển
source upload của Beam; nó không làm Git tự bỏ qua file. Muốn tránh Git phải
dùng `.gitignore` hoặc không `git add` file đó.

## 8. Kiểm tra code local trước khi gửi job

**Chạy ở: Ubuntu / WSL**

```bash
cd /mnt/d/udsc2026
source ~/beam-venv/bin/activate
python -m py_compile scripts/beam/beam_p13_run_all.py scripts/beam/beam_p13_diag.py scripts/beam/beam_gpu_test.py
python scripts/training/finetune_task1_bge_reranker.py --help
```

Expected: không có traceback. Local CPU không cần CUDA cho compile/help/unit
test; không chạy `--device cuda` trong Python Windows CPU-only.

## 9. Chạy CPU diagnostic

Diagnostic đọc tail log đã lưu trên Volume, không chạy training và không cần
GPU. Source hiện dùng `Sandbox`, 2 CPU, 8 GiB, mount cùng Volume.

**Chạy ở: Ubuntu / WSL**

```bash
cd /mnt/d/udsc2026
source ~/beam-venv/bin/activate
python scripts/beam/beam_p13_diag.py
```

Nếu chưa có `fold_0.log.tail`, diagnostic sẽ báo rõ file chưa tồn tại. Điều đó
không phải lỗi cài đặt; cần chạy job hoặc upload output cũ trước.

## 10. Chạy GPU smoke

Smoke tạo Sandbox GPU, chạy `nvidia-smi` và liệt kê `/workspace/p13`. Nó không
fine-tune model.

**Chạy ở: Ubuntu / WSL**

```bash
cd /mnt/d/udsc2026
source ~/beam-venv/bin/activate
python scripts/beam/beam_gpu_test.py
```

Expected output có tên GPU, thông tin `nvidia-smi` và listing Volume. Nếu smoke
fail, chưa được phép gửi production training; kiểm tra token, quota, GPU
availability và Volume permission trên Dashboard.

## 11. Submit production Function

Function production là `train_all_p13` trong
`scripts/beam/beam_p13_run_all.py`. File có
entrypoint local gọi chính xác:

```python
result = train_all_p13.remote()
```

**Chạy ở: Ubuntu / WSL**

```bash
cd /mnt/d/udsc2026
source ~/beam-venv/bin/activate
python scripts/beam/beam_p13_run_all.py
```

Expected local output bắt đầu bằng `Enqueuing P13 full 5-fold training...` và
sau đó Beam trả remote result/job reference. Nếu `.remote()` không còn hợp lệ
với SDK mới, dừng và kiểm tra API version; không dùng
`scripts/beam/beam_p13_serverless.py` vì file đó đã deprecated.

Sau khi submit thành công, có thể đóng terminal hoặc tắt máy local. `headless`
giúp job tiếp tục trên Beam; không được hiểu là checkpoint nằm ở local.

## 12. Mở lại terminal và kiểm tra job

**Chạy ở: Windows PowerShell**

```powershell
wsl
```

**Chạy ở: Ubuntu / WSL**

```bash
cd /mnt/d/udsc2026
source ~/beam-venv/bin/activate
<VERIFY WITH CURRENT BEAM CLI> beam jobs/status/logs <JOB_REFERENCE>
```

**Thực hiện trên: Beam Dashboard**

Mở Jobs/Functions, chọn `udsc-p13-full-5fold`, xem trạng thái, GPU usage,
runtime và log. Dashboard là nơi xem trạng thái trực tiếp; log bền vững cần
đọc từ Volume vì launcher lưu full log/tail ở đó.

Launcher không stream toàn bộ trainer log lên Dashboard. Khoảng mỗi phút có
heartbeat và tail nhỏ; sau khi subprocess kết thúc, full log được copy vào:

```text
/workspace/p13/runtime/artifacts/task1/models/bge_reranker_finetune/full_oof/beam_logs/fold_N.log
```

## 13. State, lỗi và chạy lại

Mỗi fold hoàn tất khi có marker:

```text
runtime/artifacts/task1/models/bge_reranker_finetune/full_oof/fold_N/completion.json
```

Khi chạy lại cùng Volume/output:

- fold có `completion.json` được giữ và skip;
- fold chưa hoàn thành được chạy;
- fold partial bị xóa rồi chạy lại từ đầu vì trainer chưa có resume giữa batch;
- lỗi `EAGAIN` của Volume chỉ được retry ở giai đoạn an toàn trước
  `Training batches:`;
- sau khi training thật sự bắt đầu, launcher không tự retry để tránh đốt thêm
  nhiều giờ GPU.

Không xóa Volume, marker hoặc output khi muốn resume. Không chạy hai Function
cùng ghi vào một `full_oof` directory.

**Chạy ở: Ubuntu / WSL**

```bash
python scripts/beam/beam_p13_run_all.py
```

Nếu job fail, xem Dashboard trước, sau đó đọc tail/full log trên Volume. Không
đổi seed, input, path archive hoặc hyperparameter giữa lần resume.

## 14. Kiểm tra output cuối

**Chạy ở: Ubuntu / WSL**

```bash
for f in 0 1 2 3 4; do
  test -f /workspace/p13/runtime/artifacts/task1/models/bge_reranker_finetune/full_oof/fold_$f/completion.json \
    && echo "fold_$f COMPLETE" \
    || echo "fold_$f MISSING"
done
```

Đọc các artifact tổng hợp:

```text
/workspace/p13/runtime/artifacts/task1/models/bge_reranker_finetune/full_oof/oof_predictions.jsonl
/workspace/p13/runtime/artifacts/task1/models/bge_reranker_finetune/full_oof/comparison.json
/workspace/p13/runtime/artifacts/task1/models/bge_reranker_finetune/full_oof/comparison.md
/workspace/p13/runtime/artifacts/task1/models/bge_reranker_finetune/full_oof/final_decision.json
```

Chỉ dùng checkpoint khi đủ năm fold, manifest/hash khớp input và quyết định
đánh giá cho phép. Không tự động ghi đè `models/reranker`, production config
hay pipeline Task2.

## 15. Checklist nhanh

- [ ] Đang ở Ubuntu/WSL và `pwd` là `/mnt/d/udsc2026`.
- [ ] `~/beam-venv` đã activate.
- [ ] Beam import, token và CLI đã kiểm tra.
- [ ] Volume `udsc-p13` tồn tại, đọc/ghi được.
- [ ] Ba archive đúng tên ở root Volume.
- [ ] Compile/help local pass.
- [ ] CPU diagnostic pass hoặc đã hiểu file log chưa tồn tại.
- [ ] GPU smoke thấy `RTX5090` và Volume.
- [ ] Production submit trả job reference.
- [ ] Có thể đóng terminal sau khi job đã submit headless.
- [ ] Reconnect dùng cùng Volume, không xóa completion marker.
- [ ] Đủ 5 `completion.json` và artifact tổng hợp trước khi kết luận.
- [ ] Không commit token, weights, archive lớn hoặc log.

## 16. Source of truth trong repository

- `scripts/beam/beam_p13_run_all.py`: Function production, tài nguyên, Volume, staging,
  logging và resume theo fold.
- `scripts/beam/beam_gpu_test.py`: GPU/Volume smoke.
- `scripts/beam/beam_p13_diag.py`: CPU diagnostic đọc tail log.
- `scripts/beam/beam_p13_serverless.py`: deprecated, không dùng.
- `scripts/training/finetune_task1_bge_reranker.py`: trainer được invoke bằng
  subprocess.
- `.beamignore`: phạm vi source Beam upload; không phải `.gitignore`.

Task2 không dùng launcher này. Runbook và ví dụ này chỉ phục vụ compute của
Task1, không thay đổi model/config production của Task2.
