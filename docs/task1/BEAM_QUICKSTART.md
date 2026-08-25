# Beam Quickstart cho repo UDSC2026

## 1. Nguyên tắc

- Chạy Beam CLI trong **WSL** (tải ubuntu wsl - xem ytb).
- Repo: `/mnt/d/udsc2026`.
- Dùng môi trường CLI riêng `.venv_beam312`
- Python CLI nên là **3.12**.
- Beam Volume hiện dùng tên `udsc-p13`.
- Launcher mount volume tại `/workspace/p13`; dữ liệu runtime nằm dưới `/workspace/p13/runtime`.
- Trước job tốn GPU, luôn chạy `--preflight` nếu launcher hỗ trợ.

## 2. Tạo môi trường Beam CLI

Nếu máy chưa có Python 3.12, cách đã dùng ổn là cài bằng `uv`:

```bash
cd /mnt/d/udsc2026

~/.local/bin/uv python install 3.12
~/.local/bin/uv venv --python 3.12 .venv_beam312

source .venv_beam312/bin/activate
```

Cài CLI:

```bash
python -m pip install --upgrade pip
python -m pip install \
  'beam-client==0.2.207' \
  'beta9==0.1.265' \
  'websockets==15.0.1'
```


## Mỗi lần mở WSL mới:

```bash
cd /mnt/d/udsc2026
source .venv_beam312/bin/activate
```

## 3. Kết nối account Beam

Xem ảnh để lấy token tài khoản của mình

Không commit hoặc chia sẻ ảnh/API token. Lấy token trực tiếp trong Beam Onboarding và cấu hình local bằng `beam configure default --token <TOKEN>`.

Vào Onboarding -> chạy lệnh 1 ở ubuntu wsl -> chạy lệnh 2 kết nối token với tài khoản


## 4. Beam Volume

Tạo volume nếu account/workspace mới chưa có:

```bash
beam volume create udsc-p13
beam volume list
```

Quy ước hiện tại:

```text
Beam volume:     udsc-p13
Mount:           /workspace/p13
Runtime root:    /workspace/p13/runtime
```

Volume là vùng lưu trữ các file mình tải lên or output khi chạy trên beam.

## 5. Upload / download  -- ví dụ thôi (chỉ upload các file cần thiết với task của mình)

### Upload một file

```bash
beam cp \
  local/path/file.json \
  beam://udsc-p13/runtime/path/file.json
```

### Upload một thư mục

Ví dụ model reranker:

```bash
beam cp \
  models/reranker \
  beam://udsc-p13/runtime/models/reranker
```

### Download output

```bash
beam cp \
  beam://udsc-p13/runtime/artifacts/task1/path/output.jsonl \
  artifacts/task1/path/output.jsonl
```

`beam cp` giữ dữ liệu trên Volume; code launcher không nên chứa model/data lớn nếu có thể tránh.

## 6. Cách run files

Sẽ quy định gpu trước (ví dụ RTX 4090 5090 A10G ...)

```bash
python scripts/beam/<launcher>.py --preflight
```

sau đó, khi preflight PASS:

```bash
python scripts/beam/<launcher>.py
```

Không tự đổi sang `beam run file.py:function` nếu launcher hiện tại đã tự gọi `.remote()`.

## 7. Preflight trước GPU

Ví dụ:

```bash
python scripts/beam/beam_task1_v3_frozen_features.py --preflight --public
```

Chỉ submit khi thấy trạng thái PASS và đúng

## 8. Đọc log đúng cách - kinh nghiệm cá nhân thôi

> Thường nếu chọn gpu phổ biến sẽ phải chờ beam cấp gpu và trên web ở mục task sẽ hiện là pending, nếu ở ubuntu wsl báo `erorr` mà web vẫn hiện pending thì chấc chắn bị lôi, (xem có tải đủ môi trường chưa, kết nối mạng, gpu quá hiếm, torch sai phiên bản ...)

> Nếu lỗi thì copy log ở mục task trên web, chọn task mới nhất bị lỗi (load trang lại) chọn all log, và copy từ chỗ bị lỗi tới hết

Web UI có thể cập nhật trạng thái chậm hơn CLI hoặc ngược lại. Khi nghi ngờ, kiểm tra cả task log và output trên Volume; không retry mù một job GPU chỉ vì một màn hình vẫn hiện `Pending`.

## 9. Một số lệnh hữu ích

```bash
# task
beam task list

# volume
beam volume list

# xem log của 1 task
beam logs --task-id <id task bị lỗi (hiện khi nhấn vào task đó, ở trên cùng)> (ko có dấu < >)

```
