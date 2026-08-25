# Beam Quickstart cho repo UDSC2026

Tài liệu này thay cho các note Beam rời rạc. Mục tiêu: tạo môi trường CLI riêng, kết nối account, dùng Volume, upload/download artifact và chạy launcher an toàn.

## 1. Nguyên tắc

- Chạy Beam CLI trong **WSL**.
- Repo: `/mnt/d/udsc2026`.
- Dùng môi trường CLI riêng `.venv_beam312`, không trộn với `venv_linux`.
- Python CLI nên là **3.12**.
- Beam Volume hiện dùng tên `udsc-p13`.
- Launcher mount volume tại `/workspace/p13`; dữ liệu runtime nằm dưới `/workspace/p13/runtime`.
- Trước job tốn GPU, luôn chạy `--preflight` nếu launcher hỗ trợ.
- Không lưu API token vào repo, file `.md`, shell script hoặc commit Git.

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

Kiểm tra:

```bash
python --version
which python
which beam
beam --version
python -m pip check
```

Mỗi lần mở WSL mới:

```bash
cd /mnt/d/udsc2026
source .venv_beam312/bin/activate
```

## 3. Kết nối account Beam

Tạo một config mới:

```bash
beam config create <ten-config>
```

CLI sẽ yêu cầu thông tin/token cần thiết. Không paste token vào source code hoặc tài liệu.

Chọn config:

```bash
beam config select <ten-config>
```

Test kết nối:

```bash
beam task list
```

Nếu lệnh này trả danh sách task hoặc danh sách rỗng mà không báo Unauthorized thì CLI đã kết nối được account/workspace.

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

Volume là vùng lưu trữ bền hơn container của một task. Container có thể được tạo/xóa cho từng job, còn model, data và artifact cần dùng lại nên đặt trong Volume.

## 5. Upload / download

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

## 6. Cách launcher hoạt động

Các launcher trong `scripts/beam/` thường:

1. chạy local để kiểm tra input/config;
2. khai báo Beam `Image`, CPU/GPU, RAM và `Volume`;
3. submit function remote;
4. container Beam mount `udsc-p13` vào `/workspace/p13`;
5. code remote đọc/ghi dưới `/workspace/p13/runtime`.

Với các launcher của repo này, cách dùng thông thường là chạy file Python trực tiếp, ví dụ:

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

Chỉ submit khi thấy trạng thái PASS và đúng:
- mode;
- model/input path;
- query count;
- GPU;
- output path;
- `remote_submitted=false` trong preflight.

Preflight là kiểm tra local; nó giúp tránh mất credit vì thiếu file hoặc sai contract.

## 8. Đọc log đúng cách

Mốc log nên hiểu như sau:

```text
Chưa có [BOOT]
    -> thường là scheduler/container/GPU provisioning chưa khởi động được Python.

Có [BOOT]
    -> Python của launcher đã thực sự chạy trong container.

Có [SCORE]
    -> inference/scoring GPU đã bắt đầu.

Có [WRITE]
    -> đang ghi/publish output.

Có [DONE] hoặc task Complete + output hợp lệ
    -> job hoàn tất.
```

Web UI có thể cập nhật trạng thái chậm hơn CLI hoặc ngược lại. Khi nghi ngờ, kiểm tra cả task log và output trên Volume; không retry mù một job GPU chỉ vì một màn hình vẫn hiện `Pending`.

## 9. Một số lệnh hữu ích

```bash
# task
beam task list

# volume
beam volume list

# chuyển account/workspace config
beam config select <ten-config>

# kiểm tra CLI env
python --version
beam --version
python -m pip check
```

## 10. Quy tắc vận hành

- Một job fail ở `nvidia-container-cli` trước khi có log Python thường là lỗi provisioning/runtime GPU, không phải lỗi scoring code.
- Không sửa model/scoring logic để chữa lỗi provisioning.
- Không auto-retry GPU job.
- Nếu remote báo thiếu đúng một file, upload đúng file đó; không upload cả cây dữ liệu khổng lồ nếu không cần.
- Sau khi job quan trọng Complete, download output và report về local ngay.
- Giữ manifest/report/hash đi cùng artifact production để kiểm tra provenance.
