# TV5 — Chạy Task 2 P14 trên Beam

Task 2 chạy trong Beam container có CUDA và dùng Volume `udsc-task2-p14` để
lưu input, Qwen cache, checkpoint, trạng thái resume và kết quả. Qwen 3.2 GB
được tải trực tiếp từ Hugging Face vào Volume, không nằm trong file upload local.
Mọi script/import chạy từ snapshot local `/mnt/code`; Volume chỉ chứa dữ liệu và
artifact bền vững, không được dùng để thực thi Python source.

Bundle local hiện tại:

```text
artifacts/task2/task2_p14_beam_input.tar.zst
size: 821249755 bytes (0.765 GiB)
sha256: 4b5823c049ca9a77d40d84f148d09415ddbe334abdca69151c5859167127c2eb
```

## Lệnh mặc định: FastServerless

`FastServerless` là mặc định. Mỗi lần launch gửi **một request duy nhất** chứa
toàn bộ danh sách ưu tiên:

```text
RTX5090,RTX4090,A10G
```

Đây là danh sách lựa chọn, không phải yêu cầu ba GPU cùng lúc. Beam chỉ cấp một
GPU và có thể chọn RTX4090 hoặc A10G ngay khi RTX5090 đang thiếu capacity. Nếu
một allocation thật sự hỏng do host bẩn hoặc `nvproxy`, lần retry kế tiếp vẫn
gửi đủ ba lựa chọn nhưng xoay thứ tự ưu tiên:

```text
RTX5090,RTX4090,A10G
RTX4090,A10G,RTX5090
A10G,RTX5090,RTX4090
```

Pipeline đặt `allow_marketplace=False`, vì vậy luồng này chỉ dùng serverless và
không tự reserve/chuyển sang máy H100 trả phí.

Chạy toàn bộ quy trình bằng **một lệnh** từ thư mục gốc repository:

```powershell
.\scripts\cloud\run_task2_p14_beam.ps1 -Stage Run -GpuAttempts 6
```

`-Stage Run` tự thực hiện tuần tự:

1. tạo/kiểm tra Volume và chỉ upload bundle khi SHA-256 trên Beam chưa khớp;
2. xác minh/tải đúng Qwen revision đã pin trên CPU worker, không tốn giờ GPU để tải model;
3. chạy Smoke thật gồm một bước Qwen LoRA forward/backward/optimizer ở đúng
   `batch x 3072 token` của GPU; parent-reranker đọc group size từ dữ liệu thật,
   thử listwise worst-case 34/68/136 cặp theo GPU và cấp phát cả AdamW state;
4. chỉ chạy Full khi Smoke trả về `SMOKE_PASS` đúng bundle và source; nếu Full nhận
   loại GPU khác thì tự chạy lại preflight chính xác trên GPU đó;
5. tải và kiểm tra result ZIP về máy local.

`-GpuAttempts 6` cho phép tối đa 6 lần xin allocation mới **cho mỗi GPU stage**
(Smoke và Full). Mỗi attempt đều chứa đủ ba GPU serverless; wrapper chỉ xoay thứ
tự ưu tiên và thử lại khi đã nhận diện chắc chắn lỗi allocation, gồm
`nvidia-container-cli ... unknown device` hoặc host đã bị chiếm VRAM. Lỗi code,
model, dữ liệu, CUDA OOM và lỗi client chung sẽ dừng ngay. Đây không phải Beam
auto-retry của model job: các Beam function vẫn đặt `retries=0` để tránh chạy
trùng và tốn credit.

Không cần chạy riêng Prepare, Smoke rồi Full trong luồng bình thường. Các stage
riêng vẫn có thể dùng để chẩn đoán khi cần.

## Kiểm tra capacity không phát sinh reservation

Lệnh dưới đây chỉ đọc snapshot inventory/capacity mà Beam đang công bố tại thời
điểm gọi; nó không tạo task, không reserve máy và không bắt đầu tính giờ GPU:

```powershell
.\scripts\cloud\run_task2_p14_beam.ps1 -Stage Capacity
```

`machine list` không bảo đảm RTX5090/RTX4090 serverless sẽ được cấp cho task kế
tiếp; inventory có thể đổi ngay sau khi xem và còn có thể bao gồm offer
on-demand. Chỉ một allocation thật mới xác nhận được GPU. Muốn kiểm tra mà chưa
chạy Full, dùng:

```powershell
.\scripts\cloud\run_task2_p14_beam.ps1 -Stage Guard
.\scripts\cloud\run_task2_p14_beam.ps1 -Stage Smoke -GpuAttempts 6
```

Khi GPU đã được cấp, log `GPU_PREFLIGHT_JSON` cho biết chính xác `name`,
`actual_type`, VRAM và CUDA capability. Smoke là workload serverless thật nên sẽ
dùng credit serverless trong lúc chạy; khác với `machine list`, nó không phải
phép kiểm tra read-only.

## Guard và thay job pending an toàn

Trước khi chạy Smoke/Full/Run, wrapper tự kiểm tra các Task 2 job có trạng thái
`PENDING`, `RUNNING` hoặc `RETRY`. Có thể chạy riêng phép kiểm tra read-only:

```powershell
.\scripts\cloud\run_task2_p14_beam.ps1 -Stage Guard
```

`Guard` không dừng task và không tạo task mới. Nó bỏ qua đúng những ID đã có bằng
chứng terminal `Function complete/failed` trong log local, nhưng chặn launch nếu
còn job hoạt động thật.

Nếu Guard báo một job Task 2 chỉ đang `PENDING` và bạn muốn bỏ hàng đợi cũ để gửi
lại priority list mới, chạy:

```powershell
.\scripts\cloud\run_task2_p14_beam.ps1 -Stage ReplacePending
```

`ReplacePending` chỉ stop job `PENDING`, chờ Beam xác nhận job đã rời trạng thái
active rồi chạy Guard lại. Nó **từ chối** tự động dừng job `RUNNING`, `RETRY` hoặc
trạng thái không xác định. Với các job đó, xem Status/Logs và chỉ dùng `-Stage
Stop -TaskId ...` sau khi xác nhận đúng ID. Không mở thêm terminal để chạy một
`Run` khác trong khi Guard chưa PASS.

## Không dùng H100 với free trial hiện tại

Không chạy `scripts/cloud/run_task2_p14_h100.ps1`, không dùng `beam machine
reserve`, và không truyền `-GpuType H100 -PoolName ...` khi tài khoản chỉ có
credit ghi `Serverless only`. H100 reserve/on-demand là luồng trả phí riêng và
không thuộc phương án trong tài liệu này. Phương án được hỗ trợ cho TV5 hiện tại
là `FastServerless` mặc định ở trên.

## Theo dõi task và log

Các GPU function chạy `headless=True`; mất kết nối client không đồng nghĩa remote
task đã dừng. Không khởi chạy thêm một `Run` chỉ vì terminal bị ngắt. Kiểm tra
trạng thái trước:

Smoke bị giới hạn 30 phút (TTL 45 phút) và Full bị giới hạn 8 giờ (TTL 12 giờ),
đều `retries=0`. Vì vậy task lỗi không thể treo vô hạn trên Beam. Mỗi Smoke mới
cũng vô hiệu hóa `SMOKE_PASS` cũ trước khi preflight và chỉ ghi pass mới theo kiểu
atomic sau khi tất cả kiểm tra thành công.

```powershell
.\scripts\cloud\run_task2_p14_beam.ps1 -Stage Status
```

Copy Task ID thật từ bảng status, không gõ dấu `< >`, rồi lấy 300 dòng log có
timestamp:

```powershell
.\scripts\cloud\run_task2_p14_beam.ps1 -Stage Logs -TaskId TASK_ID_THAT
```

Chỉ khi đã xác nhận đó là task cũ bị lỗi hoặc muốn hủy có chủ đích, dừng đúng ID:

```powershell
.\scripts\cloud\run_task2_p14_beam.ps1 -Stage Stop -TaskId TASK_ID_THAT
```

Log client và log task local nằm tại:

```text
artifacts/task2/beam_logs/beam_smoke_client.log
artifacts/task2/beam_logs/beam_full_client.log
artifacts/task2/beam_logs/beam_smoke_attempt_<N>_<NONCE>.log
artifacts/task2/beam_logs/beam_full_attempt_<N>_<NONCE>.log
artifacts/task2/beam_logs/beam_task_<TASK_ID>.log
```

## Lock và resume trên Volume

Smoke và Full dùng exclusive lock:

```text
/mnt/task2/.task2_p14.lock
```

Lock ngăn hai headless job cùng ghi vào Volume. File lock có thể vẫn tồn tại sau
khi job kết thúc; trạng thái khóa do `flock` quyết định, không phải do file có
tồn tại hay không. Nếu log báo job khác đang chạy, hãy chờ hoặc dừng đúng task
trùng trước khi chạy lại.

Trạng thái bền vững trên Volume:

| Nội dung | Đường dẫn |
| --- | --- |
| Input bundle và manifest | `/mnt/task2/input/` |
| Workspace đã giải nén | `/mnt/task2/workspace/` |
| Qwen cache | `/mnt/task2/models/qwen3-legal/` |
| Smoke pass | `/mnt/task2/results/smoke.json` |
| Run theo contract | `/mnt/task2/results/runs/<CONTRACT>/` |
| Full summary | `/mnt/task2/results/full_run_summary.json` |
| Failure payload | `/mnt/task2/results/smoke_failure.json` hoặc `full_failure.json` |
| Result ZIP trên Beam | `/mnt/task2/results/task2_p14_beam_result.zip` |

`<CONTRACT>` là mã 24 ký tự sinh từ SHA-256 của input bundle, source code và
contract version `p14-fast-v4`. Đổi code sẽ tự tạo run mới, không tái dùng nhầm
output cũ. Khi chạy lại cùng contract, pipeline bỏ qua các output đã hoàn thành.
Parent cross-encoder giữ `checkpoint-latest`; Qwen LoRA giữ
`checkpoint-epoch-*`; cả hai gọi training với `--resume`. Output dở dang không
có checkpoint/manifest hợp lệ sẽ được làm sạch thay vì dùng nhầm. Vì vậy sau lỗi
có thể chạy lại đúng lệnh `-Stage Run` ở trên, nhưng phải kiểm tra Status/Logs
trước để chắc chắn task headless cũ đã kết thúc.

## Kết quả

`-Stage Run` tự download kết quả. Có thể download lại riêng bằng:

```powershell
.\scripts\cloud\run_task2_p14_beam.ps1 -Stage Download
```

Đường dẫn remote và local:

```text
beam://udsc-task2-p14/results/task2_p14_beam_result.zip
artifacts/task2/beam_results/task2_p14_beam_result.zip
```

Wrapper ghi `artifacts/task2/beam_launch_manifest.json` trước khi chạy và chỉ
chấp nhận ZIP có `full_run_summary.json`, đúng SHA-256 input, source code và
contract `p14-fast-v4` của lần launch đó:

- `HELDOUT_REJECTED`: không nộp;
- `PUBLIC_CANDIDATE_READY`: `submission.zip` bên trong chỉ là **ứng viên** để
  nộp Codabench.

Gate held-out METEOR `0.57` chỉ là điều kiện nội bộ. Pipeline không hứa điểm,
thứ hạng hoặc kết quả leaderboard; chỉ kết quả Codabench chính thức mới xác lập
điểm public.
