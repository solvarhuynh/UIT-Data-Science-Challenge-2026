# TV5 — chạy Task 2 P14 trên Modal

Pipeline này thay lớp hạ tầng Beam bằng Modal nhưng giữ nguyên dữ liệu, model và
logic đánh giá P14 đã kiểm thử. GPU được chọn theo thứ tự `H100 → A100 80GB →
L40S`; Modal chỉ dùng một GPU.

## Chạy lần đầu

Mở PowerShell tại thư mục gốc repository:

```powershell
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Setup
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Check
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Run
```

`Setup` cài Modal SDK và mở bước đăng nhập. URL trang endpoint trong dashboard
không phải tham số đầu vào; runner tự tạo app `udsc-task2-p14-modal` và Volume
`udsc-task2-p14-modal` trong đúng workspace Modal đang đăng nhập.

`Run` thực hiện tuần tự:

1. kiểm tra SHA-256 archive 821 MB;
2. chỉ upload archive nếu Modal chưa có đúng bản đó;
3. tải/cache đúng revision model Qwen trên Modal Volume;
4. chạy Smoke và Full trên cùng một GPU allocation;
5. chấm held-out, chỉ sinh submission nếu đạt gate METEOR;
6. tải result ZIP về máy.

Không chạy thêm một lệnh `Run` khác trong lúc job còn hoạt động. Lệnh đã dùng
chế độ detach, nên GPU job đã submit vẫn tiếp tục nếu terminal mất kết nối.

## Xem và tải kết quả

```powershell
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Logs
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Status
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Download
```

`Logs` stream tiến độ trực tiếp. Nhấn `Ctrl+C` chỉ đóng màn hình log; job đã
detach vẫn chạy. Chỉ khi thật sự muốn dừng GPU và ngừng tính credit mới dùng:

```powershell
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Stop
```

File tải về:

```text
artifacts/task2/modal_download/task2_p14_modal_result.zip
```

Chỉ nộp `submission.zip` nằm bên trong result ZIP khi summary có:

```text
status = PUBLIC_CANDIDATE_READY
```

Nếu summary là `HELDOUT_REJECTED`, không nộp bản đó. Result vẫn giữ metrics và
checkpoint để phân tích; không tự động chạy lại và không đốt thêm credit.

## Sinh nhanh public candidate từ checkpoint đã train

Sau khi Full đã hoàn tất nhưng Qwen đơn đạt `0.516506`, dùng stage `Public`:

```powershell
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Public
```

Stage này không train lại. Nó tái dùng parent listwise và Qwen LoRA trong Modal
Volume, kiểm tra ensemble label-free đạt strict held-out `0.562459`, rồi chỉ
rerank và sinh 1.000 câu public. Exact organizer matches được giữ nguyên; các
câu còn lại ghép 352 từ chứng cứ extractive trước câu trả lời Qwen. Lượt nộp cũ
cho thấy exact overlay từng tăng public METEOR khoảng `0.0233`, nên mốc public
`0.57–0.58` là khả thi nhưng không được bảo đảm trước khi Codabench chấm.

Chỉ lấy `submission.zip` khi summary mới có
`status = PUBLIC_CANDIDATE_READY`.

## Chạy từng bước khi cần chẩn đoán

```powershell
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Prepare
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Smoke
.\scripts\cloud\run_task2_p14_modal.ps1 -Stage Full
```

Luồng bình thường nên dùng `Run`, vì Smoke và Full dùng cùng một GPU allocation,
nhanh hơn và tránh phải chờ cấp GPU lần thứ hai.
