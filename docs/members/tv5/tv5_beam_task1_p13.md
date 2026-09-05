# TV5 — Task 1 P13 trên Beam RTX 4090/5090

Đây là đường chạy chính khi đã có Beam. Pipeline mặc định dùng một RTX 4090, lưu
archive/checkpoint trong Beam Volume và không phụ thuộc phiên terminal còn mở.
Không chạy đồng thời hai job: volume có khóa độc quyền để ngăn ghi đè fold.

Không thể đảm bảo trước điểm Codabench lớn hơn 0.9591. Pipeline chỉ tạo public
candidate khi fine-tuned reranker thắng pretrained BGE trên đủ 5 fold OOF; report
ghi thêm `target_proxy_pass` khi OOF Recall đạt ít nhất 0.9591. Leaderboard vẫn là
bằng chứng cuối cùng.

## 1. Cấu hình đã khóa

- GPU mặc định: `RTX4090`, đúng một GPU; có thể chọn lại `RTX5090` bằng
  `-GpuType RTX5090` khi cụm 5090 ổn định;
- client đã kiểm: `beam-client==0.2.207`, `beta9==0.1.265`;
- image: `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`;
- model: `BAAI/bge-reranker-v2-m3` local, không tải lại từ Hub;
- candidate depth: 200, candidate Recall đã đo là 0.985574;
- 24.000 negative pairs cân bằng/query/fold, tương ứng 48.000 binary examples;
- 5 fold group-disjoint, 2 epoch cố định, không chọn epoch trên outer fold;
- train batch 8, gradient accumulation 4, inference batch 32, max length 512;
- full job luôn bật resume và lưu từng fold trong Beam Volume.

Smoke dùng đúng batch/max length của full job, chạy 800 pairs và 20 validation
queries. Nó phải thực hiện CUDA matmul, forward/backward, save/reload checkpoint
và ghi marker theo run contract trong `results/smoke/` trước khi full job được
phép chạy.

## 2. File và Beam Volume

Input local:

```text
artifacts/task1/task1_p13_kaggle_upload.tar.zst
```

- bytes: `1.763.448.793`;
- SHA-256: `21dda96ecab20686ef81b85d1f6813d8c6fa0d80f892e43c61604a3f90469921`;
- Beam Volume: `udsc-task1-p13`;
- remote input: `beam://udsc-task1-p13/input/task1_p13_kaggle_upload.tar.zst`.

`.beta9ignore` (tên mà SDK Beam 0.2.207 thực sự đọc) và `.beamignore` loại toàn bộ
data/model/artifact lớn khỏi source sync. Archive chỉ upload một lần bằng
`Prepare`; code hiện tại được Beam sync và overlay lên workspace đã giải nén ở
đầu mỗi job. Kiểm tra bằng đúng `FileSyncer` cho thấy source còn dưới 200 file,
khoảng 1,4 MiB; không còn tải nhầm `.cache`, `__pycache__`,
`frontend/node_modules`, data hoặc model trong mỗi lần gọi job.

Mỗi lần chạy được khóa bằng `run_contract_sha256`, gồm hash toàn bộ `configs/`,
`scripts/`, `src/`, `pyproject.toml`, archive, image, package và hyperparameter.
Checkpoint của hai phiên bản code khác nhau nằm ở hai thư mục khác nhau nên
`--resume` không thể trộn fold cũ với code mới.

## 3. Lệnh PowerShell

Chạy từ repo root. `Prepare`, `Smoke` và `Full` đều kiểm tra chính xác remote
path, kích thước và SHA archive trước khi tạo job. Probe CPU đọc trực tiếp file
qua mount `/mnt/task1`, không tin kích thước cache của `stat_path`. Nếu file
1.763.448.793 byte đã hợp lệ thì bỏ qua upload; nếu thiếu/sai thì wrapper xóa
đúng file partial, kiểm tra SHA local, upload multipart lại và probe qua mount
lần nữa trước khi báo PASS. Trên Windows, wrapper sửa riêng phép nối remote path
của Beam CLI 0.2.207 để không đổi dấu `/` thành `\`. Không đóng terminal trong
lần upload thực sự đầu tiên.

`Smoke` và `Full` được gọi bằng `handler.remote()` trong Python environment của
Beam. Không thay chúng bằng `beam run ...`: ở Beam 0.2.207, lệnh `run` chỉ nhận
`Pod`, không nhận hàm trang trí bằng `@function`.

Mỗi handler bắt lỗi runtime và trả `SMOKE_FAILED`/`FULL_FAILED` kèm traceback
trực tiếp về terminal, đồng thời lưu `results/<stage>_failure.json`. Cơ chế này
không phụ thuộc `beam logs`: hướng dẫn cài đặt Beam yêu cầu WSL khi dùng Windows,
còn client native Windows có thể lỗi TLS với WebSocket log dù Gateway vẫn hoạt
động. Vì vậy `Logs` chỉ là công cụ tùy chọn, không phải bước bắt buộc.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\scripts\cloud\run_task1_p13_beam.ps1 -Stage Check
.\scripts\cloud\run_task1_p13_beam.ps1 -Stage Prepare
.\scripts\cloud\run_task1_p13_beam.ps1 -Stage Smoke
.\scripts\cloud\run_task1_p13_beam.ps1 -Stage Full
.\scripts\cloud\run_task1_p13_beam.ps1 -Stage Status
.\scripts\cloud\run_task1_p13_beam.ps1 -Stage Logs -TaskId <TASK_ID>
.\scripts\cloud\run_task1_p13_beam.ps1 -Stage Download
```

Wrapper tự tìm `beam.exe` của `uv tool`, kể cả khi lệnh `beam` chưa có trong
`PATH` của terminal đang kích hoạt `.venv`. `Smoke` và `Full` tự ghi client log
vào `artifacts/task1/beam_smoke_client.log` và
`artifacts/task1/beam_full_client.log`; `Logs` thử lưu log remote theo Task ID
nhưng có thể không dùng được trên native Windows do endpoint TLS của Beam.

`Check` không tạo máy và không tốn GPU. Nó chặn hai lỗi riêng của Beam 0.2.207
trên Windows trước khi submit: handler chứa dấu `\` không import được trên Linux
và volume mount bị serialize thành `\mnt\task1` thay vì `/mnt/task1`. Nó cũng
chặn việc vô tình chạy wrapper bằng một phiên bản SDK khác với bản đã kiểm thử.

Không chạy `Full` nếu `Smoke` lỗi. `Full` có thể kéo dài nhiều giờ; function dùng
`headless=True`, `timeout=-1`, `retries=2`. Lỗi trong code được guarded handler
trả về ngay; retry chủ yếu bảo vệ lỗi cấp container/GPU host trước khi code chạy.
Nếu terminal bị ngắt, xem trạng thái:

```powershell
.\scripts\cloud\run_task1_p13_beam.ps1 -Stage Status
```

Xem `artifacts/task1/beam_<stage>_client.log` trước. Chỉ thử `-Stage Logs` nếu cần
và endpoint realtime hoạt động. Nếu full job dừng giữa chừng, chạy lại
`-Stage Full`; đúng run contract sẽ tiếp tục fold hoàn chỉnh, còn code/config
đổi sẽ tự dùng namespace mới. Không xóa checkpoint trong volume.

## 4. Gate và output

Sau full OOF, `final_decision.json` phải có:

```json
{
  "status": "PROMOTE_CANDIDATE",
  "promotable": true,
  "complete_oof": true
}
```

Nếu không đạt, job ghi `OOF_REJECTED`, không chạy public inference và không tạo
submission mới. Nếu đạt, job ensemble 5 fold, tạo submission và chạy validator.

`Download` lấy file:

```text
artifacts/task1/beam_results/task1_p13_beam_result.zip
```

Trên Windows, `Download` cũng dùng bộ nối POSIX riêng để tránh lỗi `unable to
find volume`. File được tải vào đuôi `.partial`, kiểm CRC/trạng thái/các artifact
bắt buộc và đối chiếu hash với toàn bộ code local rồi mới đổi tên nguyên tử. Vì
vậy một ZIP cũ từ lần Full trước không thể bị nhận nhầm là kết quả của code hiện
tại; bản ZIP tốt trước đó cũng không bị ghi đè nếu tải hoặc kiểm tra thất bại.

Trong ZIP có summary, OOF comparison, 5 fold metrics, public report,
`predictions.json` và `submission.zip` nếu gate pass. Checkpoint nặng vẫn nằm ở:

```text
beam://udsc-task1-p13/workspace/artifacts/task1/models/bge_reranker_finetune/beam_full_oof
```

Không ghi đè `artifacts/task1/submission.zip`; đó là bản rollback leaderboard
0.9309. Không tự động upload Codabench.
