# Task Results — TV4

Đây là bản tóm tắt theo trình tự thời gian của các task TV4 đã hoàn tất hoặc bị chặn. Bằng chứng chi tiết vẫn nằm trong các report riêng của từng experiment.

Chưa có task TV4 nào được ghi nhận là đã hoàn tất hoặc bị chặn trong ledger này. B1 Direct LTR hiện vẫn ở trạng thái **NOT_YET_EVALUATED**; chưa có kết quả Recall/Precision để báo cáo và không được tự tạo kết quả thay thế.

KHỞI ĐỘNG UBUNTU_MODAL

# PowerShell
wsl hoặc nhìn lên trên cùng có dấu mũi tên đi xuống ấn vào đó vào thẳng Ubuntu

# Vào thư mục 
cd "/mnt/d/Trung Khang/Documents/Project_UIT DATA SCIENCE CHALLENGE/UIT-Data-Science-Challenge-2026"

# Activate TV4 environment
source ~/venvs/udsc2026/bin/activate  (Như báo cáo Audit vừa chỉ ra, phiên bản Python 3.14 không có sẵn file cài đặt cho scikit-learn 1.7.2 => chuyển xuống python 3.12 bằng lệnh ở bên dưới)

source ~/venv_b1_py312/bin/activate

# Check
python --version
which python
python -m modal --version

## 2026-09-02 — B1 — Resolve Environment Block & Smoke Test

Mục tiêu: Kiểm tra môi trường Python 3.12 với LightGBM 4.5.0 và sklearn 1.7.2 để gỡ block.
Đã làm: Chạy B1 compatibility smoke test bằng môi trường ~/venv_b1_py312.
Kết quả chính: Lỗi force_all_finite đã được giải quyết, smoke test chạy thành công.
Trạng thái: PASS
Điều rút ra: Môi trường đã tương thích hoàn toàn với hợp đồng đóng băng của B1.
Chưa được kết luận: Điểm Recall/Precision cuối cùng (vì chưa chạy mô hình thực tế).
Artifact chính: reports/task1/workflow_b/tv4/b1/contracts/workflow_b_b1_compatibility_smoke.json
Bước tiếp theo: Chạy full pipeline B1 (Runtime recovery & OOF scoring).