# Gói Bàn Giao Kỷ Lục Mới Task 1 (Recall 0.92054)

## 1. Kết quả CodaBench Chính thức (1.977 câu unseen)
- **Macro Recall: 0.920544597** (Kỷ lục cao nhất của đội, vượt mốc 0.920)
- **Macro Precision: 0.196661609** (1.944/5 hits đúng trong Top 5)
- **File nộp:** `submission_constrained_dual_anchor_rrf.zip`

---

## 2. Danh mục Gói Bàn Giao
1. **Thư mục `code/`**:
   - `build_constrained_dual_anchor_rrf_submission.py`: Script chính sinh ra bản kỷ lục **0.92054**.
   - `build_dual_anchor_safeguard_submission.py`: Script sinh ra bản tiền đề **0.91978**.
2. **Thư mục `artifacts/`**:
   - `submission_constrained_dual_anchor_rrf.zip`: File zip submission đạt **0.92054**.
   - `submission_dual_anchor_v2_guarded.zip`: File zip submission đạt **0.91978**.
3. **Thư mục `logs/`**:
   - `scoring_result_09205.zip`: File zip trả về từ CodaBench (chứa `scores.json` và `metadata` của BTC).
   - `run_log.txt`: Nhật ký chạy script và các chỉ số validation.

---

## 3. Tóm tắt Ý Tưởng Kỹ Thuật (Vì sao điểm tăng?)
- Thay vì lấy các mô hình yếu (v1 0.887, v3 0.870, v4 0.889) để bầu chọn làm tụt điểm, ta kết hợp **2 mô hình mạnh nhất đội (>0.9167) độc lập nhau**:
  1. `highest_submission_private_guarded_direct.zip` (0.9174) của bạn.
  2. `submission_v2.zip` (0.9168) từ 4-Source Weighted RRF.
- **Phương pháp Constrained RRF**:
  - Thực hiện Rank Reciprocal Fusion ($k=60$) với trọng số 0.55 cho Guarded Direct và 0.45 cho V2.
  - **Khóa bảo hiểm (Hard Guardrails)**: Tài liệu Rank 1 của cả 2 mô hình **bắt buộc luôn nằm trong Top 5** (không bao giờ bị đá văng ra ngoài).
  - Kết quả: Các văn bản Rank 2 và Rank 3 có độ tin cậy cao của cả 2 bên cùng tương hỗ kéo nhau lên Top 5, giúp Recall tăng thêm +1.5 câu đúng nữa và Precision tiếp tục tăng.
