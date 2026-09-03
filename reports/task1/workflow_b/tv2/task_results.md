# Task Results — TV2

Đây là bản tóm tắt theo trình tự thời gian của các task TV2 đã hoàn tất hoặc bị chặn. Bằng chứng chi tiết vẫn nằm trong các report riêng của từng experiment.

## 2026-09-01 — B2a-0 — Context feasibility audit

**Mục tiêu:** Kiểm tra model provenance và khả năng xử lý context trước khi chạy Qwen zero-shot trên K77.

**Đã làm:** Xác nhận checkpoint Qwen/Qwen3-Reranker-0.6B tại revision e61197ed45024b0ed8a2d74b80b4d909f1255473. Audit toàn bộ 431.200 query–document pairs và ghi nhận phân bố độ dài.

**Kết quả chính:** Model audit PASS; context audit execution PASS; feasibility BLOCKED_LONG_DOCUMENT. p99 = 158.503 token, max = 2.145.502 token.

**Trạng thái:** BLOCKED

**Điều rút ra:** Chưa đủ điều kiện chạy full inference hợp lệ.

**Chưa được kết luận:** Qwen có cải thiện Recall/Precision hay không.

**Artifact chính:** Chưa ghi nhận trong ledger này.

**Bước tiếp theo:** Long-document root-cause audit.

## 2026-09-03 — B2a-0 — Frozen Qwen F1-F4 scientific evaluation

**Mục tiêu:** Đánh giá prediction artifact Qwen3-Reranker-0.6B đã freeze trên đúng F1–F4 bằng metric set-based macro Recall/Precision.

**Đã làm:** Xác minh SHA256 `9f73f9424de78a0824c1e7153d080890cfe599104c103a4e2e81b5763b03fcd9`, rồi dùng canonical scorer với `strict_cv_v2/folds.json` và `data/raw/btc/LegalIR/train.json`. Comparator là Workflow-A baseline tại `reports/task1/workflow_a/step2p1_oof_policy_decisions.jsonl`.

**Kết quả chính:** Qwen Recall `0.1593363095238095`, Precision `0.03410714285714286`; delta lần lượt `-0.7665922619047619` và `-0.16328571428571428`. Recall delta theo F1–F4 đều âm (`-0.7854166666666668`, `-0.7729761904761905`, `-0.7611309523809524`, `-0.7468452380952381`).

**Trạng thái:** FAIL — `REJECT_B2A_QWEN_06B`.

**Điều rút ra:** Không có tín hiệu Qwen đạt materiality hoặc fold consistency; 25/5.600 query tốt hơn, 1.165 bằng, 4.410 kém hơn theo Recall.

**Chưa được kết luận:** Không có quyết định tuning, inference mới hoặc prediction thay thế.

**Artifact chính:** Frozen prediction path ở `artifacts/task1/workflow_b/tv2/b2a/qwen06b/predictions.jsonl`; kết quả đầy đủ chỉ in terminal; chưa tạo report persistent.

**Bước tiếp theo:** Professor review kết quả B2a.
