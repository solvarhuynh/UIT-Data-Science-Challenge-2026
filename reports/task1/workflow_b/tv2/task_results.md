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

