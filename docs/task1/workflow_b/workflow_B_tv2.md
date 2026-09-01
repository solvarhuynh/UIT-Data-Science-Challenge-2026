# Workflow B — TV2

## Mục tiêu
Giải thích hiện trạng nghiên cứu TV2, câu hỏi khoa học, các cổng (gates) thực thi, các nhánh tương lai và logic quyết định.

---
### GIAI ĐOẠN TV2-0 — HIỆN TẠI
**B2a-0 ZERO‑SHOT SEMANTIC RERANKER**

**TRẠNG THÁI:** ACTIVE — CONTEXT GATE BLOCKED

- **Mô hình:** `Qwen/Qwen3-Reranker-0.6B` (revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`)
- **Candidate pool:** K77 (không thay đổi)
- **Câu hỏi khoa học:** Với cùng 77 candidate documents/query, liệu Qwen đọc trực tiếp Query + Document có tạo được top‑5 tốt hơn hệ thống hiện tại không?

**Pipeline yêu cầu:**
1. Model provenance
2. CPU context/truncation audit
3. Modal/CUDA GPU smoke
4. Full zero‑shot inference
5. 77 semantic scores/query
6. Direct top‑5
7. Prediction freeze
8. F1‑F4 Recall/Precision
9. Professor result review

**Ghi chú:** B2a‑0 chưa có kết quả Recall/Precision; đây vẫn là giai đoạn preflight/audit.

**Trạng thái hiện tại:**
- Provenance PASS
- Context audit execution: PASS
- Context feasibility gate: BLOCKED_LONG_DOCUMENT
- Modal GPU smoke: BLOCKED (context gate)
- Full inference chưa chạy
- Kết quả khoa học NOT_YET_EVALUATED

---
### GIAI ĐOẠN TV2-1 — FINE‑TUNING
**B2a-1 Fine‑tuned Reranker**

**TRẠNG THÁI:** DEFERRED / NOT_AUTHORIZED

**CHỦ SỞ HỮU (nếu mở):** TV2

Fine‑tuning thuộc TV2 vì:
- GPU‑heavy
- Leverage điểm cao
- Thay đổi trực tiếp kiến trúc neural

Fine‑tuning **KHÔNG** được thực hiện trước B2a‑0 hoàn thiện (top‑5, prediction freeze, Recall/Precision, professor review).

**Phương pháp tiềm năng:** full fine‑tuning, LoRA / PEFT, hard‑negative training (chưa được ủy quyền).

**Yêu cầu preregistration:** training folds, negative sampling, loss, optimizer, LR, epochs, checkpoint, context handling, prediction freeze, evaluation contract, success rule, kill rule. **Fold0** và **public labels** bị CẤM.

---
### GIAI ĐOẠN TV2-2 — LONG‑DOCUMENT NEURAL RERANKING
**TRẠNG THÁI:** CONDITIONAL

Chỉ chạy khi B2a‑0 cho thấy vấn đề context/truncation thực tế.

**Hướng tương lai:** chunk‑aware reranking, multi‑chunk aggregation, section‑aware reranking, long‑context strategy.

---
### GIAI ĐOẠN TV2-3 — STRONGER / ALTERNATIVE RERANKER
**TRẠNG THÁI:** CONDITIONAL

Chỉ chạy khi bằng chứng chứng tỏ checkpoint 0.6B là giới hạn.

---
### GIAI ĐOẠN TV2-4 — EMBEDDING / RETRIEVAL ARCHITECTURE
**TRẠNG THÁI:** DEFERRED

Thuộc TV2 nếu trở thành nhánh kiến trúc có leverage cao nhất.

---
### GIAI ĐOẠN TV2-5 — MULTI‑STAGE INTEGRATION
**TRẠNG THÁI:** LATE_STAGE / DEFERRED

Kết hợp retrieval → cheap ranking → semantic reranker → final top‑5 khi mỗi thành phần đã có bằng chứng giá trị.

---
## Quy tắc phân GPU

Nếu tại một thời điểm chỉ có **MỘT** nhánh GPU quan trọng → **TV2** sở hữu nhánh đó.

Nếu có **HAI** hoặc nhiều nhánh GPU độc lập được chứng minh đáng chạy → **TV2** giữ nhánh có **expected score leverage** cao nhất hoặc **architecture/integration risk** lớn nhất, **TV4** có thể nhận nhánh GPU quan trọng thứ hai.

---
## Chủ sở hữu Fine‑tuning

Qwen fine‑tuning / LoRA **thuộc TV2** nếu được ủy quyền, vì nó thay đổi kiến trúc neural chính, yêu cầu GPU và có leaverage điểm cao.

TV4 có thể cung cấp evidence (hard‑negative, error analysis, ranking diagnostics) nhưng **KHÔNG** sở hữu fine‑tuning.

---
## Địa chỉ báo cáo

- `reports/task1/workflow_b/tv2/` ← artifact Workflow-B của TV2
- `reports/task1/workflow_b/tv4/` ← artifact Workflow-B của TV4
- `reports/task1/workflow_b/shared/` ← định nghĩa Workflow-B dùng chung
- `reports/task1/progress_log.md` ← tiến độ
