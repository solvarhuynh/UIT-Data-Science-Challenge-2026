# Workflow B — TV4

## Mục tiêu
Giải thích hiện trạng nghiên cứu TV4, câu hỏi khoa học, các cổng (gates) thực thi, các nhánh tương lai và logic quyết định.

---
### GIAI ĐOẠN TV4-0 — HIỆN TẠI
**B1 DIRECT LEARNING‑TO‑RANK**

**TRẠNG THÁI:** ACTIVE / NOT_YET_EVALUATED

- **Mô hình:** (đã frozen theo hợp đồng B1)
- **Candidate pool:** K77 (không thay đổi)
- **Câu hỏi khoa học:** Với canonical K77 và bộ thông tin handcrafted đã frozen, liệu Direct Learning‑to‑Rank cải thiện top‑5 cuối cùng không?

**Pipeline yêu cầu:**
1. Runtime recovery
2. Detached execution
3. Full B1 model run
4. OOF scoring
5. Top‑5 generation
6. Recall/Precision evaluation
7. Per‑fold report

**Ghi chú:** Đây là thí nghiệm chính hiện tại của TV4 và **không** cần chạy mọi thí nghiệm tương lai đồng thời.

---
### GIAI ĐOẠN TV4-0B – HISTORICAL RERANKER AUTOPSY
**TRẠNG THÁI:** RUN_NOW_WHEN_CAPACITY_AVAILABLE

Kiểm tra nguyên nhân hồi quy của neural reranker cũ (checkpoint, candidate, input, truncation, training positives, negative sampling, objective, splits, fusion, post‑processing). Đây là công việc phụ, không ảnh hưởng đến B1.

---
### GIAI ĐOẠN TV4-1 — TARGETED LTR FEATURE EXPANSION
**TRẠNG THÁI:** CONDITIONAL

Chạy **sau** khi có kết quả B1 hợp lệ. Xác định các tính năng bổ sung (P4 probability, source‑specific ranks, rank gaps, document length, legal metadata, query structure…) có thể cải thiện Direct LTR.

---
### GIAI ĐOẠN TV4-2 — SEMANTIC‑AUGMENTED LTR
**TRẠNG THÁI:** CONDITIONAL

Nếu B2a‑0 (TV2) cho điểm semantic hữu ích, kết hợp điểm Qwen với LTR hiện tại để kiểm tra việc tăng điểm.

---
### GIAI ĐOẠN TV4-3 — HARD‑NEGATIVE RANKING
**TRẠNG THÁI:** CONDITIONAL

Sau khi có lỗi phân loại tài liệu gây nhầm lẫn (từ B1 hoặc B2a‑0), xây dựng tập hard‑negative để cải thiện ranking.

---
### GIAI ĐOẠN TV4-4 — SECONDARY RANKER
**TRẠNG THÁI:** CONDITIONAL

Nếu B1 để lại headroom, thử một comparator (LambdaRank, pairwise, listwise) một lúc một.

---
### GIAI ĐOẠN TV4-5 — K77 RECOVERABILITY
**TRẠNG THÁI:** CONDITIONAL

Khi Recall bắt đầu bão hòa, đánh giá phần query còn recoverable trong K77.

---
### GIAI ĐOẠN TV4-6 — SECOND GPU / RETRIEVAL BRANCH
**TRẠNG THÁI:** DEFERRED

Chỉ thuộc TV4 nếu có bằng chứng cho một nhánh GPU‑heavy độc lập và TV2 đã sở hữu nhánh cao hơn.

---
## Quy tắc phân GPU

Nếu chỉ có **MỘT** nhánh GPU quan trọng → **TV2** sở hữu.
Nếu có **HAI** hoặc hơn nhánh GPU độc lập → **TV2** giữ nhánh có **expected score leverage** hoặc **architecture risk** cao nhất, **TV4** có thể nhận nhánh thứ hai.

---
## Chủ sở hữu Fine‑tuning

Qwen fine‑tuning / LoRA **thuộc TV2** nếu được ủy quyền vì nó thay đổi kiến trúc neural chính.
TV4 chỉ cung cấp evidence (hard‑negative, error analysis, ranking diagnostics) nhưng **KHÔNG** sở hữu fine‑tuning.

---
## Địa chỉ báo cáo

- `reports/task1/workflow_b/tv2/` ← artifact Workflow-B của TV2
- `reports/task1/workflow_b/tv4/` ← artifact Workflow-B của TV4
- `reports/task1/workflow_b/shared/` ← định nghĩa Workflow-B dùng chung
- `reports/task1/progress_log_4.md` ← tiến độ
