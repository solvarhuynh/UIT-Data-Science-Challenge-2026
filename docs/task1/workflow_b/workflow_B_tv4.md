# Workflow B — TV4

## Mục tiêu
Giải thích hiện trạng nghiên cứu TV4, câu hỏi khoa học, các cổng (gates) thực thi, các nhánh tương lai và logic quyết định.

## Trạng thái vận hành hiện tại

**[CONTINUE] TV4 không bị ảnh hưởng bởi việc đổi reranker của TV2.** B1 là
nhánh khoa học độc lập và không restart từ đầu.

---
### GIAI ĐOẠN TV4-0 — CONTINUE POINT
**B1 DIRECT LEARNING‑TO‑RANK**

**TRẠNG THÁI:** [CONTINUE] ACTIVE / NOT_YET_EVALUATED

- **Mô hình:** (đã frozen theo hợp đồng B1)
- **Candidate pool:** K77 (không thay đổi)
- **Câu hỏi khoa học:** Với canonical K77 và bộ thông tin handcrafted đã frozen, liệu Direct Learning‑to‑Rank cải thiện top‑5 cuối cùng không?

**TV4 CONTINUE POINT: B1 FULL PIPELINE**
1. Runtime recovery.
2. Direct LTR training/scoring under the frozen B1 contract.
3. OOF predictions.
4. F1-F4 Recall/Precision evaluation.
5. Frozen prediction artifact and SHA256.
6. Per-fold deltas.
7. Scientific status.

**[REUSE]** B0, K77, canonical folds, existing data contracts and valid
compatibility artifacts remain unchanged. TV4 does not rerun B0. The B1
compatibility smoke is `PASS`; rerun it only if the environment changes or
the existing smoke artifact becomes invalid.

**[WAIT]** Không được phát sinh hoặc điền Recall/Precision B1 khi full OOF
pipeline chưa hoàn tất.

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
## Ranh giới TV2 / TV4

TV2 B2a và TV4 B1 chạy độc lập; chưa được fuse. Chỉ bắt đầu fusion/comparison
sau khi cả hai nhánh đều có frozen prediction artifact, SHA256, F1-F4 Recall,
F1-F4 Precision, per-fold results và scientific status. Cho đến lúc đó, TV2
tiếp tục B2a 2B còn TV4 tiếp tục B1.

Việc TV2 đổi `Qwen/Qwen3-Reranker-0.6B` sang `Qwen/Qwen3-VL-Reranker-2B`
không reset, không sửa và không làm TV4 rerun B1.

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
- `reports/task1/tv4/progress_log_4.md` ← tiến độ
