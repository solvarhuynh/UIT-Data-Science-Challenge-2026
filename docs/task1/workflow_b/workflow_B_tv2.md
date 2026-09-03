# Workflow B — TV2

## Mục tiêu
Giải thích hiện trạng nghiên cứu TV2, câu hỏi khoa học, các cổng (gates) thực thi, các nhánh tương lai và logic quyết định.

## Trạng thái vận hành hiện tại

**[CONTINUE] TV2 không khởi động lại Workflow B.** Thay đổi chỉ nằm ở
reranker; retrieval, candidate population và các hợp đồng khoa học vẫn giữ
nguyên.

---
### GIAI ĐOẠN TV2-0 — MODEL CHANGE / RESTART POINT
**B2a-0 ZERO‑SHOT SEMANTIC RERANKER**

**TRẠNG THÁI:** [RERUN] ACTIVE — RESTART TẠI B2a MODEL EXECUTION

- **Mô hình:** `Qwen/Qwen3-VL-Reranker-2B` (revision xem trong `models/download_manifest.json`)
- **Candidate pool:** K77 (không thay đổi)
- **Câu hỏi khoa học:** Với cùng 77 candidate documents/query, liệu Qwen đọc trực tiếp Query + Document có tạo được top‑5 tốt hơn hệ thống hiện tại không?

**TV2 RESTART POINT: B2a MODEL EXECUTION — QWEN3-VL-RERANKER-2B**

**[RERUN] Required sequence:**
1. Verify exact model revision/provenance.
2. Verify Qwen3-VL processor/model/scoring interface.
3. Run True-S2 compatibility smoke using the existing worklist.
4. Run GPU batch benchmark for 2B and select a safe batch size.
5. Run full F1-F4 inference on the same 431,200 query-document worklist.
6. Write outputs in a new 2B checkpoint/artifact namespace.
7. Freeze the 2B prediction artifact and compute SHA256 before labels.
8. Evaluate F1-F4 Recall/Precision against the same Workflow-A scientific reference.
9. Submit for professor review.

Không được dùng lại Qwen 0.6B SQLite checkpoint hoặc Qwen 0.6B scores. Artifact
0.6B là bằng chứng lịch sử bất biến, không phải input cho kết quả 2B.

**Ghi chú:** B2a‑0 của checkpoint 2B chưa có kết quả Recall/Precision; đây vẫn
là giai đoạn compatibility/inference chưa hoàn tất.

**Trạng thái hiện tại:**
- Provenance 2B: [WAIT] phải verify trước khi chạy.
- Processor/model/scoring interface: [WAIT].
- True-S2 compatibility smoke: [WAIT].
- Full inference: [WAIT].
- Kết quả khoa học 2B: NOT_YET_EVALUATED.

### TV2 — [REUSE] DO NOT RERUN

Các thành phần sau được kế thừa nguyên trạng; đổi reranker không làm chúng
thay đổi:

- B0 / existing baseline và canonical folds.
- Scientific Recall/Precision metric contract và Workflow-A comparator.
- Full candidate pool provenance và derived F1-F4 K77 candidate set.
- K77 membership, canonical raw processed chunks và True-S2 selected-chunk worklist.
- 431,200 F1-F4 query-document pairs và 1,293,198 selected chunks.
- Evaluator và scientific decision rules.

**[LOCKED]** Không rebuild K77, không rebuild True-S2, không rerun B0.

### TV2 — [HISTORICAL] Qwen3-Reranker-0.6B

Kết quả B2a cũ được giữ nguyên làm historical evidence, không được diễn giải
lại thành pending experiment:

- Status: `REJECT_B2A_QWEN_06B`.
- Frozen Recall: `0.1593363095238095`.
- Frozen Precision: `0.03410714285714286`.
- Prediction SHA256: `9f73f9424de78a0824c1e7153d080890cfe599104c103a4e2e81b5763b03fcd9`.
- Sanity audit: `TRUE_QWEN_06B_MODEL_FAILURE`.

Các score, SQLite checkpoint và frozen predictions 0.6B không được dùng cho
checkpoint 2B.

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

Chỉ mở sau khi B2a-2B có kết quả hợp lệ và có review; không phải công việc hiện tại.

---
### GIAI ĐOẠN TV2-4 — EMBEDDING / RETRIEVAL ARCHITECTURE
**TRẠNG THÁI:** DEFERRED

Thuộc TV2 nếu trở thành nhánh kiến trúc có leverage cao nhất.

Đổi riêng reranker không yêu cầu rebuild K77 hoặc True-S2. Nếu sau này đổi
embedding, ví dụ sang `Qwen/Qwen3-VL-Embedding-2B`, phải regenerate theo chuỗi:
`embeddings → vector index → retrieval candidates → candidate pool/K77 →
True-S2 selections → reranker inference`. Chuỗi này không thuộc model change
hiện tại.

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
- `reports/task1/tv2/progress_log_2.md` ← tiến độ
