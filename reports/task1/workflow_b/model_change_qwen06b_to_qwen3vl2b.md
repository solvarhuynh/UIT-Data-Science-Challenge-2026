# Task1 Workflow-B Model Change Impact

## 1. What changed

Old:
`Qwen/Qwen3-Reranker-0.6B`

New:
`Qwen/Qwen3-VL-Reranker-2B`

Production/runtime configuration đã được cập nhật sang reranker mới. Đây là
thay đổi model-only ở TV2 B2a; không thay đổi retrieval, K77, True-S2 hay
metric/scientific contract.

## 2. Why the old model was retired

Kết quả frozen của thí nghiệm cũ được giữ nguyên, không reinterpret:

- Status: `REJECT_B2A_QWEN_06B`
- Recall: `0.1593363095238095`
- Precision: `0.03410714285714286`
- Prediction SHA256: `9f73f9424de78a0824c1e7153d080890cfe599104c103a4e2e81b5763b03fcd9`
- Sanity audit: `TRUE_QWEN_06B_MODEL_FAILURE`

## 3. What remains reusable

Các artifact sau vẫn canonical và được reuse:

- B0, canonical folds và Workflow-A scientific comparator.
- Scientific Recall/Precision metric contract và evaluator/decision rules.
- Full candidate pool provenance và derived F1-F4 K77 candidate set.
- K77 membership, canonical raw processed chunks và True-S2 selected-chunk worklist.
- 431,200 F1-F4 query-document pairs và 1,293,198 selected chunks.

Không cần restart Workflow B từ đầu.

## 4. What TV2 must rerun

TV2 restart point là **B2a MODEL EXECUTION — QWEN3-VL-RERANKER-2B**:

1. Verify exact model revision/provenance.
2. Verify Qwen3-VL processor/model/scoring interface.
3. True-S2 compatibility smoke trên existing worklist.
4. GPU batch benchmark cho 2B và chọn safe batch size.
5. Full F1-F4 inference trên cùng 431,200 query-document pairs.
6. Dùng namespace checkpoint/artifact mới cho 2B.
7. Freeze prediction artifact 2B và tính SHA256 trước khi xem labels.
8. Evaluate F1-F4 Recall/Precision.
9. Compare cùng Workflow-A scientific reference.
10. Professor review.

## 5. What TV4 must do

- B1 environment compatibility smoke: `PASS`.
- B1 scientific status: `NOT_YET_EVALUATED`.
- TV4 tiếp tục full B1 runtime recovery / OOF pipeline.
- Không rerun B0.
- Không rerun B1 compatibility smoke trừ khi environment thay đổi hoặc smoke
  artifact hiện tại không còn hợp lệ.

Việc đổi Qwen 0.6B sang Qwen3-VL-Reranker-2B không reset hoặc alter TV4 B1.

## 6. What must NOT be reused

- Qwen 0.6B model scores.
- Qwen 0.6B SQLite checkpoint như checkpoint 2B.
- Qwen 0.6B frozen predictions như predictions 2B.

Các artifact 0.6B chỉ được giữ làm immutable historical evidence.

## 7. Branch synchronization rule

TV2 B2a và TV4 B1 remain independent. Không fuse/comparison trước khi cả hai
nhánh đều có frozen prediction artifact, SHA256, F1-F4 Recall, F1-F4 Precision,
per-fold results và scientific status.

## 8. Current next actions

TV2: chạy Qwen3-VL-Reranker-2B compatibility smoke trên existing True-S2
worklist.

TV4: tiếp tục B1 full runtime recovery / OOF pipeline.
