# Task1 Private — Production Runbook

Đây là source of truth vận hành cho Private Test. Mục tiêu duy nhất là khôi
phục đúng pipeline Public production đã từng tạo submission; không nghiên cứu
hoặc chọn model thêm.

## CURRENT DECISION

- `offline_ab_research = CLOSED`
- `used_for_private_production = NO`
- Không dùng `FINAL_OFFLINE_DECISION`, `A_BASELINE`, `A_NO_OVERLAY`,
  `PRIVATE_SAFE_BASELINE`, F1–F4 model selection, Qwen, Fold0 hay
  `baseline_093_oof` cho Private production.
- Không xóa artifact lịch sử.

## PUBLIC TRACE RESULT

Reference chính:

- `artifacts/task1/submission.zip`
- Exact immutable reproduction:
  `artifacts/task1/recovery_096/public_anchor_093/submission_093.zip`
- Provenance:
  `artifacts/task1/recovery_096/public_anchor_093/producer_manifest.json`

Đã xác định exact producer:

```text
data/raw/btc/LegalIR/public-official.json
→ dense + reranked artifacts
→ scripts/submission/build_legal_ir_ensemble.py
→ scripts/submission/write_legal_ir_submission.py
→ submission.zip
```

Nhưng producer manifest ghi rõ:

- labeled inputs: `data/raw/btc/LegalIR/train.json` và `data/task1/warmup.json`
- `exact_label_overlay = true`
- `normalized_exact_matches = 57`
- `documents_added_by_overlay = 5`

Vì vậy đây không phải pipeline production label-free có thể tái sử dụng cho
query Private mới. Gate “Public production không dùng answer” **FAIL**.

Step4 cũng không giải quyết blocker: `artifacts/task1/handoff/step4_bge_handoff/`
dùng baseline/anchor đã được tạo từ pipeline label-overlay; calibrated JSON
chỉ là output/swap của anchor đó, không phải pipeline Private độc lập.

## CURRENT PIPELINE STATUS

```text
private_task1/input/private-official.json       READY (2080, answer=null)
→ private dense FAISS retrieval                 READY (CPU)
→ private_candidate_union.jsonl                 READY (132376 q-docs)
→ exact Public production rerank                BLOCKED (answer dependency)
→ Top5                                            MISSING
→ submission_private.json/zip                    MISSING
```

Private retrieval artifacts hiện có:

- `private_task1/retrieval/candidates/private_dense_predictions.jsonl`
- `private_task1/retrieval/candidates/private_candidate_union.jsonl`
- `private_task1/retrieval/manifests/private_dense_predictions_manifest.json`
- `private_task1/retrieval/manifests/private_candidate_union_manifest.json`

Contract đã xác nhận: 2,080 queries; 132,376 q-doc candidates; 8,180 unique
documents; local chunk resolution PASS. Đây chưa phải exact Public candidate
contract vì Public trace có thêm labeled KNN/overlay semantics.

## GPU/MODAL

Không chạy GPU, không chạy Modal inference, không upload Volume và không tạo
worklist `private_production_bge_worklist.jsonl` khi Public production contract
chưa được chứng minh là label-free.

`private_safe_validation_k20.jsonl` là artifact lịch sử/validation, không dùng
cho Private production.

## FINAL OUTPUT PATHS (CHƯA TẠO)

- `private_task1/submissions/final/submission_private.json`
- `private_task1/submissions/final/submission_private.zip`
- `private_task1/submissions/final/submission_manifest.json`

## NEXT ACTION — DUY NHẤT

Cần một pipeline production label-free đã được BTC/nguồn code xác nhận, hoặc
được ủy quyền rõ ràng để thiết kế một pipeline mới. Cho đến lúc đó giữ trạng
thái `BLOCKED`; không chạy GPU và không tạo submission.
