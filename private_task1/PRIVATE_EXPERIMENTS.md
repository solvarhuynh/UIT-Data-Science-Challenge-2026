# Task1 Private — Experiment State

Historical offline A/B research: **CLOSED, NOT USED FOR PRIVATE.**

- Đóng toàn bộ A/B, F1–F4 model selection, slot4/5 comparison, Qwen,
  `PRIVATE_SAFE_BASELINE`, `A_BASELINE`, `A_NO_OVERLAY` và `baseline_093_oof`.
- Không xóa artifact lịch sử; chỉ cấm dùng chúng làm Private production.
- Không có Private gold label và không dùng leaderboard để chọn model.

## Decision

`PUBLIC_PRODUCTION_PIPELINE_RECOVERY = BLOCKED`

Bằng chứng tại:

- `artifacts/task1/recovery_096/public_anchor_093/producer_manifest.json`
- `artifacts/task1/submission.zip`

Manifest exact reproduction ghi `train.json`, `warmup.json` và
`exact_label_overlay=true` (57 normalized matches, 5 documents added). Do đó
Public submission cũ không chứng minh được pipeline inference không cần answer.

Step4 calibrated chỉ dùng anchor/baseline đã có provenance label-overlay; không
được coi là Private production pipeline.

## Private status

- Input: READY — 2,080 query, tất cả `answer=null`.
- Retrieval: READY — dense FAISS CPU.
- Candidates: READY — 132,376 q-doc, 8,180 unique documents, local chunks PASS.
- Exact production rerank: READY — frozen label-free `RETRIEVAL_RRF_NO_LABEL`
  K20 worklist; GPU command prepared but not run.
- Top5: MISSING.
- Submission: MISSING.
- GPU/Modal inference runs trong task này: 0; storage-only chunk upload: 119.

Không chạy GPU hoặc tạo submission trong task này. Chỉ sau khi người dùng chủ
động chạy GPU và scores hoàn tất mới được tạo Top5 bằng policy frozen.

## V3 private-safe validation — FAILED (CPU audit of completed GPU output)

Artifact đã hoàn tất được audit:

- Manifest: `private_task1/experiments/private_safe_baseline/validation_download/manifests/production.json`
- Scores: `private_task1/experiments/private_safe_baseline/validation_download/results/production/scores.jsonl`
- Worklist: `private_task1/experiments/private_safe_baseline/private_safe_validation_k20.jsonl`
- Namespace: `runtime/task1_private_safe_validation`
- GPU thực tế: `NVIDIA L4`
- Contract: `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`

Integrity PASS: manifest `COMPLETE`, `112000/112000` q-doc, `335838/335838`
inference units, checkpoint generation `438`, score identities unique, worklist SHA
khớp, expected units recompute khớp, missing/extra `0/0`, non-finite `0`, và MAX
document aggregation khớp cho toàn bộ rows.

Canonical evaluator trên F1–F4 (không dùng Fold0):

- pooled Recall@5: `0.8423571428571428`
- pooled Precision@5: `0.1787857142857143`
- fold Recall: F1 `0.8560714285714286`, F2 `0.8397619047619048`, F3 `0.8363095238095237`, F4 `0.8372857142857143`
- candidate-oracle Recall@20: `0.9664940476190477`; oracle-to-Top5 gap: `0.12413690476190486`
- deployment gate: FAIL (pooled yêu cầu `>=0.92`, từng fold yêu cầu `>=0.90`)

Kết luận: đây là validation result FAIL, không được freeze thành
`V3_PRODUCTION_BENCHMARK`, không tạo Private submission, không authorize production.
Giữ `exact_production_rerank = BLOCKED_ANSWER_DEPENDENCY` và dừng để review metric.
