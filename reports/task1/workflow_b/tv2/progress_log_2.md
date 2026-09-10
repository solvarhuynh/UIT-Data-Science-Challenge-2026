# Progress Log

## 2026-08-27 — STEP 0: Metric + Output-Capacity Contract Audit

- Status: **BLOCKED**
- Mục tiêu: kiểm tra metric chính thức, evaluator nội bộ, evaluator CV/V3A, validator submission và thống kê gold trên folds 1–4.
- Kết quả metric chính thức: `macro_recall` là primary, `macro_precision` là secondary; Recall/Precision đều là set-based macro metrics, không phụ thuộc thứ tự ranking.
- Phát hiện blocker: historical V3 CV dùng `scripts/beam/task1_v3_residual/common.py:29-30` thay vì evaluator chính thức; launcher cũng gọi bước đánh giá Fold0.
- Vì vậy: không đọc gold labels, không chạm Fold0, không tính statistics folds 1–4.
- Fold0 touched: `false`
- Public labels used: `false`
- Metric sources consistent: `false`
- Output: `reports/step0_metric_contract_report.json`

## 2026-08-27 — STEP 0 corrected audit

- Status: **PASS**
- Đã phân biệt đúng semantic metric với khác biệt implementation guard.
- Công thức Recall cốt lõi của official scorer và V3 local helper giống nhau: `|gold ∩ pred| / |gold|`.
- Guard có khác biệt (official xử lý prediction rỗng hoặc >5), nhưng 5.600 prediction folds 1–4 được kiểm tra đều là top-5 unique nên không kích hoạt guard.
- Score equivalence: 5.600 query, 0 mismatch, sai khác lớn nhất `0.0`.
- Thống kê folds 1–4: 5.600 query; multi-gold 438 (`7.8214286%`); gold >5 là 0 (`0%`); duplicate gold ID là 0.
- Phân phối gold: 1 document = 5.162 query; 2 = 386; 3 = 43; 4 = 8; 5 = 1.
- Historical launcher có stage Fold0, nhưng Step 0 không execute launcher/stage đó; không đọc Fold0 labels/data và không dùng public labels.
- Output: `reports/step0_metric_contract_report.json`

## 2026-08-27 — STEP 1A: Recall-gap / Candidate-repair Ceiling

- Status: **BLOCKED**
- Đã trace được baseline V3 (`artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl`) và full candidate pool trước shortlist (`artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl`).
- Cả hai artifact chuẩn đều gộp Fold0; chưa có partition folds 1–4 được xác minh cho cả hai input.
- Để giữ ràng buộc không đọc/chạm Fold0, không chạy oracle và không diễn giải các số gain placeholder là kết quả đo lường.
- Fold0 touched: `false`; public labels used: `false`.
- Output: `reports/exp_1a_recall_gap_report.json`

## 2026-08-27 — STEP 1A: Recall-gap / Candidate-repair Ceiling (corrected aggregate reader)

- Status: **PASS**
- Aggregate baseline và candidate pool được stream bằng structural target-only reader: 1.400 non-target records mỗi nguồn bị skip trước khi decode payload; chỉ 5.600 query folds 1–4 được materialize/audit.
- Fold0 payload materialized/used in statistics/oracle: `false`; Fold0 labels và public labels used: `false`.
- One-swap candidate ceiling gain: `0.06381547619047619` (PASS, > `0.003`); unconstrained top-5 repair ceiling: `0.06687797619047618`.
- Recall gap: 505 query; missing gold: 573; missing gold hiện trong full pool: 507; one-swap opportunity: 463 query.
- Per-fold one-swap gain: F1 `0.05482142857142857`; F2 `0.06398809523809525`; F3 `0.0655952380952381`; F4 `0.07085714285714284`.
- Output: `reports/task1/exp_1a_recall_gap_report.json`; script: `scripts/analysis/exp_1a_recall_gap_audit.py`.

## 2026-08-27 — STEP 1B: One-swap Oracle Gap Decomposition

- Status: **BLOCKED**
- Đã trace current V3A action-space: shortlist → `policy/actions.jsonl`; `build_actions.py` dùng incoming documents ngoài baseline và chỉ tạo action drop rank 4/5.
- Không có artifact canonical lưu quyết định/prediction V3A OOF theo từng query cho folds 1–4. `policy_training_report.json` chỉ lưu metric aggregate; file prediction duy nhất là Fold0 và bị cấm.
- Không chạy lại inner-CV policy vì đó là training, vượt phạm vi nhiệm vụ. A/B/C/D và các gap để `null`, không dùng số placeholder.
- Fold0/public payload or labels used: `false`.
- Output: `reports/task1/exp_1b_oracle_gap_decomposition_report.json`.

## 2026-08-27 — STEP 1B: One-swap Oracle Gap Decomposition (core A/B/C)

- Status: **FAIL**
- Core A/B/C hoàn tất cho 5.600 query bằng target-only structural readers; không train/replay policy, không materialize Fold0 payload, không dùng public labels.
- Step 1A cross-check: A = `357.3666666666666` / `0.06381547619047619`, khớp tuyệt đối.
- B = C = `0.04177083333333334`; action-space gap = `0.02204464285714285`; rank-limit gap = `0.0` với 0 query dương.
- Gate FAIL: không có evidence mở nhánh rank1–3 theo threshold đã đăng ký (`< 0.001`). D không có vì historical folds1–4 OOF decisions không được serialized; không retrain để tái tạo.
- Output: `reports/task1/exp_1b_oracle_gap_decomposition_report.json`; script: `scripts/analysis/exp_1b_oracle_gap_decomposition.py`.

## 2026-08-27 — Workflow A document update

- Đã đồng bộ `docs/task1/workflow_A_new.md` thành flow hiện hành: Step 0 → 1A → 1B → 1C → Step 4.
- Ghi nhận Step 1B rank-limit gate **FAIL**, rank1–3 branch **REJECTED**, và action-space gap là trọng tâm.
- Đưa Step 1C vào workflow như diagnostic-only; loại Step 2 cũ khỏi active workflow và repurpose Step 3 cũ.
- Không chạy experiment, không sửa production artifacts, không dùng Fold0/public labels.

## 2026-08-27 — STEP 1C: Action-Space Filter Loss Attribution Forensic (retry after professor-review artifact restore)

- Status: **PASS** (diagnostic-only; no branch gate).
- Professor review restored và contract khớp canonical Step 0/1A/1B reports.
- Exact reconstruction: `union_rank <= 20` shortlist cutoff là stage deterministic duy nhất giữa full pool và incoming action-space; final set match 5.600/5.600 query.
- Stage loss = toàn bộ action-space gap: `0.02204464285714285` macro Recall (share `1.0`), 162 affected query, per fold 40/39/41/42.
- Professor direction: `DOMINANT_SINGLE_FILTER_STAGE`; heterogeneity chỉ descriptive. Fold0/public labels: `false`; không train/inference/GPU.
- Outputs: `reports/task1/professor_review_after_step1b.md`, `reports/task1/exp_1c_action_space_filter_loss_attribution_report.json`, `reports/task1/exp_1c_action_space_filter_loss_trace.jsonl`, `docs/task1/workflow_A_new.md`.

## 2026-08-27 — STEP 1D: Union-Rank Cutoff Expansion Oracle Sweep

- Status: **PASS** (diagnostic-only; no production K selected).
- Pre-registered K: `20, 23, 29, 45, 77, 135, 197`, derived from 169 lost-beneficial candidate ranks (21–197).
- K20 cross-check = `0.04177083333333333`; K197 reaches A = `0.06381547619047619`, recovering all A-B oracle gap.
- Incoming/action growth: 86.169/172.338 at K20 to 1.075.215/2.150.430 at K197. Monotonicity and all per-fold K20 checks pass.
- No production cutoff changed; no Step 1B rerun, training, inference, GPU, Fold0 payload or public labels used.
- Outputs: `reports/task1/exp_1d_union_rank_sweep_report.json`; `docs/task1/workflow_A_new.md`.

## 2026-08-27 — STEP 1E: Action-Space Noise/Harm Composition Sweep

- Status: **PASS** (diagnostic-only; production K not selected).
- Reused K grid `20, 23, 29, 45, 77, 135, 197`; candidate/action count cross-checks against Step 1D all pass.
- At K197 vs K20: added candidates = 989.046 (187 gold; candidate noise `0.9998109289153386`); added actions = 1.978.092 (363 beneficial, 1.924.231 neutral, 53.498 harmful; harmful rate `0.02704525370912981`).
- Candidate non-gold and structural harmful actions are reported separately; no policy harm or production behavior inferred. No training/inference/GPU/Fold0/public labels used.
- Advisory post-review joint score is pre-registered; it does not adopt K. Output: `reports/task1/exp_1e_action_space_noise_report.json`; workflow updated.

## 2026-08-27 — STEP 1F: Action-Space Tractability / K-Selection Protocol

- Status: **PASS**; scientific status: exploratory protocol selection.
- Frozen rule: minmax(fraction A-B recovered) − minmax(actions per beneficial action), smallest-K tie-break; historical candidate-noise score retained but rejected for K selection.
- Pooled selected K = 77. LOFO selections F1/F2/F3/F4 = 77/77/77/77; exact-match = 4 and neighbor-match = 4, so stability passes.
- Decision: `ADOPT_RESEARCH_K`; adopted research K = 77. Production K remains unselected.
- Next required step: exactly one Step 1B rerun at K77; not executed here. Fold0/public/training/inference: `false`.
- Output: `reports/task1/exp_1f_k_selection_protocol_report.json`; workflow updated.

## 2026-08-27 — STEP 1B-R77: Oracle Gap Decomposition at Adopted Research K=77

- Status: **PASS** technical execution; adopted research K = 77, production K remains unselected.
- A77/B77/C77 = `0.06381547619047619` / `0.058785714285714274` / `0.058785714285714274`; residual action-space gap = `0.005029761904761912`.
- Rank-limit gap = `0.0` with 0 positive query in every fold; gate **FAIL**, rank1–3 remains closed at K77.
- K77 candidate/action counts = 403.322 / 806.644. D77 unavailable; C-D unresolved. No training/policy replay/inference/GPU/Fold0/public labels used.
- Future Step2 is ready for professor specification, but not run. Outputs: `reports/task1/exp_1b_rerun_k77_oracle_gap_decomposition_report.json`, `reports/task1/exp_1b_rerun_k77_oracle_gap_trace.jsonl`; workflow updated.

## 2026-08-27 — STEP 1C: Action-Space Filter Loss Attribution Forensic

- Status: **BLOCKED**
- Mandatory input `reports/task1/professor_review_after_step1b.md` không tồn tại; tìm trong `reports/` và `docs/` không thấy file thay thế.
- Vì Step 1C và việc cập nhật `docs/task1/workflow_A_new.md` phải tuân theo review này, không suy đoán quyết định giáo sư hoặc tự sửa workflow.
- Không đọc/materialize Fold0 payload, không dùng gold/public labels, không chạy oracle/training/inference.
- Output: `reports/task1/exp_1c_action_space_filter_loss_attribution_report.json`.

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2 — Strict-OOF Policy Realizability Test at Fixed Research K=77
Date: 2026-08-27
Technical status: PASS
Scientific gate: FAIL
Fixed research K: 77
Production K selected: false
Model: sklearn HistGradientBoostingClassifier
Outer OOF: F1/F2/F3/F4 held out independently
Threshold selection: nested 3-way inner OOF
Inputs: artifacts/task1/evaluation/strict_cv_v2/folds.json; data/raw/btc/LegalIR/train.json; artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl; artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl
Outputs: reports/task1/step2_k77_model_contract.json; reports/task1/step2_k77_oof_policy_decisions.jsonl; reports/task1/step2_k77_oof_policy_realizability_report.json; docs/task1/workflow_A_new.md
Key metrics: baseline_macro_recall=0.9259285714285714; policy_macro_recall=0.9259285714285714; D77_gain=0.0; C77=0.058785714285714274; C_minus_D=0.058785714285714274; realization_ratio=0.0; precision_delta=0.0; per_fold_D=F1:0.00035714285714283367,F2:0.0,F3:-0.00035714285714283367,F4:0.0
Policy decisions: selected=362; no_op=5238; beneficial=7; neutral=348; harmful=7
Bootstrap: NOT_RUN_NO_EXISTING_COMPATIBLE_PROTOCOL
Decision: STEP2_FAILURE_ANALYSIS_REQUIRED
Notes: Fold0=false; public=false; K_sweep=false; reranker_inference=false; GPU=false; workflow_updated=true
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-2B B1 versus B16 bounded scientific recovery diagnostic
Date: 2026-09-05
Authorization: user-requested single detached diagnostic submission; no canonical-runner, K77, True-S2, model, prompt, revision, or frozen-prediction changes.
Scope: deterministic 100 evenly-spaced F1-F4 queries, complete 77-document candidate sets, all frozen selected chunks; expected 7,700 q-doc pairs.
Experimental design: one requested-A10 model instance scores identical immutable q-doc-chunk inputs first B1 then B16; labels are loaded only after both score maps/rankings are frozen in memory.
Pre-submit checks: temporary script scripts/modal/_tmp_qwen3vl2b_scientific_recovery.py; .venv Python py_compile PASS; prior Modal app list preflight reported no active Qwen3-VL 2B FunctionCall.
Lifecycle: planned exactly one `modal run --detach`; entrypoint uses recovery.spawn(); no retry authority.
Status: SUBMISSION_PENDING
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B batch1 scientific recovery test
Date: 2026-09-05
Authorization: one bounded A10 100-query F1-F4 scientific diagnostic only (all 77 K77 q-doc candidates/query and their frozen True-S2 chunks), compare batch1 versus batch16 on one loaded model; load the selected F1-F4 labels only after both score sets are complete. No production runner change, no full 5600-query run, no prediction or SQLite reuse/overwrite, no Fold0/public labels, no K77/True-S2/model/prompt/revision/scoring change, and no submission.
Preflight: all prior Qwen3-VL Modal apps stopped with zero tasks; no existing FunctionCall will be duplicated. A temporary detached diagnostic with one async spawn will be removed after results are collected.
Status: STARTED
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B deterministic batch/permutation/alignment diagnostic
Date: 2026-09-05
App/FunctionCall: ap-W1Br4jNDgnyiunaGIInyQa / fc-01M1PVYGVY1DMZ9WAWTB7KVXZF
GPU/model lifecycle: NVIDIA A10G; one frozen revision model loaded once (reloads=0), model.eval() and torch.inference_mode(), detached local entrypoint plus one async spawn, no retry.
Sample: 64 evenly spaced ordered F1-F4 True-S2 q-doc worklist groups; 192 chunk keys (query_id,doc_id,chunk_id). No labels/Fold0/public data used. Key-score mapping was verified at runtime for every forward batch: outputs matched inputs in count and exact immutable-key sequence; no duplicates/missing keys or re-sorting occurred.
Batch deltas versus B1: B2 max/median/p95=0.02734375/0.005859375/0.017578125; B16=0.03125/0.005859375/0.01953125; B32=0.03515625/0.005859375/0.017578125. Every B2/B16/B32 chunk and q-doc MAX ordering differed from B1. B16 permutation joined by immutable key had max delta 0.02734375 versus original B16 and ordering FAIL, while key-score alignment remained PASS.
Neighbor test: all 8 targets retained identical non-padding IDs, effective attention length, final non-padding ten IDs, and final-token semantic position. Same-item score spreads ranged to 0.0234375 as batch padding width changed. For the largest-spread witness, fixed width 447 reduced the B1-vs-B16 delta from 0.0078125 (dynamic alone vs long-neighbor) to 0.001953125 (75% reduction), but did not eliminate it.
Numerics: B16-vs-B1 hidden-state max delta 2.75 and pre-sigmoid max delta 0.1484375; FP32 final projection did not remove the issue (B16-vs-B1 FP32 score delta 0.03390538692474365; native-vs-FP32 maximum 0.002234935760498047). Thus the first divergence is before final projection, and output alignment is ruled out.
Classification: DYNAMIC_PADDING_BATCH_SHAPE_BUG (fixed padding strongly reduces, but does not fully eliminate, the batch-composition sensitivity); no key-to-score association bug and no canonical runner change. No full inference/checkpoint/prediction/SQLite/model revision/K77/True-S2 changes or submission. Temporary diagnostic script removed.
Next: FIX_PROVEN_BATCH_BUG
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B deterministic batch/permutation/alignment diagnostic
Date: 2026-09-05
Authorization: one bounded A10 diagnostic only: 64 F1-F4 True-S2 q-doc pairs and all selected chunks, batch matrix 1/2/16/32, one deterministic B16 permutation, repeated-item neighbor test, FP32 final-projection control, and fixed-padding control only if triggered. No full inference, no canonical runner change, no Fold0/public labels, no K77/True-S2/model/revision/prompt changes, and no submission.
Preflight: all existing Qwen3-VL Modal apps are stopped with zero tasks. One detached temporary diagnostic with one async spawn may be submitted; it will be removed after results are collected.
Status: STARTED
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B complete 32-qdoc official parity confirmation
Date: 2026-09-05
App/FunctionCall: ap-xaZBSSMkFz2gb3nl9a291P / fc-01M1PVH0A5JKM1C8839T3MYB52
GPU/batch/lifecycle: NVIDIA A10 / 1 / one detached local entrypoint with one async spawn; no retry.
Deterministic selection: take the first 32,768 ordered F1-F4 True-S2 worklist groups; selectively decode only those already-targeted F1-F4 train records; then take the first 16 gold and first 16 non-gold q-doc groups in original worklist order. Completed 32/32 q-docs and all 96 selected chunks, with no skips. Labels were used only for construction of this bounded sample; Fold0 and public labels were not decoded or used.
Parity result: rendered prompt/input IDs/attention mask/length all exact 96/96; final 10 IDs exact for every chunk; truncations 0; hidden position PASS; max canonical-vs-official hidden/pre-sigmoid/final-score deltas 0.0/0.0/0.0; chunk and q-doc MAX aggregation orderings PASS; first divergence NONE; failed q-docs 0.
Result: PASS_32QDOC_OFFICIAL_PARITY. Root-cause interpretation: BASIC_OFFICIAL_IMPLEMENTATION_PARITY_CONFIRMED. No canonical runner, frozen prediction SHA256 9e5750ef71f2f0d300ab4fb850358e913b2d1bc5f923f940a2ebcd766999dfa1, SQLite, K77, True-S2, model revision, or scientific artifacts were changed. No full inference or submission occurred. Temporary diagnostic script removed after result collection.
Next: RUN_BATCH_PERMUTATION_ALIGNMENT_TEST
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B complete 32-qdoc official parity confirmation
Date: 2026-09-05
Authorization: exactly one bounded A10 batch-1 confirmation using 16 gold and 16 non-gold F1-F4 True-S2 q-doc pairs, all their selected chunks, no runner change, no full inference, no Fold0/public labels, no K77/True-S2/model revision change, and no submission.
Preflight: all prior Qwen3-VL parity Modal apps are stopped with zero tasks; no existing FunctionCall will be duplicated.
Lifecycle: temporary standalone diagnostic only, submitted detached with one async spawn. It will be deleted locally after terminal results are collected.
Status: STARTED
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B exact batch-1 micro-parity diagnosis
Date: 2026-09-05
App/FunctionCall: ap-c643AIIT41KRvvI5pqZiim / fc-01M1PTMWREEC966D84DZJKV8V4
GPU/batch: NVIDIA A10 / 1. Detached local entrypoint with one async spawn; no retry.
Deterministic selection: scan the first 8,192 ordered F1-F4 True-S2 worklist q-doc groups, decode train records only for those already-targeted F1-F4 query IDs, then take the first 4 gold and first 4 non-gold q-doc groups in worklist order. Completed 8/8 q-docs and all 24 selected chunks (no chunk skipped); labels were used only for this selection.
Official frozen paths: models/qwen3-vl-reranker-2b/scripts/qwen3_vl_reranker.py; models/qwen3-vl-reranker-2b/1_LogitScore/config.json; models/qwen3-vl-reranker-2b/sentence_bert_config.json; models/qwen3-vl-reranker-2b/config_sentence_transformers.json; models/qwen3-vl-reranker-2b/modules.json. yes/no IDs: 9693/2152.
Parity result: rendered prompts 24/24, input IDs 24/24, attention masks 24/24, lengths 24/24, truncations 0, hidden position PASS, max hidden/pre-sigmoid/final canonical-vs-official deltas 0.0/0.0/0.0, chunk and q-doc ordering PASS. Every item first-divergence stage was NO_DIVERGENCE.
Exact extraction rules: official self.model(**inputs).last_hidden_state[:, -1]; canonical model.model(**encoded).last_hidden_state[:, -1]. Both execute BF16 through model, hidden state, LM-head vectors, direction, dot product, and sigmoid. FP32 final-score diagnostic differed from native BF16 by at most 0.0025377273559570312, but this is shared arithmetic and not a canonical-versus-official divergence.
Result: PASS_MICRO_PARITY. No concrete canonical bug proven. No runner, prediction, SQLite, K77, True-S2, model revision, Fold0, public labels, or full inference changed/used. The temporary diagnostic script was removed after result collection.
Next: RUN_32_QDOC_PARITY_CONFIRMATION
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B exact batch-1 micro-parity diagnosis
Date: 2026-09-05
Authorization: one bounded A10 diagnostic only: exactly 8 F1-F4 True-S2 q-doc pairs (4 gold, 4 non-gold) and all their selected chunks; no full inference, no 32-qdoc run, no production runner change, no Fold0/public labels, no K77/True-S2/model revision change, and no submission.
Preflight: current Modal app listing shows ap-sI5Ujo4ovfQCddhgw2AfsG and all prior Qwen3-VL 2B apps stopped with zero tasks. It is therefore safe to submit one new bounded diagnostic call.
Lifecycle: temporary standalone Modal diagnostic (not the canonical runner), detached local entrypoint plus async FunctionCall.spawn(). The temporary script will be removed after logs/result collection.
Status: STARTED
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B official scorer parity resolution — preflight block
Date: 2026-09-05
Existing-call evidence: Modal app ap-sI5Ujo4ovfQCddhgw2AfsG reports "ephemeral (detached)", tasks=1. Filtered logs for fc-01M1PMNB73Q2GX75Z56C6PQZJ3 contain only worker heartbeat failures (2026-09-04 23:56 and 2026-09-05 00:06 +07); no selector, tokenizer, model, or scoring marker was emitted. The function timeout was one hour, so this call has no usable 32/32 result and is an orphan-risk, but Modal still marks its container active.
Official snapshot evidence inspected: models/qwen3-vl-reranker-2b/scripts/qwen3_vl_reranker.py; models/qwen3-vl-reranker-2b/1_LogitScore/config.json; models/qwen3-vl-reranker-2b/sentence_bert_config.json; models/qwen3-vl-reranker-2b/config_sentence_transformers.json; models/qwen3-vl-reranker-2b/modules.json. Locked true/false token IDs verified as 9693/2152.
Static comparison: both paths use the base model final position [-1], BF16 yes-minus-no LM-head direction, and sigmoid. The official tokenizer route uses processor(...) then restores the final five tokens; canonical manually invokes tokenizer on rendered text. Runtime/token/hidden parity cannot be claimed without the required bounded call.
Safety action: inspected stop command only. An attempted stop required confirmation; the automatic stop request was rejected because the user has not explicitly authorized terminating the currently active remote container. No Modal app/call was changed, no new GPU was submitted, and no runner/scientific artifact was modified.
Status: BLOCKED_PENDING_EXPLICIT_ORPHAN_CALL_CANCELLATION_AUTHORITY
Next: USER_CONFIRM_CANCEL_ap-sI5Ujo4ovfQCddhgw2AfsG_THEN_RUN_ONE_STREAMING_32_PAIR_A10_PARITY_DIAGNOSTIC
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B official scorer parity resolution
Date: 2026-09-05
Authorization: bounded A10 parity diagnostic only; no full F1-F4 inference, no frozen prediction or SQLite reuse, no Fold0/public labels, no True-S2/K77/model-revision changes, and no submission.
Primary existing FunctionCall to inspect before any new GPU work: fc-01M1PMNB73Q2GX75Z56C6PQZJ3 (App ap-sI5Ujo4ovfQCddhgw2AfsG).
Plan: inspect the existing call; recover or deterministically reproduce the exact 32 q-doc sample only if needed; compare official and canonical paths at batch 1, then isolate batch-16 numerical effects. Patch the canonical runner only after a first-divergence implementation defect is proven.
Status: STARTED
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B model-load dependency validation
Date: 2026-09-03
Technical status: PASS_MODEL_LOAD
Prompt/task: Fix missing torchvision dependency and validate model loading only
Canonical runner: scripts/modal/task1_b2a_qwen3vl2b.py
Modal profile: hoconlinea10
Environment: main
GPU: NVIDIA A10
Torch: 2.8.0
Torchvision: 0.23.0
Model: Qwen/Qwen3-VL-Reranker-2B
Requested/resolved revision: 4bd860ac4f15ad1897a214615cccc700f8f71818
HF download: PASS
Processor: Qwen3VLProcessor
Yes token ID: 9693
No token ID: 2152
Model: Qwen3VLForConditionalGeneration
CUDA: PASS
Score direction finite: PASS
Inference: NOT_RUN
Worklist/chunks/labels: NOT_READ
Temporary probe: removed
Next: RETURN_TO_A10_B16_TRUE_S2_SMOKE
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Workflow B B2a — Qwen true S2 top-3 worklist build (superseded by integrity correction)
Date: 2026-09-02
Technical status: BLOCKED_INCOMPLETE_WORKLIST
Worklist build status: INCOMPLETE
Model: Qwen/Qwen3-Reranker-0.6B
Mode: ZERO_SHOT
Checkpoint pairs: 420000
Current unique pairs: 423835
Canonical expected pairs: 431200
Missing canonical pairs: 7365
Missing mappings: 0
Wrong document mappings: 0
Worklist artifact: reports/task1/workflow_b/tv2/b2a/manifests/b2a_qwen_true_s2_top3_worklist.jsonl
Checkpoint: reports/task1/workflow_b/tv2/b2a/runtime/b2a_qwen_true_s2_top3_worklist_checkpoint.json
Inference status: NOT_RUN
GPU smoke: NOT_RUN
Whole-document context gate: SUPERSEDED_FOR_TRUE_S2
Scientific metrics computed: false
Fold0 used: false
Public labels used: false
Conclusion: 420000 was an intermediate checkpoint, not K75 truncation. The exact frozen v2 builder is absent, so the missing canonical pairs cannot be safely appended.
Detailed report: reports/task1/workflow_b/tv2/b2a/reports/b2a_qwen_true_s2_top3_worklist_report.md
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Workflow B B2a-0 Zero-Shot Semantic Reranker
Technical status: BLOCKED
Scientific status: NOT_YET_EVALUATED
Model: Qwen/Qwen3-Reranker-0.6B
Model revision: e61197ed45024b0ed8a2d74b80b4d909f1255473
Mode: ZERO_SHOT
K77 unchanged: true
Direct document reranking: true
One-swap used: false
Threshold used: false
Training labels used: false
Fold0 used: false
Public labels used: false
Expected query-document pairs: 431200
Scored pairs: 0
Max length: NOT_SELECTED_GPU_GATE_BLOCKED
Truncation rate: NOT_RUN
Comparator Recall: NOT_RUN
B2a Recall: NOT_RUN
Recall delta: NOT_RUN
Comparator Precision: NOT_RUN
B2a Precision: NOT_RUN
Precision delta: NOT_RUN
Non-negative Recall folds: 0/4
Materiality floor: 0.003 provisional
Next: PROFESSOR B2A RESULT REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Task1 Repository Workflow-Lineage Reorganization
Technical status: PASS
Scientific results changed: false
Historical artifacts deleted: false
Workflow A docs preserved: true
Workflow A status: CLOSED_HISTORICAL
Workflow B B1 status: NOT_YET_EVALUATED
Workflow B B2a status: ACTIVE_NEURAL_RERANKER_TRACK
Migration manifest: reports/task1/artifact_migration_manifest.json
Broken live references after migration: 0
Next: B2A MODEL PROVENANCE / CONTEXT / FEASIBILITY AUDIT
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Workflow-A P5 Public Submission Build
Date: 2026-08-30
Technical status: BLOCKED_PUBLIC_DEPLOYMENT_CONTRACT_UNDEFINED
Task type: DEPLOYMENT-CONTRACT PREFLIGHT ONLY
Training executed: false
Public inference executed: false
Fold0 used: false
Public labels used: false
Tuning executed: false
Blocking artifact: reports/task1/public_p5_submission_preflight.json
Blocking reason: No exact no-Fold0 production P1/P4 final-fit contract, no P5 K77 public action/feature contract, and no development-only rule for a single public threshold. Historical public deployment is V3A, not P5.
Upload-ready file: NONE
Validator: NOT_RUN
Next: PROFESSOR REVIEW — DEFINE OR AUTHORIZE A REPRODUCIBLE P5 PUBLIC DEPLOYMENT CONTRACT
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P5 — P1-Proxy Percentile-Fusion Relative Reranking Intervention
Date: 2026-08-30
Technical status: PASS
Experiment type: CONTROLLED_SINGLE_VARIABLE_INTERVENTION
Training executed: false
New model inference executed: false
P5-A trusted artifact verified: true
Inner join preflight: PASS
Outer join preflight: PASS
K: 77
Beta: 0.05
P1 transform: identity
Proxy normalization: within-query zero-based averaged percentile
Fresh nested-OOF threshold: true
Baseline R: 357/426
P5 R: 356/426
Delta R: -1
Baseline D77_exact: -0.0007142857142857143
P5 D77_exact: -0.000565476190476191
Delta D77_exact: 0.000148809523809524
Baseline gain_sum: -4.0
P5 gain_sum: -3.16666666666667
Delta gain_sum: 0.833333333333333
Baseline S/T/R: 6/63/357
P5 S/T/R: 6/64/356
Folds non-worsened: 4/4
Top-action changes: 145
P1 failure -> P5 success: 1
P1 success -> P5 failure: 0
Pooled R improves: true
D77 non-worsens: true
Fold robustness met: true
Overall preregistered validation: true
policy_oof_strict: true
end_to_end_selection_oof: false
Fold0/public: false
Beta sweep: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P5-A — P1 Inner Full-Action Score Artifact Completion
Date: 2026-08-30
Technical status: PASS
Experiment type: ARTIFACT COMPLETION ONLY
Scientific intervention executed: false
P5 executed: false
Frozen P1 inner models expected: 12
Frozen P1 inner models found: 12/12
Frozen-model path used: true
Fallback retraining used: false
Fallback models retrained: 0
K: 77
Features: 36 unchanged
Reference top-score rows: 16800
Queries reproduction-checked: 16800
Top-action identity matches: 16800/16800
Top-score matches within 1e-9: 16800/16800
Max absolute top-score error: 0
Full inner action-score rows persisted: 806644
Artifact trusted: true
Policy OOF strict: true
Fold0/public used: false
P5 fusion computed: false
P5 threshold selected: false
P5 R/D77 computed: false
P5 fusion hypothesis: AUTHORIZED_BUT_NOT_RUN
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P5 — P1-Proxy Percentile-Fusion Relative Reranking Intervention
Date: 2026-08-30
Technical status: BLOCKED_ARTIFACT_PROVENANCE_CONFLICT
Scientific result: NONE — NOT RUN
Training executed: false
New model inference executed: false
Beta: 0.05
Normalization: within-query zero-based percentile
P1 outer artifact verified: true
P1 inner artifact verified: false
P4 outer proxy verified: true
P4 inner proxy verified: true
Outer join complete: false
Inner join complete: false
Fold0/public: false
Blocking reason: persisted P1 inner artifact has only one top action per query (16800 rows), not full action-level P1 scores required for within-query percentile fusion and fresh nested-OOF threshold selection; regenerating missing scores requires prohibited P1 inference
Next: PROFESSOR CLARIFICATION
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P5-A — P1 Inner Full-Action Score Artifact Completion
Date: 2026-08-30
Technical status: PASS
Experiment type: ARTIFACT COMPLETION ONLY
Scientific intervention executed: false
P5 executed: false
Frozen P1 inner models expected: 12
Frozen P1 inner models found: 12/12
Frozen-model path used: true
Fallback retraining used: false
Fallback models retrained: 0
K: 77
Features: 36 unchanged
Reference top-score rows: 16800
Queries reproduction-checked: 16800
Top-action identity matches: 16800/16800
Top-score matches within 1e-9: 16800/16800
Max absolute top-score error: 0
Full inner action-score rows persisted: 2419932
Artifact trusted: true
Policy OOF strict: true
Fold0/public used: false
P5 fusion computed: false
P5 threshold selected: false
P5 R/D77 computed: false
P5 fusion hypothesis: AUTHORIZED_BUT_NOT_RUN
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P4-Q — Query-Conditioned B-vs-N Proxy Concordance Diagnostic
Date: 2026-08-30
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Training executed: false
New inference executed: false
Frozen P4 outer-OOF predictions reused: true
K: 77
Canonical oracle-positive queries: 426
Included B/N queries: 426
Excluded queries: 0
Excluded NO_B: 0
Excluded NO_N: 0
Excluded NO_B_AND_NO_N: 0
B actions: 897
N actions: 59138
B/N pair count: 125576
B above N pairs: 102221
Tie pairs: 75
B below N pairs: 23280
Pair-pooled B-vs-N concordance: 0.8143156335605529
Query-majority concordance rate: 0.8708920187793427
F1 concordance: 0.8027104187226843
F2 concordance: 0.8449552006232957
F3 concordance: 0.806654460319366
F4 concordance: 0.8034548422198041
Chance reference: 0.50
Proxy threshold selected: false
Proxy gate authorized/executed: false
Proxy reranking executed: false
P1 policy modified: false
New D77 policy computed: false
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P4 — Label-Free Beneficial-Action Proxy Discriminability Diagnostic
Date: 2026-08-30
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Training executed: true
Policy modified: false
Outer diagnostic models trained: 4/4
Inner diagnostic models trained: 12/12
K: 77
Features: 36 unchanged
Target: BENEFICIAL vs NEUTRAL/HARMFUL
Class weighting: none
Resampling: none
Fold0/public: false
Outer AUC F1: 0.8142359298766111
Outer AUC F2: 0.8486435053975017
Outer AUC F3: 0.8421840104747816
Outer AUC F4: 0.8091343492111038
Pooled outer-OOF AUC: 0.8273434845649417
AUC orientation: higher score = more likely BENEFICIAL
Directional AUC reversal used: false
Feature importance status: NOT_AVAILABLE_WITHOUT_NEW_UNPREREGISTERED_METHOD
Proxy threshold selected: false
P1 ranking modified: false
New D77 policy computed: false
P3 sub-branch: FORMALLY_CLOSED
Relative-ordering hypothesis: SUPPORTED
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P3-T — BN Probability-Transformation Relative-Ranking Intervention
Date: 2026-08-29
Technical status: BLOCKED_SPEC_LABEL_LEAK
Scientific result: NONE — NOT RUN
Training executed: false
New scoring executed: false
Transformed probabilities computed: false
BN identity available without truth: false
Outer labels required to identify BN: true
Public gold required for deployable transformation: true
Blocking definition: B/N/H are assigned from gold-derived recall delta after the proposed swap; using them to select BN/NB transformations leaks held-out truth into scoring
Approximation/proxy used: false
Fold0/public used: false
K: 77
Delta: 0.03
Next: PROFESSOR CLARIFICATION
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P3-R — BN Head-to-Head Decision-Margin Intervention
Date: 2026-08-29
Technical status: BLOCKED_SPEC_AGGREGATION_CONFLICT
Experiment type: CONTROLLED SINGLE-VARIABLE INTERVENTION — SEMANTIC PRE-FLIGHT BLOCKED
Training executed: false
New scoring executed: false
K: 77
Requested thresholds: BN=0.55; BH/NH=0.50
Aggregation changed: false
Blocking contract: canonical ranking is MEAN_PAIRWISE_WIN_PROBABILITY over unthresholded P(a>b); no binary pair decision exists or affects ranking
Directional conflict: under ordinary threshold semantics, 0.55 makes B harder than 0.50 to beat N, not easier
Approximation used: false
Fold0/public: false
Next: PROFESSOR CLARIFICATION
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1A-F — MEAN-vs-MEDIAN Aggregation Failure Attribution
Date: 2026-08-29
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Training executed: false
New scoring executed: false
New formula tested: false
K: 77
Queries: 5600/5600
Oracle-positive queries: 426/426
Top-action changes: 1318/1318
Oracle-positive top-action changes: 95/95
MEAN gain sum: -4.0 / -4.0
MEDIAN gain sum: 0.3333333333333333 / 0.3333333333333333
Delta gain: 4.333333333333333 / 4.333333333333333
MEAN executions: 410/410
MEDIAN executions: 17/17
MEAN executed B/N/H: 6 / 392 / 12
MEDIAN executed B/N/H: 1 / 16 / 0
Harm-avoidance top flips: 4
Opportunity-loss top flips: 18
Opportunity-discovery top flips: 10
Harm-introduction top flips: 1
Same-class reshuffles: 1285
Removed harmful loss via top-action change: 1.0
Removed harmful loss via same-top EXECUTE->NOOP: 6.666666666666667
Lost beneficial gain via top-action change: 1.0
Lost beneficial gain via same-top EXECUTE->NOOP: 2.3333333333333335
TOP_SAME + EXECUTE->NOOP queries: 337
TOP_CHANGED + EXECUTE->NOOP queries: 56
F3 oracle-positive top-change count: 34
Depth definition source: STEP 2-P1F-R2 highest-scoring MEAN BENEFICIAL action incoming_union_rank
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1A — Controlled Median-Aggregation Action-Score Experiment
Date: 2026-08-29
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Experiment type: DETERMINISTIC_POSTHOC_RECOMBINATION
Model fit executed: false
K: 77
Features: 36
Objective changed: false
Pair weighting changed: false
Aggregation sweep: false
Old aggregation: MEAN_PAIRWISE_WIN_PROBABILITY
New aggregation: MEDIAN_PAIRWISE_WIN_PROBABILITY
Outer median action rows: 806644/806644
Inner median OOF rows: 16800/16800
Mean reconstruction audit: PASS
Mean thresholds: F1=0.8871766063648394; F2=0.9158352964999534; F3=0.8885338990548798; F4=0.9072451320464932
Median thresholds: F1=Infinity; F2=Infinity; F3=0.9837219720799008; F4=Infinity
Mean gain sum: -4.0
Mean D77: -0.0007142857142857143
Median gain sum: 0.3333333333333333
Median D77: 0.000059523809523809524
Delta D77 median-mean: 0.0007738095238095238
Median D77 > mean: true
Median D77 > 0: true
All fold median D >= mean: false
Mean top B/N/H: 69 / 5405 / 126
Median top B/N/H: 61 / 5416 / 123
Mean executed B/N/H: 6 / 392 / 12
Median executed B/N/H: 1 / 16 / 0
Mean S/T/R: 6 / 63 / 357
Median S/T/R: 1 / 60 / 365
Mean median(best_B-best_N): -0.0944695955058123
Median aggregation median(best_B-best_N): -0.047746588404987356
Mean negative B-N: 357/426
Median negative B-N: 359/426
Mean harmful count/loss: 12 / -7.666666666666667
Median harmful count/loss: 0 / 0
Mean-vs-median top-action changes: 1318/5600
Outer-label threshold optimization performed: false
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1C — Nested-OOF Threshold/Calibration Experiment for Frozen Pairwise Score
Date: 2026-08-28
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Experiment type: AUTHORIZED_NESTED_OOF_THRESHOLD_EXPERIMENT_TRAINING
K: 77
Features: 36
Inner model fits: 12/12
Inner models persisted: 12/12
Outer models retrained: false
Objective changed: false
Pair weighting changed: false
Action score changed: false
Historical threshold reproduction: PASS
Old thresholds: F1=0.8871766063648394; F2=0.9158352964999534; F3=0.8885338990548798; F4=0.9072451320464932
New thresholds: F1=0.8871766063648394; F2=0.9158352964999534; F3=0.8885338990548798; F4=0.9072451320464932
Old gain sum: -4.0
Old D77: -0.0007142857142857143
New gain sum: -4.0
New D77: -0.0007142857142857143
Delta D77 new-old: 0.0
New D77 > old: false
New D77 > 0: false
All fold D_new >= D_old: true
Old executed B/N/H: 6 / 392 / 12
New executed B/N/H: 6 / 392 / 12
Old S/T/R: 6 / 63 / 357
New S/T/R: 6 / 63 / 357
Old B/N/H crossing rates: 6/69 / 392/5405 / 12/126
New B/N/H crossing rates: 6/69 / 392/5405 / 12/126
Old harmful loss: -7.666666666666667
New harmful loss: -7.666666666666667
Outer-label threshold optimization performed: false
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1T — Pairwise-Specific Threshold/Calibration Forensic
Date: 2026-08-28
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Training executed: false
K: 77
Features changed: false
Objective changed: false
Pair weighting changed: false
Score formula changed: false
Thresholds changed: false
Queries analyzed: 5600/5600
B-top / N-top / H-top: 69 / 5405 / 126
B-top oracle-positive queries: 69/69
S / T: 6 / 63
S conversion rate among B-top: 0.08695652173913043
Median S score: 0.9083505600543628
Median T score: 0.8036790371020378
Median S threshold margin: 0.019816660999482982
Median T threshold margin: -0.0922472723226565
T within 0.005 below threshold: 1/63
T within 0.010 below threshold: 3/63
T more than 0.050 below threshold: 49/63
Threshold crossing rate B-top: 0.08695652173913043
Threshold crossing rate N-top: 0.0725254394079556
Threshold crossing rate H-top: 0.09523809523809523
Executed B / N / H: 6 / 392 / 12
Executed harmful Recall-delta sum: -7.666666666666667
Median executed-H threshold margin: 0.02017194368118208
Inner-OOF threshold trace status: NOT_AVAILABLE_FROM_PERSISTED_ARTIFACTS
Outer-label threshold optimization performed: false
Calibration model fit: false
Fold0/public: false
Remaining ranking bottleneck: PRIMARY
Threshold/calibration bottleneck: SECONDARY_NEWLY_SIGNIFICANT
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Xác nhận file báo cáo kết quả STEP 2-P1F-R2
Date: 2026-08-28
Kết quả: Báo cáo tổng hợp chính là step2p1f_failure_attribution_report.json; các file gate, metric resolution và pair-family là báo cáo bổ trợ.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1F-R2 — Canonical-Metric Resolution and Pairwise Failure Attribution
Date: 2026-08-28
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Metric contract resolution: ACCEPTABLE_METRIC_CONTRACT_CORRECTION
Canonical D semantics: SUM(per_query_recall_delta)/N
Canonical P1 gain sum: -4.0
Canonical P1 D77: -0.0007142857142857143
Training terminology: DETERMINISTIC_FROZEN_MODEL_RECONSTRUCTION
Mechanical model.fit executed: true
New scientific training: false
Primitive reproduction gate: PASS
Thresholds exact: true
Top scores exact: 5600/5600
Executed identities exact: 410/410
NO_OP exact: 5190/5190
Class composition: B=6; N=392; H=12
Models persisted: 4/4
Full action rows: 806644/806644
Oracle-positive queries: 426
S_SUCCESS: count=6; headroom=3.6666666666666665; share=0.01113811259619279
T_THRESHOLD_LOSS: count=63; headroom=51.833333333333336; share=0.15745240988254355
R_RANKING_DISCRIMINATION_LOSS: count=357; headroom=273.7; share=0.8314094775212637
Median best_B-best_N: -0.0944695955058123
Negative best_B-best_N: 357/426
Pointwise F2 R share: 0.8921628189550426
Executed harmful: 12
Harmful Recall-delta sum: -7.666666666666667
Pair counts: B-N=125576; B-H=1625; N-H=1539444
Query-local Spearman log(BN/NH) vs B-N gap: 0.0957899011266975
Query-local Pearson log(BN/NH) vs B-N gap: 0.16299073573310122
Pair weighting changed: false
Objective changed: false
Features changed: false
K changed: false
Threshold retuned: false
Fold0/public: false
Next: STEP2_P1F_R2_COMPLETE_PENDING_PROFESSOR_REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1F-R — kiểm tra trạng thái chạy và reproduction gate
Date: 2026-08-28
Technical status: RECONSTRUCTION_MISMATCH
Process status: FINISHED; PID 20364 không còn tồn tại
Result: Gate đã hoàn tất; thresholds, 5,600 decisions và top scores khớp. Tuy nhiên D77/per-fold D77 có sai khác số thực rất nhỏ so với canonical nên gate không pass.
Outputs: reports/task1/step2p1f_reproduction_gate_report.json; reports/task1/step2p1f_run.log (rỗng)
Decision: Không có model/diagnostic attribution nào được tạo; không chạy lại tự động và không diễn giải thành kết quả khoa học.
Notes: Fold0=false; public=false; GPU=false; P2/Step3/Step4/Workflow B chưa chạy.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1F — Pairwise Policy Failure Attribution
Date: 2026-08-27
Technical status: BLOCKED
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
P1 D77: -0.0007142857142857784
K: 77
Features: 36
Full actions: NOT_CREATED/806644
Reproduction gate: NOT_RUN
Oracle-positive queries: NOT_ANALYZED (requires authorized forward scoring)
S_SUCCESS: NOT_ANALYZED
T_THRESHOLD_LOSS: NOT_ANALYZED
R_RANKING_DISCRIMINATION_LOSS: NOT_ANALYZED
Median best_B-best_N: NOT_ANALYZED
Negative best_B-best_N: NOT_ANALYZED/426
Pointwise comparator R-share: 0.8921628189550426
Executed harmful: NOT_ANALYZED; harmful recall-delta sum=NOT_ANALYZED
Pair counts reproduced: B-N=125576; B-H=1625; N-H=1539444 (canonical P1 verified)
Spearman log(BN/NH) vs B-N gap: NOT_ANALYZED
Pearson log(BN/NH) vs B-N gap: NOT_ANALYZED
F1 D=0.0009523809523808158; S/T/R=NOT_ANALYZED
F2 D=0.0; S/T/R=NOT_ANALYZED
F3 D=-0.0023809523809524835; S/T/R=NOT_ANALYZED
F4 D=-0.0014285714285713347; S/T/R=NOT_ANALYZED
Pair weighting changed: false
Objective changed: false
Features changed: false
K changed: false
Threshold retuned: false
Fold0/public: false
Blocker: no persisted exact P1 model objects; task forbids refitting for forward scoring
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1 — Controlled pairwise policy objective at fixed K77
Date: 2026-08-27
Technical status: PASS; scientific status: DIAGNOSTIC ONLY; gate: FAIL
Result: D77_pairwise=-0.0007142857142857784; C77=0.058785714285714274;
realization_ratio=-0.012150668286756865; precision_delta=-0.0002142857142857224
Coverage: 5,600 OOF queries; 410 executed (6 beneficial, 392 neutral,
12 harmful); 5,190 no-op; folds 1–4 only; Fold0/public labels=false
Artifacts: step2p1_realizability_report.json; step2p1_oof_policy_decisions.jsonl;
step2p1_model_contract.json; workflow_A_new.md updated
Decision: K77 exploratory only; production K NOT SELECTED; no downstream step run
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1 — Controlled Query-Aware Pairwise Policy Objective Experiment at Fixed K77
Date: 2026-08-27
Technical status: BLOCKED
Scientific status: DIAGNOSTIC ONLY
Policy OOF strict: true
End-to-end selection OOF: false
K selected exploratorily on folds1-4: true
K: 77
Feature count: 36
Model: HistGradientBoostingClassifier
Objective: pairwise difference binary log-loss
Truth ordering: BENEFICIAL > NEUTRAL > HARMFUL
Pair generation: deterministic all cross-class + mirrored
Pair weights: uniform 1.0
Action score: mean pairwise win probability against all other query actions
Threshold protocol: original nested inner-OOF protocol unchanged
Inputs: canonical Step2-R1/F2/F3 artifacts and K77 raw artifacts
Outputs: reports/task1/step2p1_model_contract.json; reports/task1/step2p1_realizability_report.json; docs/task1/workflow_A_new.md
Pair counts: B-N=125576; B-H=1625; N-H=1539444; mirrored rows=3333290
Baseline macro Recall: not measured
Pairwise policy macro Recall: not measured
D77 pairwise: not measured
C77: 0.058785714285714274
C-D: not measured
Realization ratio: not measured
Per-fold D77: not measured
Precision delta: not measured
Executed decisions: not measured
Step2-R1 pointwise D77: 0.0
Pairwise-minus-pointwise D77: not measured
Original Step2 gate comparison: NOT_RUN_BLOCKED (NON-BINDING)
Fold0/public: false
Feature/model-family/K changes: false
Next: PROFESSOR REVIEW — complete exact pairwise CPU execution feasibility required
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-F3 — Frozen-Feature Discriminative Sufficiency Forensic
Date: 2026-08-27
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
K: 77
Feature count: 36
Full action rows: 806644
Oracle-positive queries: 426
Ranking-loss queries: 380
Primary contrast: BENEFICIAL vs NEUTRAL
Primary query-balanced B/N queries: 426
Feature groups frozen pre-statistics: true
Upstream Group6 feature count: 28
Upstream signal pattern: SEPARATION_VISIBLE
Top upstream feature by directional AUC: incoming_union_rank; ROC_AUC=0.7097026604068858; PR_AUC=0.7233536170873198; MI=0.11959345039371892
Top non-upstream feature by directional AUC: no separate non-upstream leader; incoming_union_rank is the pooled maximum
Depth upstream summary: 1-20=reported; 21-29=reported; 30-45=reported; 46-77=reported
R-loss conditional upstream summary: reported; incoming_union_rank is the top directional-AUC feature
Permutation importance: NOT_RUN_NO_PERSISTED_FROZEN_MODEL
Training/refitting: false
Ablation: false
Feature changes: false
Model changes: false
K changes: false
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-F2 — Full-Action Rescoring Diagnostic via Deterministic Reconstruction
Date: 2026-08-27
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
K: 77
Production K selected: false
Reconstruction gate: PASS
Reconstructed thresholds: F1=0.9336471149727648; F2=+Infinity; F3=0.9009323675256211; F4=0.8974011769826262
Decision reproduction: executed_action_match=362/362; noop_match=5238/5238; top_score_match=5600/5600
D77 reproduced: 0.0
Full action rows: 806644/806644
Oracle-positive queries: 426
S_SUCCESS: count=7; headroom=6.0; share=0.01822600243013366
T_THRESHOLD_LOSS: count=39; headroom=29.5; share=0.08961117861482382
R_RANKING_DISCRIMINATION_LOSS: count=380; headroom=293.7; share=0.8921628189550426
Fold2: T=13; R=94; S=0; selected_tau=+Infinity; best_finite_inner_gain=0.0
Deep best-beneficial <=20: see diagnostic report
Deep best-beneficial 21-29: see diagnostic report
Deep best-beneficial 30-45: see diagnostic report
Deep best-beneficial 46-77: see diagnostic report
Training/tuning: false
Deterministic frozen-model reconstruction: true
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-F — Decision-Level Failure Attribution for Step2-R1
Date: 2026-08-27
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Inputs: reports/task1/step2_k77_v2_oof_policy_decisions.jsonl; deterministic target-only K77 oracle reconstruction from canonical folds/train/baseline/candidate artifacts
Outputs: reports/task1/step2f_failure_attribution_report.json; docs/task1/workflow_A_new.md
Queries: 5600
Oracle-positive queries: 426
Oracle headroom sum: 329.2
Realized policy gain sum: 0.0
Missed headroom sum: 329.2
Category A oracle-positive no-op: count=367; oracle_headroom_sum=280.3333333333333; headroom_share=0.851559335763467
Category B oracle-positive beneficial: count=7
Category C oracle-positive neutral: count=52
Category D oracle-positive harmful: count=0
Category E oracle-zero no-op: count=4871
Category F oracle-zero neutral: count=296
Category G oracle-zero harmful: count=7
Threshold attribution: UNRESOLVED_REQUIRES_STEP2_F2_FULL_ACTION_RESCORING
Ranking attribution: UNRESOLVED_REQUIRES_STEP2_F2_FULL_ACTION_RESCORING
Training: false
Model inference: false
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-R1 — Contract-Repair Rerun at Fixed K77
Date: 2026-08-27
Technical status: PASS
Scientific gate: FAIL
Parent attempt: STEP2 attempt #1 = CONTRACT_ERROR / INVALID_FOR_SCIENTIFIC_INFERENCE_AUDIT_ONLY
Repair: ordered_unique feature schema deduplication only
Parent feature entries: 38
V2 unique features: 36
Removed duplicates: dropped_baseline_rank; incoming_is_baseline_top5
K: 77
Drop ranks: 4/5
Model: sklearn HistGradientBoostingClassifier
Inputs: artifacts/task1/evaluation/strict_cv_v2/folds.json; data/raw/btc/LegalIR/train.json; artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl; artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl
Outputs: reports/task1/step2_k77_model_contract_v2.json; reports/task1/step2_k77_v2_oof_policy_decisions.jsonl; reports/task1/step2_k77_v2_oof_policy_realizability_report.json; docs/task1/workflow_A_new.md
Key metrics: baseline_macro_recall=0.9259285714285714; policy_macro_recall=0.9259285714285714; D77=0.0; C77=0.058785714285714274; C_minus_D=0.058785714285714274; realization_ratio=0.0; precision_delta=0.0; per_fold_D=F1:0.00035714285714283367,F2:0.0,F3:-0.00035714285714283367,F4:0.0
Policy decisions: selected=362; no_op=5238; beneficial=7; neutral=348; harmful=7
Parent artifacts preserved: true
Fold0/public: false
Training hardware: CPU
Next state: STEP2_VALID_FAIL_PENDING_FAILURE_ANALYSIS_REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2 — Strict-OOF Policy Realizability Test at Fixed Research K=77 (final validation correction)
Date: 2026-08-27
Technical status: CONTRACT_ERROR
Scientific gate: NOT_RUN
Fixed research K: 77
Production K selected: false
Model: sklearn HistGradientBoostingClassifier
Outer OOF: generated but invalid for scientific use
Threshold selection: nested 3-way inner OOF; invalidated with feature contract
Inputs: canonical Step 2 inputs as recorded in the model contract
Outputs: reports/task1/step2_k77_model_contract.json; reports/task1/step2_k77_oof_policy_decisions.jsonl; reports/task1/step2_k77_oof_policy_realizability_report.json; docs/task1/workflow_A_new.md
Key metrics: NOT VALID; D77 unresolved; C_minus_D unresolved
Policy decisions: audit-only; do not use for research conclusion
Bootstrap: NOT_RUN_NO_EXISTING_COMPATIBLE_PROTOCOL
Decision: STEP2_CONTRACT_ERROR_REQUIRES_PROFESSOR_REVIEW
Notes: Final audit found duplicate feature columns dropped_baseline_rank and incoming_is_baseline_top5 in the frozen contract. No rerun or further experiment was performed. Fold0=false; public=false; K_sweep=false; reranker_inference=false; GPU=false; workflow_updated=true
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1A-C — Decision-Confidence / Abstention Failure Forensic
Date: 2026-08-29
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Training executed: false
New scoring executed: false
New formula tested: false
K: 77
MEAN executed B/N/H: 6 / 392 / 12
Same-top EXECUTE->NOOP: 337/337
Top-changed EXECUTE->NOOP: 56/56
Available confidence signals: mean/median top1-top2 margins; signed/absolute same-action MEAN-MEDIAN gap; persisted incoming_union_rank and drop_rank
Unavailable confidence signals: raw pairwise dispersion; inner-model agreement; other exact frozen-36 feature values
Pairwise raw dispersion available: false
Inner-model agreement available: false
Mean top1-top2 margin B-vs-H directional AUC: 0.5833333333333333
Median top1-top2 margin B-vs-H directional AUC: 0.5138888888888888
Mean-median disagreement B-vs-H directional AUC: 0.875
Strongest available signal: same_action_mean_median_gap (lower associated with BENEFICIAL)
Strongest directional AUC: 0.875
Headline confidence status: STRONG_BUT_SMALL_N_CONFIDENCE_SEPARATION
Small-N warning: true
Threshold optimization performed: false
Policy counterfactual performed: false
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1A-R — MEAN-MEDIAN Disagreement Signal Robustness Forensic
Date: 2026-08-29
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Training executed: false
New scoring executed: false
Threshold optimization performed: false
Policy counterfactual performed: false
K: 77
Signal selected exploratorily on folds1-4: true
End-to-end selection OOF: false
Original B/H: 6 / 12
Original strongest directional AUC: 0.875/0.875
Original strongest variant: signed
Original raw AUC: 0.125
Original orientation: LOWER_GAP_ASSOCIATED_WITH_B
All-MEAN-top B/H: 69 / 126
All-top signed raw/directional AUC: 0.48780768345985737 / 0.5121923165401426
All-top signed orientation matches original: true
All-top absolute raw/directional AUC: 0.539797561536692 / 0.539797561536692
All-top absolute orientation matches original: true
Oracle-positive B/H: 69 / 8
Oracle-positive signed directional AUC: 0.6340579710144927
Oracle-positive absolute directional AUC: 0.5434782608695652
Same-top abstention total: 337/337
Same-top abstention B/N/H: 3 / 323 / 11
Same-top abstention signed directional AUC: 0.8181818181818181
Same-top abstention absolute directional AUC: 0.7878787878787878
Signed fold orientation status: MIXED_ORIENTATION
Absolute fold orientation status: MIXED_ORIENTATION
Signed robustness status: ROBUSTNESS_NOT_SUPPORTED
Absolute robustness status: ROBUSTNESS_NOT_SUPPORTED
Decision-confidence intervention performed: false
Raw pairwise rescoring performed: false
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P1A-CLOSE — Decision-Confidence / Abstention Sub-Branch Closure Summary
Date: 2026-08-29
Technical status: COMPLETE
Task type: DOCUMENTATION ONLY — NO BRANCH GATE
Training executed: false
Inference/rescoring executed: false
Threshold optimization executed: false
Policy intervention executed: false
New diagnostic executed: false
K: 77
P1A MEDIAN D77: 0.000059523809523809524
P1A MEDIAN executed B/N/H: 1 / 16 / 0
P1A-F abstention delta: +4.333333333333333
P1A-F same-top harmful loss removed: 6.666666666666667
P1A-F same-top beneficial gain lost: 2.3333333333333335
P1A-C original disagreement AUC: 0.875
P1A-C original B/H: 6 / 12
P1A-R all-top B/H: 69 / 126
P1A-R signed directional AUC: ~0.5122
P1A-R absolute directional AUC: ~0.5398
P1A-R signed fold orientation: 2 match / 2 reverse
P1A-R absolute fold orientation: 2 match / 2 reverse
Abstention mechanism: DIRECTLY_MEASURED
MEAN-MEDIAN gap proxy: STRONGLY_WEAKENED
Broad decision-confidence representation: UNRESOLVED
Raw pairwise dispersion: SUPPORTED_HYPOTHESIS_ONLY
Raw pairwise rescoring authorized: false
Confidence intervention authorized: false
Third reducer authorized: false
Pair reweighting authorized: false
HGB: UNRESOLVED
Feature expansion: NOT_JUSTIFIED
K77: NOT_REJECTED
Smaller K: DEFERRED
Workflow B: NOT_ACTIVE
Decision-confidence branch: CLOSED_AS_CURRENTLY_SCOPED
Next: PROFESSOR REVIEW — SELECT NEXT BRANCH FRESH
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P2 — Ranking-Failure Localization Diagnostic
Date: 2026-08-29
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Training executed: false
New scoring executed: false
Pair weighting changed: false
Objective changed: false
Aggregation changed: false
K: 77
Oracle-positive queries: 426/426
Ranking success: 69/69
Ranking failure: 357/357
Failure query share: 0.8380281690140845
Total oracle headroom: 329.2/329.2
Failure oracle headroom: 273.7
Failure headroom share: 0.8314094775212637
Depth 1-20 success/failure: 69/232
Depth 21-29 success/failure: 0/43
Depth 30-45 success/failure: 0/43
Depth 46-77 success/failure: 0/39
Shallow failure rate: 0.770764119601329
Deep failure rate: 1.0
Median failure B-minus-N: -0.11180351945655054
Decisively-wrong failures: 67
Moderately-wrong failures: 220
Near-tie-wrong failures: 70
Median B count success/failure: 2.0/2.0
Median N count success/failure: 142.0/142.0
Median H count success/failure: 0.0/0.0
BN-fraction quartile failure-rate range: 0.07661788044436602
NH-fraction quartile failure-rate range: 0.04849232939516834
Fold failure counts: F1=81; F2=79; F3=96; F4=101
Localization summary: DIFFUSE_FAILURE_PATTERN
Pair-family causal claim made: false
Pair reweighting authorized: false
B-vs-N-only authorized: false
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P2-S — Shallow Ranking-Failure Isolation Diagnostic
Date: 2026-08-29
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Training executed: false
New scoring executed: false
Pair weighting changed: false
Objective changed: false
Aggregation changed: false
K: 77
Shallow oracle-positive: 301/301
Shallow success: 69/69
Shallow failure: 232/232
Shallow failure rate: 0.770764119601329
Shallow failure headroom: 178.41666666666666/178.41666666666666
Shallow failure headroom share: 0.7627360171001069
Shallow top class N/H: 229/3
Best-B score success/failure median: 0.8117236223951685/0.7324749073494621
Best-B score shallow directional AUC: 0.7280109945027486
Pooled best-B score AUC comparator: 0.8102951325457719
Best-N score success/failure median: 0.7860027663452466/0.8190439396506517
Best-N score shallow directional AUC: 0.6355572213893054
Best-B incoming rank success/failure median: 2.0/11.0
Best-B incoming rank shallow directional AUC: 0.9343140929535232
Pooled rank AUC comparator: 0.9573133601266594
Near-tie shallow failures: 70
Moderately-wrong shallow failures: 158
Decisively-wrong shallow failures: 4
Within-rank-bin mechanism: MIXED_DEPTH_AND_SCORE_EFFECT
Score decomposition: B_SUPPRESSION_DOMINANT
Action-count explanatory: false
Shallow mechanism summary: SHALLOW_FAILURE_BOTH_SCORE_AND_RANK_DEPENDENT
Pair reweighting authorized: false
B-vs-N-only authorized: false
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P2-R47 — Rank-Bin-4-7 Joint B/N Score-Shift Structure Diagnostic
Date: 2026-08-29
Technical status: PASS
Scientific status: DIAGNOSTIC ONLY — NO BRANCH GATE
Training executed: false
New scoring executed: false
Pair weighting changed: false
Objective changed: false
Aggregation changed: false
K: 77
Rank4-7 total: 74/74
Rank4-7 success: 14/14
Rank4-7 failure: 60/60
Reference success B median: 0.8318562423746685/0.8318562423746685
Reference success N median: 0.80171758726799/0.80171758726799
Failure JOINT B-down+N-up: 21 / 0.35
Failure B-down only: 24 / 0.4
Failure N-up only: 15 / 0.25
Failure neither adverse: 0 / 0.0
Failure joint share among any adverse: 0.35
Success JOINT B-down+N-up: 0 / 0.0
Success B-down only: 7 / 0.5
Success N-up only: 7 / 0.5
Success neither adverse: 0 / 0.0
Failure best-B/best-N Spearman rho: 0.9265907196443457
Success best-B/best-N Spearman rho: 0.9516483516483516
Dominant failure category: B_DOWN_ONLY
Ranking intervention authorized: false
Pair reweighting authorized: false
B-vs-N-only authorized: false
Fold0/public: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 2-P3 — Margin-Augmented Pairwise B-vs-N Relative-Ordering Intervention
Date: 2026-08-29
Technical status: BLOCKED_SPEC_CONFLICT
Experiment type: CONTROLLED SINGLE-VARIABLE INTERVENTION — PRE-FLIGHT BLOCKED
Training executed: false
New scoring executed: false
K: 77
Requested BN margin: 0.05
Blocking contract: sklearn.ensemble.HistGradientBoostingClassifier fixed binary log_loss; no custom per-example loss/gradient API
Exact incompatibility: canonical trainer learns pairwise probabilities from x(a)-x(b), not verified train-time s(a)-s(b); action scores are post-training MEAN probabilities, so m=0.05 has no verified equivalent loss scale and BN-only softplus(m-y*z) cannot be implemented exactly under frozen HGB
Approximation used: false
Fold0/public: false
Next: PROFESSOR CLARIFICATION
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 3-A — Confirmatory Holdout Data-Scope Audit
Date: 2026-08-30
Technical status: PASS
Task type: READ-ONLY AUDIT
Models run: false
Training executed: false
Inference executed: false
Evaluation executed: false
Untouched label values inspected: false
Research Constitution found: true
Research Constitution path: docs/task1/workflow_A_new.md
Partitions identified: LegalIR_train_parent; F1; F2; F3; F4; Fold0; public; LegalIR_warmup; synthetic_qa_benchmark
F1-F4 confirmatory eligible: false
Post-hoc F1-F4 split confirmatory eligible: false
Fold0 constitution status: CONDITIONALLY_AUTHORIZED
Fold0 untouched status: false
Fold0 Step3 eligible: false
Public constitution status: CONDITIONALLY_AUTHORIZED
Public untouched status: false
Public Step3 eligible: false
Other eligible untouched partitions: NONE
Untouched authorized holdout exists: NO
Step3 recommendation: BLOCKED_NO_INDEPENDENT_AUTHORIZED_HOLDOUT
Step3 execution performed: false
Next: PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: STEP 4 — Workflow A Finalization
Date: 2026-08-30
Technical status: PASS
Task type: ARTIFACT FREEZE AND DOCUMENTATION ONLY
Training executed: false
Inference executed: false
New evaluation executed: false
Fold0 used: false
Public labels used: false
Step2 status: CLOSED
Step3 status: BLOCKED_NO_INDEPENDENT_AUTHORIZED_HOLDOUT
Independent confirmation achieved: false
Final Workflow-A policy: P5
Policy status: PROVISIONAL_ONLY
Scientific status: SUPPORTED_WITHIN_EXPLORATORY_OOF_SCOPE
K: 77
Beta: 0.05
Baseline R: 357/426
P5 R: 356/426
Baseline D77_exact: -0.0007142857142857143
P5 D77_exact: -0.0005654761904761906
Baseline gain_sum: -4.0
P5 gain_sum: -3.166666666666667
Baseline S/T/R: 6/63/357
P5 S/T/R: 6/64/356
R non-worsened folds: 4/4
Failure->success: 1
Success->failure: 0
P5 preregistered validation: PASS
Workflow A: CLOSED
Workflow B: ELIGIBLE_TO_OPEN
Specific Workflow-B experiment authorized: false
Next: WORKFLOW B PREREGISTRATION / PROFESSOR REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: P5 Production P1/P4 Final Fit
Date: 2026-08-30
Technical status: PASS
Operation: DEPLOYMENT_OF_FROZEN_EXPLORATORY_POLICY
Production P1 fit: true
Production P4 fit: true
Training population: F1-F4
Fold0 used: false
Public labels used: false
K: 77
Features: 36
P1 canonical contract match: true
P4 canonical contract match: true
P4 target: BENEFICIAL vs {NEUTRAL,HARMFUL}
P4 class weighting: none
P4 resampling: none
Hyperparameter tuning: false
Threshold selection executed: false
Public inference executed: false
Submission generated: false
Production P1 artifact: reports/task1/production_p1_model.pkl
Production P4 artifact: reports/task1/production_p4_model.pkl
Next: PROFESSOR REVIEW OF PRODUCTION FIT ARTIFACTS
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: P5 Production Threshold Selection
Date: 2026-08-30
Technical status: PASS
Operation: DEPLOYMENT_OF_FROZEN_EXPLORATORY_POLICY
Threshold-selection population: ALL F1-F4 queries
Query count: 5600
Inner OOF: true
Truth joined after held-out scores/actions frozen: true
Oracle-positive-only: false
K: 77
Features: 36
Beta: 0.05
P4 target: BENEFICIAL vs {NEUTRAL,HARMFUL}
Production threshold: Infinity
Canonical gain-sum selection: true
Fold0 used: false
Public labels used: false
Public inference executed: false
Submission generated: false
Next: PROFESSOR REVIEW OF PRODUCTION THRESHOLD
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Frozen P5 All-NO_OP Public Submission
Date: 2026-08-30
Technical status: PASS
Operation: DEPLOYMENT_OF_FROZEN_P5_WITH_ALL_NO_OP
K77/36-feature equivalence gate: PASS
Production threshold: Infinity
Finite-threshold substitution: false
All public queries NO_OP: true
Public P1 inference executed: false
Public P4 inference executed: false
P5 fusion inference executed: false
Baseline public artifact: artifacts/task1/recovery_096/public_anchor_093/submission_093.zip
Baseline/public final mismatch queries: 0
Fold0 used: false
Public labels used: false
Leaderboard feedback used: false
Tuning executed: false
Validator: PASS
Upload-ready submission: artifacts/task1/submission_p5_all_noop/public_submission.json
Submission SHA256: f1462cb1280ef942f0ceee1dcaf9e384bd90c8de976342ae41484a7b1b77c00f
Scientific improvement claimed: false
Workflow A reopened: false
Workflow B evidence created: false
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Task1 Active Metric Contract Amendment
Date: 2026-08-30
Technical status: PASS
Primary metric: set-based macro Recall
Secondary metric: set-based macro Precision
Max predictions per query: 5
Rank order within unchanged top-5 affects primary Recall: false
Top-5 membership affects primary Recall: true
MRR/Recall@3 status: STALE
Metric semantics resolved: true
Exact scorer binary provenance resolved: false
Scientific reference separated from deployment incumbent: true
Deployment incumbent score metadata: 0.9391
Fold0 scientific exclusion: active
Workflow A status: CLOSED
Workflow B scientific development authorized: true
Next: PROFESSOR PREREGISTRATION OF WORKFLOW-B B1
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Workflow B B1 Direct LTR
Date: 2026-08-30
Technical status: BLOCKED
Scientific branch: B1
Estimator: LightGBM LGBMRanker
Objective: LambdaRank / NDCG@1
K: 77
Features: 36
P4 feature used: false
Target encoding: H=0 N=1 B=2
Outer OOF queries: 0
Fold0 used: false
Public labels used: false
Hyperparameter sweep: false
Primary metric: set-based macro Recall
P1 pooled Recall: NOT_RUN
B1 pooled Recall: NOT_RUN
Recall delta: NOT_RUN
P1 pooled Precision: NOT_RUN
B1 pooled Precision: NOT_RUN
Precision delta: NOT_RUN
F1 Recall delta: NOT_RUN
F2 Recall delta: NOT_RUN
F3 Recall delta: NOT_RUN
F4 Recall delta: NOT_RUN
Primary success >= +0.003: false
All folds non-negative: false
Precision delta >= -0.001: false
Surrogate/result classification: NOT_RUN
Scientific status: BLOCKED
Next: PROFESSOR B1 REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Workflow B B1 Runtime Recovery + Frozen Execution
Date: 2026-08-31
Recovery classification: IMPLEMENTATION_RUNTIME
Original B1 preregistration changed: false
Recovery equivalence: PASS
Recovery mismatch count: 0
Models fit before equivalence PASS: 0
Bounded-memory materialization: true
K: 77
Features: 36
LightGBM version: 4.5.0
P4 used: false
Fold0 used: false
Public labels used: false
OOF queries: 0
P1 pooled Recall: NOT_RUN
B1 pooled Recall: NOT_RUN
Recall delta: NOT_RUN
P1 pooled Precision: NOT_RUN
B1 pooled Precision: NOT_RUN
Precision delta: NOT_RUN
F1 Recall delta: NOT_RUN
F2 Recall delta: NOT_RUN
F3 Recall delta: NOT_RUN
F4 Recall delta: NOT_RUN
B1 scientific result: NOT_YET_EVALUATED
Scientific status: BLOCKED
Next: PROFESSOR B1 RESULT REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Workflow B B1 Full-Scale Runtime Provenance Diagnostic
Date: 2026-08-31
Scientific experiment completed: false
Scientific evidence produced: false
Canonical data equivalence: PASS
Environment compatibility smoke: PASS
Real instrumented run executed: true
Previous failure reproduced: true
First full project-data fit completed: false
Exit code: NOT_CAPTURED_WRAPPER_EXTERNALLY_TERMINATED
Elapsed seconds: 49.6458855
Peak process RSS bytes: 4218880
Minimum system available RAM bytes: NOT_CAPTURED_CIM_ACCESS_DENIED
Known orchestration timeout seconds: UNKNOWN
Python traceback present: false
Native crash evidence present: false
Windows resource exhaustion evidence: false
Synthetic same-scale stress test run: false
Root cause class: EXTERNAL_TERMINATION
Root cause confidence: MEDIUM
Scientific contract changed: false
Fold0 used: false
Public labels used: false
Scientific B1 status: NOT_YET_EVALUATED
Next: PROFESSOR B1 RUNTIME ROOT-CAUSE REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Workflow B B1 Environment Amendment + Frozen Execution
Date: 2026-08-31
Original B1 preregistration changed: false
Dependency block classification: IMPLEMENTATION_RUNTIME
LightGBM version: 4.5.0
Previous sklearn version: 1.9.0
Authorized sklearn version: 1.7.2
Compatibility shim used: false
Synthetic compatibility smoke: PASS
Project labels used in smoke: false
Recovery equivalence: PASS
Recovery mismatch count: 0
Canonical action rows: 806644
K: 77
Features: 36
P4 used: false
Fold0 used: false
Public labels used: false
OOF queries: 0
P1 pooled Recall: NOT_RUN
B1 pooled Recall: NOT_RUN
Recall delta: NOT_RUN
P1 pooled Precision: NOT_RUN
B1 pooled Precision: NOT_RUN
Precision delta: NOT_RUN
F1 Recall delta: NOT_RUN
F2 Recall delta: NOT_RUN
F3 Recall delta: NOT_RUN
F4 Recall delta: NOT_RUN
Primary success >= +0.003: NOT_RUN
All folds non-negative: NOT_RUN
Precision guard: NOT_RUN
B1 scientific result: NOT_YET_EVALUATED
Scientific status: BLOCKED
Next: PROFESSOR B1 RESULT REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Workflow B B2a-0 Preflight Recovery
Technical status: BLOCKED
Scientific result produced: false
Model: Qwen/Qwen3-Reranker-0.6B
Model revision: e61197ed45024b0ed8a2d74b80b4d909f1255473
Previous provenance audit reused: true
Context audit executed on CPU: true
Context audit CUDA dependency removed: true
Query count: 5600
Expected pairs: 431200
Context pairs audited: 431200
p99 token length: 158503
Selected max_length: NOT_SELECTED_P99_GT_32768
Truncation rate: NOT_COMPUTED_NO_VALID_MAX_LENGTH
GPU provider: NOT_RUN_CONTEXT_GATE_BLOCKED
GPU: NOT_RUN
GPU smoke: BLOCKED
Scientific metrics computed: false
Training labels used: false
Fold0 used: false
Public labels used: false
Full B2a inference run: false
Next: PROFESSOR B2A PREFLIGHT REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2a-0 Long-Document Root-Cause Audit
Owner: TV2
Type: UNLABELED_DIAGNOSTIC
Context audit execution: PASS
Context feasibility before audit: BLOCKED_LONG_DOCUMENT
Pairs previously audited: 431200
Labels used: false
Fold0 used: false
Public labels used: false
GPU used: false
Model inference run: false
Primary root-cause classification: D_MIXED_REAL_AND_PIPELINE_PROBLEM
Pipeline bug found: true
Long-document handling change authorized: false
Scientific B2a result produced: false
Next: PROFESSOR LONG-DOCUMENT REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2a-0 Passage Provenance and Long-Document Corruption Map
Owner: TV2
Type: READ_ONLY_DATA_PROVENANCE
Technical status: BLOCKED
Passage producer identified: false
GT32768 document count reconciled: 668
EQ32768 document count: 0
Audit boundary mismatch found: true
Long documents classified: 668
Confirmed pipeline corruption: 0
Confirmed upstream corruption: 0
Likely genuine long documents: 588
Mixed/ambiguous: 80
Unresolved provenance: 0
Safe deterministic repair candidates: 0
Projected pairs >32768 after safe repair: 125568
Data modified: false
GPU used: false
Model inference: false
Labels used: false
Fold0 used: false
Public labels used: false
Chunking introduced: false
Scientific B2a result produced: false
Next: NEEDS_MORE_PROVENANCE
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
2026-09-02 — B1 — Resolve Environment Block & Smoke Test
Mục tiêu: Kiểm tra môi trường Python 3.12 với LightGBM 4.5.0 và sklearn 1.7.2 để gỡ block.
Đã làm: Chạy B1 compatibility smoke test bằng môi trường ~/venv_b1_py312.
Kết quả chính: Lỗi force_all_finite đã được giải quyết, smoke test chạy thành công.
Trạng thái: PASS
Điều rút ra: Môi trường đã tương thích hoàn toàn với hợp đồng đóng băng của B1.
Chưa được kết luận: Điểm Recall/Precision cuối cùng (vì chưa chạy mô hình thực tế).
Artifact chính: reports/task1/workflow_b/tv4/b1/contracts/workflow_b_b1_compatibility_smoke.json
Bước tiếp theo: Chạy full pipeline B1 (Runtime recovery & OOF scoring).

=== B2A TRUE-S2 CANONICAL PROMOTION ===
Status: COMPLETE_CANONICAL_TRUE_S2_WORKLIST
Canonical worklist: reports/task1/workflow_b/tv2/b2a/manifests/b2a_qwen_true_s2_top3_worklist.jsonl
Canonical queries: 5600
Canonical query-document pairs: 431200
Docs per query: 77/77/77
Duplicate query-document pairs: 0
Missing canonical pairs: 0
Wrong-doc mappings: 0
Missing pairs processed: 7365
Historical semantics validation: 100/100 pairs, 300/300 stored BM25 scores
Post-completion reproduction: 100/100
Canonical SHA256: a935e356cf8fe62a000e56c3817fd31dfc5ebbc079c4eef849803a7552c3c25a
Qwen: NOT_RUN
GPU: NOT_RUN
=== B2A TRUE-S2 CANONICAL PROMOTION END ===

=== PROGRESS_LOG_ENTRY START ===
Task: Modal pending-call diagnosis
Date: 2026-09-04
Observed app: ap-jauKb7QQ8kfk6QlGGovWla (tmp-qwen3vl2b-load-probe)
Finding: stale temporary probe remains ephemeral and repeatedly retries container startup
Exact repeated error: ModuleNotFoundError: No module named 'scripts'
Evidence: app logs show failed containers approximately every 10 minutes; canonical smoke apps are stopped
Cause: orphaned temporary probe call retrying import failure, not A10 capacity and not Qwen True-S2 execution
Action: no cancellation performed
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Experiment: B2A-0 — Frozen Qwen3-Reranker-0.6B F1-F4 Scientific Evaluation
Date: 2026-09-03
Technical status: PASS
Scientific status: FAIL — REJECT_B2A_QWEN_06B
Prediction SHA256 verified: 9f73f9424de78a0824c1e7153d080890cfe599104c103a4e2e81b5763b03fcd9
Prediction queries: 5600
Evaluation folds: 1,2,3,4
Qwen Recall: 0.1593363095238095
Scientific reference Recall: 0.9259285714285714
Recall delta: -0.7665922619047619
Qwen Precision: 0.03410714285714286
Scientific reference Precision: 0.19739285714285715
Precision delta: -0.16328571428571428
Per-fold Recall delta: F1=-0.7854166666666668,F2=-0.7729761904761905,F3=-0.7611309523809524,F4=-0.7468452380952381
Nonnegative Recall folds: 0/4
Top-5 set differences vs scientific reference: 5573
Recall query comparison better/same/worse: 25/1165/4410
Fold0 used: false
Public labels used: false
Predictions modified: false
Canonical scorer: src/udsc2026/evaluation/legal_ir.py::evaluate_legal_ir
Canonical F1-F4 source: artifacts/task1/evaluation/strict_cv_v2/folds.json + data/raw/btc/LegalIR/train.json
Reference artifact: reports/task1/workflow_a/step2p1_oof_policy_decisions.jsonl
Persistent B2a evaluation report: REPORT_PERSISTENCE_NEEDS_USER_DECISION
Next: PROFESSOR B2A RESULT REVIEW
=== PROGRESS_LOG_ENTRY END ===
=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B A10 B16 True-S2 compatibility smoke
Date: 2026-09-03
Prompt/task: Run canonical 2B True-S2 smoke after model-load validation
Modal profile: hoconlinea10
Environment: main
GPU requested: NVIDIA A10
Batch size: 16
App ID: ap-Nk5UFW1Pomgpjh1C2lSQOe
FunctionCall ID: fc-01M1M31HGRJA7NTW0V228TA40N
Execution: one smoke call submitted; existing FunctionCall reattached
Result: BLOCKED
Exact exception: modal.exception.RemoteError('')
Metrics: NOT_AVAILABLE
True-S2/worklist/chunks/labels: no rebuild; labels not loaded
Full inference: NOT_RUN
Next: FIX_EXACT_SMOKE_ERROR
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A self-diagnosing True-S2 smoke
Date: 2026-09-04
Modal profile: hoconlinea10
Environment: main
GPU requested: A10
Batch size: 16
App ID: ap-fCWXFpy8C6WPNvVNwzlvFK
FunctionCall ID: fc-01M1M6YFAMK2WCNE5P4K1T4XRT
Execution: exactly one smoke FunctionCall submitted; reattachment used without replacement call
Result: BLOCKED
Local result: modal.exception.RemoteError('')
Diagnostic path checked: /runtime/reports/task1/workflow_b/tv2/b2a/runtime/qwen3vl2b_smoke_diagnostic.json
Diagnostic artifact: NOT_FOUND
Classification: FAIL_BEFORE_SMOKE_FUNCTION_BODY_OR_VOLUME_WRITE
Labels/Fold0/full inference: NOT_RUN
Next: FIX_PRE_FUNCTION_MODAL_FAILURE
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A canonical decorated smoke() reversible no-op startup isolation
Date: 2026-09-04
Modal profile/environment: hoconlinea10/main
Pre-test runner SHA256: 974c28cd694171599d8596a6a1a9f5036de0f255913fe9f65c52a747bc627815
Temporary change: smoke() body only; emitted CANONICAL_NOOP_SMOKE_ENTERED, wrote canonical_noop_smoke_probe.json, committed volume, and returned PASS. Decorator/resources were unchanged (A10, CPU 8, 32768 MiB).
Local temporary validation: py_compile PASS; git diff --check PASS.
Duplicate guard: no active Qwen3-VL 2B app listed; prior exact-resource probe was stopped.
App ID: ap-WzN6Aumo6WXDYQigfNNx06
FunctionCall ID: fc-01M1MVH6SBE76ZHA0ZGMTPQFFF
Result: modal.exception.RemoteError(''); reattachment made no new FunctionCall.
No-op marker: NOT_FOUND in udsc-p13 runtime directory.
Classification: FAIL_CANONICAL_MODULE_OR_FUNCTION_STARTUP (failure before entering exact decorated smoke() body).
Qwen/worklist/chunks/inference: NOT_RUN.
Restoration: exact canonical smoke() wrapper restored; post-restore SHA256 matches pre-test; post-restore py_compile PASS; git diff --check PASS.
Scientific/model/True-S2 contract changed: false
Next: FIX_CANONICAL_MODULE_STARTUP
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A exact canonical smoke resource-contract probe
Date: 2026-09-04
Modal profile: hoconlinea10
Environment: main
Canonical/probe resource contract: A10; CPU 8; memory 32768 MiB; timeout 3600 seconds; Volume udsc-p13 mounted at /workspace/p13; canonical heavy image dependencies
App ID: ap-5WRVuz2kB0wkZUNJe114dy
FunctionCall ID: fc-01M1MTM3MQ21VCN71RV3PTMCHK
Result: CANONICAL_RESOURCE_REQUEST_HEALTHY
Function body entered: true
CUDA/GPU/mount: true/NVIDIA A10/true
Smoke resource modified: false
Real True-S2 smoke submitted: false (resource blocker not proven)
Temporary probe removed: true
Next: INVESTIGATE_NON_RESOURCE_PREFUNCTION_FAILURE
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B A10 B16 True-S2 compatibility smoke retry after orphan cancellation
Date: 2026-09-04
Modal profile: hoconlinea10
Environment: main
GPU requested: A10
Batch size: 16
App ID: ap-ZKoIipQsAKdVfAJxxj1a0E
FunctionCall ID: fc-01M1M6HXHW00ZC6XGH7RJ8S686
Execution: exactly one smoke FunctionCall submitted; reattachment used without spawning a second call
Result: BLOCKED
Exact exception: modal.exception.RemoteError('')
FunctionCall-scoped logs: no output retained by Modal CLI
Metrics: NOT_AVAILABLE
Labels/Fold0/full inference: NOT_RUN
Next: FIX_EXACT_SMOKE_ERROR
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A canonical Modal bootstrap probe
Date: 2026-09-04
Modal profile: hoconlinea10
Environment: main
Resource contract: canonical image dependencies, volume udsc-p13, mount /workspace/p13, A10, CPU 8, memory 32768 MiB
App ID: ap-KSkds8nUvZld5mhzL9KCq5
FunctionCall ID: fc-01M1M7H6FPHYPQJ69CS8C4QVZ0
Result: FAIL_MODAL_PLATFORM_PRE_FUNCTION
Remote return: modal.exception.RemoteError('')
Volume marker: /runtime/reports/task1/workflow_b/tv2/b2a/runtime/modal_bootstrap_probe.json NOT_FOUND
Function body entered: false
Qwen/worklist/chunks/inference: NOT_RUN
Temporary probe removed: true
Next: FIX_CANONICAL_FUNCTION_STARTUP
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: UDSC-P13 minimal CPU Volume mount probe
Date: 2026-09-04
Modal profile: hoconlinea10
Environment: main
Volume: udsc-p13
Mount: /workspace/p13
App ID: ap-7rJMfbe7xynXnFwPG107MA
FunctionCall ID: fc-01M1M8EWPE1T6MGSK4RE6EFF58
Result: PASS_VOLUME_MOUNT
Function body entered: true
Mount exists: true
train.json exists: true
chunks directory exists: true
canonical worklist exists: true
GPU/Qwen/scientific data parsing/inference: NOT_RUN
Temporary probe removed: true
Next: TEST_QWEN_IMAGE_WITH_VOLUME_NO_MODEL
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A heavy Qwen image plus Volume bootstrap probe
Date: 2026-09-04
Modal profile: hoconlinea10
Environment: main
Volume/mount: udsc-p13 /workspace/p13
GPU: NVIDIA A10
App ID: ap-c80775O2e88ix4W9SaIeHC
FunctionCall ID: fc-01M1M8MPBXT6AMXTFHEADA1QFQ
Result: PASS_QWEN_IMAGE_VOLUME_BOOTSTRAP
Function body entered: true
Mount/train.json visible: true/true
Dependencies: torch 2.8.0+cu128; torchvision 0.23.0+cu128; transformers 4.57.6; accelerate 1.14.0; huggingface_hub 0.36.2
CUDA available: true
Diagnostic marker persisted: runtime/reports/task1/workflow_b/tv2/b2a/runtime/qwen_image_volume_probe.json
Qwen model/worklist/chunks/inference: NOT_RUN
Temporary probe removed: true
Next: COMPARE_CANONICAL_SMOKE_FUNCTION_GRAPH
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A canonical Qwen3-VL smoke function graph audit
Date: 2026-09-04
Scope: no GPU submission; no Qwen/data/inference execution
Finding: canonical module has only stdlib plus modal imports; no scripts.* package imports or module-level data reads found.
Resource graph comparison: heavy-image+A10+udsc-p13 probe passed, so image, Volume, CUDA and basic function startup are independently healthy.
Local Modal definition validation: PASS via `modal run scripts/modal/task1_b2a_qwen3vl2b.py::smoke --help`.
Wrapper observability added: qwen3vl2b_smoke_wrapper_entry.json persisted before smoke_remote() is referenced.
Scientific/model/True-S2 contract changed: false
Root cause: NOT_PROVEN
Next: RUN_ONE_A10_B16_SMOKE
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A decisive A10 B16 True-S2 smoke with wrapper markers
Date: 2026-09-04
Modal profile: hoconlinea10
Environment: main
App ID: ap-mKRWM3AnbyLk4FHhW2FS8V
FunctionCall ID: fc-01M1M9N1RK30HW4NJ8FWT6KC2N
GPU/batch: A10/16
Execution: exactly one smoke FunctionCall; reattachment performed without replacement call
Result: BLOCKED
Local result: modal.exception.RemoteError('')
Wrapper marker: NOT_FOUND
Smoke diagnostic: NOT_FOUND
Classification: FAIL_BEFORE_DECORATED_SMOKE_BODY
Labels/Fold0/full inference: NOT_RUN
Next: FIX_PRE_FUNCTION_MODAL_FAILURE
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A canonical Qwen3-VL 2B detached A10 B16 True-S2 compatibility smoke
Date: 2026-09-04
Modal profile/environment: hoconlinea10/main
Lifecycle: modal run --detach plus canonical smoke.spawn(); runner unchanged.
App ID: ap-nKgDn3I7QBQ3iRzubLBWOx
FunctionCall ID: fc-01M1MW76TQX11YB2FBQ4E3B3YV
Result: PASS_2B_COMPATIBILITY_SMOKE
Lifecycle classification: DETACH_FIXED_PREFUNCTION_LIFECYCLE
Wrapper marker: qwen3vl2b_smoke_wrapper_entry.json present.
Diagnostic: PASS / SMOKE_PASS; GPU NVIDIA A10; batch 16; pairs 64/64; chunks 192; finite document scores 64/64; NaN/Inf 0; OOM false.
Runtime/throughput/peak VRAM/max tokens: 5.322410393 s / 36.07388115965601 chunks/s / 4.360501289367676 GiB / 481.
Labels/Fold0/full inference: false/false/false.
Runner/scientific contract changed: false
Next: RUN_2B_BATCH_BENCHMARK
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B controlled A10 batch benchmark
Date: 2026-09-04
Modal profile/environment: hoconlinea10/main
Lifecycle: modal run --detach plus benchmark.spawn(); no full inference.
App ID: ap-0ClSRTBV9Vu75Lq5T3TGj6
FunctionCall ID: fc-01M1N58VQS5XGD3Q5C95N11JTM
Sample: existing deterministic 512 True-S2 q-doc pairs / 1536 selected chunks; labels and Fold0 not loaded.
Batch 16: 40.476026434999994 s; 37.948389090679285 chunks/s; 4.526285648345947 GiB; max tokens 683; truncated 0; NaN/Inf 0; OOM false; reference.
Batch 32: 42.79800470999999 s; 35.88952359830703 chunks/s; 5.081110000610352 GiB; NaN/Inf 0; OOM false; max delta 0.033203125; chunk/document parity false; ineligible.
Batch 64: 46.31273936900001 s; 33.165820483254336 chunks/s; 6.188146114349365 GiB; NaN/Inf 0; OOM false; max delta 0.03515625; chunk/document parity false; ineligible.
Selection: batch 16, the only parity-safe batch and also highest measured throughput.
Model/revision/scoring/prompt/max_length/K77/True-S2 changed: false
Full inference/predictions: NOT_RUN
Next: RUN_FULL_F1_F4_INFERENCE
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B full F1-F4 scientific inference launch
Date: 2026-09-04
Modal profile/environment: hoconlinea10/main
Runner: scripts/modal/task1_b2a_qwen3vl2b.py
Model/revision: Qwen/Qwen3-VL-Reranker-2B / 4bd860ac4f15ad1897a214615cccc700f8f71818
Execution: exactly one detached full call; no labels, Fold0, evaluation, or public submission.
GPU/batch: NVIDIA A10 / 16. Full decorator changed only from A100-40GB to A10 per frozen benchmark selection.
App ID: ap-4ebAucBSSKuqqee1sC7T3J
FunctionCall ID: fc-01M1N6S0003JNTR1AK80ADRQJC
Initial checkpoint: integrity=absent; completed_chunks=0; completed_q_doc_pairs=0; canonical 2B namespace isolated.
Observed logs: CHECKPOINT_VERIFIED, HF snapshot download in progress; app state ephemeral (detached), tasks=1.
Status: RUNNING
Full scientific inference/evaluation: NOT_COMPLETE / NOT_RUN
Next: REATTACH_EXISTING_FUNCTIONCALL
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-Reranker-2B F1-F4 scientific evaluation
Date: 2026-09-04
Prediction artifact: artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl
Prediction SHA256: 9e5750ef71f2f0d300ab4fb850358e913b2d1bc5f923f940a2ebcd766999dfa1 (verified)
Evaluation: F1-F4 only; 5600 queries; exactly 5 unique predictions/query; Fold0/public labels/GPU/submission not used.
Comparator: reports/task1/workflow_a/step2p1_realizability_report.json plus canonical baseline top-5 artifact; reference Recall 0.9259285714285714; Precision 0.19739285714285715.
Qwen 2B metrics: Recall 0.05084821428571429; Precision 0.011464285714285715; Recall delta -0.8750803571428571; Precision delta -0.18592857142857144.
Per-fold Recall: F1 0.050119047619047626 (delta -0.88625); F2 0.050238095238095234 (delta -0.8780952380952382); F3 0.04755952380952381 (delta -0.8718452380952382); F4 0.05547619047619048 (delta -0.8641309523809524).
Materiality gate: FAIL; nonnegative Recall folds 0/4; Precision guard FAIL; top-5 sets differ 5600/5600; Recall better/same/worse 25/550/5025.
Final decision: REJECT_B2A_QWEN3VL_2B.
Persistence: no canonical 2B evaluation report path defined; REPORT_PERSISTENCE_NEEDS_USER_DECISION.
Next: PROFESSOR_REVIEW_OR_NEW_AUTHORIZED_B2A_BRANCH
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A catastrophic-failure sanity audit (no full rerun)
Date: 2026-09-04
Prediction SHA256: 9e5750ef71f2f0d300ab4fb850358e913b2d1bc5f923f940a2ebcd766999dfa1 verified; predictions unchanged.
Official contract/source: models/qwen3-vl-reranker-2b/scripts/qwen3_vl_reranker.py and sentence-transformers metadata. Contract conceptually matches current runner: final hidden state, yes(9693)-no(2152), sigmoid, higher-is-relevant; template/truncation/padding are materially consistent for text-only input.
Frozen score diagnostics (F1-F4, no Fold0/public labels): normal Recall@5 0.05084821428571429; reversed Recall@5 0.03927083333333333; gold/non-gold q-doc medians 0.443359375/0.44140625; AUC 0.511060740015279.
True-S2 diagnostic: 5984 gold q-docs, 17935 selected gold chunks; selected BM25 rank median 2 (range 1-3), score median 20.97980564953179. Raw available-chunk inventory not materialized from frozen artifact, so TRUE_S2_NOT_PROVEN_BROKEN.
Official parity probe: initial 23-pair bounded probe completed with max delta 0.0234375, median delta 0.0078125, chunk/q-doc ordering parity FAIL (official script initially lacked scipy; no scientific artifacts written). Corrected exact-32 probe submitted as one detached FunctionCall fc-01M1PMNB73Q2GX75Z56C6PQZJ3 (App ap-sI5Ujo4ovfQCddhgw2AfsG); remained detached/pending with no terminal logs at audit stop.
Classification: INCONCLUSIVE_NEEDS_PROFESSOR_REVIEW (exact-32 parity call not terminal; partial parity evidence materially mismatched).
Full inference rerun: NO. GPU used only for bounded parity probes; no predictions/checkpoints modified.
Next: PROFESSOR_REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-2B B1 versus B16 bounded scientific recovery diagnostic
Date: 2026-09-05
Modal profile/environment: hoconlinea10/main
Lifecycle: exactly one detached temporary diagnostic submission; recovery.spawn() FunctionCall fc-01M1PX3HJWMGZ7C46GYBFXT219, App ap-GLC6uPp4XowJyyt2Dvu0mr. No retry.
Hardware: requested NVIDIA A10; runtime observed NVIDIA A10 (exact requested class, no hardware mismatch).
Scope/result: 100 deterministic evenly-spaced F1-F4 query units; 7,700 q-doc pairs; 23,093 True-S2 selected chunks. Labels loaded only after B1 and B16 score maps were held in memory.
B1 vs B16: maximum/median/p95 chunk-score delta 0.046875/0.005859375/0.01953125; top-5 sets differed for 40/100; mean top-5 overlap 0.908.
Scientific metrics: B1 Recall/Precision/AUC 0.81/0.168/0.9242780772352798; B16 0.79/0.16400000000000003/0.9237191188522721; improvements 0.020000000000000018/0.003999999999999976/0.0005589583830076794. Query Recall B1 better/same/worse 4/94/2; B1 added a gold document for 4 queries.
Decision: BATCH1_MODEST_RECOVERY (the +0.02 recall threshold was met; B1-vs-B16 AUC change was negligible). Both conditions scored successfully in a single loaded model instance.
Guards: Fold0 false; public labels false; full inference false; frozen predictions false; canonical runner false. Temporary script removed after terminal result retrieval.
Next: PROFESSOR_REVIEW
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A frozen-full versus 100-query diagnostic-B16 contradiction audit
Date: 2026-09-05
Authorization/safety: CPU-only read-only audit; no model inference, GPU, Modal submission, runner or artifact modification, Fold0, or public labels.
Frozen artifact: predictions.jsonl SHA256 verified exactly 9e5750ef71f2f0d300ab4fb850358e913b2d1bc5f923f940a2ebcd766999dfa1.
Exact sample recovery: canonical distinct-worklist query indices {round(i * 5599 / 99) for i in range(100)}; 100 IDs recovered. Frozen subset has 100 queries and 7,700 q-doc scores, Recall 0.055, Precision 0.012000000000000002, AUC 0.5204742937775815.
Checkpoint audit: read-only SQLite integrity ok; 1,293,198 chunks, 431,200 q-doc groups, 5,600 queries. For the exact 100 queries, checkpoint MAX aggregation and frozen serialization match exactly: 100/100 candidate sets, 7,700 scores, document order, top-5, and every document score (max/median/p95 delta 0.0).
Diagnostic evidence: the temporary recovery source deliberately retained B1/B16 keyed scores only in memory and wrote only RECOVERY_RESULT aggregates; the source was removed after the permitted diagnostic and no exact B16 score/ranking artifact exists. Per-doc frozen-vs-diagnostic comparison is therefore unavailable.
Representativeness: sample/full fold counts 25/25/25/25 versus 1400/1400/1400/1400; K77 gold-presence 0.97 versus 0.99125; K77 oracle Recall@5 0.665 versus 0.6687113095238095; no obvious easy-sample bias.
Decision: INCONCLUSIVE_MISSING_DIAGNOSTIC_SCORES. The frozen same-query subset is low-performing, while the recorded diagnostic B16 aggregate is high-performing; candidate, True-S2, model/prompt, aggregation, and checkpoint-to-frozen-serialization contracts are not the shown divergence.
Next: RECOVER_EXACT_DIAGNOSTIC_SCORES
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A exact frozen-versus-fresh B16 per-score trace
Date: 2026-09-05
Authorization: recover existing task-scoped B16 scores if available; otherwise one deterministic 100-query B16-only rerun with exact keyed score persistence. No B1, full inference, canonical-runner or frozen-artifact modification, Fold0, or public labels.
Preflight: prior temporary source is absent; only its bytecode remains. Relevant runtime/artifact paths contain no keyed recovery B16 score artifact, so a single B16 rerun is required.
Status: PREPARING_SINGLE_B16_RERUN
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A exact frozen-versus-fresh B16 per-score trace outcome
Date: 2026-09-05
Lifecycle: exactly one detached B16-only trace call, App ap-5eJz5iUq5SZDySnZKw0cTI, FunctionCall fc-01M1Q1VDNC5198Q59W1M8YBCXT; no B1 or retry.
Hardware/scope: NVIDIA A10; 100/100 deterministic queries, 7,700/7,700 q-doc pairs, 23,093 chunks; frozen SHA verified before comparison.
Persisted bounded diagnostic: /workspace/p13/runtime/reports/task1/workflow_b/tv2/b2a/runtime/qwen3vl2b_fresh_b16_score_trace.jsonl, one JSONL row per immutable q-doc-chunk score.
Fresh versus frozen document comparison: 7,700 rows; no missing or duplicate keys; max/median/p95 absolute score delta 0.521484375/0.109375/0.33203125; Pearson 0.1137284087764302; Spearman 0.07716167246485887.
Metrics: fresh Recall/AUC 0.79/0.9237191188522721; frozen Recall/AUC 0.055/0.5204742937775815.
Checkpoint evidence: all 23,093 fresh keys present; fresh versus checkpoint chunk max/median/p95 delta 0.609375/0.1015625/0.306640625. Prior read-only audit established checkpoint MAX aggregation, document ordering, top-5, and frozen serialization exact (0.0 deltas), localizing the first observed divergence to CHECKPOINT_STORED_SCORE rather than aggregation/key association/serialization.
Classification: MODEL_FORWARD_FULL_VS_FRESH_DIVERGENCE. The exact source-level trigger remains unlocalized; no checkpoint corruption or mapping bug is proven. Temporary runner script removed; bounded score trace retained as authorized.
Next: FIX_PROVEN_FULL_RUN_BUG
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A batched kernel stability search against batch-1 reference
Date: 2026-09-05
Authorization: one small A10 diagnostic only. Reuse the 64 evenly-spaced ordered F1-F4 True-S2 q-doc worklist groups (192 chunks), score batch1 reference and batch16 attention-backend alternatives; no labels, Fold0, public data, full inference, or canonical-runner change.
Status: PREPARING_SINGLE_STABILITY_DIAGNOSTIC
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A batched kernel stability search outcome
Date: 2026-09-05
Lifecycle: exactly one detached A10 diagnostic, App ap-ntG92jiX4R5VrlnS2dLsL7, FunctionCall fc-01M1Q39Y57M9ZWJ7JJN10N9S9K; no retry.
Sample/reference: 64 evenly spaced ordered F1-F4 True-S2 q-doc groups, 192 immutable chunk keys; default B1 reference PASS. No labels, Fold0, public data, production checkpoint/predictions, or runner modification.
Installed supported paths observed: default attention=sdpa; model supports SDPA; torch SDPA API exposes MATH (also CUDNN/EFFICIENT/FLASH/OVERRIDEABLE). Tested only default, SDPA-MATH-only, and model-accepted eager; FlashAttention was not tested.
CURRENT_DEFAULT_B16: max/median/p95 B1 delta 0.02734375/0.0078125/0.021484375; >1e-4/>1e-3/>1e-2=172/170/52; Spearman 0.9971302372469503; chunk/doc ordering false/true; neighbor spread 0.029296875; 39.94432168489901 chunks/s; 4.333388328552246 GiB; gate FAIL.
SDPA_MATH_ONLY_B16: 0.02734375/0.0078125/0.01953125; 171/168/62; Spearman 0.9977533691114752; chunk/doc false/true; neighbor 0.017578125; 29.990661353228464 chunks/s; 4.824692249298096 GiB; gate FAIL.
EAGER_B16: 0.02734375/0.00634765625/0.01953125; 172/170/56; Spearman 0.9974418037060654; chunk/doc false/true; neighbor 0.0234375; 34.54781279747685 chunks/s; 4.65053129196167 GiB; gate FAIL.
Decision: NO_STABLE_BATCHED_CONFIGURATION. Math-only reduced neighbor spread but remained above the 0.005 stability gate. Temporary diagnostic script removed.
Next: TEST_BATCH1_EFFICIENT_MICROBATCH_STRATEGY
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A micro-batch stability selection
Date: 2026-09-05
Authorization: one small A10 diagnostic using the exact prior 64 evenly-spaced ordered F1-F4 True-S2 q-doc groups and 192 chunks. B1/B2/B4/B8 only; no B16, backend changes, labels, Fold0, public data, production checkpoint, or canonical-runner change.
Status: PREPARING_SINGLE_MICROBATCH_DIAGNOSTIC
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A micro-batch stability selection outcome
Date: 2026-09-05
Lifecycle: exactly one detached A10 diagnostic, App ap-ld1nTULMlhIqkfBJkf0Sqi, FunctionCall fc-01M1Q3X87SQS7KY55VJBS7XPNC; no B16 or retry.
Sample/reference: same 64/64 q-doc groups and 192 keyed chunks; B1 throughput/VRAM 37.5939056406784 chunks/s and 3.993192195892334 GiB.
B2/B4/B8: all q-doc order parity true but chunk ordering false and strict/practical gates fail. Max/p95/neighbor deltas respectively B2 0.029296875/0.01953125/0.01953125, B4 0.02734375/0.01953125/0.01953125, B8 0.02734375/0.015625/0.01953125.
Selection: BATCH1_ONLY_SAFE. Temporary diagnostic script removed; no labels/Fold0/public data, production checkpoint/prediction, full inference, or runner change.
Next: TRACE_REMAINING_FULL_RUN_DIVERGENCE
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A canonical B1 100-query post-score trace
Date: 2026-09-05
Authorization/lifecycle: CPU-only, read-only diagnosis of existing isolated replay artifacts; temporary local downloads and audit script classified TEMPORARY_DELETE. No Modal call, GPU inference, runner/full-run/frozen-artifact change, Fold0, or public labels.
Artifacts inspected: Modal Volume `udsc-p13:/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-100q-replay/{predictions.jsonl,run_state.json}`; 100 queries, 7,700 document rows, 23,093 chunks, prediction SHA256 `879f66168798b2ad59170120cc05489d7f1ac7e8c550f6946b5521d4d7a52e66`.
Result: canonical frozen replay predictions equal descending MAX(document chunk score) top-5 exactly for 100/100 queries (bottom-5 0/100); reconstructed Recall/Precision `0.79/0.164`, AUC `0.9237562524511432`. Direct source trace proves the evaluator instead used unsorted `grouped[query][:5]` after `work.sort(query, doc, chunk)`, reproducing exactly Recall/Precision `0.03/0.006`.
Classification: EVALUATOR_INPUT_MISMATCH; post-score prediction construction is correct. Next: PATCH_PROVEN_POST_SCORE_BUG.
FILE LIFECYCLE REVIEW: UPDATE_IN_PLACE both TV2 ledgers; delete three temporary local audit/download files after logging; no consumer impact.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A canonical batch-1 100-query replay outcome
Date: 2026-09-05
App/FunctionCall: ap-8dcyGpyl9Id6peRKzpqC72 / fc-01M1Q4TAG2NWWA2ZNVQA5XVM00
Runner: scripts/modal/task1_b2a_qwen3vl2b.py; canonical effective model-forward batch patched to 1; isolated replay namespace only.
Runtime: requested NVIDIA A10; actual NVIDIA A10G; 100 queries, 7,700 q-doc pairs, 23,093 chunks; missing/duplicate/checkpoint errors 0.
Result: PASS infrastructure/key coverage but PRODUCTION_B1_FIX_NOT_CONFIRMED: Recall 0.03, Precision 0.006000000000000001, AUC 0.9237562524511432. Elapsed 714.0007578279999 seconds; throughput 32.34310292645798 chunks/sec; peak VRAM 4.006190776824951 GiB.
Replay prediction SHA256: 879f66168798b2ad59170120cc05489d7f1ac7e8c550f6946b5521d4d7a52e66; historical frozen prediction SHA unchanged. No Fold0/public labels/full inference. User supplied terminal result; no rerun.
Next: TRACE_REMAINING_FULL_RUN_DIVERGENCE
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A replay100 evaluator input bug fix and artifact re-evaluation
Date: 2026-09-05
Authorization/lifecycle: narrow source patch plus CPU-only evaluation of the existing immutable replay artifact. No Modal FunctionCall, GPU inference, score/model/prediction/frozen-artifact change, full F1-F4 run, Fold0, or public labels.
Scope proof: the faulty `grouped[query][:5]` occurred only in `scripts/modal/task1_b2a_qwen3vl2b.py::replay100_batch1_remote`, reached only through `replay100()` / `main(mode="replay100")`; `run_remote()` persists predictions with `metrics=NOT_RUN` and has no call to this evaluator.
Patch: the replay metric path now applies the same descending `(score, canonical_k77_rank, doc_id)` ordering as materialized `predicted_doc_ids` before selecting top-5. Formulae, AUC, scores, aggregation, candidate pool, and output ordering are unchanged.
Re-evaluation: downloaded existing Volume artifact SHA256 `879f66168798b2ad59170120cc05489d7f1ac7e8c550f6946b5521d4d7a52e66`; 100 queries/7,700 pairs/23,093 chunks; 100/100 top-5 parity, missing/duplicate IDs 0; Recall `0.79`, Precision `0.16400000000000003`, AUC `0.9237562524511432`.
Gate: PRODUCTION_B1_FIX_CONFIRMED for this bounded replay only. Historical full 2B and Workflow-A were not called through this evaluator; fresh recovery evaluator provenance not proven. Next: RUN_LARGER_B1_VALIDATION.
FILE LIFECYCLE REVIEW: UPDATE_IN_PLACE runner and two ledgers; downloaded replay JSONL and local re-evaluator are TEMPORARY_DELETE; no downstream artifact consumer changed.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A disjoint 400-query F1-F4 batch-1 validation
Date: 2026-09-05
Authorization: exactly one new bounded A10 validation before any possible full run; no full 5,600-query execution, no batch above 1, Fold0, public labels, retrieval/True-S2/model/prompt/metric change, or historical artifact overwrite.
Preparation: added isolated `validation400_batch1_remote` / `validation400` mode to the canonical runner. It reconstructs the prior 100 replay IDs by their documented 100 evenly-spaced positions, deterministically selects 100 evenly spread non-overlapping queries in each F1-F4, and refuses an existing output namespace.
Contract: unchanged Qwen3-VL 2B revision `4bd860ac4f15ad1897a214615cccc700f8f71818`, A10, model.eval/inference_mode, effective forward batch 1, K77, True-S2, MAX chunk-to-document score, and fixed score-sorted top-5 evaluator. Pre-forward checks will print F1-F4=100 each, total=400, overlap=0; expected 30,800 q-doc pairs.
Status: READY_FOR_SINGLE_VALIDATION_SUBMISSION.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A disjoint 400-query F1-F4 batch-1 validation outcome
Date: 2026-09-05
Lifecycle: first non-detached local entrypoint call `fc-01M1Q9VNCHT5CK2BRBNQGH4FQ0` stopped before container/GPU allocation (no remote logs); one detached validation FunctionCall `fc-01M1QA0ZSMSVBP14E71J2NCMEQ` then completed on App `ap-ZuWwUUJx080xdsOfXciywe`. No full inference or retry after a GPU-backed call.
Contract/integrity: NVIDIA A10; effective forward batch 1; immutable revision `4bd860ac4f15ad1897a214615cccc700f8f71818`; F1/F2/F3/F4=100/100/100/100; total 400; overlap with replay100=0; 30,800 q-doc pairs; 92,380 chunks; missing/duplicate query IDs 0; 5 predictions/query; score-sorted prediction/evaluation top-5 consistency 400/400. Fold0/public labels/historical frozen predictions were not used/changed.
Pooled: Recall `0.7820833333333332`, Precision `0.169`, AUC `0.9139076992763887`; elapsed `2233.8103294459997` s, `41.355346415159104` chunks/s, peak `4.019114971160889` GiB. F1 `0.835/0.18/0.9312806121453382`; F2 `0.785/0.17/0.9147000497599813`; F3 `0.6983333333333333/0.15200000000000002/0.9038866644348269`; F4 `0.81/0.17400000000000002/0.9068348017406237` (Recall/Precision/AUC).
Gate: B1_LARGER_VALIDATION_PASS. Artifact: `udsc-p13:/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-400q-disjoint-validation/{predictions.jsonl,run_state.json}`, prediction SHA256 `011fa8859cffe90ccc84e96b90c8bce425860af375449dae1d0294b5ae62aeb3`. Next: RUN_FULL_F1_F4_BATCH1.
FILE LIFECYCLE REVIEW: UPDATE_IN_PLACE runner and ledgers; no local scientific artifact retained; remove syntax-check bytecode if present; only the new isolated Volume namespace is consumed by downstream validation review.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A full F1-F4 batch-1 no-GPU pre-flight
Date: 2026-09-05
Lifecycle: CPU/config and read-only Modal Volume checks only; no GPU FunctionCall, full run, artifact write, Fold0, or public labels.
Workload/storage PASS: local canonical worklist SHA256 `a935e356cf8fe62a000e56c3817fd31dfc5ebbc079c4eef849803a7552c3c25a`; runner constants declare F1-F4 `5600` queries, `431200` q-doc pairs, `1293198` chunks, `77` docs/query, batch `1`, fixed 2B revision/prompt/token scoring/MAX aggregation. Active Modal profile/workspace `hoconlinea10`; workspace default environment `main`; Volume `udsc-p13` contains the worklist and chunk files.
BLOCKER: `run_remote()` still binds full mode to `OUTPUT_ROOT=/workspace/p13/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b` and `CHECKPOINT_DB=.../chunk_scores.sqlite3`. That historical namespace exists remotely with checkpoint, WAL/SHM, predictions, and run_state, so full mode would inspect/resume it. Proposed clean namespace is absent: `/workspace/p13/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-full-corrected` (checkpoint `chunk_scores.sqlite3`, predictions `predictions.jsonl`, status `run_state.json`).
Status: BLOCKED — do not launch until a separately authorized runner namespace/provenance patch isolates full Batch1. Future command after that patch: `.\\.venv\\Scripts\\modal.exe run --detach --profile hoconlinea10 --env main scripts\\modal\\task1_b2a_qwen3vl2b.py::main --mode full`.
FILE LIFECYCLE REVIEW: UPDATE_IN_PLACE ledgers only; no new namespace/artifact created; historical and frozen artifacts untouched.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A full-mode checkpoint routing trace
Date: 2026-09-05
Lifecycle: strict read-only source and Volume trace; no code/Volume/checkpoint mutation, GPU call, full run, Fold0, or public labels.
Call chain: `main(mode="full")` line 1506 → `run.spawn()` line 1509 → `run()` line 1419 → `run_remote()` line 686 → `inspect_checkpoint()` line 691 (read-only SQLite URI line 399) → `init_db(CHECKPOINT_DB)` line 711 → `sqlite3.connect(path)` line 384. There are no full-mode arguments, environment-path reads, resume flags, fallback paths, or checkpoint auto-discovery helpers.
Resolution: module lines 33–38 set `OUTPUT_ROOT=/workspace/p13/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b`, `CHECKPOINT_DB=OUTPUT_ROOT/chunk_scores.sqlite3`, prediction `OUTPUT_ROOT/predictions.jsonl`, status `OUTPUT_ROOT/run_state.json`; full mode passes that exact checkpoint unchanged to SQLite. Volume confirms the same namespace contains checkpoint/WAL/SHM/predictions/run_state. Thus actual full and historical Batch16 checkpoint/namespace are literally identical.
Corrected candidate namespace `.../qwen3-vl-reranker-2b-batch1-full-corrected` is absent on `udsc-p13`, so no corrected SQLite exists (size/rows/provenance unavailable). The prior `Historical checkpoint reused: YES` was a preflight conclusion from this literal equality, not a runner boolean/expression. Classification: FULL_MODE_HARDCODED_HISTORICAL_PATH. Next: PATCH_EXACT_CHECKPOINT_ROUTING_BUG.
FILE LIFECYCLE REVIEW: UPDATE_IN_PLACE ledgers only; no temporary files or artifact changes.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A full Batch1 corrected checkpoint routing patch and CPU/config pre-flight
Date: 2026-09-05
Authorization/lifecycle: one routing-only runner edit plus CPU/config and read-only storage verification; no Modal FunctionCall, GPU inference, full F1-F4, Fold0, public labels, or historical artifact mutation.
Patch: `scripts/modal/task1_b2a_qwen3vl2b.py` updates shared full-mode `OUTPUT_ROOT` to `.../qwen3-vl-reranker-2b-batch1-full-corrected`. Thus `CHECKPOINT_DB`, predictions, errors, and run state resolve only inside the corrected namespace; replay100 and validation400 retain their explicitly separate local namespaces.
Pre-flight: py_compile PASS. Corrected namespace/checkpoint is absent on `udsc-p13` before launch, while historical `.../qwen3-vl-reranker-2b` remains untouched. Contract/storage PASS: 5,600 queries, 431,200 q-doc pairs, 1,293,198 chunks, worklist SHA `a935e356cf8fe62a000e56c3817fd31dfc5ebbc079c4eef849803a7552c3c25a`, effective batch 1, profile `hoconlinea10`, env `main`.
Status: READY_TO_LAUNCH. Next: RUN_FULL_F1_F4_BATCH1.
FILE LIFECYCLE REVIEW: UPDATE_IN_PLACE canonical runner and both ledgers; runner remains KEEP_ACTIVE, consumed by the future full launch; no new local/Volume artifact, cleanup, or historical consumer impact.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A full Batch1 SQLite hot-path optimization
Date: 2026-09-07
Authorization/lifecycle: canonical runner persistence/progress-only patch; no Modal call, GPU inference, checkpoint/Volume mutation, model-forward batch, scientific-contract, Fold0, or public-label change.
Patch: full mode retains sequential batch-1 scoring but buffers scored rows and commits at 256 q-doc pairs or 1,024 chunk rows, then final flushes. Startup counts remain authoritative; hot-loop full-table COUNT/GROUP BY was removed. In-memory counters advance from committed rows/pairs. Logs retain `throughput` and add `cumulative_new_chunks_per_sec` and 60-second `rolling_new_chunks_per_sec`.
Validation: CPU SQLite old-vs-buffered row/key/value/count equivalence PASS; interruption simulation confirms committed rows skip, uncommitted rows rescore, and final result is identical (duplicates/missing 0). py_compile and git diff --check PASS. Existing schema/WAL/FULL durability and checkpoint resume format unchanged.
Status: PASS_SQLITE_OPTIMIZATION. Next: RESUME_FULL_BATCH1_FROM_EXISTING_CHECKPOINT.
FILE LIFECYCLE REVIEW: UPDATE_IN_PLACE canonical runner and both ledgers; runner is KEEP_ACTIVE and directly consumed by full resume; test databases used TemporaryDirectory and were removed; no historical or current Volume artifact changed.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Qwen3-VL-2B Batch1 full F1-F4 frozen scientific evaluation
Date: 2026-09-07
Evaluation-only: SHA256 verified frozen prediction `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl` = `556af4f7d83484c5fdabced049ea98e983923038180346b7a7e59b939b7c3cb3`. Canonical set evaluator `src/udsc2026/evaluation/legal_ir_recovery.py::metrics`, target-only fold reader, and target-only train reader evaluated 5,600 F1-F4 queries.
Result: pooled Recall `0.05327380952380952`, Precision `0.012071428571428573`; folds Recall F1=`0.05648809523809523`, F2=`0.050833333333333335`, F3=`0.05125`, F4=`0.05452380952380952`. Workflow-A pooled comparator Recall/Precision `0.9259285714285714/0.19739285714285715`; deltas `-0.8726547619047619/-0.18532142857142858`; nonnegative folds `0/4`.
Decision: REJECT_B2A_MATERIALITY (all three fixed gates fail). Integrity: missing/duplicate/unknown predictions `0/0/0`; Fold0 NO; public labels NO; GPU/Modal/model inference NO; frozen predictions unchanged. Report: `reports/task1/workflow_b/tv2/b2a/reports/b2a_qwen3vl2b_batch1_full_scientific_evaluation.json`.
FILE LIFECYCLE REVIEW: VERSIONED_NEW_FILE canonical B2a evaluation report plus UPDATE_IN_PLACE ledgers; report is consumed by professor review; no temporary artifact or frozen input changed. Next: SEND_FROZEN_FULL_EVAL_TO_PROFESSOR_REVIEW.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A frozen JSON top-5 reconstruction diagnostic
Date: 2026-09-07
Read-only/CPU-only: verified frozen predictions SHA256 `556af4f7d83484c5fdabced049ea98e983923038180346b7a7e59b939b7c3cb3`; did not read or use the failed local SQLite checkpoint. JSON has exactly `query_id`, `predicted_doc_ids`, and `document_scores`: 5,600 records, 431,200 numeric document scores, exactly 77/query, no missing score list.
Trace/result: `run_remote()` lines 807-816 aggregates `MAX(score)` by `(query,doc)`, sorts `(-score, canonical_k77_rank, doc_id)`, and serializes both top-5 and full ordered document scores. Frozen and independently reconstructed descending top-5 match exactly in set/order for 5,600/5,600; overlap 5.0/5; frozen equals both first five serialized score records and descending top-5 because document_scores are serialized score-sorted.
Diagnostic evaluation is unchanged: pooled Recall/Precision `0.05327380952380952/0.012071428571428573`; classification FROZEN_DOCUMENT_SCORE_FAILURE. No GPU/Modal/inference/Fold0/public labels or frozen-artifact mutation. Next: SEND_CONFIRMED_FAILURE_TO_PROFESSOR.
FILE LIFECYCLE REVIEW: UPDATE_IN_PLACE ledgers only; no new scientific artifact, temporary file, frozen prediction, or checkpoint changed.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A Batch1 bounded-400Q versus full score divergence trace
Date: 2026-09-07
Read-only/CPU-only: frozen full SHA verified. Exact 400Q evidence is the completed Volume artifact `udsc-p13:/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-400q-disjoint-validation/{predictions.jsonl,run_state.json}` recorded with matching 400Q metrics and revision in the TV2 ledger; it is not local, so its top-5/document scores cannot be joined without download.
Same-query diagnostic: reconstructed the deterministic 400 IDs from `validation400_batch1_remote` and canonical local worklist (100 per F1-F4, 30,800 q-doc). Full frozen coverage is 400/400 but on those exact queries Recall/Precision is `0.04166666666666667/0.009500000000000001`, versus bounded `0.7820833333333332/0.169`.
Static trace: both paths call the same `download_locked_snapshot`, `load_model`, and `score_batch` contract (revision, bf16/eval, prompt/tokenization, batch 1, yes/no vector, sigmoid, MAX doc aggregation). Bounded path is independent in-memory/sorted work; full uses sequential worklist plus checkpoint/resume SQLite. Same-item scores and resume-key membership cannot be proven because 400Q scores and a valid full checkpoint are unavailable locally. Classification: UNRESOLVED. Next: RECOVER_400Q_DOCUMENT_SCORES_AND_FINAL_CHECKPOINT_PROVENANCE.
FILE LIFECYCLE REVIEW: UPDATE_IN_PLACE ledgers only; no frozen artifact, checkpoint, GPU/Modal run, Fold0, or public label use.
=== PROGRESS_LOG_ENTRY END ===

=== PROGRESS_LOG_ENTRY START ===
Task: B2A same-item 400Q versus valid full checkpoint score/provenance trace
Date: 2026-09-07
Inputs PASS: frozen prediction SHA and worklist SHA verified; `chunk_scores_FULL.sqlite3` read-only integrity `ok`, 1,293,198 rows, 431,200 q-doc pairs, duplicate PK 0. Exact nested 400Q `predictions.jsonl`/`run_state.json` found; run state exactly matches known PASS metrics, revision, batch 1, provenance, and worklist SHA. No chunk-score artifact is present in that directory.
Same-item result: all 30,800 q-doc scores joined (missing 0/0). Absolute score delta max/median/p95/mean `0.57421875/0.10546875/0.322265625/0.12786021839488637`; Pearson/Spearman `0.11502696061016578/0.0810068890816878`. Good-vs-full reconstructed top5 set/ordered matches `0/400`, mean overlap `0.3275/5`; full frozen equals final SQLite top5 `400/400`, and full same-400 Recall/Precision remains `0.04166666666666667/0.009500000000000001`.
Row-level pre/post-resume origin is absent from schema, so count 693322 cannot prove prefix membership. Classification: FULL_VS_400Q_SCORE_DIVERGENCE_UNEXPLAINED. Next: FORENSIC_COMPARE_PRE_RESUME_SOURCE_CHECKPOINT_PROVENANCE.
FILE LIFECYCLE REVIEW: UPDATE_IN_PLACE ledgers only; no SQLite, frozen output, GPU/Modal, Fold0, or public-label modification.
=== PROGRESS_LOG_ENTRY END ===
