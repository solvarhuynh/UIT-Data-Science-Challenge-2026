------------------------------------------------------------
Status: CLOSED_HISTORICAL

Scientific lineage retained for provenance.

This document is NOT the current Workflow-B execution plan.

This document is NOT the deployment incumbent.
------------------------------------------------------------

# UDSC2026 Task1 LegalIR — Workflow A (Current)

**Phạm vi:** forensic audit Task1 LegalIR, dùng official macro Recall. CPU-only.
Mỗi step độc lập; không tự động chạy step kế tiếp.

## Research constitution

- Chỉ dùng validation queries folds `[1, 2, 3, 4]`.
- Không đọc/materialize/sử dụng Fold0 labels/data hoặc public labels/gold.
- Không sửa scorer, production artifacts, model weights, baseline hay candidate artifacts.
- Không dùng MRR, Recall@3, leaderboard score, raw model score hoặc confidence làm ground-truth utility.
- Không train, calibration, GPU, Modal, Beam job, Fold0 evaluation hoặc public inference trừ khi một prompt được duyệt yêu cầu rõ.
- Official Recall: `|set(gold) ∩ set(prediction)| / |set(gold)|`.

## Current workflow

```text
STEP 0 — Metric contract
    ↓
STEP 1A — Recall-gap / candidate-repair ceiling
    ↓
STEP 1B — One-swap oracle gap decomposition
    ↓
STEP 1C — Action-space filter-loss attribution (diagnostic only)
    ↓
STEP 4 — Consolidation / professor-approved downstream decision
```

Step 2 OOF Query-Conditioned Calibration cũ được loại khỏi active Workflow A,
defer sang Workflow B hoặc post-forensic modeling stage. Step 3 Candidate Pool
Noise Forensic cũ được repurpose thành Step 1C.

## STEP 0 — Metric + Output-Capacity Contract Audit

Canonical status: **PASS** (`reports/task1/step0_metric_contract_report.json`).
Primary metric là `macro_recall`; ranking order không ảnh hưởng official score.
Có 5.600 query folds 1–4; phân phối gold là 1/2/3/4/5 document tương ứng
5.162/386/43/8/1. Multi-gold = 438 (`0.07821428571428571`), gold >5 = 0,
nên capacity ceiling là 1.0. Fold0 và public labels không được sử dụng.

## STEP 1A — Recall-gap / Candidate-repair Ceiling

Canonical status: **PASS** (`reports/task1/exp_1a_recall_gap_report.json`).
Audit 5.600 query, one-swap positive 463 query, missing gold 573, trong đó
507 document có trong full pool. One-swap ceiling gain là
`0.06381547619047619`; unconstrained top-5 repair ceiling là
`0.06687797619047618`.

## STEP 1B — One-swap Oracle Gap Decomposition

Canonical status: **FAIL** cho rank-limit gate
(`reports/task1/exp_1b_oracle_gap_decomposition_report.json`).

- `pooled_A = 0.06381547619047619`
- `pooled_B = 0.04177083333333334`
- `pooled_C = 0.04177083333333334`
- action-space gap `A-B = 0.02204464285714285`
- rank-limit gap `B-C = 0.0`; positive queries = 0
- action-space-gap queries = 162; per fold F1/F2/F3/F4 = 40/39/41/42

Kết luận giáo sư được áp dụng: **OPEN RANK1–3 BRANCH = REJECTED**. Trọng tâm
là candidate/action-space coverage. Historical V3A D không serialized cho folds
1–4 và không cần cho quyết định Workflow A; không retrain để tái tạo D.

## STEP 1C — Action-Space Filter Loss Attribution Forensic

Step 1C là **DIAGNOSTIC ONLY — NO BRANCH GATE**. Mục tiêu là phân rã measured
action-space gap `0.02204464285714285` theo các deterministic filter thực sự
nằm giữa full pre-shortlist pool và V3A incoming action-space.

Endpoints:

- Full pool: `artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl`
- Baseline: `artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl`
- Current action-space: `artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl`

Phương pháp: structural target-only reader cho folds 1–4; reconstruct từng
stage deterministic; tại mỗi stage tính exact one-swap Recall ceiling với drop
rank 1–5; stage loss là chênh lệch giữa hai ceiling liên tiếp. Final stage phải
khớp current incoming set từ actions.jsonl cho 5.600/5.600 query; phải kiểm tra
oracle non-increasing, per-query telescoping và per-fold accounting.

Không đưa operation trước `candidate_refs_full.jsonl` vào attribution và không
trộn rank4/5 restriction vào Step 1C (đã đo ở Step 1B, gap = 0).

Outputs bắt buộc:

- `reports/task1/exp_1c_action_space_filter_loss_attribution_report.json`
- `reports/task1/exp_1c_action_space_filter_loss_trace.jsonl`
- `scripts/analysis/exp_1c_action_space_filter_loss_attribution.py`

`dominant_stage_share` chỉ là descriptive: trên 0.80 thì
`DOMINANT_SINGLE_FILTER_STAGE`, ngược lại `DISTRIBUTED_ACROSS_MULTIPLE_STAGES`.
Không dùng kết quả này để tự sửa filter, tăng shortlist, mở rank1–3 hay chạy
workflow khác.

**Kết quả canonical:** `PASS` kỹ thuật; branch gate vẫn là diagnostic-only.

- Pipeline A→B có một deterministic loss stage: `union_rank <= 20` shortlist
  cutoff (`build_shortlist_features.py:40-53`).
- Stage này giảm incoming candidates từ 1.092.015 xuống 86.169 và giải thích
  toàn bộ action-space gap: pooled Recall loss `0.02204464285714285` (share 1.0).
- Endpoint checks: A `0.06381547619047619`, B `0.04177083333333334`; final
  reconstructed incoming set khớp canonical actions cho 5.600/5.600 query.
- Affected queries: 162, theo fold 40/39/41/42; tất cả có first loss stage là
  `union_rank <= 20`.
- Professor direction: `DOMINANT_SINGLE_FILTER_STAGE`.
- Heterogeneity: descriptive only, không có threshold mới.

**Report:** `reports/task1/exp_1c_action_space_filter_loss_attribution_report.json`.
**Trace:** `reports/task1/exp_1c_action_space_filter_loss_trace.jsonl`.

## STEP 4 — Consolidation

Chỉ đọc các report đã tồn tại và viết một quyết định nghiên cứu duy nhất.
Không cộng các oracle ceiling như các production gain độc lập; file thiếu phải
ghi rõ chưa có dữ liệu.

## STEP 1D — Union-Rank Cutoff Expansion Oracle Sweep

Canonical status: **PASS — DIAGNOSTIC COMPLETE**
(`reports/task1/exp_1d_union_rank_sweep_report.json`). Branch gate vẫn là
**DIAGNOSTIC ONLY — NO BRANCH GATE**; Step 1D không chọn production K.

K được derive bằng nearest-rank percentiles 10/25/50/75/90/100 của 169
lost-beneficial candidates (union-rank 21–197), cộng anchor K=20. Các K đo:
`[20, 23, 29, 45, 77, 135, 197]`.

- K=20: oracle gain `0.04177083333333333`, 86.169 incoming candidates,
  172.338 actions.
- K=23/29/45/77/135/197: recovery fraction of A-B lần lượt
  `0.1228567571` / `0.2875658161` / `0.5474551100` / `0.7718374511` /
  `0.9189955448` / `1.0`.
- K=197 đạt oracle full-pool A `0.06381547619047619`, với 1.075.215 incoming
  candidates và 2.150.430 actions; đây là descriptive sweep result, không phải
  production recommendation.

K=20 và per-fold cross-check đều khớp; incoming sets và oracle gains monotonic.
Current rank1–3 decision dưới K=20 vẫn **REJECTED**. Step 1D sweep không tự mở
lại nhánh này và không rerun Step 1B. Chỉ sau khi một K cụ thể được review và
adopt làm action-space thực tế mới rerun Step 1B.

Calibration remains removed/deferred. Chỉ xem xét lại sau khi production K được
adopt và Step 1B được rerun ở K đó. Workflow B không được vào trực tiếp từ
Step 1D; không training/fine-tuning trước downstream review.

## STEP 1E — Action-Space Noise/Harm Composition Sweep

Canonical status: **PASS — DIAGNOSTIC COMPLETE**
(`reports/task1/exp_1e_action_space_noise_report.json`). Branch gate vẫn là
**DIAGNOSTIC ONLY — NO BRANCH GATE**. K grid: `[20, 23, 29, 45, 77, 135, 197]`.

Step 1E tách structural candidate composition khỏi action utility: non-gold
candidate không phải policy harm; harmful action chỉ là swap rank4/5 làm giảm
official Recall nếu action đó được thực hiện. Không policy nào được evaluate.

At K197 versus K20: thêm 989.046 candidates (187 gold, 988.859 non-gold;
noise rate `0.9998109289`) và 1.978.092 actions (363 beneficial, 1.924.231
neutral, 53.498 harmful; harmful-action rate `0.0270452537`). Các rate là
descriptive, không phải monotonic invariant.

Later review score được pre-register, không phải production selector:

```text
benefit_score(K) = Step 1D fraction_of_action_space_gap_recovered(K)
noise_score(K) = Step 1E cumulative candidate_noise_rate(K)
lambda = 1.0
advisory_joint_score(K) = benefit_score(K) - noise_score(K)
```

Không tự adopt production K. Future order: professor review → adopt K nếu được
duyệt → rerun Step 1B tại K đã adopt → future Step 2 OOF
Policy-Realizability Test. Step 2 chưa active cho đến khi đủ ba điều kiện.
Rank1–3 vẫn rejected tại K20; C-D vẫn unresolved. Workflow B, reranker
fine-tuning và query-conditioned K vẫn out of scope pending review.

## STEP 1F — Action-Space Tractability / K-Selection Protocol

Canonical status: **PASS** (`reports/task1/exp_1f_k_selection_protocol_report.json`).
Scientific status: **EXPLORATORY PROTOCOL SELECTION**, not confirmatory
preregistration: the rule was designed after Step 1D/1E evidence existed but
frozen before its mechanical application.

The old `benefit_score - candidate_noise_rate` score remains a historical
diagnostic and is **rejected for K selection**. Step 1F instead uses:

```text
benefit = Step 1D fraction_of_action_space_gap_recovered
cost = Step 1E actions_per_beneficial_action
balance_score = minmax(benefit) - minmax(cost)
tie-break = smallest K
```

Pooled selected K is **77**. LOFO selections for held-out F1/F2/F3/F4 are
`77/77/77/77`: exact match = 4, neighbor match = 4, so the frozen stability
contract passes. Decision: **ADOPT_RESEARCH_K**, with
`adopted_research_K = 77`; this is explicitly not a production/deployment K.

At adopted research K=77, cumulative harmful action rate is `0.0270342705`;
the selected shell (45→77) has 4,161.6512 actions per beneficial action and
harmful rate `0.0270520981`. These are descriptive guardrails with no new
numerical threshold.

### Downstream state

The next required step is exactly one **Step 1B rerun at adopted research K=77**.
It is not run by Step 1F. Rank1–3 status is now
`REOPEN_PENDING_STEP1B_RERUN_AT_ADOPTED_K`; no claim is made for K77 before
that rerun. Policy C-D remains unresolved.

Future Step 2 — OOF Policy-Realizability Test is gated and not active until:

1. adopted research K exists; and
2. Step 1B rerun has completed at exactly that K.

Step 2 must test this one fixed K, not sweep K values. Production K remains
unselected. Step 4 stays deferred until the Step 1B rerun and future Step 2
are resolved.

## STEP 1B-R77 — Oracle Gap Decomposition at Adopted Research K=77

Canonical status: **PASS** technical execution
(`reports/task1/exp_1b_rerun_k77_oracle_gap_decomposition_report.json`).
K=77 is an adopted research configuration, not a production K.

- A77 = `0.06381547619047619`
- B77 = `0.058785714285714274`
- C77 = `0.058785714285714274`
- Remaining A77−B77 action-space gap = `0.005029761904761912`
- B77−C77 rank-limit gap = `0.0`; positive rank-limit queries = 0
- Action-space gap queries = 37; B77/C77 positive queries = 426
- K77 incoming candidates/actions = 403.322 / 806.644

Per-fold B77 values are F1/F2/F3/F4 = `0.04922619047619048` /
`0.060416666666666674` / `0.06107142857142857` /
`0.06442857142857143`; every fold has rank-limit gap 0.

Rank-limit gate: **FAIL**. Therefore rank1–3 is **REJECTED AT K77** and no
rank1–3 branch opens. D77/C-D remains unresolved because no historical K77
policy exists and no policy was trained/replayed.

Future Step 2 — OOF Policy-Realizability Test is now **READY FOR PROFESSOR
SPECIFICATION**, but is not run by this task. Production K remains unselected;
Workflow B, fine-tuning, query-conditioned K and Step 4 remain deferred.

## STEP 2 — Strict-OOF Policy Realizability Test at Fixed Research K=77

### Attempt #1 — permanent audit artifact

Technical status: **CONTRACT_ERROR**
(`reports/task1/step2_k77_oof_policy_realizability_report.json`). Its frozen
feature schema duplicated `dropped_baseline_rank` and
`incoming_is_baseline_top5`; its scientific status is
**INVALID_FOR_SCIENTIFIC_INFERENCE_AUDIT_ONLY**. D77 and C77-D77 from that
attempt remain unresolved. Fold0/public audit passed. The original contract,
OOF decision file, and report are preserved permanently and are not used as
inputs or evidence for the rerun.

### STEP 2-R1 — contract v2

Technical status: **PASS**
(`reports/task1/step2_k77_v2_oof_policy_realizability_report.json`). The sole
repair was `ordered_unique` with first-occurrence order preservation: 38
parent entries became 36 unique v2 features, removing exactly
`dropped_baseline_rank` and `incoming_is_baseline_top5`. All model settings,
OOF protocol, threshold rule, K=77 and ranks 4/5 remained unchanged.

- Baseline/policy macro Recall = `0.9259285714285714` / `0.9259285714285714`.
  D77 = `0.0`; C77-D77 = `0.058785714285714274`; realization ratio = `0.0`.
- Baseline/policy macro Precision = `0.19739285714285715` /
  `0.19739285714285715`; precision delta = `0.0`; guard-rail passes.
- Per-fold D77: F1 `0.00035714285714283367`, F2 `0.0`, F3
  `-0.00035714285714283367`, F4 `0.0`.
- Selected/no-op = 362/5,238; selected beneficial/neutral/harmful = 7/348/7.
  V2 decisions are serialized in
  `reports/task1/step2_k77_v2_oof_policy_decisions.jsonl`.
- Bootstrap remains `NOT_RUN_NO_EXISTING_COMPATIBLE_PROTOCOL`.
- No compatible Task1 query-level bootstrap protocol was found in the checked
  source/tests/docs, so bootstrap is `NOT_RUN_NO_EXISTING_COMPATIBLE_PROTOCOL`.

The valid v2 scientific gate is **FAIL**: pooled D77 is not positive and Fold3
is negative, though the precision guard-rail holds. Next state is
**STEP2_VALID_FAIL_PENDING_FAILURE_ANALYSIS_REVIEW**. Rank1--3 remains
**CLOSED AT K77**, production K remains **NOT SELECTED**, and no downstream
work is authorized automatically.

## STEP 2-F — Decision-Level Failure Attribution for Step2-R1

Technical status: **PASS**; scientific status: **DIAGNOSTIC ONLY — NO BRANCH
GATE** (`reports/task1/step2f_failure_attribution_report.json`). It used the
valid v2 OOF decisions plus deterministic target-only reconstruction of the
fixed K77 rank4/5 oracle; no training, model loading/scoring, rescoring, or
threshold retuning was performed.

The exhaustive seven-category partition reconciles all 5,600 queries and the
full C77 headroom of 329.2 sum-units. The dominant category is oracle-positive
plus no-op: 367 queries, 280.3333333333333 headroom sum, or
`0.851559335763467` of total oracle headroom/gap. The remaining
oracle-positive cases are 7 beneficial selections (6.0 realized sum), 52
neutral selections (42.86666666666667 headroom), and 0 harmful selections.
There are 4,871 oracle-zero no-ops, 296 oracle-zero neutral selections, and 7
oracle-zero harmful selections; these harmful actions lose 6.0 sum-units.

Fold2 had threshold `+Infinity`, 0 selected actions and 1,400 no-ops: this
only records that inner-CV selected universal no-op under the frozen protocol.
Fold3's two beneficial and two harmful choices have sums +1.5 and -2.0,
respectively, explaining its negative net. Selected incoming union-ranks were
resolved from canonical candidates and all were <=77.

Threshold and ranking/discrimination attribution both remain
`UNRESOLVED_REQUIRES_STEP2_F2_FULL_ACTION_RESCORING`; Step2-F2 is not
authorized here. K77 is **NOT REJECTED**, rank1--3 stays **CLOSED AT K77**,
and results return to professor review.

## STEP 2-F2 — Full-Action Rescoring Diagnostic

Technical status: **PASS**; scientific status: **DIAGNOSTIC ONLY — NO BRANCH
GATE** (`reports/task1/step2f2_rescoring_diagnostic_report.json`). The exact
frozen v2 reconstruction reproduced all four thresholds, 362/362 executed
action identities, 5,238/5,238 no-op decisions, and 5,600/5,600 top scores
with zero score error. It then serialized all 806,644 held-out K77 actions.

For the 426 oracle-positive queries, T/R/S attribution is:

- S_SUCCESS: 7 queries; headroom 6.0 (`1.8226%`).
- T_THRESHOLD_LOSS: 39 queries; headroom 29.5 (`8.9611%`).
- R_RANKING_DISCRIMINATION_LOSS: 380 queries; headroom 293.7 (`89.2163%`).

Thus most available K77 oracle headroom is ranking/discrimination loss under
the reconstructed frozen policy, while threshold loss is secondary. Fold2
retains selected `+Infinity`; among its 4,188 finite inner candidates, the
best finite threshold was `0.986503082282858` with inner macro Recall gain
`0.0`. No alternate outer threshold was evaluated or applied.

This authorizes no fix: K77 remains **NOT REJECTED**, rank1--3 remains
**CLOSED**, production K is **NOT SELECTED**, and the next state is
**STEP2_F2_COMPLETE_PENDING_PROFESSOR_REVIEW**.

## STEP 2-F3 — Frozen-Feature Discriminative Sufficiency Forensic

Technical status: **PASS**; scientific status: **DIAGNOSTIC ONLY — NO BRANCH
GATE** (`reports/task1/step2f3_feature_sufficiency_report.json`). All 36
frozen inference-safe features were reconstructed deterministically for the
806,644 F2 actions, grouped before outcome statistics, and analyzed without
model refitting, ablation, feature changes, or K/threshold changes.

The primary query-balanced BENEFICIAL-vs-NEUTRAL view contains all 426
oracle-positive queries. The upstream rank/reranker-signal overlay has 28
features and its predeclared qualitative pattern is **SEPARATION_VISIBLE**.
The strongest upstream descriptive feature is `incoming_union_rank` (ROC-AUC
`0.7097026604068858`, PR-AUC `0.7233536170873198`, MI
`0.11959345039371892`); `diff_union_rank` has the strongest upstream MI.
Visible all-depth ordering is present for incoming rank/support signals and
their specified difference features. Query-global and structural families are
near chance descriptively; this is evidence, not a feature-selection outcome.

The R-only (380-query) conditional view and all four depth buckets were
reported. Permutation importance is
`NOT_RUN_NO_PERSISTED_FROZEN_MODEL`: F2 persisted scores but no frozen outer
model objects, and refitting solely for F3 is prohibited. Feature sufficiency,
model-family adequacy, and objective adequacy remain unresolved pending
professor interpretation. Next state: **STEP2_F3_COMPLETE_PENDING_PROFESSOR_REVIEW**;
Workflow B is still not active.

## STEP 2-P1 — Controlled Query-Aware Pairwise Policy Objective Experiment

Technical status: **BLOCKED** at the mandatory full-pair feasibility preflight
(`reports/task1/step2p1_realizability_report.json`). The contract was written
before any fit and the full deterministic population was counted without
sampling: 1,666,645 unordered / 3,333,290 mirrored pair rows across folds;
the largest final outer fit requires 2,529,488 mirrored rows. No query has
fewer than two actions. No model was fitted because the available
CPU-only/non-streaming HGB execution could not establish completion of the
required complete pair construction, repeated nested fits, and exhaustive
ordered-pair scoring without altering the frozen experiment semantics.

No pair sampling, reweighting, model/K/feature/threshold change, or fallback
was used. D77_pairwise is therefore not measured. K77 remains **NOT REJECTED**,
rank1--3 remains **CLOSED**, production K remains **NOT SELECTED**, and this
requires professor review before any resource or execution decision.

## Reopen conditions

Rank1–3 chỉ được xem xét lại nếu upstream candidate/action-space thay đổi, sau
đó phải rerun Step 1B theo một review mới. Historical V3A policy quality trên
folds 1–4 hiện `UNRESOLVED`, không được suy ra từ Fold0 artifact.

## Execution policy

Sau mỗi prompt ghi entry ngắn vào `reports/task1/progress_log.md`. Sau khi hoàn
tất đúng step được yêu cầu, in report, summary và progress entry rồi dừng hoàn
toàn; không tự chạy step khác, training, calibration, Fold0 evaluation hoặc
public inference.

## Canonical active-metric amendment (2026-08-30)

This amendment does not rewrite or alter any historical Workflow-A result.
The historical local scientific Recall/Precision interpretation is consistent
with the locked active contract: primary **set-based macro Recall**, secondary
**set-based macro Precision**, and no more than five document IDs per query.
MRR and Recall@3 language encountered elsewhere is stale and is not a
scientific optimization objective for Task1.

Document rank order inside an unchanged valid top-5 set is irrelevant to
primary Recall. Action evaluation must consequently be interpreted through
top-5 set membership and Recall delta; rank-4/rank-5 changes can matter when
they change membership. Workflow-A remains **CLOSED**. The scientific reference
is its F1-F4 exploratory lineage, distinct from the deployment incumbent
`artifacts/task1/submission.zip` (score metadata 0.9391), which is an
operational regression guard and not a scientific baseline.

Metric semantics are locked for scientific development. Exact active CodaBench
scorer binary/package provenance remains an unresolved deployment requirement;
it does not revise historical results or reopen Workflow-A.

## STEP 2-P1 — Final Result (2026-08-27)

Technical status: **PASS**; scientific status: **DIAGNOSTIC ONLY**. The
controlled pairwise policy experiment completed on folds 1–4 at exploratory
fixed K77 with the frozen 36-feature contract, identical HGB v2 settings,
complete deterministic mirrored pairs, strict outer OOF policy, and nested
inner-OOF threshold selection.

The run produced 5,600 unique query decisions. Baseline macro Recall was
`0.9259285714285714`; pairwise-policy macro Recall was
`0.9252142857142857`; therefore `D77_pairwise = -0.0007142857142857784`.
Precision changed from `0.19739285714285715` to
`0.19717857142857143` (`precision_delta = -0.0002142857142857224`). The
baseline oracle headroom was `C77 = 0.058785714285714274`, with
`realization_ratio = -0.012150668286756865`.

The policy executed 410 queries (6 beneficial, 392 neutral, 12 harmful) and
made 5,190 no-op decisions. The original scientific gate is **FAIL** because
pooled D77 is not positive, not all fold gains are non-negative, and the
precision guardrail does not hold. K77 remains exploratory only; production K
is **NOT SELECTED**, rank 1–3 remains **CLOSED**, and no downstream experiment
was run. Fold0 and public labels were not used.

Detailed artifacts: `reports/task1/step2p1_realizability_report.json`,
`reports/task1/step2p1_oof_policy_decisions.jsonl`, and
`reports/task1/step2p1_model_contract.json`.

## STEP 2-P1F — Pairwise Policy Failure Attribution (2026-08-27)

Technical status: **BLOCKED**; scientific status: **DIAGNOSTIC ONLY — NO
BRANCH GATE**. Canonical P1 was verified as valid: technical PASS,
DIAGNOSTIC_ONLY, fixed K77/frozen 36 features, 5,600 OOF decisions, 410
executions, and `D77_pairwise = -0.0007142857142857784`.

P1F cannot perform the mandatory exact forward-scoring reproduction because
the trained P1 `HistGradientBoostingClassifier` objects were not persisted.
The approved P1 implementation creates models only in memory during its
inner/outer fitting loop and writes the contract, final decisions, and report;
it does not serialize model objects. The task prohibits deterministic refitting
in this situation, so no model was refit and no reconstructed action scores,
S/T/R attribution, B-minus-N diagnostics, harmful attribution, pair-family
correlation, or depth analysis was produced.

This preserves all constraints: no objective/pair construction/pair weighting/
feature/K/threshold change, no Fold0 or public labels, and Workflow B remains
inactive. Status is `BLOCKED_NO_PERSISTED_P1_MODELS_FOR_AUTHORIZED_FORWARD_SCORING`;
next state is **STEP2_P1F_BLOCKED_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p1f_reproduction_check_report.json`.

## STEP 2-P1F-R — Deterministic Frozen-P1 Reconstruction (2026-08-28)

The CPU process finished, but the exact reconstruction gate returned
`RECONSTRUCTION_MISMATCH`. All four thresholds, 5,600 decisions, and query-top
scores matched the canonical P1 outputs. The measured D77 and per-fold D77
values differed from the canonical serialized values only by floating-point
rounding, so the strict gate did not pass. No models or downstream diagnostic
reports were persisted, and no scientific conclusion or branch gate is valid.

The original historical blocker remains unchanged:
`BLOCKED_NO_PERSISTED_P1_MODELS_FOR_AUTHORIZED_FORWARD_SCORING`.

## STEP 2-P1F-R2 — Canonical-Metric Resolution and Pairwise Failure Attribution (2026-08-28)

Professor resolution classified the earlier strict-gate mismatch as
`DERIVED_METRIC_FLOATING_POINT_CONTRACT_MISMATCH`, not a true model mismatch.
Primitive outputs were accepted as functionally equivalent: thresholds 4/4,
query-top scores 5,600/5,600 with zero error, executed identities 410/410,
NO_OP decisions 5,190/5,190, and class composition B=6/N=392/H=12.

The project-wide canonical D metric is now
`SUM(per_query_recall_delta)/N`; macro-recall subtraction is a non-binding
display cross-check. Canonical P1 gain sum is -4.0 and canonical D77 is
-0.0007142857142857143. The one authorized deterministic frozen replay passed,
persisted all four diagnostic outer models, and produced full scoring for
806,644 actions. P1 attribution was S=6, T=63, R=357 oracle-positive queries;
pair counts were BN=125,576, BH=1,625, NH=1,539,444. Pair-family correlations
are association-only, not causal (Spearman 0.0957899011; Pearson 0.1629907357).

Scientific status remains **DIAGNOSTIC ONLY — NO BRANCH GATE**. P1 formulation
is REJECTED; pairwise principle and HGB remain UNRESOLVED; pair reweighting,
B-vs-N-only training, and threshold retuning are NOT AUTHORIZED/DEFERRED.
Workflow B, P2, Step3, and Step4 remain inactive. Next state:
**STEP2_P1F_R2_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p1f_metric_contract_resolution_report.json`,
`reports/task1/step2p1f_r2_reproduction_gate_report.json`,
`reports/task1/step2p1f_failure_attribution_report.json`, and
`reports/task1/step2p1f_pairfamily_correlation_report.json`.

## STEP 2-P1T — Pairwise-Specific Threshold/Calibration Forensic (2026-08-28)

Technical status: **PASS**; scientific status: **DIAGNOSTIC ONLY — NO BRANCH
GATE**. This was a read-only analysis of the persisted 806,644 frozen P1
action scores: no `model.fit`, refit, rescoring, threshold change, threshold
optimization, score-formula change, pair weighting change, or calibration fit.
All canonical inputs were reproduced: 5,600 query tops; B/N/H=69/5,405/126;
S/T/R=6/63/357; and executed B/N/H=6/392/12.

Of the 69 oracle-positive queries whose frozen top action was BENEFICIAL, only
6 crossed their already-selected nested-OOF threshold (8.6956521739%). The
other 63 are mostly substantially below it: 49/63 (77.7777777778%) are more
than 0.050 below threshold; median T margin is -0.0922472723226565. Crossing
rates are B=6/69, N=392/5,405, and H=12/126, descriptively supporting weak
class selectivity of the frozen absolute score at the fixed threshold. Harmful
executions are all at most 0.050 above threshold, with median margin
0.02017194368118208; their aggregate recall-delta is -7.666666666666667. The 6
beneficial executions realize +3.6666666666666665, yielding canonical net
gain -4.0.

The inner-OOF candidate threshold curve was not available from persisted
artifacts; no inner refit is permitted or required, and the four existing
thresholds remain frozen. The interpretation is unchanged: ranking loss is
the primary remaining bottleneck; threshold/score calibration is a newly
significant secondary bottleneck; mean pairwise win-probability calibration is
a supported hypothesis, not an established cause. Fold0/public labels were
not used. Pair reweighting, B-vs-N-only training, HGB/feature/K changes, and
Workflow B remain unauthorized/inactive. Next state:
**STEP2_P1T_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p1t_threshold_calibration_report.json`.

## STEP 2-P1C — Nested-OOF Threshold/Calibration Experiment for Frozen Pairwise Score (2026-08-28)

Technical status: **PASS**; scientific status: **DIAGNOSTIC ONLY — NO BRANCH
GATE**. The authorized nested-OOF experiment fit and persisted 12 inner
models and serialized 16,800 inner-context query rows. Historical P1
thresholds were reproduced exactly, and canonical gain-sum selection chose
the same thresholds in every fold: F1=0.8871766063648394,
F2=0.9158352964999534, F3=0.8885338990548798, F4=0.9072451320464932.

Using the existing frozen outer action scores, the new policy is identical to
the old policy: gain sum=-4.0, D77=-0.0007142857142857143, delta D77=0,
executed B/N/H=6/392/12, S/T/R=6/63/357, and harmful loss=-7.666666666666667.
All fold thresholds, execution counts, gains, and D values are unchanged.
Thus proper nested-OOF threshold reselection did not improve held-out D77;
weak absolute score selectivity persists. This does not authorize threshold
changes or alternative score aggregation. Ranking loss remains the primary
bottleneck; threshold placement is measured but unchanged under the proper
inner-OOF procedure.

No outer models were retrained, no Fold0/public data were used, and no
calibration model or scientific branch was run. Next state:
**STEP2_P1C_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p1c_realizability_report.json` and
`reports/task1/step2p1c_inner_oof_threshold_curves.json`.

## STEP 2-P1A — Controlled Median-Aggregation Action-Score Experiment (2026-08-29)

Technical status: **PASS**; scientific status: **DIAGNOSTIC ONLY — NO BRANCH
GATE**. This was a deterministic posthoc recombination using the persisted
P1 outer/inner models. No model fitting, retraining, feature/objective/K/pair
weighting change, or aggregation sweep was performed. The mean reconstruction
audit passed on 100 queries and 14,404 actions with zero absolute/relative
error and 100/100 top-action identity matches.

The only tested intervention was
`MEAN_PAIRWISE_WIN_PROBABILITY` -> `MEDIAN_PAIRWISE_WIN_PROBABILITY`.
Outer median scores covered 806,644 actions; inner median OOF scores covered
16,800 rows. Median nested-OOF thresholds were F1=Infinity, F2=Infinity,
F3=0.9837219720799008, and F4=Infinity. Under these frozen inner-selected
thresholds, median gain sum was 0.3333333333333333 and D77 was
0.000059523809523809524, versus mean gain -4.0 and D77
-0.0007142857142857143 (delta +0.0007738095238095238).

Median execution was B/N/H=1/16/0, with top classes B/N/H=61/5,416/123;
S/T/R=1/60/365. Harmful loss was 0, while beneficial realized gain was
0.3333333333333333. Median changed 1,318/5,600 top-action identities and the
median best-B-minus-best-N was -0.047746588404987356, with 359 negative and
18 within 1e-12 of zero. Therefore the median result improves pooled D77 and
removes harmful executions in this diagnostic, but it also changes ranking
substantially and increases the oracle-positive R count from 357 to 365;
this is not a production or branch decision.

The scalar-threshold-only branch remains closed. Pair-family causality remains
unresolved, no alternative formula is authorized, and Workflow B/P2/Step3/
Step4 remain inactive. Next state:
**STEP2_P1A_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p1a_realizability_report.json`,
`reports/task1/step2p1a_threshold_selection_report.json`, and
`reports/task1/step2p1a_median_scores.jsonl`.

## STEP 2-P1A-F — MEAN-vs-MEDIAN Aggregation Failure Attribution (2026-08-29)

Technical status: **PASS**; scientific status: **DIAGNOSTIC ONLY — NO BRANCH
GATE**. This was a pure post-hoc action/query join: no model fit, rescoring,
formula test, threshold change, Fold0, or public labels. The join reproduced
all canonical results: mean/median gain -4.0/0.3333333333333333, delta
4.333333333333333, executions 410/17, B/N/H 6/392/12 versus 1/16/0,
S/T/R 6/63/357 versus 1/60/365, and oracle headroom 329.2 over 426 queries.

Of 1,318 changed tops, the complete mean-to-median class matrix was B->N=18,
N->B=10, N->N=1,261, N->H=1, H->N=4, H->H=24; all other cells were zero.
For the 95 oracle-positive changed tops it was B->N=18, N->B=10, N->N=64,
H->N=1, H->H=2. The decisive pooled gain change came mostly from threshold
abstention with unchanged top action: TOP_SAME + EXECUTE_TO_NOOP removed
6.666666666666667 harmful loss but also gave up 2.3333333333333335 beneficial
gain. Changed-top abstention removed another 1.0 harmful loss and lost another
1.0 beneficial gain; changed-top flips alone did not generate the positive
pooled gain.

All 12 mean harmful executions were accounted for: 6.666666666666667 loss was
removed through same-top execute-to-no-op and 1.0 through changed-top behavior.
All six mean beneficial executions were accounted for: 2.3333333333333335 gain
was lost through same-top abstention and 1.0 through top changes. The sole
median beneficial execution was retained query 34748 in F3, gain 1/3. F3 has
a higher top-change rate (27.5% vs 22.21% elsewhere), removed 5.166666666666667
harmful loss, and contains the only median beneficial realization.

Median remains **MECHANISTICALLY_INFORMATIVE_BUT_NOT_REALIZABILITY_SUCCESS**:
aggregation sensitivity is directly observed, while ranking remains the primary
bottleneck. No other aggregation, threshold change, or downstream branch is
authorized. Next state: **STEP2_P1AF_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p1af_topaction_change_report.json` and
`reports/task1/step2p1af_fold_and_depth_breakdown.json`.

## STEP 2-P1A-C — Decision-Confidence / Abstention Failure Forensic (2026-08-29)

Technical status: **PASS**; scientific status: **DIAGNOSTIC ONLY — NO BRANCH
GATE**. This was a read-only post-hoc forensic over the persisted MEAN and
MEDIAN action-score artifacts: 806,644 action rows joined exactly across 5,600
research-fold queries. No model fit, model loading for scoring, rescoring,
formula test, threshold optimization, policy counterfactual, Fold0, or public
labels were used. The canonical MEAN populations reproduced exactly:
B/N/H=6/392/12, with 337 same-top EXECUTE->NOOP and 56 top-changed
EXECUTE->NOOP cases.

The persisted score-level signals were available. The strongest descriptive
separation was MEAN-minus-MEDIAN score for the same selected action: its
directional B-vs-H AUC was 0.875 (lower signed gap associated with BENEFICIAL;
B/H medians -0.05885789464094121/-0.00024852357905857936). Absolute gap gave
directional AUC 0.8611111111111112. In contrast, MEAN and MEDIAN top1-top2
margins were weak (0.5833333333333333 and 0.5138888888888888). The 337
same-top abstentions had a signed-gap median of -0.03394576328653587, between
the B and H medians. The 56 top-changed abstentions had lower score margins
than the same-top abstentions, while their same-action gap profile was close.
Fold-level B/H support was sparse (F2 had neither class; F4 had no B), so no
fold result is confirmatory.

Raw pairwise probability vectors/dispersion statistics, inner-model agreement
traces, and exact values for the other frozen 36 features were not persisted
and were explicitly left unavailable rather than reconstructed. The headline
is **STRONG_BUT_SMALL_N_CONFIDENCE_SEPARATION**, strictly descriptive with
only 6 B and 12 H executions; it does not authorize any confidence rule,
threshold, aggregation, feature, model, K, pair-weighting, Workflow B, or
downstream intervention. P1A-F remains valid: all +4.333333333333333 pooled
improvement was decision abstention, with same-top abstention removing
6.666666666666667/7.666666666666667 harmful loss and losing
2.3333333333333335/3.333333333333333 beneficial gain; top-change net policy
contribution was 0, 1261/1318 top changes were N->N, and 0/10 new B-top
discoveries were realized. Ranking remains primary; decision-confidence /
abstention remains a strongly supported hypothesis; aggregation-as-ranking-fix
is weakened and a third reducer remains unauthorized. Next state:
**STEP2_P1AC_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p1ac_confidence_signal_report.json`.

## STEP 2-P1A-R — MEAN-MEDIAN Disagreement Signal Robustness Forensic (2026-08-29)

Technical status: **PASS**; scientific status: **DIAGNOSTIC ONLY — NO BRANCH
GATE**. P1A-C's selected same-action MEAN-MEDIAN disagreement was evaluated
only in its already-declared signed and absolute forms, using persisted MEAN
top actions and their serialized MEDIAN scores. No model fit/loading for
inference, rescoring, signal search, threshold optimization, policy
counterfactual, Fold0, or public labels was used. All contracts reproduced:
806,644 exact MEAN/MEDIAN action joins, 5,600 queries, 69/5,405/126 MEAN top
classes, 6/392/12 MEAN executions, 426 oracle-positive queries, and the
337/56 abstention populations.

The original executed n=18 result reproduced exactly: signed gap had raw AUC
0.125 and directional AUC 0.875 (lower signed gap associated with B); absolute
gap had raw/directional AUC 0.8611111111111112 (higher absolute gap associated
with B). This is the post-selection exploratory reference, not an independent
validation. In the primary larger all-MEAN-top population (B/H=69/126), signed
raw/directional AUC was 0.48780768345985737/0.5121923165401426 and absolute
was 0.539797561536692 for both. Although each pooled raw orientation matched
its own original reference, neither had meaningful larger-population
separation. Oracle-positive B/H=69/8 yielded directional AUC 0.6340579710144927
for signed (matching orientation) and 0.5434782608695652 for absolute (raw
orientation reversed). The 337 same-top abstentions contained B/N/H=3/323/11:
signed/absolute directional AUCs were 0.8181818181818181 and
0.7878787878787878, respectively, both in the original orientations but still
small B support. The 56 changed-top abstentions were B/N/H=2/53/1 and are
context only.

All four folds had B/H support, but both signed and absolute variants matched
their original raw orientation in only 2/4 folds, so both orientation summaries
are **MIXED_ORIENTATION**. Accordingly, both robustness statuses are
**ROBUSTNESS_NOT_SUPPORTED**. This does not invalidate the observed P1A-C
small-n separation; it establishes that it did not persist in the primary
larger B-top/H-top population. Separate decision-confidence representation
remains supported as a hypothesis, but no intervention, direction, threshold,
or policy is authorized. Ranking remains primary; top1-top2 margin is weakened;
raw pairwise rescoring and a third reducer remain unauthorized. Next state:
**STEP2_P1AR_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p1ar_disagreement_robustness_report.json`.

## STEP 2-P1A-CLOSE — Decision-Confidence / Abstention Sub-Branch Closure Summary (2026-08-29)

This is a documentation-only closure, not an experiment. It references the
completed research-fold results at fixed exploratory K=77; `policy_oof_strict`
remains true, `end_to_end_selection_oof` remains false, and none of these
results is independent confirmatory end-to-end OOF.

### Why this sub-branch opened

STEP2-P1's controlled pairwise formulation produced gain sum -4.0, D77
-0.0007142857142857143, executed B/N/H=6/392/12, and S/T/R=6/63/357. Its exact
formulation was rejected end-to-end, while the pairwise principle remained
alive/unresolved. P1F-R2 nevertheless showed measurable relative-ranking
improvement: pointwise R=380 versus pairwise R=357; R-headroom shares were
about 89.2163% and 83.14095%, and median best-B-minus-best-N improved from
about -0.2652850863 to -0.0944695955. P1T then showed weak absolute execution
selectivity: 49/63 B-top threshold losses lay more than 0.050 below threshold.
P1C reconstructed all 12 inner models and all 16,800 inner-OOF rows, reproduced
the four historical thresholds exactly, and selected the identical thresholds
again (D77 delta=0). The raw-MEAN scalar threshold-placement branch is therefore
closed/rejected within current scope.

### What MEDIAN and attribution established

P1A tested only controlled MEDIAN aggregation without retraining. MEDIAN had
D77=0.000059523809523809524 and gain sum=0.3333333333333333, executing B/N/H
=1/16/0, with thresholds Infinity/Infinity/0.9837219720799008/Infinity across
F1–F4. It changed 1,318/5,600 top actions, worsened ranking (R=365/426 versus
canonical MEAN R=357/426), and was mechanically informative but not a
realizability success. P1A-F directly measured the mechanism: the full
+4.333333333333333 pooled change came from SAME_TOP EXECUTE->NOOP. Across 337
same-top abstentions, it removed 6.666666666666667 harmful loss while giving up
2.3333333333333335 beneficial gain. Top-action changes had net policy
contribution 0; 1,261/1,318 changes were NEUTRAL->NEUTRAL, and none of the 10
new MEDIAN B-top discoveries became a realized beneficial execution.

Thus the abstention mechanism and its harm/opportunity tradeoff are directly
measured. This mechanism is not the same thing as a validated confidence
signal.

### Decision-confidence signal and robustness result

P1A-C's exploratory, post-selection n=18 comparison (MEAN-executed B/H=6/12)
found signed same-action MEAN-MEDIAN disagreement with directional AUC=0.875;
this was classified as strong descriptive small-N separation. Top1-top2 margin
was weakened, and no intervention was authorized. P1A-R evaluated only the
preselected signed and absolute forms on larger fixed populations. In the
primary all-MEAN-top B/H=69/126 population, signed and absolute directional
AUCs were about 0.5122 and 0.5398. Each variant matched its original raw
orientation in only 2/4 folds, with 2/4 reversals. Both robustness results are
`ROBUSTNESS_NOT_SUPPORTED`; the high-AUC same-top subset does not rescue the
broad hypothesis. The original AUC=0.875 is therefore a small-N,
selection-driven non-generalization, not model overfitting.

The decision-confidence mechanism remains plausible but currently lacks a
validated operational signal. Signed and absolute MEAN-MEDIAN gap as general
confidence proxies are strongly weakened; a global confidence direction and
any confidence intervention are not authorized.

### Final closure status

- Abstention mechanism: **DIRECTLY_MEASURED**.
- Broad decision-confidence representation: **UNRESOLVED**.
- MEAN-MEDIAN gap proxy: **STRONGLY_WEAKENED**.
- Confidence intervention: **NOT_AUTHORIZED**.
- Raw pairwise dispersion/agreement: **SUPPORTED_HYPOTHESIS_ONLY**; it was not
  measured directly.
- Raw pairwise rescoring, pair reweighting, B-vs-N-only, new aggregation, and
  Workflow B: **NOT_AUTHORIZED** / **NOT_ACTIVE**.
- Third scalar-reducer search: **REJECTED/CLOSED_WITHIN_CURRENT_SCOPE**.
- Ranking/discrimination remains the **PRIMARY** bottleneck; return-to-ranking
  is **NOT_YET** and requires a fresh professor review.
- Pair-family causal hypothesis and HGB: **UNRESOLVED**. Feature expansion is
  **NOT_JUSTIFIED**; K77 is **NOT_REJECTED** and smaller K is **DEFERRED**.

No next mechanism is selected here. A future single preregistered raw-pairwise
dispersion diagnostic could be considered only if explicitly authorized by the
professor; all other mechanisms also require fresh review. Next state:
**STEP2_P1A_DECISION_CONFIDENCE_BRANCH_CLOSED_PENDING_PROFESSOR_REVIEW**.

## STEP 2-P2 — Ranking-Failure Localization Diagnostic (2026-08-29)

Fresh professor review closed the decision-confidence sub-branch as currently
scoped and reopened ranking because within-query pairwise ranking remains the
largest unresolved, directly responsive gap. This read-only diagnostic used
only the frozen serialized MEAN action scores: no training, inference,
rescoring, pair reweighting, objective/aggregation/threshold/K change, Fold0,
or public labels. Raw pairwise dispersion remains deferred; pair reweighting
and B-vs-N-only remain unauthorized; HGB is unresolved, feature expansion is
not justified, K77 is not rejected, and Workflow B remains inactive.

The canonical ranking state reproduced exactly: 426 oracle-positive queries,
69 successes, 357 failures, no ties, median best-B-minus-best-N
-0.0944695955058123, and total oracle headroom 329.2. Failures hold 273.7
headroom (83.14094775212637%), while successes hold 55.5 (16.859052247873635%).
All 69 successes are shallow (depth 1–20); shallow has S/R=69/232 and failure
rate 77.0764119601329%, whereas deep 21–77 has S/R=0/125 and failure rate 100%.
However, shallow failures are 232/357 (65.0%), below the preregistered 75%
threshold for a shallow concentration label, while deep failures are 125/357
(35.0%).

Local B/N/H action composition and local BN/NH pair-family composition do not
differentiate success from failure materially: both groups have median B/N/H
counts 2/142/0 and median BN/NH fractions 1.0/0.0. Rank-based BN and NH
quartile failure-rate ranges are only 0.07661788044436602 and
0.04849232939516834, respectively, below the 0.20 concentration rule. The
failure gap is mostly moderately wrong (220/357), but this is severity of the
definition of failure rather than a discovered mechanism. Fold failure counts
are F1=81, F2=79, F3=96, F4=101; no causal four-fold inference is made.

The preregistered localization result is **DIFFUSE_FAILURE_PATTERN**: none of
the shallow, deep, pair-composition, or single-gap-bucket concentration rules
held. The descriptive contrasts show failures tend to have deeper best-B
incoming rank (median 15 versus 2), while best-N remains early (median 2
versus 4), but this is localization rather than an intervention recommendation.
Ranking/discrimination remains the primary bottleneck; no ranking mechanism is
selected or authorized. Next state:
**STEP2_P2_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p2_ranking_failure_localization_report.json`.

## STEP 2-P2-S — Shallow Ranking-Failure Isolation Diagnostic (2026-08-29)

Professor interpretation of P2 was preserved: depth is a strong localization
marker but not a sufficient explanation; all 125 deep queries fail, while the
shallow region still has 232 failures among 301 oracle-positive queries. The
working association was upstream-rank-dependent beneficial-score suppression,
with best-B suppression primary, best-N elevation secondary, pair composition
strongly weakened, and no authorization for pair reweighting, B-vs-N-only,
HGB/feature/K changes, or Workflow B.

This read-only P2-S diagnostic reproduced the exact shallow population:
301 total, 69 successes, 232 failures, and shallow failure headroom
178.41666666666666 of 233.91666666666666 (76.27360171001069%). Shallow failures
are overwhelmingly N-top (229) rather than H-top (3). Best-B score remained
lower in shallow failures (median 0.7324749073494621 versus 0.8117236223951685;
directional AUC 0.7280109945027486), so the pooled score separation weakened
from 0.8102951325457719 but remained visible. Best-N score was higher in
failures (0.8190439396506517 versus 0.7860027663452466; directional AUC
0.6355572213893054). Score decomposition is **B_SUPPRESSION_DOMINANT**:
failure-minus-success median shifts were -0.07924871504570641 for B and
+0.033041173305405125 for N.

The residual shallow rank gradient remains very strong: best-B incoming rank
median is 11 in failures versus 2 in successes (directional AUC
0.9343140929535232); best-N rank is 2 versus 4 (AUC 0.7619940029985007).
Drop ranks and action counts show no descriptive separation. Shallow failures
are mainly moderately wrong (158/232), then near-tie wrong (70/232), with only
4 decisively wrong. Across fixed best-B rank bins 1–3, 4–7, 8–12, and 13–20,
the B-score shift was mixed, not uniformly suppressed after rank conditioning.
The within-bin label is **MIXED_DEPTH_AND_SCORE_EFFECT**, and the prescribed
overall shallow label is **SHALLOW_FAILURE_BOTH_SCORE_AND_RANK_DEPENDENT**.
This is descriptive association only, not causal attribution or intervention
authorization. Per-fold shallow failures are F1=52, F2=48, F3=63, F4=69.

Ranking remains the primary bottleneck. No model/objective/pair construction,
pair weighting, threshold, aggregation, feature, HGB, or K intervention is
authorized. Next state: **STEP2_P2S_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p2s_shallow_failure_isolation_report.json`.

## STEP 2-P2-R47 — Rank-Bin-4-7 Joint Best-B/Best-N Score-Shift Structure Diagnostic (2026-08-29)

Professor interpretation of P2-S was preserved: the shallow mechanism is
**RANK_CONDITIONED_B_N_SCORE_GEOMETRY**, refined as rank-conditioned B-vs-N
score distortion. Rank 1–3 had an N-elevation-dominant descriptive pattern;
rank 4–7 had mixed B-down + N-up medians; rank >=8 lacked enough success
support for score-mechanism comparison. Drop rank and action count were
strongly weakened, and no ranking intervention is authorized.

This read-only diagnostic reproduced the exact rank-4–7 population: 74
oracle-positive queries, 14 successes, 60 failures, and no ties. It used only
the frozen success reference medians B=0.8318562423746685 and
N=0.80171758726799. Among failures, 21/60 (35%) were JOINT B-down+N-up,
24/60 (40%) B-down only, 15/60 (25%) N-up only, and none were neither adverse.
Thus every failure had at least one adverse flag, but the joint pattern was
35% of adverse failures rather than the dominant category; the dominant
failure category is **B_DOWN_ONLY**. Successes split B-down only/N-up only
at 7/14 each, with no joint or neither cases.

Spearman best-B versus best-N scores were strongly positive in both groups
(failure 0.9265907196443457; success 0.9516483516483516). This continuous
association does not show strong inverse co-movement; together with the fixed
four-category table it documents co-occurrence in a subset, not a coupled or
independent causal mechanism. No objective, rank feature, pair weighting,
pair construction, model, aggregation, threshold, feature, or K intervention
is selected. Next state: **STEP2_P2R47_COMPLETE_PENDING_PROFESSOR_REVIEW**.
See `reports/task1/step2p2r47_joint_shift_structure_report.json`.

## STEP 2-P3 — Margin-Augmented Pairwise B-vs-N Relative-Ordering Intervention (2026-08-29)

Professor review completed the localization chain and authorized exactly one
proposed intervention: BN-only margin-augmented pairwise logloss with m=0.05,
while preserving pair construction, uniform weighting, HGB, 36 features, K77,
and MEAN aggregation. Pre-flight inspection found a specification conflict
before any training began. Canonical P1 uses
`sklearn.ensemble.HistGradientBoostingClassifier(loss='log_loss')` on mirrored
feature-difference rows `x(hi)-x(lo)` with binary labels. It learns a pairwise
probability only; it has no individual train-time action scalars `s(a), s(b)`.
The action score is later formed by averaging inference pairwise probabilities.

The frozen sklearn HGB API has no custom per-example loss/gradient hook. It
therefore cannot implement the exact requested BN-only symmetric
`softplus(m - y*z)` loss with m=0.05 while retaining m=0 for BH/NH, without
changing model framework or approximating via weights, labels, or shifts. The
required mapping of the professor's m=0.05 to the canonical train-time scale is
also not verifiable: the persisted/action-scale score is a MEAN-aggregated
probability, not an identifiable raw pair score difference. All listed
substitutes are forbidden.

Status: **BLOCKED_SPEC_CONFLICT**. Training, inference/rescoring, models,
action-score serialization, threshold selection, and evaluation were not run;
no approximation was used. Next state:
**STEP2_P3_BLOCKED_PENDING_PROFESSOR_CLARIFICATION**. See
`reports/task1/step2p3_realizability_report.json`.

## STEP 2-P3-R — BN Head-to-Head Decision-Margin Intervention (2026-08-29)

The original P3 margin-loss formulation remains **BLOCKED_SPEC_CONFLICT** and
has no scientific result. The professor reformulated the intervention as an
inference-time BN rule `p_model(B>N) > 0.55`, while BH/NH remain 0.50 and the
canonical MEAN_PAIRWISE_WIN_PROBABILITY aggregation remains frozen.

Semantic pre-flight found no valid integration point. Canonical P1/P1F-R2
generates a probability for every ordered pair, then ranks each action by the
mean of its unthresholded pairwise probabilities. It has no binary pair
win/loss decision that feeds the action ranking. Thus a BN threshold of 0.55
cannot alter a rank without introducing a prohibited new aggregation or
ranking formulation. Separately, under ordinary binary semantics, raising the
B-vs-N threshold from 0.50 to 0.55 makes B harder to declare above N, which is
opposite to the stated B-vs-N improvement target.

Status: **BLOCKED_SPEC_AGGREGATION_CONFLICT** (with the directional conflict
documented). No training, scoring, retraining, threshold selection, output
serialization, or approximation was run; Fold0/public data were not used.
Next state: **STEP2_P3R_BLOCKED_PENDING_PROFESSOR_CLARIFICATION**. See
`reports/task1/step2p3r_realizability_report.json`.

## STEP 2-P3-T — BN Probability-Transformation Relative-Ranking Intervention (2026-08-29)

Professor direction was preserved: transform BN pair probabilities with
`clip(p+0.03,0,1)`, transform NB with `clip(p-0.03,0,1)`, retain the frozen P1
models and arithmetic MEAN_PAIRWISE_WIN_PROBABILITY reducer, use no retraining,
and select a fresh nested-OOF threshold. The relative-ordering hypothesis
remains **SUPPORTED**.

Mandatory label-availability pre-flight is **BLOCKED_SPEC_LABEL_LEAK**. In the
canonical action builder, BENEFICIAL, NEUTRAL, and HARMFUL are respectively
defined by whether replacing a baseline document changes recall against the
query gold answer by a positive, zero, or negative amount. Therefore deciding
whether an inference pair is BN/NB requires its held-out gold-derived action
outcomes. Canonical P1 uses B/N/H during supervised training-pair construction,
but its inference scorer itself uses only label-free feature differences; P3-T
would newly put these gold-derived labels into that scoring function.

The transformation is mathematically defined but operationally requires oracle
B/N action identity at inference, so it is not a valid executable OOF policy
under the Research Constitution and is oracle-only/non-deployable. No frozen
model was loaded or reused, no probability was transformed, and no score,
threshold, or metric output was created. Fold0/public data were not used.
Next state: **STEP2_P3T_BLOCKED_PENDING_PROFESSOR_CLARIFICATION**. See
`reports/task1/step2p3t_realizability_report.json`.

## STEP 2-P4 — Label-Free Beneficial-Action Proxy Discriminability Diagnostic (2026-08-30)

The P3 sub-branch is **FORMALLY_CLOSED**: P3 was
**BLOCKED_SPEC_CONFLICT**, P3-R was **BLOCKED_SPEC_AGGREGATION_CONFLICT**, and
P3-T was **BLOCKED_SPEC_LABEL_LEAK**. Those interventions produced no
scientific R/D77 result. The relative-ordering hypothesis remains
**SUPPORTED**; the new binding problem is **MISSING LABEL-FREE MECHANISM
IDENTIFICATION SIGNAL**.

P4 trained a separate pointwise BENEFICIAL-vs-{NEUTRAL,HARMFUL} diagnostic
using only the frozen 36 observable action features, uniform influence, and
the frozen HGB log-loss configuration. It fit 4 outer and 12 inner
fold-separated diagnostic models; no Fold0/public data, threshold, gate,
policy score, ranking, aggregation, or P1 decision was changed. Outer held-out
ROC-AUC was F1=0.8142359298766111, F2=0.8486435053975017,
F3=0.8421840104747816, F4=0.8091343492111038, and pooled=0.8273434845649417
(fixed orientation: higher score means more likely BENEFICIAL). This is
mechanism-recognition evidence only, not authorization for a proxy-based
intervention. The BENEFICIAL prevalence was 897/806644
(0.001112014717768929).

Feature importance is **NOT_AVAILABLE_WITHOUT_NEW_UNPREREGISTERED_METHOD**:
no canonical procedure exists and sklearn HGB exposes no direct
`feature_importances_` attribute. P1 success/failure proxy-score distributions
were reported only after outer predictions were frozen. Next state:
**STEP2_P4_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p4_discriminability_report.json`.

## STEP 2-P4-Q — Query-Conditioned B-vs-N Proxy Concordance Diagnostic (2026-08-30)

Professor adjudication of P4 is preserved: P4 is **VALID** with a
**STRONG_DESCRIPTIVE_OOF_SIGNAL**; label-free signal existence is **SUPPORTED**,
feature signal is **FEATURE_SIGNAL_PRESENT**, and HGB beneficiality-recognition
capacity is **SUPPORTED_CAPACITY_FOR_BENEFICIALITY_RECOGNITION**. P3 remains
closed. Within-query discriminability was the remaining unresolved question;
proxy threshold/gating and proxy reranking remain **NOT_AUTHORIZED**.

This read-only diagnostic reused exactly the frozen P4 outer-OOF predictions,
without training or new inference. All 426 canonical oracle-positive queries
contained at least one true BENEFICIAL and one true NEUTRAL action, so all were
included (none excluded). Across 125576 Cartesian B/N pairs, the preregistered
pair-pooled concordance was 0.8143156335605529 against the fixed 0.50
reference: 102221 B-above-N pairs, 75 exact ties, and 23280 B-below-N pairs.
The query-majority concordance rate was 371/426 = 0.8708920187793427, with one
exact-50/50 query. Per-fold pair concordance was F1=0.8027104187226843,
F2=0.8449552006232957, F3=0.806654460319366, and F4=0.8034548422198041.

These are descriptive concordance results only. No threshold, gate, reranking,
proxy/P1 fusion, policy modification, or new D77 computation occurred. Next
state: **STEP2_P4Q_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p4q_concordance_report.json`.

## STEP 2-P5 — P1-Proxy Percentile-Fusion Relative Reranking Intervention (2026-08-30)

Professor authorization was recorded: P4-Q is **VALID**;
within-query B-vs-N proxy discriminability is
**SUPPORTED_DESCRIPTIVELY**; relative proxy reranking is authorized at
beta=0.05 with within-query percentile normalization; absolute proxy gating
and proxy-only ranking remain **NOT_AUTHORIZED**; another diagnostic is not
required; feature sufficiency for proxy discrimination is supported; feature
expansion is not justified; and P3 remains closed.

P5 stopped in mandatory artifact-provenance preflight with status
**BLOCKED_ARTIFACT_PROVENANCE_CONFLICT**. The full frozen P1 outer action-score
artifact and full frozen P4 outer proxy artifact exist. However, the resolved
P1 inner artifact `step2p1c_inner_oof_top_scores.jsonl` contains only one
already-selected P1 top action per inner-held-out query (16800 rows), whereas
P5 requires scores for all actions in every query to calculate the prescribed
within-query proxy percentile, fuse scores, rerank, and then perform fresh
nested-OOF threshold selection. P4 inner predictions contain all actions, but
there is no matching persisted P1 inner full-action score artifact. Regenerating
it would require prohibited P1 inference.

No fusion, threshold selection, ranking, policy evaluation, model inference,
or training was performed, and no Fold0/public data were used. Next state:
**STEP2_P5_BLOCKED_PENDING_PROFESSOR_CLARIFICATION**. See
`reports/task1/step2p5_realizability_report.json`.

## STEP 2-P5-A — P1 Inner Full-Action Score Artifact Completion (2026-08-30)

P5 được professor phân loại là **ARTIFACT_AVAILABILITY_REPRODUCIBILITY_BLOCK_ONLY**:
fusion beta=0.05 đã được cấp phép nhưng chưa chạy. P5-A đã hoàn tất việc
khôi phục artifact còn thiếu bằng PRIMARY PATH: nạp đúng 12 frozen P1C inner
models tại `reports/task1/step2p1c_models/inner_fold_models`, không retrain.

Full per-action P1 scores được tái tạo trên đúng K77, 36 features, pair input
`x(a)-x(b)`, MEAN_PAIRWISE_WIN_PROBABILITY và canonical tie-break. Reproduction
gate đạt **PASS**: kiểm tra 16,800/16,800 inner-held-out queries; top-action
identity khớp 16,800/16,800; top-score khớp trong 1e-9 là 16,800/16,800; sai
số tuyệt đối lớn nhất là 0. Artifact gồm 2,419,932 full inner-action score rows
và được đánh dấu **trusted**.

P5 fusion, percentile fusion, threshold selection, R/D77 và mọi ranking mới
chưa được thực hiện. Không dùng Fold0/public và không có OOF violation. Next
state: **STEP2_P5A_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p5a_reproducibility_report.json`.

## STEP 2-P5 — P1-Proxy Percentile-Fusion Relative Reranking Intervention (2026-08-30)

P5-A đã PASS và artifact blocker được giải quyết. Theo professor authorization,
P5 đã chạy bằng đúng frozen P1/P4 outer + inner artifacts, không train và không
model inference mới. Inner và outer action-level join đều PASS 1:1; percentile
dùng zero-based averaged within-query rank; công thức là
`P1_score + 0.05 * proxy_percentile`.

Kết quả kỹ thuật **PASS**. Fresh nested-OOF thresholds được chọn riêng cho F1–F4
chỉ từ inner OOF. Trên 426 oracle-positive queries, R giảm từ 357 xuống **356**
(delta -1). D77_exact tăng từ -0.0007142857142857143 lên
**-0.000565476190476191** (delta 0.000148809523809524); gain_sum tăng từ -4.0
lên **-3.16666666666667**. S/T/R là **6/64/356**. R không xấu đi ở 4/4 folds
(F1 81→81, F2 79→79, F3 96→96, F4 101→100), nên cả ba tiêu chí
preregistered đều đạt; overall validation = **true**.

P5 đổi top action ở 145/5600 queries; trong 426 oracle-positive queries có
1 failure→success, 0 success→failure, 356 failure→failure và 69
success→success. Đây là kết quả của đúng beta=0.05, không phải lý do để đổi
beta hay mở nhánh mới. Không dùng Fold0/public; `policy_oof_strict=true`,
`end_to_end_selection_oof=false`. Next state:
**STEP2_P5_COMPLETE_PENDING_PROFESSOR_REVIEW**. See
`reports/task1/step2p5_realizability_report.json`.

## STEP 3-A — Confirmatory Holdout Data-Scope Audit (2026-08-30)

Step2 is **CLOSED** and P5 remains **PROVISIONAL_ONLY**. Step3 was opened only
as `OPENED_BUT_BLOCKED_PENDING_AUDIT`; no confirmatory execution was authorized
or performed.

The canonical Research Constitution is the `Research constitution` section of
this document. It permits only validation queries F1–F4 for the standing
research workflow and forbids Fold0 labels/data, Fold0 evaluation, and public
labels/gold unless a separately approved prompt clearly requires an exception.
It does not state a general final-confirmatory authorization.

`reports/task1/step3_data_scope_audit.json` records the full read-only
inventory and prior-use audit. Result:

- `untouched_authorized_holdout_exists = NO`
- `eligible_holdouts = NONE`
- F1–F4: not independently confirmatory due to Step2 selection dependence.
- Fold0: exists, has labels, was historically used for independent/oracle
  evaluation; constitution status `CONDITIONALLY_AUTHORIZED`, untouched `false`,
  Step3 eligible `false`.
- public: exists as a label-free 1,000-query public population and was used for
  final inference/leaderboard observation; constitution status
  `CONDITIONALLY_AUTHORIZED`, untouched `false`, Step3 eligible `false`.
- Other identified Task1 populations (warm-up and internal synthetic benchmark)
  are outside the expressly permitted F1–F4 research scope.

No label values were inspected and no model, training, inference, evaluation,
threshold selection, or policy modification occurred. The next state is
**STEP3_DATA_SCOPE_AUDIT_COMPLETE_PENDING_PROFESSOR_REVIEW**. A fresh professor
review is required; this audit does not authorize confirmatory execution.

## STEP 4 — Workflow A Finalization (2026-08-30)

Technical status: **PASS**. Artifact-only integrity verification passed for the
frozen P1/P4/P5 chain and the Step3 audit; no model execution, training,
inference, evaluation, threshold selection, or score recomputation occurred.

- Workflow A: **CLOSED**.
- Final Workflow-A policy: **P5** — `P1_score + 0.05 * proxy_percentile` at
  K=77, using 36 frozen features and
  `WITHIN_QUERY_ZERO_BASED_AVERAGED_PERCENTILE` normalization.
- Policy status: **PROVISIONAL_ONLY** and **BEST-EVIDENCED WORKFLOW-A POLICY**.
- Scientific status: **SUPPORTED_WITHIN_EXPLORATORY_OOF_SCOPE**.
- Independent confirmation: **NOT ACHIEVED**; reason: **NO INDEPENDENT
  AUTHORIZED HOLDOUT**.
- Step3: **BLOCKED_NO_INDEPENDENT_AUTHORIZED_HOLDOUT**.
- Workflow B: **ELIGIBLE_TO_OPEN**; specific Workflow-B experiment:
  **NOT_AUTHORIZED_YET**.

P5 preserves the frozen result R `356/426`, D77_exact
`-0.0005654761904761906`, gain_sum `-3.166666666666667`, S/T/R `6/64/356`,
four of four R folds non-worsened, one failure-to-success, zero
success-to-failure, and a PASS preregistered validation. The effect remains
small but directionally consistent. The authoritative closure and integrity
manifest are in `reports/task1/step4_workflow_a_finalization.md`.

Next state: **WORKFLOW_A_CLOSED_PENDING_WORKFLOW_B_PREREGISTRATION**. Workflow
B is a new exploratory, large-leverage/structural-improvement lineage; it is
not Step3 confirmation and no specific experiment is authorized here.
