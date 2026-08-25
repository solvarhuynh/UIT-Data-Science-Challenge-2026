# UDSC2026 Task1 LegalIR — Experiment History and Decision Log

Status: reconstructed from local source, manifests, JSON reports and preserved artifacts. No training, retrieval, inference, GPU, Beam, public inference, or new Fold0 evaluation was run while writing this document.

Evidence priority used here: metric/evaluation JSON > manifest > final report > source > notes > directory names. When a report does not prove a claim, it is marked `UNKNOWN` or `LIKELY`; no score is inferred from a folder name.

## 1. Score ledger and evaluation boundaries

These numbers are not interchangeable:

| Score family | Split / meaning | Recall | Interpretation |
|---|---|---:|---|
| Strict OOF baseline | 7,000 train questions, five disjoint normalized groups, pooled folds | 0.9244452381 | strict CV baseline (`baseline_093_oof/report.json`) |
| V3 residual baseline | folds 1–4, 5,600 questions; fold 0 held out | 0.9259285714 | training/selection reference, not public score |
| V3A | folds 1–4 validation | 0.9296488095 | +0.0037202381 vs V3 residual baseline |
| V3A Fold0 | 1,400 untouched evaluation questions | 0.9217261905 | +0.0032142857 vs Fold0 baseline 0.9185119048 |
| V3B raw | folds 1–4 | 0.9297976190 | +0.0038690476 vs baseline, but negative folds versus V3A |
| V3A one-swap oracle | folds 1–4, labels used diagnostically | 0.9676994048 | ceiling, not deployable policy |
| V3A one-swap oracle | Fold0 | 0.9646428571 | ceiling, evaluation only |
| Public leaderboard | user/leaderboard observation | ~0.930 → ~0.939 | not CV evidence; no component attribution is claimed |

The strict CV manifest records 7,000 questions, five folds, 1,400 validation questions per fold, 5,600 training questions per fold, disjoint normalized groups, and zero normalized train/validation overlap. Fold0 is explicitly evaluation-only in `v3_residual/policy/fold0_evaluation.json`; it was not used for model selection.

## 2. Timeline by hypothesis

### EXP-001 — Dense / baseline retrieval and scorer plumbing

**Hypothesis.** A deterministic dense retrieval and correct document-level scorer establish a trustworthy baseline.

**Data and split.** Train LegalIR data; the strict evaluation contract is 7,000 questions and five folds. Early P1/P2 diagnostics also used bounded 500-question probes. Public benchmark rows are unlabeled and cannot be evaluated against a fake chunk label.

**Method and controls.** Candidate depths 5/10/20/50/100/200 (and optional 500), first-occurrence/max document collapse, official macro Recall/Precision, no neural model in P1. The P1 report corrected the label-space mismatch by separating unlabeled transport rows from LegalIR document references.

**Score.** P2 candidate analysis on a 500-question diagnostic gives CandidateDocRecall@200 0.9696667, with 11 queries having no gold document in the cached pool. This is candidate coverage, not final official Recall.

**Result.** PASS as infrastructure/baseline; not a final model. The old generic zero metric was a scorer plumbing error, confirmed by `p1_report.json`, not a retrieval conclusion.

**Decision / successor.** Keep strict scorer and baseline evidence; proceed to compatible BGE and candidate-union experiments. Evidence: `artifacts/task1/evaluation/p1_report.json`, `p2_candidate/summary.json`, `evaluation/strict_cv_v2/manifest.json`.

### EXP-002 — BGE reranker experiments

**Hypothesis.** A cross-encoder reranker improves dense candidates.

**Data.** Historical public/train BGE runs include 200/500/all and short17 variants. Exact sample/split and final top-5 metrics are not uniformly recorded in their manifests.

**Result.** `UNKNOWN` for a single stable aggregate: the preserved comparison manifests prove these are historical variants, but not a common comparable metric table. They are not production V3A.

**Decision.** Retain as historical evidence under `research/legacy/reranker/`; use the inference-matched reranker path and later frozen-feature contract instead. Evidence: moved `research/legacy/reranker/**`, their `run_manifest.json` and `evaluation/comparison.json`.

### EXP-003 — Strict CV v2

**Hypothesis.** Group-normalized, disjoint five-fold evaluation prevents question paraphrase leakage and makes model comparisons reproducible.

**Data/split.** 7,000 questions; 6,984 normalized groups; 16 duplicate groups / 32 duplicate questions; five folds; 1,400 validation and 5,600 training per fold.

**Result.** PASS. Leakage checks are zero for normalized train/validation overlap. This is the evaluation protocol, not a model improvement.

**Decision.** Keep as the source of truth for later experiments. Evidence: `evaluation/strict_cv_v2/{manifest,folds,fold_stats}.json`, `evaluation/p1_report.json`.

### EXP-004 — Inference-matched reranker

**Hypothesis.** A reranker trained/materialized with the same inference semantics as production will improve candidate ordering without train/inference mismatch.

**Data/split.** Train LegalIR candidate materialization, with Fold0 overlap audit and fold-specific artifacts. Exact aggregate improvement is not isolated in one final report.

**Result.** `INCONCLUSIVE` as a standalone promotion claim; the dataset and overlap audits are preserved, but subsequent V3 residual evidence is the decision-bearing artifact.

**Decision.** Keep reproducibility checkpoint under `recovery_096/inference_matched_reranker_v1/` and `models/inference_matched_reranker_v1_fold0/`; do not call it the final production policy. Evidence: `inspection_report.json`, `dataset/audit_report*.json`, `models/.../run_manifest.json`.

### EXP-005 — BM25-grounded BGE variants

**Hypothesis.** Reranking BM25-selected chunks within a document recovers lexical/legal matches missed by dense retrieval.

**Data.** 6,884 queries, 161,763 chunks scored, no missing errors; GPU inference with max length 512 and fp16. `COMPLETE_NO_TOP5_EVALUATION` is explicit.

**Result.** No top-5 official score was produced, so promotion is `INCONCLUSIVE`. Candidate/worklist evidence supports continued candidate construction, not a public ranking claim.

**Decision.** Keep as reproducibility/research evidence; successor is the bounded multi-source union and true-S2 evidence path. Evidence: `recovery_096/bm25_grounded_bge_b4_v1/report.json`, `..._evaluation/report.json`.

### EXP-006 — Selector / gating experiments

**Hypothesis.** A selective gate can spend BGE compute on likely rescue queries and improve recall at bounded cost.

**Data/split.** Fold-aware train diagnostics and held-fold reports; candidate oracle is explicitly diagnostic, not final Recall.

**Result.** Candidate ceilings around 0.9767–0.9801 are recorded for selector plans, but no production top-5 score is established. `INCONCLUSIVE/REJECTED` as final policy because oracle ceilings and rescue counts do not prove policy benefit.

**Failure mode.** Gate aggressiveness and candidate/policy separation; a high oracle can coexist with a poor selector. This is supported by `candidate_oracle_is_diagnostic_not_final_recall=true`.

**Decision / successor.** Keep diagnostics; move toward adaptive K500 and bounded union. Evidence: `bm25_grounded_selector_gate_v1/`, `..._v2/`, `v2b_slim_preflight/` reports.

### EXP-007 — Bounded candidate union

**Hypothesis.** Adding independent lexical sources (BM25, word-KNN, char-KNN) to adaptive candidates raises candidate recall while retaining bounded cost.

**Data/split.** Train fold diagnostics with exact source alignment; public uses fresh K500-derived inputs. Source ablation evidence reports adaptive-union oracle values, not final top-5 scores.

**Evidence.** Checkpointed lexical ablation reports source oracle gains versus K200: word KNN +0.0078452, char KNN +0.0080238, BM25 +0.01235 on its diagnostic protocol; BM25 is most expensive. These are oracle/cost diagnostics, not public attribution.

**Result.** PASS as candidate construction. `public_candidate_union_manifest.json` is the compatibility contract; the dense-K500-incompatible variant is rejected and archived.

**Decision.** Keep the compatible union; successor is shortlist/evidence and frozen features. Evidence: `candidate_union_ablations/`, `public_candidate_union_manifest.json`, `post_k500_bounded_union_v2_audit/report.json`.

### EXP-008 — Adaptive K500

**Hypothesis.** Expand exactly a calibrated fraction of queries from K200 to K500 instead of rescoring all queries.

**Data/split.** Train final-fit model from fold-aware artifacts; public application is 1,000 unlabeled queries with 250 expanded (25%).

**Method.** Logistic calibration/ranking; public rule is `ceil(query_count * selected_budget)`, exact fresh K500 prefix alignment, no public labels.

**Result.** PASS. Public report: 1,000 queries, 250 expanded, selected budget 0.25, alignment PASS. This is a correctness-compatible retrieval expansion, not an independently attributable leaderboard component.

**Decision.** Keep `adaptive_k500_v1` final-fit and `final_public_v3/public_adaptive_k500*`. Evidence: `adaptive_k500_v1/{report.json,adaptive_k500_final_fit_manifest.json}`, public report.

### EXP-009 — Evidence selector / true-S2

**Hypothesis.** Selecting evidence chunks within candidate documents can expose a true second-stage signal under a bounded chunk budget.

**Data/split.** Fold-aware diagnostics; smoke and partial reports are preserved. Selection rule explicitly avoids held-fold selection.

**Result.** `INCONCLUSIVE` for a final independent score: reports contain candidate-oracle ceilings and policy plans but do not establish a stable production top-5 gain. Keep as research evidence.

**Decision / successor.** Use the verified public shortlist/evidence contract as input to frozen features, not the partial selector reports as a leaderboard claim. Evidence: `evidence_selector_recovery_v1*/report*.json`, `final_public_v3/public_shortlist_report.json`.

### EXP-010 — V2 branches

**Hypothesis.** Earlier V2 retrieval/reranking and bounded-rescue variants could improve the baseline.

**Data/split.** Historical fold/candidate artifacts; exact comparable metrics are branch-specific.

**Result.** `REJECTED/LEGACY` as production direction. The preserved V2 artifacts are useful for reproducibility but were superseded by inference-matched V3 residual design.

**Decision.** Do not delete; keep under recovery/archive and the V2 evidence directories. Evidence: `scripts/beam/task1_v2/**`, `recovery_096/v2b_slim_preflight/`, `selective_bge_*`, migration reports.

### EXP-011 — V3 residual policy baseline

**Hypothesis.** A residual one-swap policy can improve a strong baseline while protecting the top three ranks.

**Data/split.** Fold 1–4 train/validation for selection; Fold0 untouched evaluation. Action labels are derived from baseline versus gold only in training diagnostics; public apply uses no labels.

**Method.** Fixed 58-feature contract, max one swap per query, rank 1–3 protected, rank 4/5 eligible.

**Result.** Baseline folds1–4 0.9259285714; V3A 0.9296488095; delta +0.0037202381; per-fold deltas +0.0032143, +0.0063095, +0.0042857, +0.0010714; no negative folds. Fold0 baseline 0.9185119048, V3A 0.9217261905.

**Decision.** PASS and production candidate. Evidence: `v3_residual/policy/policy_training_report.json`, `v3a_final_fit_manifest.json`, `fold0_evaluation.json`.

### EXP-012 — V3A final-fit and public apply

**Hypothesis.** Fit the selected stable V3A residual policy on the allowed training evidence and apply the frozen feature contract to public.

**Data.** Public: 1,000 queries, 20,392 docs, 61,150 chunks; no public labels, no folds. Frozen report records CUDA fp16 inference, batch 32, model unchanged.

**Public result.** Policy PASS: 328 swaps, 672 unchanged; 110 rank-4 drops and 218 rank-5 drops; max-one-swap and top1–3 protection true.

**Decision.** KEEP production. The public leaderboard change (~0.930 to ~0.939) is a user/leaderboard observation only; no +0.009 component attribution is claimed. Evidence: `final_public_v3/public_frozen_features_report.json`, `public_v3a_policy_report.json`, `submission.zip`.

### EXP-013 — V3B two-stage BENEFIT/HARM policy

**Hypothesis.** A two-stage BENEFIT/HARM classifier and gate can improve action selection beyond V3A.

**Data/split.** Folds 1–4, 5,600 queries; Fold0 labels untouched. BENEFIT is rare: 301 queries with any BENEFIT pooled; stage-1 top1 BENEFIT recall is 0.2691 in forensic analysis.

**Score and stability.** Raw V3B 0.9297976190 (+0.0038690 vs baseline, +0.0001488 vs V3A), but fold deltas versus V3A are +0.0016667, -0.0003571, -0.0025, +0.0017857: two negative folds. Best eligible model is null.

**Root cause.** CONFIRMED: stage-1 ranking misses BENEFIT actions (220 pooled queries have BENEFIT but non-BENEFIT stage-1 top1); the gate-only ablation is harmful (-0.0027827 vs baseline, four negative folds). This is not merely a peak-score issue.

**Decision.** REJECT for production; retain forensic artifacts. Successor: stable V3A max-one-swap policy. Evidence: `v3b_policy/{v3b_policy_training_report.json,forensic_report.json}` and `v3_delta_report.json`.

### EXP-014 — Pairwise BENEFIT

**Hypothesis.** Pairwise ranking of beneficial versus harmful actions will improve action ordering.

**Data/method.** Historical pairwise rescue checkpoints and reports; exact final comparable score is not present in the retained canonical report.

**Result.** REJECTED/INCONCLUSIVE. BENEFIT sparsity and action imbalance are documented failure risks; no stable fold evidence justifies promotion.

**Decision.** Keep under `archive/incomplete` / research; replaced by V3A. Evidence: `archive/incomplete/bge_reranker_rescue_v1/`, pairwise scripts and manifests.

### EXP-015 — LambdaMART / LTR

**Hypothesis.** A listwise/tree ranker can learn nonlinear residual ranking better than the linear policy.

**Result.** `UNKNOWN` as a numeric decision: no complete stable metric report was found in the authoritative artifacts. It is not production.

**Decision.** REJECT/LEGACY until a complete fold-aware report exists. No claim about algorithm quality is made from filenames alone.

### EXP-016 — Delta K10

**Hypothesis.** Restricting residual actions to a small delta-K candidate window reduces harm.

**Result.** Rejected historical branch; no authoritative stable aggregate found. Likely risk is candidate-window omission, but direct proof is `UNKNOWN`.

**Decision.** Keep forensic branch; replaced by bounded V3A rank-4/5 action space. Evidence: `scripts/beam/archive/v3_residual_rejected_20260823/`.

### EXP-017 — Opportunity-Gated K20 V1/V2

**Hypothesis.** A gate over opportunity scores can reduce harmful swaps.

**Result.** Rejected. The gate-only ablation provides direct negative evidence: 0.9231458333, delta -0.0027827381, four negative folds. Root cause is confirmed as overly aggressive gating with many neutral/harmful executions.

**Decision.** Replace with V3A's conservative action rule and top-rank protection. Evidence: `v3_delta_report.json` gate ablations and rejected scripts.

### EXP-018 — Hard-negative experiment

**Hypothesis.** Hard negatives improve useful discrimination among near-miss chunks.

**Result.** `INCONCLUSIVE/REJECTED`: no stable authoritative improvement report was found. Increased complexity without verified useful discrimination is not enough for production.

**Decision.** Keep artifacts for research only; no retraining is implied. Evidence: hard-negative scripts/experiment manifests under recovery/archive.

### EXP-019 — Rescue branches

**Hypothesis.** Rescue reranking or selective BGE can recover missed candidates cheaply.

**Evidence.** Selective BGE from Beam scored 1,627 selected queries / 487,914 new chunks, with no training and no submission; other rescue reports explicitly label candidate oracle as diagnostic.

**Result.** Useful for forensic/cost analysis, not a stable final policy. Old incompatible BGE cache was rejected as provenance for fresh K500.

**Decision.** Keep rescue evidence; V3A production uses fresh compatible K500/BGE and bounded union. Evidence: `selective_bge_k500_v1_from_beam/{manifest,cost_report}.json`, `run_adaptive_k500_rescue.py` reports.

### EXP-020 — Metric audit and forensic analyses

**Hypothesis.** Scorer, swap relevance, and incoming/opportunity signals can identify whether remaining errors are retrieval or policy errors.

**Result.** PASS as diagnostics, never as production scores. The official scorer boundary, one-swap oracle, stage-1 failure counts, and candidate-oracle ceilings are explicitly labeled.

**Decision.** Keep reports; do not use oracle values as leaderboard claims. Evidence: `audit_task1_metric_integrity.py`, `forensic_swap_relevance_algebra.py`, `forensic_incoming_relevance_signal.py`, `fold0_evaluation.json`.

## 3. Why V3A was selected over V3B

V3B's peak pooled score is only +0.0001488 over V3A and is not stable: it is below V3A on two of four validation folds, while V3A improves every fold over baseline. V3A has a protected top1–3 contract, max one swap, a positive Fold0 result, a final-fit manifest, and a reproducible public feature/policy hash chain. V3B has no eligible stable configuration and forensic evidence of BENEFIT ranking failure. Therefore V3A is selected on stability, eligibility, Fold0 behavior, and reproducibility—not on peak aggregate alone.

## 4. Oracle gap

Using folds1–4, V3A is 0.9296488 and the one-swap oracle is 0.9676994: gap ≈ **0.0380506**. Fold0 is 0.9217262 versus oracle 0.9646429: gap ≈ **0.0429167**.

The gap proves that the action space contains recoverable opportunities under gold-aware diagnostics. It does **not** prove that a deployable model can reach the oracle, nor that all gap is retrieval. Forensic evidence points to both candidate/policy separation and action-ranking failures; the exact attribution remains mixed.

## 5. Public pipeline: from ~0.930 to ~0.939

The production chain is:

`fresh public dense K500 → compatible BGE K200 prefix → adaptive K500 (25%) → BM25 + word-KNN + char-KNN union → public shortlist → true-S2 evidence → frozen reranker features → 58-feature V3A → max-one-swap → protect rank1–3 → allow rank4/5 correction`.

Classification of claims:

- Offline evidence exists for candidate coverage, adaptive alignment, feature parity, and policy fold/Fold0 behavior.
- Fresh compatibility, hashes, no-label guarantees, and public-contract checks are correctness/reproducibility evidence.
- The ~+0.009 public change is a leaderboard observation. There is no public ablation that permits allocating it among dense, adaptive, union, evidence, frozen features, or policy components.

## 6. Negative-result ledger

| Experiment | Dataset/split | Score | Delta | Status | Supported failure mode | Replaced by |
|---|---|---:|---:|---|---|---|
| Gate-only opportunity policy | folds1–4 | 0.9231458 | -0.0027827 | REJECT | aggressive gate; harmful/neutral executions | V3A |
| V3B raw | folds1–4 | 0.9297976 | +0.0038690 vs baseline | REJECT | unstable; two folds below V3A; BENEFIT top1 recall 0.2691 | V3A |
| Old BGE/cache variants | historical | UNKNOWN common score | UNKNOWN | LEGACY | incompatible provenance / incomplete comparison | fresh compatible K500/BGE |
| BM25-grounded BGE | 6,884 queries | no top-5 score | N/A | INCONCLUSIVE | runtime/candidate evidence only | bounded union + evidence |
| Pairwise/LTR/delta/hard-negative branches | historical | UNKNOWN authoritative aggregate | UNKNOWN | REJECT/LEGACY | insufficient stable evidence | V3A |

## 7. Final decision tree

```text
strict scorer + grouped CV baseline
  ├─ old dense/BGE variants ──> legacy/inconclusive
  ├─ candidate diagnostics ──> bounded union
  │                              ├─ incompatible cache ──> rejected
  │                              └─ fresh K500 + compatible BGE ──> adaptive K500
  ├─ selector/gating/rescue ──> diagnostics; unstable/aggressive branches rejected
  ├─ evidence / true-S2 ──> verified shortlist contract
  ├─ V3B BENEFIT/HARM ──> rejected: unstable and stage-1 BENEFIT misses
  └─ V3A residual one-swap ──> stable folds + positive Fold0
                                   └─ public frozen features/policy
                                        └─ final submission.zip
```

## 8. Lessons learned

1. Candidate coverage and final ranking are separate bottlenecks; oracle scores are not deployable scores.
2. Grouped strict CV and untouched Fold0 prevent selecting a peak that does not reproduce.
3. Conservative residual one-swap with rank1–3 protection is more reliable than broad gating.
4. BENEFIT is sparse; a two-stage classifier can fail before the gate if stage-1 ordering misses the useful action.
5. Public leaderboard movement cannot be decomposed without public ablations.
6. Fresh provenance and exact feature/schema parity matter as much as model choice.
7. GPU work is justified for frozen inference and bounded rescoring only when the contract is already verified; it is not evidence of score by itself.

## 9. Master table

| Experiment | Hypothesis | Train/Public | Split | Baseline | Score | Delta | Stable? | Fold0? | Public? | GPU? | Decision | Replacement | Evidence |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|
| Dense/scorer plumbing | trustworthy baseline | Train | 5-fold | — | 0.9244452 | — | Yes | Yes | No | No | Keep | V3 residual | `evaluation/p1_report.json` |
| BGE variants | rerank candidates | Train/Public historical | mixed | UNKNOWN | UNKNOWN | UNKNOWN | Unknown | Unknown | No | Some | Legacy | inference-matched | `research/legacy/reranker/` |
| Strict CV v2 | prevent leakage | Train | 5-fold grouped | — | protocol PASS | — | Yes | Yes | No | No | Keep | all later models | `evaluation/strict_cv_v2/` |
| Inference-matched reranker | match train/inference | Train | fold-aware | UNKNOWN | UNKNOWN | UNKNOWN | Unknown | audited | No | Yes | Repro | V3 residual | `recovery_096/inference_matched_reranker_v1/` |
| BM25-grounded BGE | lexical rescue | Train | 6,884 query worklist | — | no top5 score | — | Unknown | Unknown | No | Yes | Inconclusive | union | `bm25_grounded_bge_b4_v1/report.json` |
| Selector/gating | selective compute | Train | fold-aware | — | oracle diagnostics | — | Unstable | — | No | Some | Reject final | adaptive K500 | `bm25_grounded_selector_gate_v*/` |
| Bounded union | add lexical sources | Train/Public | fold-aware/public contract | — | oracle/correctness PASS | — | Yes contract | — | Yes | Some | Keep | shortlist | `candidate_union_ablations/` |
| Adaptive K500 | expand 25% | Train/Public | final-fit/1000 public | — | 250/1000 expanded | — | Yes contract | — | Yes | No/Some | Keep | evidence | `public_adaptive_k500_report.json` |
| Evidence/true-S2 | select useful chunks | Train/Public | fold-aware/1000 public | — | contract PASS; no isolated public attribution | — | Unknown | — | Yes | Yes | Keep contract | frozen features | `public_shortlist_report.json` |
| V2 branches | earlier rescue | Train | historical folds | UNKNOWN | UNKNOWN | UNKNOWN | No final proof | Unknown | No | Some | Legacy | V3 | `recovery_096/v2*` |
| V3 baseline | residual action | Train | folds1–4 + Fold0 | 0.9259286 / 0.9185119 | same | — | Yes | Yes | No | No | Keep reference | V3A | `policy_training_report.json` |
| V3A | conservative one swap | Train/Public | folds1–4 + Fold0/1000 public | 0.9259286 | 0.9296488 / 0.9217262 | +.0037202 / +.0032143 | Yes | Yes | Yes | Frozen GPU | Production | — | `v3a_final_fit_manifest.json` |
| V3B | BENEFIT/HARM gate | Train | folds1–4 | 0.9259286 | 0.9297976 | +.0038690 | No | forensic only | No | No | Reject | V3A | `v3b_policy_training_report.json` |
| Pairwise BENEFIT | pairwise action ranking | Train | historical | UNKNOWN | UNKNOWN | UNKNOWN | Unknown | Unknown | No | Some | Reject/legacy | V3A | rescue checkpoints |
| LambdaMART/LTR | nonlinear ranker | Train | historical | UNKNOWN | UNKNOWN | UNKNOWN | Unknown | Unknown | No | Unknown | Reject/legacy | V3A | no complete report |
| Delta K10 | narrow action window | Train | historical | UNKNOWN | UNKNOWN | UNKNOWN | Unknown | Unknown | No | Unknown | Reject/legacy | V3A | rejected scripts |
| Opportunity K20 | gate opportunities | Train | folds1–4 | 0.9259286 | 0.9231458 | -.0027827 | No | No | No | No | Reject | V3A | gate ablation |
| Hard negatives | improve discrimination | Train | historical | UNKNOWN | UNKNOWN | UNKNOWN | Unknown | Unknown | No | Some | Reject/legacy | V3A | no complete report |
| Rescue variants | cheap recovery | Train/Public diagnostic | mixed | — | oracle/cost only | — | Unknown | — | No | Yes | Research | fresh union |
| Metric/forensic audit | explain failures | Train | fold-aware/Fold0 | — | oracle/diagnostic | — | N/A | Yes | No | No | Keep diagnostic | — | forensic reports |

## 10. Evidence completeness

Evidence is complete enough for the production decision for baseline, strict CV, adaptive K500 contract, bounded union contract, V3A, V3B rejection, frozen public features, and final public apply. Evidence is incomplete or non-comparable for historical BGE variants, Pairwise BENEFIT, LambdaMART/LTR, Delta K10, and hard-negative branches; these remain explicitly marked `UNKNOWN`/legacy rather than receiving invented scores.

TASK1_EXPERIMENT_HISTORY_COMPLETE
