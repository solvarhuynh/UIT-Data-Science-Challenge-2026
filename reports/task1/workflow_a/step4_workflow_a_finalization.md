# Workflow A Finalization

## 1. Final Status

- Workflow A: **CLOSED**
- Step2: **CLOSED**
- Step3: **BLOCKED_NO_INDEPENDENT_AUTHORIZED_HOLDOUT**
- Current policy: **P5 PROVISIONAL_ONLY**
- Policy designation: **BEST-EVIDENCED WORKFLOW-A POLICY**
- Scientific status: **SUPPORTED_WITHIN_EXPLORATORY_OOF_SCOPE**
- Independent confirmation: **NOT ACHIEVED** — this is a data-scope limitation, not failed confirmatory validation.
- `end_to_end_selection_oof`: `false`

Technical finalization status: **PASS**. Required artifacts exist, required JSON reports parse, JSONL artifacts are readable and non-empty, documented inner-artifact record counts match, and the frozen P5/Step3 scientific state is consistent across reports.

## 2. Frozen Policy

Workflow-A final policy: **P5**.

```text
fused_score(a) = P1_score(a) + 0.05 * proxy_percentile(a)
```

- P1: canonical P1/P1F-R2, mean pairwise win-probability action scoring.
- Proxy: canonical P4 beneficiality proxy using the frozen 36 features.
- K: `77`.
- Beta: `0.05`.
- Proxy normalization: `WITHIN_QUERY_ZERO_BASED_AVERAGED_PERCENTILE` (`midrank0/(n-1)`; averaged proxy ties).
- Threshold policy: already-produced canonical P5 nested-OOF threshold artifacts; no threshold was generated in Step4.
- Policy status: **PROVISIONAL_ONLY**.

## 3. Final Workflow-A Metrics

| Measure | Canonical P1 baseline | P5 | P5 minus baseline |
|---|---:|---:|---:|
| R | 357 / 426 | 356 / 426 | -1 |
| D77_exact | -0.0007142857142857143 | -0.0005654761904761906 | +0.00014880952380952371 |
| gain_sum | -4.0 | -3.166666666666667 | +0.833333333333333 |
| S / T / R | 6 / 63 / 357 | 6 / 64 / 356 | — |

- R folds non-worsened: `4 / 4`.
- Failure to success: `1`.
- Success to failure: `0`.
- P5 preregistered validation: **PASS** (`pooled_R_improves=true`, `D77_non_worsens=true`, `fold_robustness_met=true`, `overall_preregistered_validation=true`).

## 4. Step2 Scientific Conclusion

Step2 established that residual post-P1 ranking failures reflect a real, non-trivial relative B-vs-N ordering defect, not a homogeneous global score-shift. A label-free proxy built on the existing frozen 36 features recovers this signal: global AUC is approximately `0.827`, within-query concordance is approximately `0.814`, and query-majority concordance is approximately `0.871`.

That signal was converted into the deployable query-relative policy `P1_score + 0.05 * proxy_percentile`, which passed the preregistered R, D77, and fold-robustness contract. The result is **SUPPORTED_WITHIN_EXPLORATORY_OOF_SCOPE**, because `end_to_end_selection_oof=false`.

## 5. Effect-Size Limitation

P5 rescued one ranking failure. The effect-size classification is **SMALL BUT DIRECTIONALLY CONSISTENT IMPROVEMENT**. It does not establish independent confirmation and does not justify a stronger claim than the frozen exploratory OOF evidence supports.

## 6. Step3 Independent-Validation Limitation

"Independent confirmatory validation of the Step2/P5 policy was scientifically desired to resolve end_to_end_selection_oof=false, but a read-only audit established that no untouched, authorized, labeled data partition exists: F1-F4 were used in design/selection, Fold0 has already undergone historical evaluation (independence lost irrespective of future authorization), and Public contains no labels. Independent confirmation is therefore presently impossible, not because P5 failed any test, but because no qualifying data resource exists. P5 remains supported within its original exploratory scope; this is a data-scope limitation, not a failed replication."

## 7. Artifact Integrity Verification

All hashes below were recorded before Step4 documentation edits. JSON report parse checks passed. JSONL checks established readability and non-emptiness without scientific score recomputation.

| Artifact | Exists | Size (bytes) | SHA256 | Parse/read status |
|---|---|---:|---|---|
| `reports/task1/step2p5_realizability_report.json` | true | 5,172 | `90ac859f8a851f7e3f0e5ba5d2543a570bb8bcbef5f91c2282d2b8791593231f` | JSON parse PASS; frozen P5 values PASS |
| `reports/task1/step2p5_threshold_report.json` | true | 3,052 | `19c571473e048ddeefe2e8c3ba1f1a3a3ebad242672927ed0754ca69822949c8` | JSON parse PASS |
| `reports/task1/step2p5_fused_scores.jsonl` | true | 385,387,357 | `30d1bebce5919e30023231abf60c3a87592a5d7e936d16d85be66b235eaa023c` | readable, non-empty PASS |
| `reports/task1/step2p5_p1_inner_full_action_scores.jsonl` | true | 992,494,938 | `9666176881fa6c368e11507ff9ea987b7c13079b342b4fabf707fd53f1294dc6` | readable, non-empty PASS; 2,419,932 records verified |
| `reports/task1/step2p4_proxy_inner_oof_predictions.jsonl` | true | 676,842,605 | `0efe8f9beb0893642ae8f6b981f201f5e701c14ec26980058ecc498dbf6e19c6` | readable, non-empty PASS; 2,419,932 records verified |
| `reports/task1/step2p1f_full_action_scores.jsonl` | true | 265,981,695 | `4246d81b6c517b21aed7f63c163bbd7cd7c73aa36940d72e8eb970c2d2e24ec8` | readable, non-empty PASS |
| `reports/task1/step2p4_proxy_predictions.jsonl` | true | 270,836,222 | `ba899b3d3414469f1e2f01cc1c19f81c47743d825160396397b4147b0df05783` | readable, non-empty PASS |
| `reports/task1/step2p5a_reproducibility_report.json` | true | 6,171 | `3c8d1093f74725744e5a868a880751e6e023f894a1bafb6f825af810b2561610` | JSON parse PASS; `status=PASS`, `artifact_trusted=true` |
| `reports/task1/step3_data_scope_audit.json` | true | 14,090 | `b8c28ad6f19ed29130926b5eec8c2648db26fb6700a04353d33bb70473f5a1cd` | JSON parse PASS; Step3 conclusion PASS |
| `docs/task1/workflow_A_new.md` | true | 62,258 | `c65593b387cdb693ac04f179af867a7d5889e983b131dba20e227f89df0d9cbc` | readable PASS |
| `reports/task1/progress_log.md` | true | 47,946 | `8a3b542336825b85081d6729dbbf559e76e09cd6cc5e0ebf84a60b1c759bab76` | readable PASS |

Cross-report frozen-value checks passed:

- P5: `status=PASS`, `K=77`, `beta=0.05`, `R=356`, `D77_exact=-0.0005654761904761906`, `gain_sum=-3.166666666666667`, `S/T=6/64`.
- P5 execution flags: `policy_oof_strict=true`, `end_to_end_selection_oof=false`, `training_executed=false`, `new_model_inference_executed=false`.
- P5 preregistered contract: `pooled_R_improves=true`, `D77_non_worsens=true`, `folds_non_worsened=4`, `fold_robustness_met=true`, `overall_preregistered_validation=true`.
- P5-A: `status=PASS`, `artifact_trusted=true`, documented full-action rows `2,419,932`.
- Step3: `status=PASS`, `untouched_authorized_holdout_exists=NO`, `eligible_holdouts=[]`, `step3_recommendation=BLOCKED_NO_INDEPENDENT_AUTHORIZED_HOLDOUT`.

## 8. Frozen Artifact Manifest

The frozen artifacts required to reproduce the Workflow-A policy state are:

- Canonical P1 action scores: `reports/task1/step2p1f_full_action_scores.jsonl`.
- Canonical P4 outer proxy predictions: `reports/task1/step2p4_proxy_predictions.jsonl`.
- Canonical P1 inner full-action scores: `reports/task1/step2p5_p1_inner_full_action_scores.jsonl`.
- Canonical P4 inner OOF proxy predictions: `reports/task1/step2p4_proxy_inner_oof_predictions.jsonl`.
- Canonical P5 fused action scores: `reports/task1/step2p5_fused_scores.jsonl`.
- Canonical P5 threshold record: `reports/task1/step2p5_threshold_report.json`.
- Canonical P5 realizability report: `reports/task1/step2p5_realizability_report.json`.
- Canonical P5-A reproducibility report: `reports/task1/step2p5a_reproducibility_report.json`.
- Canonical data-scope audit: `reports/task1/step3_data_scope_audit.json`.

## 9. Research Integrity Status

- Training in Step4: none.
- Inference in Step4: none.
- Evaluation in Step4: none.
- Score recomputation in Step4: none.
- Threshold selection in Step4: none.
- Fold0: not used.
- Public labels: not used.
- Workflow-B experiment: not executed.

## 10. Final Hypothesis Hierarchy

| Hypothesis / state | Final status |
|---|---|
| ranking bottleneck | SUPPORTED |
| relative B-vs-N ordering defect | SUPPORTED |
| global label-free BENEFICIAL signal | SUPPORTED |
| within-query proxy signal | SUPPORTED_DESCRIPTIVELY |
| deployable mechanism recognition | SUPPORTED_WITHIN_EXPLORATORY_SCOPE |
| P5 relative reranking | SUPPORTED_WITHIN_EXPLORATORY_SCOPE |
| absolute proxy gate | REJECTED_WITHIN_SCOPE |
| proxy-only ranking | REJECTED_WITHIN_SCOPE |
| feature sufficiency for incremental proxy policy | SUPPORTED |
| feature expansion | DEFERRED / NOT_JUSTIFIED FOR WORKFLOW A |
| HGB beneficiality recognition | SUPPORTED |
| HGB within-query discrimination | SUPPORTED |
| HGB augmented-ranking capacity | SUPPORTED_WITHIN_EXPLORATORY_SCOPE |
| HGB custom pairwise-margin flexibility | BLOCKED_BY_IMPLEMENTATION |
| pair-family composition | UNRESOLVED |
| pair reweighting | DEFERRED |
| B-vs-N-only | DEFERRED |
| rank-feature intervention | DEFERRED |
| raw pairwise dispersion | DEFERRED |
| K77 | SUPPORTED_WITHIN_EXPLORATORY_SCOPE |
| smaller K | DEFERRED |
| P3 | CLOSED |
| Step2 | CLOSED |
| Step3 | BLOCKED_NO_INDEPENDENT_AUTHORIZED_HOLDOUT |
| Workflow A | CLOSED |
| Workflow B | ELIGIBLE_TO_OPEN |

## 11. Workflow B Handoff

- Workflow B: **ELIGIBLE_TO_OPEN**.
- Type: **NEW EXPLORATORY IMPROVEMENT LINEAGE**.
- Goal: **LARGE-LEVERAGE / STRUCTURAL IMPROVEMENT SEARCH**.
- Specific experiment: **NOT YET AUTHORIZED**.

Workflow B is neither continuation of Step3 confirmation nor automatic beta/normalization micro-tuning. Before large compute expenditure, a future intervention should have a preregistered practical-effect rationale materially larger than P5's approximately one-query improvement. This finalization sets no exact numeric effect bar and authorizes no specific Workflow-B model or experiment.

## 12. Closure

Workflow A is formally complete. Its final state is the frozen P5 policy, designated **BEST-EVIDENCED WORKFLOW-A POLICY** and **PROVISIONAL_ONLY**, with scientific status **SUPPORTED_WITHIN_EXPLORATORY_OOF_SCOPE**. The next state is `WORKFLOW_A_CLOSED_PENDING_WORKFLOW_B_PREREGISTRATION`.
