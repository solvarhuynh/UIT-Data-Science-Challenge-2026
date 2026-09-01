# Pre-Workflow-B incumbent regression forensic audit

Status: **PASS**. This was a read-only provenance audit: no model training, public-label access, or Fold0-label loading occurred.

## Exact artifacts and delta

- Incumbent 0.9391: `artifacts/task1/submission.zip`, internal `submission.json` SHA256 `d00d5612efa9ee354a9629a399f967b1319c4decdf121c090d03e8a90e396a56`.
- All-NO_OP 0.9309: `artifacts/task1/submission_p5_all_noop/submission.zip`, internal `submission.json` SHA256 `f1462cb1280ef942f0ceee1dcaf9e384bd90c8de976342ae41484a7b1b77c00f`.
- The incumbent content is exactly the normalized content of `public_v3a_predictions.json`.
- Of 1,000 queries, 672 are identical and 328 differ. Every difference is one replacement: 110 at rank 4, 218 at rank 5, and 0 at ranks 1--3.
- All 328 are exactly traced to the V3A public residual-policy output by full final-top5 equality plus incoming-document candidate/action identity. Per-action model scores were not persisted and were not recomputed.

## Incumbent lineage

`public_anchor_093` baseline -> four-source RRF candidate union (cap 200) -> shortlist (union rank <=20 plus baseline, at most 25 docs) -> 58 public frozen features -> V3A `HistGradientBoostingClassifier` -> threshold 0.0 / harm weight 1.5 -> one rank-4-or-5 swap -> `public_v3a_predictions.json` -> `artifacts/task1/submission.zip`.

The final-fit manifest states training folds `[0,1,2,3,4]`; therefore incumbent training provenance is **F1_F4_PLUS_FOLD0**. This conclusion comes solely from the manifest, not Fold0 labels. V3A public inference reports `no_public_labels_used=true` and the manifest reports no public training.

## Why the deployment regressed

The 0.9309 artifact was intentionally the scientific/pre-policy baseline anchor. Workflow A froze P5 against the F1--F4 scientific reference, then the valid production threshold selection chose `Infinity`, yielding all-NO_OP. The deployment correctly reused that pre-V3A baseline, but no deployment-incumbent preservation gate compared it to the existing 0.9391 V3A submission. Thus the frozen-P5 deployment was technically faithful but operationally not a valid incumbent replacement.

## Scoring contract

The checked repository scoring program, local LegalIR evaluator, and historical recovery evaluator all implement macro set Recall/Precision on up to five documents. The reported external MRR/Recall@3 wording was not available as a local executable contract, so the active platform scorer remains **UNRESOLVED**. Rank-4/5-only changes cannot alter Recall@3, but they can alter untruncated MRR whenever the first relevant document is at rank 4 or 5; the observed public delta alone therefore cannot identify the active scorer.

## Required gates before Workflow B

1. `INCUMBENT_ARTIFACT_HASH_LOCK`
2. `INCUMBENT_PROVENANCE_LOCK`
3. `LOCAL_SCIENTIFIC_COMPARATOR`
4. `DEPLOYMENT_REGRESSION_GATE`
5. `SUBMISSION_DIFF_REPORT`
6. `SCORING_CONTRACT_GATE`
7. `NO_PUBLIC_TUNING_RULE`

Workflow-A scientific findings are not invalidated. Its deployment decision is **partially invalidated**: the frozen P5 mechanics were correct, but replacing the public incumbent without an incumbent gate was not.
