# Task 1 LegalIR — Scientific Closeout and Competition Incumbent Lock

**Status:** `INCUMBENT_PRESERVED_AND_SCORE_SEARCH_CLOSED`  
**Decision:** `PRESERVE_CURRENT_0.9391_COMPETITION_INCUMBENT`

## Competition incumbent

The exact deployment artifact is `artifacts/task1/submission.zip`, not an inferred
surrogate.  The independent incumbent forensic audit established a semantic match
between this ZIP and the V3A public prediction stream.

| Field | Locked value |
|---|---|
| Observed public score | approximately `0.9391` (external leaderboard fact; not a CV result) |
| Artifact | `artifacts/task1/submission.zip` |
| ZIP SHA-256 | `4f860cb42a5ee681894cbd96b98a27bd2ad564100f8d90d058b23826fcee5a1a` |
| Submission JSON SHA-256 | `d00d5612efa9ee354a9629a399f967b1319c4decdf121c090d03e8a90e396a56` |
| Byte artifact | existing 19,662-byte ZIP; preserve byte-for-byte |
| Creation timestamp | 2026-08-25 06:35:26 local filesystem time |
| Safe deployment status | `PRESERVE_AS_COMPETITION_INCUMBENT` |

Lineage is public V3A residual policy over the public `0.9309` anchor:

- baseline public anchor: `artifacts/task1/recovery_096/public_anchor_093/submission_093.zip`
- candidate union: four-source RRF (`adaptive_k500`, BM25, word-KNN, char-KNN), cap 200; shortlist is union rank <=20 plus baseline top-5
- feature artifact: `artifacts/task1/recovery_096/final_public_v3/public_frozen_features.jsonl`, 58 features, SHA-256 `cc6c0357a979172faf10135962e7e056fdc82256be361eb92304a2d54e52085e`
- policy family: `HistGradientBoostingClassifier`, one replacement at baseline rank 4 or 5; ranks 1–3 protected
- final-fit model SHA-256: `74633f3677df0ad04273f04ca816c2e64f89576f651b7892e4f9cff3c5324f1e`
- final-fit population: Folds 0–4 / 7,000 queries; no public labels were used for public application
- generation/inference: `scripts/beam/task1_v3_residual/train_residual_policy.py::apply_public_model`
- prediction artifact: `artifacts/task1/recovery_096/final_public_v3/public_v3a_predictions.json`, SHA-256 `a6b2791c7c2e15ac1d6426f4422535638015f5169761aaa80e5198d4ef304097`
- bound policy report: `artifacts/task1/recovery_096/final_public_v3/public_v3a_policy_report.json`, SHA-256 `ca3f9c258bfb37048d5e23cb8c96ae08590e8d40fd898adb74234d5f9e887658`

The V3A public report records 1,000 queries, 328 one-swap changes, 672 unchanged
queries, 110 rank-4 drops, 218 rank-5 drops, no public labels, and no folds in the
public application.  This public incumbent is an operational guard, not a scientific
comparator for F1–F4 research.

## Scientific incumbent

The reproducible F1–F4 comparator is
`artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl`, SHA-256
`1272cb9e8b465f433c725674081084f084a60cdb9807d6ca6a3aa3d9acc1e7d5`.

| Metric | Value |
|---|---:|
| Recall | `0.9259285714285714` |
| Precision | `0.19739285714285715` |
| F1 Recall | `0.9363690476190477` |
| F2 Recall | `0.9283333333333333` |
| F3 Recall | `0.9194047619047620` |
| F4 Recall | `0.9196071428571428` |

This artifact and its manifest are reproducible.  It must not be equated with the
external public `~0.9391` observation.

## Historical V3 policies

| Policy | Reported F1–F4 Recall | Current status |
|---|---:|---|
| V3A | `0.9296488095238095` | `HISTORICAL_POSITIVE_BUT_NOT_CURRENTLY_REPRODUCIBLE`: exact F1–F4 prediction stream is absent, historical source SHA is unavailable, and current reproduction was `0.9280416667`. |
| V3B | `0.9297976190` | `HISTORICAL_AGGREGATE_ONLY_NOT_REPRODUCIBLE`. |

Neither historical aggregate is authorization to replace the preserved public
incumbent.

## Closed branches

| Branch | Final status | Scientific meaning |
|---|---|---|
| Workflow A residual/action family | `CLOSED_METHOD_CEILING` | The investigated action-policy path has no authorized high-ROI continuation. |
| Workflow B Qwen standalone | `REJECTED_STANDALONE_REPLACEMENT` | Does not justify replacement of the incumbent. |
| Workflow B Qwen auxiliary | `NO_VALID_POSITIVE_SCIENTIFIC_RESULT` | No validated incremental result. |
| Qwen shortlist rescue | `SHORTLIST_SIGNAL_SUPPORTED_BUT_NON_DEPLOYABLE` | Shortlist evidence is not final-top-5 deployment evidence. |
| Qwen final-top-5 bridge | `CLOSED_DROP_SIGNAL_INSUFFICIENT` | Drop choice did not support safe final intervention. |
| Workflow C | `CLOSED_SCIENTIFIC_CANDIDATE_REJECTED_AT_INNER_GATE` | C2 did not pass its scientific inner gate. |
| V3A/V3B consensus | `ABANDONED_UNRECOVERABLE_REPRODUCTION` | Required historical prediction/source evidence is unavailable. |
| Direct listwise top-5 | `BLOCKED_LISTWISE_ENGINE` | Not scientifically rejected; no installed grouped listwise engine. |
| Citation-aware retrieval | `CLOSED_SIGNAL_INSUFFICIENT` | Exact matches were already covered by baseline and recovered no out-of-pool gold. |

## Unresolved but not pursued

- Direct grouped listwise ranking remains scientifically unresolved, but needs new
  ranking-environment infrastructure.
- New citation/retrieval construction could be explored later, but the audited
  deterministic citation signal has no current final-top-5 leverage.

Score search stops because the remaining directions require substantial new
infrastructure, retrieval construction, model execution, or environment work without
enough evidence of short-horizon gain.  **Theoretical headroom exists**, but **no
current evidence-supported high-ROI path remains before deadline**.  This is not a
claim of a theoretical task ceiling.

## Do not submit

| Artifact | Reason |
|---|---|
| `artifacts/task1/submission_p5_all_noop/submission.zip` | Known public `~0.9309` all-NO-OP regression artifact; SHA-256 `b44f2d2683a6ba5bd0a8d9b211f80ba76ef3f08245c391544db030a1531bb09c`. |
| `artifacts/task1/recovery_096/public_anchor_093/submission_093.zip` | Pre-V3A baseline anchor; it omits the 328 traced V3A incumbent changes. |
| Workflow C outputs under `reports/task1/workflow_c/shared/c2/` | Incomplete/rejected scientific candidate; never a submission artifact. |
| Qwen experimental outputs under `reports/task1/workflow_b/tv2/b2a/` | Standalone/auxiliary experimental outputs lack a valid positive final-top-5 result. |
| Historical V3B policy outputs under `artifacts/task1/recovery_096/v3_residual/v3b_policy/` | Aggregate-only, not reproducible, and not an authorized public replacement. |

## Final recommendation

`PRESERVE_CURRENT_0.9391_COMPETITION_INCUMBENT`

Do not recreate, overwrite, regenerate, or replace `artifacts/task1/submission.zip`
unless a future candidate has independently validated evidence and an explicit
deployment authorization.
