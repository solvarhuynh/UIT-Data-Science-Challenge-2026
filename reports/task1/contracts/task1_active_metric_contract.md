# Task1 Active Metric Contract

Status: LOCKED_AT_SEMANTIC_LEVEL

Primary: SET_BASED_MACRO_RECALL

Secondary: SET_BASED_MACRO_PRECISION

Max predictions: 5

Rank order within unchanged top-5 set affects Recall: NO

Top-5 membership affects Recall: YES

Rank4/rank5 changes can affect Recall: YES, when membership changes

MRR: STALE_FOR_ACTIVE_TASK

Recall@3: STALE_FOR_ACTIVE_TASK

Exact CodaBench scorer binary provenance: UNRESOLVED_DEPLOYMENT_REQUIREMENT

Workflow B scientific development: AUTHORIZED

Public deployment without scorer provenance: REQUIRES_DEPLOYMENT_REVIEW

## Metric formulas

For query \(i\), with relevant document set \(R_i\) and the valid submitted
top-five-or-fewer document set \(P_i\):

\[
\operatorname{Recall}_i = \frac{|R_i \cap P_i|}{|R_i|}
\]

\[
\operatorname{Precision}_i = \frac{|R_i \cap P_i|}{|P_i|}
\]

Final Recall and Precision are each macro averages across queries. Recall is
the primary scientific metric; Precision is only the secondary tie-break after
Recall. A valid prediction contains at most five document IDs.

## Scientific consequences

The primary objective is relevant-document coverage in the selected top-5
set. Reordering documents already in an unchanged valid top-5 set does not
change primary Recall. A rank-4 or rank-5 change can matter precisely when it
changes that set's membership and therefore its overlap with the relevant set.
Improving internal rank without changing top-5 membership is not itself a
primary-metric gain.

MRR and Recall@3 language is stale for the active task and must not be used as
the scientific optimization objective. Future Workflow-B Direct LTR or ranking
work may use a surrogate ranking objective, but its preregistration must keep
that surrogate distinct from end-to-end evaluation by official set-based macro
Recall.

## Custody and deployment distinction

`ACTIVE_METRIC_SEMANTICS_LOCK` is satisfied for scientific development.
Exact active CodaBench scorer binary/package provenance remains unresolved and
is a deployment/upload-readiness requirement, not a block on Workflow-B
scientific development.

The scientific reference remains the Workflow-A F1-F4 exploratory scientific
lineage. The deployment incumbent is `artifacts/task1/submission.zip` with
score metadata 0.9391, ZIP SHA256
`4f860cb42a5ee681894cbd96b98a27bd2ad564100f8d90d058b23826fcee5a1a`, and
`submission.json` SHA256
`d00d5612efa9ee354a9629a399f967b1319c4decdf121c090d03e8a90e396a56`.
It is an operational regression guard only, not a clean scientific baseline;
it used historical F1-F4 plus Fold0 final fit and must not be used for
model/hyperparameter selection or to override the F1-F4 scientific reference.

## Permanent submission safety gates

- `INCUMBENT_ARTIFACT_HASH_LOCK`
- `INCUMBENT_PROVENANCE_LOCK`
- `SCIENTIFIC_REFERENCE_LOCK`
- `ACTIVE_SCORER_CONTRACT_LOCK`
- `SUBMISSION_DIFF_REPORT`
- `DEPLOYMENT_REGRESSION_GATE`
- `NO_PUBLIC_TUNING_RULE`
- `FOLD0_SCIENTIFIC_EXCLUSION`

The active-scorer contract lock is satisfied at semantic level for scientific
development. Exact scorer binary/package provenance remains required at a
future deployment/upload checkpoint.
