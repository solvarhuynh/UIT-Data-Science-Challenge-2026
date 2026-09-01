# Professor Review After Step 1B

## 1. Executive Decision

Primary measured bottleneck: **ACTION-SPACE / SHORTLIST COVERAGE LOSS**.
Measured `A - B = 0.02204464285714285` pooled macro Recall on 162/5600 queries.
Rank1–3 restriction is not a bottleneck: `B - C = 0.0`, with zero positive
queries and zero loss in all four folds. Decision-policy quality `C-D` remains
unresolved because historical folds1–4 V3A OOF decisions were not serialized.

## 2. Step 0 / Step 1A / Step 1B Audit

- Step 0: PASS. Official primary macro Recall; secondary macro Precision;
  ranking order does not affect official Recall; 5,600 folds1–4 queries; 438
  multi-gold queries; no gold count above 5; Fold0 untouched; public labels unused.
- Step 1A: PASS. One-swap ceiling `0.06381547619047619`; unconstrained ceiling
  `0.06687797619047618`; 463 positive one-swap queries; 573 missing gold and
  507 missing gold found in the full pool.
- Step 1B: FAIL on rank-limit gate. A/B/C are
  `0.06381547619047619` / `0.04177083333333334` / `0.04177083333333334`;
  action-space gap is `0.02204464285714285`; rank-limit gap is zero.

Canonical affected A>B counts are F1/F2/F3/F4 = **40/39/41/42**. The
alternative 40/42/38/42 prose count is a typo and must not be used.

## 3. Bottleneck Diagnosis

The immediate measured bottleneck is useful candidates lost between the full
candidate pool and current V3A incoming action-space, not rank protection.

## 4. Rank1–3 Decision

**OPEN RANK1–3 BRANCH = REJECTED.** Reopen only if an upstream candidate or
action-space change materially changes the incoming universe; then rerun Step 1B.

## 5. Recommended Step 1C

Run Step 1C — Action-Space Filter Loss Attribution Forensic to identify the
deterministic stage(s) causing the measured A-B loss. Step 1C is diagnostic
only: no scientific branch gate may be invented from its observed loss.

## 6. Decision on Old Step 2

Old OOF Query-Conditioned Calibration is removed from active Workflow A and
deferred to Workflow B / post-forensic modeling because C-D is unresolved
while A-B is already a directly measured upstream bottleneck.

## 7. Decision on Old Step 3

Old Candidate Pool Noise Forensic must not run as written. Its relevant role is
repurposed into Step 1C: shortlist/action-space filter-loss attribution.

## 8. Decision on Historical D

Historical folds1–4 V3A OOF decisions are not required for Workflow A.
Decision-policy quality remains unresolved; do not retrain merely to recreate D.

## 9. Workflow Changes

Current ordering is Step 0 → Step 1A → Step 1B → Step 1C → Step 4
Consolidation / downstream professor decision. Step 1C is CPU-only.

## 10. Research Risks / Caveats

Oracle/candidate coverage is diagnostic evidence, not a production gain.
Heterogeneity is descriptive only: no numerical heterogeneity threshold was
pre-registered. No GPU, Modal, Fold0 evaluation, public inference or public
labels are required for Step 1C.
