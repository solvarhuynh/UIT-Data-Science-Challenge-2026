# Active scorer provenance and parity audit

Status: PASS (audit completed; active contract remains unresolved).

## Authoritative decision

No immutable local evidence identifies the active UIT-DSC 2026 Task1 CodaBench scorer. There is no local active bundle/package hash, competition/phase/task mapping, source revision, Docker image, or digest. Therefore `ACTIVE_SCORER_CONTRACT = UNRESOLVED`; Workflow B is not safe to open.

## Local candidates

`docs/Scoring-Program-Task-LegalIR/scoring.py` is a likely organizer scorer but its version is unproven. It calculates macro set Recall (primary) and set Precision (secondary), accepting 1–5 IDs and assigning zero for empty or >5 answers. `legal_ir.py` matches that set behavior. `legal_ir_recovery.py` is historical local code and scores the first five IDs as a set. None proves active CodaBench deployment provenance.

## Documentation contract

The local Task1 overview says Recall is primary, Precision is secondary, output IDs are treated as sets, up to five IDs are evaluated, and aggregation is macro. It does not define how a single displayed CodaBench Score is mapped: `DISPLAY_SCORE_MAPPING_UNSPECIFIED`. The document has no MRR or Recall@3 wording.

## Synthetic metamorphic parity

All tests used only the synthetic reference in `scorer_metamorphic_results.json`. The three local implementations agree: rank-4/rank-5 document-set changes can change Recall/Precision; a reordering of the same top-5 set cannot. Pure position changes cannot affect their set scores unless the returned set changes. Analytic MRR and Recall@3 comparators show that moving a relevant result from rank 3 to 4 alters both, while moving rank 4 to 5 alters MRR but not Recall@3.

The two public prediction files have identical top 1–3 by the locked structural facts. Recall@3 alone therefore cannot explain their score delta. The local top-5 set scorer could distinguish them only if the changed IDs differ in relevance under unknown public truth; no public truth was opened or scored. MRR can also distinguish some rank-4/rank-5 changes, so structural data alone is not identifying.

## Required acquisition

Obtain the active `scoring_program.zip` or competition bundle, competition/phase/task IDs and mapping, package SHA256, source revision, Docker image plus digest, and official phase metadata/export. Confirm separately whether the participant UI exposes each item; that availability is not locally verifiable.
