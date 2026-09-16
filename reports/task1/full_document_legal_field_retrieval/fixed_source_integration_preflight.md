# FULLDOC_BM25_FIXED_SOURCE_INTEGRATION_PREFLIGHT

## Status

`BLOCKED_NO_REPRODUCIBLE_NEW_CANDIDATE_SCORER`

## Frozen retrieval

- **path:** `reports/task1/full_document_legal_field_retrieval/full_document_legal_field_retrieval_candidates.jsonl`
- **SHA256:** `3519a417cbeed46491e270f11a2b63707856cad594cda6c239e37a4a78204ab0`
- **queries:** 5,600 F1–F4
- **rows:** 2,800,000 (exactly 500/query)

## Novel candidate volume

| Full-doc cutoff | New vs pool200 | New vs K20 | Mean/query | P95/query |
|---:|---:|---:|---:|---:|
| 5 | 2,509 | 12,312 | 0.45 | 3 |
| 10 | 6,523 | 30,348 | 1.16 | 5 |
| 20 | 18,115 | 74,551 | 3.23 | 12 |
| 50 | 73,487 | 226,917 | 13.12 | 32 |
| 100 | 214,445 | 496,104 | 38.29 | 69 |
| 200 | 598,071 | 1,045,948 | 106.80 | 150 |

These counts are identity-only; no relevance labels were used to compute or select a budget.

## Neural scoring requirement

| Budget | New q-docs | Existing BGE | Need BGE inference | Existing Qwen |
|---|---:|---:|---:|---:|
| TOP10 | 6,523 | 21 | 6,502 | 0 |
| TOP20 | 18,115 | 56 | 18,059 | 0 |
| TOP50 | 73,487 | 173 | 73,314 | 0 |

The corresponding unique document counts are 2,446, 4,472, and 6,849. All full-document results resolve to canonical document IDs from the raw 8,532-document corpus, whose raw-to-vector payload chunk parity was previously established; no new chunk generation is indicated. The operational issue is scoring q-doc pairs, not document/chunk identity.

## Candidate mapping

- **valid canonical doc IDs:** all 2,800,000 frozen retrieval rows
- **canonical chunks available:** all mapped canonical documents, per prior exact raw/payload chunk parity audit
- **mapping failures:** 0 observed in frozen retrieval q-doc identities
- **new chunk generation needed:** no

## Available scorer

- **surviving BGE contract:** per-query/document `MAX(bge_score)` across chunks, as recorded by the incumbent compact BGE source.
- **closest source:** `scripts/beam/beam_task1_v3_public_adaptive_k200.py`.
- **model contract:** `models/reranker`, XLM-R sequence classifier, exact expected hashes recorded by that script.
- **aggregation:** chunk BGE score then document-level maximum.
- **execution:** explicitly CUDA-required; the surviving runner is public-K200-specific and expects raw dense-K500 plus its public artifact contract.
- **reproducible for new F1–F4 full-doc candidates:** no. No existing script materializes the new F1–F4 full-doc candidates into the exact chunk-pair input expected by this scorer while preserving the incumbent scoring provenance.
- **comparable with incumbent scores:** the model/aggregation contract is conceptually comparable, but an executable F1–F4 new-candidate path is not presently preserved.

## Selected integration budget

No budget is selected because selection requires an executable comparable scorer. Structurally, `FULLDOC_TOP10` would be the candidate-volume-minimizing choice (6,502 new BGE q-doc scorings), but it is recorded only as a non-authoritative engineering observation—not as a selected experimental budget.

## Prospective next experiment

Blocked pending exactly one prerequisite: a reproducible, provenance-locked F1–F4 BGE scoring/materialization path for new full-document candidates. Once available, the next authorized experiment should predeclare one rank-only budget, union by exact `(query_id, document_id)`, score new q-doc chunks with the incumbent BGE contract, aggregate `MAX(bge_score)`, rank all merged candidates by frozen score/tie-breaks, and compare strict OOF top-5 against recall `0.9259285714285714` and precision `0.19739285714285715`.

## Hard60 note

`HISTORICAL_HARD60_STRONG_STATUS_UNRESOLVED; KNOWN66_RETRIEVAL_SIGNAL_CONFIRMED; HARD60_RECOVERY_LOWER_BOUND_6_AT_200`

## Safe next action

`RESOLVE_ONLY_THE_REPORTED_INTEGRATION_BLOCKER`
