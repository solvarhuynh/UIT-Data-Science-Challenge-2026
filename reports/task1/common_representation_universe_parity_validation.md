# COMMON_REPRESENTATION_UNIVERSE_PARITY_VALIDATION — Task1 LegalIR

## Status

`REPRESENTATION_MISMATCH_IS_SECONDARY_LIMITATION`

**Scope:** read-only, F1–F4 only. No model fitting, inference, reranking, score imputation, or historical-artifact modification occurred.

## Canonical universe

- **queries:** 5,600 (F1–F4)
- **q-doc rows:** 114,169
- **duplicates:** 0 after exact `(query_id, document_id)` deduplication; 25,831 attempted insertions overlapped while forming `baseline_top5 ∪ candidate_rank≤20`
- **exact join identity:** yes — the candidate, baseline, original-BGE, and corrected-Qwen artifacts each contain the same 5,600 F1–F4 query IDs

The primary membership was frozen from the baseline top-5 plus canonical candidate ranks 1–20 before relevance was attached. It contains 112,000 K20 rows plus 2,169 baseline-only anchors. Availability flags were attached without score imputation: original BGE, corrected Qwen, BM25, dense/BGE-hit, word KNN, char KNN, baseline membership/rank, and canonical candidate rank where present. Candidate-source availability in the frozen universe: BM25 85,708 rows; word KNN 62,507; char KNN 60,780; adaptive-K500 74,517; 2,169 baseline-only anchors have no canonical-candidate source-rank record.

## Provenance

### BGE

- **path:** `artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl`
- **SHA256:** `7286ec481d26fc134711ecb03ceac2afe1a03bf0dabd842e1ed29ffb074e85c8`
- **rows:** 1,120,000 raw chunk-hit rows; 351,762 distinct q-doc identities
- **duplicate q-doc count:** 768,238 additional raw hit rows collapse onto an existing q-doc (expected chunk-to-document multiplicity)
- **unmatched q-doc count:** 277,914 distinct BGE q-docs are outside the frozen primary universe; 73,848 BGE q-docs join it

### Qwen

- **path:** `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl`
- **SHA256:** `65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f` (required hash matched)
- **rows:** 431,200 distinct q-doc identities
- **duplicate q-doc count:** 0
- **unmatched q-doc count:** 317,153 Qwen q-docs are outside the frozen primary universe; 114,047 Qwen q-docs join it

“Unmatched” means outside the intentionally narrow K20-plus-anchor analysis universe, not an identity failure. All joins use exact query ID plus exact document ID.

## Primary coverage matrix

Labels were attached only after membership and availability flags were frozen. There are 5,847 relevant q-doc occurrences in the primary universe.

| Coverage class | Rows | Relevant | Relevant rate | Share of relevant |
|---|---:|---:|---:|---:|
| BGE+Qwen | 73,729 | 5,774 | 7.83% | 98.75% |
| BGE only | 119 | 2 | 1.68% | 0.03% |
| Qwen only | 40,318 | 70 | 0.17% | 1.20% |
| Neither | 3 | 1 | 33.33% | 0.02% |

### Candidate-rank coverage

| Candidate band | Rows | BGE covered | Qwen covered | Both | Neither |
|---|---:|---:|---:|---:|---:|
| 1–5 | 28,000 | 21,601 | 28,000 | 21,601 | 0 |
| 6–10 | 28,000 | 14,262 | 28,000 | 14,262 | 0 |
| 11–20 | 56,000 | 36,148 | 56,000 | 36,148 | 0 |

### Fold coverage

| Fold | Rows | BGE covered | Qwen covered | Both | Neither |
|---:|---:|---:|---:|---:|---:|
| F1 | 28,520 | 18,364 | 28,494 | 18,338 | 0 |
| F2 | 28,555 | 18,479 | 28,526 | 18,450 | 1 |
| F3 | 28,536 | 18,544 | 28,508 | 18,516 | 0 |
| F4 | 28,558 | 18,461 | 28,520 | 18,425 | 2 |

The raw BGE missing-row percentage is therefore real, but Qwen covers every canonical rank-1–20 candidate row. The missing BGE component is not equivalent to lack of neural evidence.

## Missed relevant candidates

Only relevant q-docs not already in the baseline top-5 are included below. There are 320 such actionable in-universe occurrences; 264 (82.5%) have both signals and 56 (17.5%) are Qwen-only. None lack both signals.

| Band | Relevant | Both | BGE only | Qwen only | Neither |
|---|---:|---:|---:|---:|---:|
| 1–5 candidate universe | 130 | 117 | 0 | 13 | 0 |
| 6–10 | 57 | 37 | 0 | 20 | 0 |
| 11–20 | 133 | 110 | 0 | 23 | 0 |

Among the 505 queries where baseline misses at least one gold document, the same 320 actionable candidate occurrences split 264 BGE+Qwen and 56 Qwen-only. Thus the evidence is not concentrated in already-correct queries.

## Baseline parity

- **slots:** 28,000
- **BGE covered:** 27,392
- **Qwen covered:** 27,878
- **both:** 27,273
- **neither:** 3
- **relevant slots missing BGE:** 15
- **relevant slots missing Qwen:** 3

The incumbent does not materially rely on documents that are absent from all neural-score universes. The 608 baseline slots without BGE are mostly non-relevant; Qwen nevertheless covers 605 of them.

## K21–77 diagnostic

This section is separate from the primary K20-plus-anchor result.

- **rows:** 319,200
- **relevant:** 179
- **BGE coverage:** 157,057 rows (134 relevant)
- **Qwen coverage:** 319,200 rows (179 relevant)
- **both:** 157,057 rows (134 relevant)
- **neither:** 0 rows (0 relevant)

Deep-candidate rescue differs from K20 because BGE reaches only 134/179 relevant occurrences in ranks 21–77, whereas corrected Qwen reaches all 179. This constrains BGE-dependent comparators, but does not create a no-signal region for Qwen-based analysis.

## Scientific interpretation

Actionable missed golds are **not** mostly missing neural evidence. Every one of the 320 relevant candidates outside incumbent top-5 has corrected Qwen; 264 already have both BGE and Qwen. In the key near-top 6–20 bands, 147/190 (77.4%) have both signals and the remaining 43/190 (22.6%) still have Qwen. The primary failure mode is therefore **SIGNAL_AVAILABLE_BUT_RANKER_FAILS_TO_USE_IT**, not `NO_SIGNAL_AVAILABLE`.

The BGE-specific representation gap touches 56/320 actionable in-universe missed relevant occurrences: 17.5% of that set, or **0.94 percentage points of total F1–F4 gold recall mass** (56/5,970). It is too small, and too completely covered by Qwen, to plausibly explain the month-long plateau on its own.

## Root-cause conclusion

Representation universes are not identical: original BGE covers fewer expanded candidate rows than corrected Qwen, and this difference grows in K21–77. However, on scientifically actionable F1–F4 missed relevant candidates, 100% retain Qwen and 82.5% retain both neural signals; no actionable missed relevant candidate lacks both. The mismatch is therefore a secondary limitation for fair BGE-dependent fusion/comparator design, not a material shared root cause of the incumbent’s score stagnation. The evidence favors fragile top-5 selection among already-scored plausible documents, with a separate residual retrieval ceiling, over a common representation failure.

## Safe next action

`MOVE_TO_RETRIEVAL_REPRESENTATION_OR_STOP_MODEL_SEARCH`

STOP.
