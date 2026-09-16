# Task1 LegalIR — Foundational System Root-Cause Audit

**Audit type:** read-only forensic audit  
**Scope:** persisted artifacts and source code only; no training, inference, model change, or historical-artifact modification  
**Population:** strict OOF folds F1–F4 (5,600 queries), unless explicitly noted  
**Verdict:** `NO_SHARED_BUG_FOUND_METHOD_CEILING_MORE_LIKELY`

## Executive conclusion

The strongest foundational explanation for the month-long score stagnation is: **the remaining errors are predominantly a ranking/selection ceiling inside an otherwise compatible candidate-and-evidence pipeline, rather than one shared data, identity, corpus, fold, or metric defect.**

No single shared bug is confirmed. There is real provenance debt (some historical producer code is absent or drifted) and a substantial BGE-coverage asymmetry in the expanded shortlist, but the available labels show that this asymmetry contains very little of the relevant mass. Treat it as a reproducibility and future-experiment risk, not as the established cause of the score plateau.

## Dependency DAG

```text
raw train.json ──> strict_cv_v2/folds.json ──> OOF train/evaluate population
      │                                              │
      └──> gold answers ─────────────────────────────┤
                                                     ├──> shared LegalIR evaluator ──> recall / precision
raw selected-contexts ──> raw chunks ──> vector payloads ──> retrieval candidates (union ≤200)
                                                        │                 │
                                                        │                 └──> shortlist evidence (K20 + baseline top-5)
                                                        │                                      │
                                                        ├──> original BGE top-200 evidence ────┼──> baseline top-5
                                                        │                                      │        │
                                                        └──> Qwen K77 universe (partial join) ─┼──> direct / residual experiments
                                                                                               │
baseline top-5 ───────────────────────────────────────────────────────────────────────────────┘
```

The critical architectural distinction is that the union shortlist is broader than the persisted original-BGE universe. It is not evidence, by itself, that candidate rows were created after BGE or that a bug occurred.

## Identity, labels, corpus, and chunk ancestry

| Audit | Evidence | Finding | Assessment |
|---|---:|---|---|
| Train query identity | `train.json`: 7,000 unique query IDs | Fold manifest has exactly 7,000 IDs; no missing, extra, or duplicate ID | clean |
| Fold partition | 5 folds × 1,400 queries | C2 F1–F4 population is 5,600 queries | clean |
| Gold-to-raw corpus parity | 7,637 gold occurrences / 3,105 distinct gold doc IDs | Every gold ID exists in the 8,532 raw selected-context documents | clean |
| Raw corpus-to-vector payload parity | 1,270,356 chunk keys | Exact key and text match; no raw-only, payload-only, or duplicate chunk key | clean |
| Vector manifest “20,865 documents” | manifest field | This is a windowed-document count, not a conflict with 8,532 canonical document IDs | not a mismatch |
| Candidate construction | full candidate oracle recall `0.992471` | Four-source capped union; construction report declares `no_gold_used_in_construction=true` | high-quality candidate pool, not perfect |

The apparent 8,532-vs-20,865 corpus count discrepancy is resolved by the vector payload parity report and must not be carried forward as a corpus-integrity finding.

## BGE / candidate / baseline coverage audit

The persisted original BGE compact source covers F1–F4 only. Counts below intentionally exclude F0.

| Shortlist band | Candidate rows | Without original BGE doc score | Relevant occurrences | Relevant occurrences without BGE |
|---|---:|---:|---:|---:|
| Union ranks 1–5 | 28,000 | 6,399 | 3,999 | 17 |
| Union ranks 6–10 | 28,000 | 13,738 | 1,209 | 22 |
| Union ranks 11–20 | 56,000 | 19,852 | 597 | 24 |
| Union ranks 21–25 | 2,169 | 332 | 42 | 8 |
| **Total** | **114,169** | **40,321 (35.32%)** | **5,847** | **71 (1.21%)** |

Baseline top-5 has 28,000 output slots, of which 608 lack a document match in the persisted original BGE source; only 15 of those 608 are gold occurrences. Therefore the BGE coverage asymmetry is material for representation consistency, but the label evidence does **not** support it as the dominant explanation for baseline recall or the observed plateau.

The separate Qwen K77 score universe is also incomplete when joined to the Workflow-A candidate universe (the prior B2a audit found 122,694 Workflow-A rows without Qwen score and 202,132 Qwen rows without Workflow-A score). This is a hard limitation on experiments that use that join, but not evidence that the baseline scorer or corpus is corrupted.

## Metric and fold audit

`src/udsc2026/evaluation/legal_ir_recovery.py` evaluates set overlap between gold documents and the first five returned IDs, rejects duplicate predictions, checks coverage, and reports recall/precision from actual returned IDs. The baseline emits exactly five IDs per query, so its precision denominator is five in practice.

The stored baseline OOF result is internally consistent across F1–F4: recall `0.9259285714`, precision `0.1973928571`; fold recalls are `0.9363690`, `0.9283333`, `0.9194048`, and `0.9196071`. No stale-fold or metric-definition defect was found in the evaluator path inspected. This does not certify every historical notebook or report, only the shared evaluator and stored strict-OOF baseline path.

## Why the direct experiments do not identify a common system defect

The direct document-relevance experiment observed 5,847 relevant and 108,322 non-relevant shortlist rows, with strong learned-score separation. Its missed-gold anatomy was 253 outside the candidate pool and 334 at known candidate ranks 6–22. The neural follow-up recovered only 58 of those 334 near-top occurrences and lost more than it gained overall:

- Direct relevance: recall `0.9241577381` vs baseline `0.9259285714` (delta `-0.0017708333`).
- Direct neural: recall `0.9209880952` vs baseline (delta `-0.0049404762`); 28 improved versus 64 harmed queries.
- The full candidate oracle is `0.992471`; the residual gap is principally selecting the right five among plausible documents, plus 253 known misses beyond the candidate pool.

This pattern is consistent with a selection/ranking ceiling and heterogeneous residual failures. It is inconsistent with a broad corpus, label, query-identity, or evaluator fault, which would be expected to create pervasive parity or fold inconsistencies.

## Historical drift and reproducibility debt

| Item | Evidence | Risk | Interpretation |
|---|---|---|---|
| V3A residual policy | historical source hash `b430…` differs from current `train_residual_policy.py` hash `9378…` | high for V3A reproduction | confirmed source drift; not proof of a shared scoring bug |
| Baseline producer | manifest references `scripts/beam/build_public093_strict_oof.py`, absent now | high for reconstruction | baseline output remains evaluable, producer reproduction is weakened |
| BGE compact source | persisted F1–F4 source available and hash-recorded | medium | enough for coverage audit, but not a complete regeneration recipe |
| Qwen K77 universe | partial overlap with Workflow-A universe | high for fusion claims | representation/universe boundary must be declared in every experiment |

These are governance and reproducibility defects. They increase the chance of misleading future comparisons, but the evidence does not connect them causally to every failed experiment or the baseline score itself.

## Ranked high-blast-radius findings

| Rank | Finding | Blast radius | Evidence strength | Status |
|---:|---|---|---|---|
| 1 | Candidate shortlist and neural-score universes are not identical | residual, fusion, and direct-rerank experiments | high | real constraint; not root-cause confirmed |
| 2 | Historical producer source is missing/drifted | reproducibility of baseline/V3A comparisons | high | confirmed provenance debt |
| 3 | Candidate oracle remains below 1.0 | any top-5 reranker | high | method ceiling contributor |
| 4 | Near-top relevant documents compete with many plausible negatives | final selection/ranking | high | method ceiling contributor |
| 5 | Qwen K77 partial join | Qwen-based fusion only | high | experiment-specific constraint |
| 6 | Corpus/vector mismatch hypothesis | all retrieval and scoring paths | disproved | do not pursue |
| 7 | Fold/metric mismatch hypothesis | all OOF claims | not found | do not pursue without new evidence |

## Root-cause ranking (maximum seven)

1. **Top-5 selection ceiling among high-plausibility candidates** — strongest support from negative direct reranks despite score separation.
2. **Residual candidate miss ceiling** — 253 relevant occurrences in the direct experiment lie outside the evaluated candidate pool; candidate oracle is not 1.0.
3. **Representation-universe mismatch for extended experiments** — BGE and Qwen coverage are incomplete on the expanded union, constraining fair comparison and fusion.
4. **Historical source/provenance drift** — impairs exact reproduction and attribution, particularly V3A and baseline production.
5. **Heterogeneous fold/query residuals** — neural intervention is negative in three folds and only trivially positive in one, arguing against one universal fix.
6. **Corpus mapping defect** — rejected by exact raw-to-vector payload parity.
7. **Shared evaluator/fold identity defect** — not found in the audited path.

## Safe next action

Run **one frozen, label-free representation-parity validation** before any new model work: regenerate or locate a single common candidate universe for F1–F4, attach availability flags for BGE and Qwen without imputing scores, and measure non-gold coverage, gold coverage, and top-5 parity against the stored baseline. The run should be read-only with respect to existing artifacts, record all source hashes, and stop if the universe cannot be made explicit. It validates the remaining architectural risk without claiming that it is already the root cause.

## Evidence locations

- `artifacts/task1/recovery_096/vector_payloads_vs_raw_chunks_b4/report.json`
- `artifacts/task1/recovery_096/candidate_refs_full_report.json`
- `artifacts/task1/recovery_096/v3_residual/shortlist_report.json`
- `artifacts/task1/recovery_096/baseline_093_oof/report.json`
- `reports/task1/direct_document_relevance_top5/direct_document_relevance_failure_attribution.json`
- `reports/task1/direct_document_neural_signal_top5/direct_neural_evaluation.json`
- `reports/task1/workflow_b/tv2/b2a/reports/b2a_qwen3vl2b_auxiliary_fusion_crossfit.json`
