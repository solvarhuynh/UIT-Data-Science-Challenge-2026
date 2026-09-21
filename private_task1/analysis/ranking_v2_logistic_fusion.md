# CROSS_FITTED_LOGISTIC_FUSION

## Contracts

- CPU-only; existing F1?F4 K20 and BGE score artifacts; no GPU/Modal/retrieval rerun, Fold0, Private labels, leaderboard data, or weight search.
- Features: `dense_rr`, `bm25_rr`, `knn_rr`, `bge_rr`, `dense_present`, `bm25_present`, `knn_present`, `retrieval_support_count`, `bge_score_percentile`, `retrieval_rr_sum`, `bge_support_interaction`.
- BGE percentile: rank-normalized per query, rank 1 = 1.0 and rank 20 = 0.0.
- Model: `StandardScaler -> LogisticRegression(class_weight=balanced, solver=lbfgs, C=1.0)`; scaler fit inside each fold.

## OOF results

- Validation: `5600/5600`; K20 q-docs: `112000/112000`; Fold0: `0`.
- Baseline Recall@5: `0.924514880952381`.
- V2 OOF Recall@5: `0.927312500000000`.
- Delta: `+0.002797619047619`.
| Fold | Baseline | V2 OOF | Delta |
|---|---:|---:|---:|
| F1 | 0.929821429 | 0.928511905 | -0.001309524 |
| F2 | 0.928452381 | 0.935833333 | +0.007380952 |
| F3 | 0.921785714 | 0.921071429 | -0.000714286 |
| F4 | 0.918000000 | 0.923833333 | +0.005833333 |

## Paired comparison

- Better: `43`; worse: `23`; same: `5534`.
- CURRENT ranking-miss rescued: `28/187`.
- New misses created: `16`; net rescue: `12`.

## Bootstrap

- Resamples: `10000`; seed: `20260920`.
- Mean delta: `+0.002797619047619`.
- 95% CI: `[+0.000416666666667, +0.005238095238095]`.

## Gate

| Criterion | Result |
|---|---|
| OOF > baseline | PASS |
| delta >= 0.002 | PASS |
| at least 3/4 folds non-decreasing | FAIL |
| no fold drop > 0.003 | PASS |
| bootstrap mean > 0 | PASS |
| **V2_GATE** | **FAIL** |


## Private application

- V2 failed its pre-registered gate; no Private fit/application was run.
- Private submission created: `NO`.
- Current submission preserved: `YES`.

## Final

- V2_GATE: **FAIL**.
- NEXT_ACTION: **RUN_PROCESSED_PV1_REPAIR**.
- GPU runs: `0`; Modal inference runs: `0`.
- No automatic upload/submission performed.

