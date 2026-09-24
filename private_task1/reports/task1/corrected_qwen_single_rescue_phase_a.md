# Corrected Qwen single-rescue — Phase A

CPU-only structural/cache audit. No labels, Recall, Precision, inference, GPU, Modal, policy selection, or submission creation.

## Canonical artifact

- Path: `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl`.
- SHA256: `65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f`; provenance gate: **PASS**.
- Scope: `5600` queries, `431200` q-doc rows, K77 (77/query); score field `document_scores[].score`.
- Model/revision: `Qwen/Qwen3-VL-Reranker-2B` / `4bd860ac4f15ad1897a214615cccc700f8f71818`; selector `true_s2_bm25_within_document_v2`; aggregation `MAX`.
- The newer 32-shard full-doc residual artifact is complete but intentionally does not cover the current PV1 K20 universe; it is not silently used for rescue joins.

## Coverage by exact `(query_id, document_id)` join

- Validation: `111816/112000` (99.835714%), full queries `5443`, partial `157`, zero `0`.
- Private: `0/41600` (0.000000%), full queries `0`, partial `0`, zero `2080`; cache **ABSENT**.
- Validation rank bands: `{'rank_1_5': {'total': 28000, 'covered': 27975, 'missing': 25, 'coverage_pct': 99.91071428571429}, 'rank_6_10': {'total': 28000, 'covered': 27958, 'missing': 42, 'coverage_pct': 99.85}, 'rank_11_20': {'total': 56000, 'covered': 55883, 'missing': 117, 'coverage_pct': 99.79107142857143}}`.
- Private rank bands: `{'rank_1_5': {'total': 10400, 'covered': 0, 'missing': 10400, 'coverage_pct': 0.0}, 'rank_6_10': {'total': 10400, 'covered': 0, 'missing': 10400, 'coverage_pct': 0.0}, 'rank_11_20': {'total': 20800, 'covered': 0, 'missing': 20800, 'coverage_pct': 0.0}}`.

## Structural rescue feasibility

- Validation queries with at least one eligible rank6–20 cached-Qwen candidate outside incumbent Top5: `5600`.
- Private queries with at least one eligible candidate: `0`.
- No candidate was selected and no threshold/policy was applied.

## Decision: `CACHE_PARTIAL`

Next action: Do not open rescue policy; identify and fill the missing cached Qwen identities before any Phase B.
