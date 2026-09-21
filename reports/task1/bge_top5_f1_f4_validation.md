# BGE TOP5 F1–F4 validation

Status: **BGE_TOP5_F1_F4_POLICY_EVALUATION_COMPLETE**

This is CPU-only replay over the downloaded BGE artifact. No Modal, GPU, Qwen inference, public labels, or Fold0 was used. F1–F4 train gold was read only after prediction construction, for retrospective metrics.

## Frozen provenance

- Raw output: `artifacts/task1/qwen_to_bge_minimal/bge_top5_production_complete.jsonl`
- Raw SHA256: `a72b243e9895b2973b63bc16a48c150982346cc1ba384b77ebf82453f61c5fd7`
- Canonical frozen copy: `artifacts/task1/qwen_to_bge_minimal/bge_top5_production_canonical.jsonl`
- Canonical SHA256: `a72b243e9895b2973b63bc16a48c150982346cc1ba384b77ebf82453f61c5fd7`
- Rows/q-docs: **25902**; F1–F4 queries: **5600**
- Model: `BAAI/bge-reranker-v2-m3`; base revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; selector: `true_s2_bm25_within_document_v2`; aggregation: `MAX`; max length: `512`
- Old BGE inventory: **not used** because its persisted score provenance is marked `MODEL_REVISION_UNPROVEN_FOR_PERSISTED_SCORE_ARTIFACT`.

## Baseline anchor

- Recall: `0.925928571428571`; precision: `0.197392857142857`
- Fold recall: F1=0.936369047619048, F2=0.928333333333333, F3=0.919404761904762, F4=0.919607142857143

## Evaluation rule

Positions 1–3 are locked. A new BGE-scored q-doc can replace slot 4/5 only when the dropped slot also has a current production score; queries without that current anchor stay unchanged. Qwen score/rank is never read. Historical Step4 weights are replayed with per-query min–max normalization of current FT/Base/RRF features.

## Selection

- Candidate policy: **RETURN_TO_09411**
- Production authorized: **NO**
- Policy result JSON: `artifacts/task1/qwen_to_bge_minimal/bge_top5_f1_f4_policy_results.json`
- Predictions: `artifacts/task1/qwen_to_bge_minimal/bge_top5_f1_f4_predictions.jsonl`

A policy would need positive pooled recall delta, non-negative recall delta on every F1–F4 fold, and precision delta no worse than −0.001. The incumbent 0.9411 remains the safe submission anchor unless a separately authorized handoff says otherwise.
