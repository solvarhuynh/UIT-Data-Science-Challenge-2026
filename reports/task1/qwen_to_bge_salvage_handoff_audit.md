# Qwen → BGE salvage handoff audit

Status: **BLOCKED — zero query-ID intersection; no inference run**

## Finding

The Step4 public calibrated submission contains `1000` query IDs. The FULLDOC Qwen merged artifact contains `5600` query IDs. Their exact string identity intersection is **0**. This is a query-namespace mismatch, not evidence of zero candidate novelty and not evidence that Qwen candidates are safe to score against Step4.

- Step4 public query IDs absent from Qwen: `1000`
- Qwen query IDs absent from Step4 public: `5600`
- Example Step4 IDs absent from Qwen: `136, 138, 144, 208, 212, 246, 362, 542, 786, 806`
- Example Qwen IDs absent from Step4: `6, 36, 52, 76, 86, 96, 180, 196, 214, 222`

The join is therefore deliberately not fabricated. TOP5/TOP10/TOP20 novelty, BGE-score coverage, missing inference pairs, and budget/workload estimates are **NOT COMPUTABLE** until the correct query-ID mapping or matching Qwen artifact is identified.

## Verified source artifacts

- Step4 anchor: `D:\udsc2026\artifacts\task1\handoff\step4_bge_handoff\public\submission_step4_calibrated.json`
- Step4 anchor population/rows: `public`, `1000/1000`
- Step4 anchor SHA256: `b655fdf5444f1809e71a9f643c065e3ccf0c5e5bd1870113fd7c4e727ad1a8ff`
- Claimed Step4 metric: Macro Recall `0.9411`, Macro Precision `0.2022`
- Step4 candidate union: `D:\udsc2026\artifacts\task1\handoff\step4_bge_handoff\candidates\public_candidate_union.jsonl`
- Step4 candidate-union SHA256: `c23605b28d634cc4e0df6cc313eabcf4a6a9122a036e652691666570b54559ce`
- Step4 candidate pairs: `200000` (`1000` queries × `200` cap)
- Retained BGE swap-log pairs: `1110`; raw `scores/raw_pair_bge_scores.npy` is explicitly missing.
- Qwen merged artifact: `D:\udsc2026\artifacts\task1\full_doc_qwen_optimized_merged\full_doc_top200_qwen_optimized_32shards.jsonl`
- Qwen expected SHA256: `25cc4b7dfb5919a47f927c545927e775c11997790984b519e2b74a71abce0415`
- Qwen observed SHA256: `25cc4b7dfb5919a47f927c545927e775c11997790984b519e2b74a71abce0415`
- Qwen SHA match: `PASS`
- Qwen rows/unique q-docs: `598192/598192`
- Qwen queries/unique documents: `5600/8261`
- Qwen duplicate identities/non-finite scores: `0/0`
- Missing handoff artifact: `D:\udsc2026\artifacts\task1\handoff\step4_bge_handoff\predictions\f1_f4_final_top5.json` is explicitly documented as MISSING.

## Safety result

- Worklist: `D:\udsc2026\artifacts\task1\qwen_to_bge_salvage\missing_bge_score_candidates.jsonl` (empty by design; 0 rows)
- BGE inference runs: `0`
- GPU/Modal/Qwen runs: `0/0/0`
- Step4 handoff and Qwen merged artifact: unchanged
- Final Top5/submission: unchanged

Do **not** run BGE scoring on this empty worklist as a production action. First resolve the namespace mapping, then repeat this audit and only score the proven missing `(query_id, document_id)` pairs. Qwen scores remain provenance-only and must never be used for final ranking.
