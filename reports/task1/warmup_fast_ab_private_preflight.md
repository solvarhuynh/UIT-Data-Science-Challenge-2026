# Warmup fast A/B private preflight

Status: **WARMUP_B_BLOCKED_NEW_GPU_REQUIRED** for a complete 500-query A/B
comparison. This was a CPU-only audit; no GPU, Modal, Qwen inference, public
scoring, label tuning, or submission write was performed.

## Frozen inputs and order of operations

Predictions were loaded and validated first. Only after the prediction maps
were frozen was `warmup.json` read for gold metrics. No warmup answer values
were sent to any model or external service.

- Warmup: `data/task1/warmup.json`
  - SHA256: `fadfbcab2923c085980239396d5564c16c232714f42004a36ed2a3d93962c0f9`
  - 500 query IDs
  - schema: top-level object; each value has `question: str` and
    `answer: list[str]`
  - answer cardinalities: 463 queries with 1 answer, 33 with 2, 3 with 3,
    and 1 with 4
- A incumbent: `artifacts/task1/handoff/step4_bge_handoff/public/submission_step4_calibrated.json`
  - SHA256: `b655fdf5444f1809e71a9f643c065e3ccf0c5e5bd1870113fd7c4e727ad1a8ff`
  - 1,000 query IDs; five predicted documents per query
- B frozen current BGE TOP5 predictions:
  `artifacts/task1/qwen_to_bge_minimal/bge_top5_f1_f4_predictions.jsonl`
  - SHA256: `a786c8a6e1ac7235ef6b24ad553e8a88d4a74eaaeefe28adc8222a6c15fa7b0b`
  - 5,600 query IDs; five predicted documents per query
  - score-source artifact:
    `artifacts/task1/qwen_to_bge_minimal/bge_top5_production_canonical.jsonl`
  - score-source SHA256:
    `a72b243e9895b2973b63bc16a48c150982346cc1ba384b77ebf82453f61c5fd7`

The B prediction artifact is the existing frozen BGE TOP5 policy output; no
new ranking policy was fitted during this audit.

## Coverage and paired-comparison gate

Direct `(query_id)` coverage on the 500 warmup IDs:

| source | available | coverage |
|---|---:|---:|
| A incumbent | 52 | 10.4% |
| B current BGE TOP5 | 264 | 52.8% |
| both A and B | 0 | 0.0% |
| neither | 184 | 36.8% |

A and B also have zero query-ID overlap outside warmup (their artifacts are
from different query populations). Therefore paired A/B outcomes, net query
gain, and net relevant-document gain are **UNAVAILABLE**, not zero.

The remaining warmup queries are not covered by the frozen B artifact. Applying
B to them would require new inference/materialization, so the complete
holdout comparison is blocked under the no-new-inference constraint.

## Retrospective metrics on available subsets only

Metrics use the warmup `answer` document IDs and the frozen five-document
prediction lists. They are descriptive and are not a valid A-vs-B model
selection comparison because the query subsets are disjoint.

| source/subset | Recall@5 | Precision@5 | query hit rate | mean relevant docs/query | relevant hits |
|---|---:|---:|---:|---:|---:|
| A, 52 available queries | 1.0000000000 | 0.2115384615 | 1.0000000000 | 1.0576923077 | 55 |
| B, 264 available queries | 0.9084595960 | 0.1931818182 | 0.9280303030 | 1.0909090909 | 255 |

Available-subset caveat: A's perfect score is based on only 52 queries and
must not be generalized to all 500 warmup queries. B's result is based on 264
queries and likewise does not establish full-warmup performance.

## Decision

- Complete 500-query A/B diagnostic: **BLOCKED**.
- B status: `WARMUP_B_BLOCKED_NEW_GPU_REQUIRED` for the 236 uncovered warmup
  IDs.
- No model or submission selection is authorized from these disjoint
  descriptive subsets.
- No Gemini overlay artifact was used as B evidence; its provenance does not
  establish that it is the current BGE TOP5 pipeline.

