# Qwen3-VL-Reranker-2B professor audit evidence bundle

Status: `READY_FOR_PROFESSOR_AUDIT_WITH_DOCUMENTED_LIMITATION`.

## Evidence hierarchy

### LEVEL 1 — ORIGINAL/RAW EXECUTION EVIDENCE

- Pre-fix raw replay: `qwen3vl2b_micro_replay_prefix_raw.json` (24 q-docs, 72 chunks). PATH_A supplied actual Vietnamese query text; PATH_B supplied query ID `13844`. PATH_A exactly matches the recorded GOOD regime and PATH_B exactly matches the recorded BAD-full regime at the 24-document comparison level.
- Post-fix raw replay: `qwen3vl2b_micro_replay_postfix_raw.json` (24 q-docs, 72 chunks). PATH_A and PATH_B have exact input parity and equal scores for all 72 chunks; both exactly match the recorded GOOD regime at the document level.
- Fresh full-run state: 1,293,198 chunks scored, zero recovered, zero skipped, zero errors. This excludes checkpoint contamination.
- Scientific evaluation: prediction SHA `65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f`; F1–F4 pooled Recall `0.799797619047619`, pooled Precision `0.16939285714285715`; Fold0 and public labels were not used.
- Official parity raw output: 72 rows, chunk Pearson `0.9997522649508032`, Spearman `0.999244013387365`, and zero material rank reversals (the three observed rank changes are zero-margin ties).

## Final prediction chain of custody

The fresh run-state records `/workspace/p13/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-full-querytext-fixed/predictions.jsonl` with SHA256 `65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f`. The local artifact evaluated by the scientific report, `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl`, has the same SHA256 (27,365,076 bytes; 5,600 rows), and the scientific report records that same SHA. Thus metric integrity is `METRIC_BOUND_TO_CORRECT_BYTES_BY_SHA` and source/destination byte identity is established by SHA. The historical transfer command is not preserved: classification `BYTES_IDENTICAL_COPY_EVENT_UNRECOVERED`, not a claimed proven copy event.

### LEVEL 2 — DERIVED AUDIT TABLES

`qwen3vl2b_same_item_pre_post_72rows.jsonl` is a convenience-only, deterministic normalization of the two Level-1 replay JSON files. It contains 72 rows in string-key order, has passed 72/72 exact semantic round-trip validation, and is not independent historical evidence.

## Root-cause conclusion

The raw micro replay proves execution-level validation/full divergence before the fix: PATH_A prepared actual Vietnamese query text while PATH_B constructed the query-ID string. After the fix, both paths prepare the same Vietnamese query text and produce identical scores across all 72 frozen chunks. Classification: `PROVEN_BY_RAW_MICRO_REPLAY`.

## Historical-source limitation

The exact historical pre-fix Python source snapshot was not preserved. A real historical source diff is `NOT_AVAILABLE`, and source-code-level validation-escape proof remains `UNPROVEN`. This limitation is explicit: the bundle is audit-ready on its raw execution, full-run, scientific, and official-parity evidence, but is not fully provenance-complete at source-snapshot level.
