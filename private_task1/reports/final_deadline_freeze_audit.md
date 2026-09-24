# Final Deadline Freeze Audit

Status: **PASS**

Incumbent giữ nguyên: `GUARDED_DIRECT_K20_RESIDUAL / V2_NESTED`  
Private observed score: `0.9174`

## Artifact gates

- Submission ZIP: SHA256 khớp expected (`cb2b65b46068fabc5d8d6601add88ec8205dbab60a4c5536af210df3a5cf87fd`).
- Final model: SHA256 khớp expected (`4b962c56570e6315876aeb6eea26d8877a8007e636d83c70ac2e84a13c2a175c`).
- JSON validator: **PASS**.
- ZIP validator: **PASS**; ZIP có đúng một member `submission.json`.

## Submission structure

- Queries: `2080/2080`; question-ID set khớp input private.
- Documents/query: đúng `5`.
- Missing queries: `0`; extra queries: `0`.
- Duplicate documents trong query: `0`.
- Null/non-string document IDs: `0`.
- Payload trong ZIP byte-level JSON parse tương đương file submission JSON.

## Frozen deployment contract

- Selected policy: `V2_NESTED`.
- OOF Recall: `0.9300505952380952`.
- OOF delta: `+0.004464285714285698`.
- Fold0 scientific use: `false`.
- Private labels used: `false`.
- Private changed queries: `36`; Private Top1–3 changed: `0`.
- Final model fit gate: **PASS**.

## Freeze and lineage

Targeted git/status checks reported no changes for the incumbent paths. The failed challenger artifacts remain present and are preserved as evidence:

- `CORRECTED_QWEN_SINGLE_RESCUE`: **CLOSED**.
- `DIRECT_K20_PLUS_ANCHORS_DOCUMENT_LISTWISE_TOP5`: **CLOSED**.

This audit performed no training, refit, inference, GPU/Modal run, submission rebuild, prediction change, or automatic submission.

## Final decision

`READY_TO_KEEP_CURRENT_SUBMISSION`

Next action: `NO_MORE_EXPERIMENTS`.
