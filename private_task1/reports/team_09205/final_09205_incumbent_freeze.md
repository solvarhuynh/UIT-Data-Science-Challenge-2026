# Final 09205 Incumbent Freeze

- Incumbent: `CONSTRAINED_DUAL_ANCHOR_RRF`
- Observed Private score: `0.920544597`
- Submission: `private_task1/submissions/team_09205/submission_private_constrained_dual_anchor_rrf_09205.zip`
- SHA256 gate: **PASS** (`aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae`)
- Canonical validator: **PASS** — `2080/2080` queries, exactly `5` unique documents/query, missing/extra/duplicate/null = `0`.

## V2 anchor status

The exact V2 anchor is `MISSING_AND_NOT_IDENTIFIABLE`. Recovery evidence recorded 189 fingerprint queries, 48 filesystem candidates, 9 structural candidates, 0 rank-1 exact candidates, and 0 git historical candidates. The inverse audit found `0/2080` unique, `2080/2080` ambiguous, and `0/2080` no-solution queries.

No fabricated anchor was created, and the incumbent prediction artifact was not altered. The forensic recovery artifacts remain preserved in this directory.

## Closed branches and constraints

- `MISSING_V2_ANCHOR_RECOVERY = CLOSED`
- `DUAL_ANCHOR_SELECTOR = BLOCKED_NO_EXACT_V2`
- `QWEN_SINGLE_RESCUE = CLOSED`
- `DIRECT_LISTWISE = CLOSED`
- Private labels: not used
- Fold0: not used
- GPU/Modal: `0`

## Final decision

`FINAL_FREEZE_GATE = PASS`

Keep the 0.920544597 submission. Reopen only if a teammate supplies the exact V2 artifact or the exact generator together with its inputs.
