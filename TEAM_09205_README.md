# Team 0.92054 handoff — project-root layout

Extract this package directly into the repository root `D:\udsc2026`.

## Verified official result

- Recall: `0.920544597`
- Precision: `0.196661609`
- Official scoring artifact: `private_task1/reports/team_09205/scoring_result_09205.zip`
- New submission artifact: `private_task1/submissions/team_09205/submission_private_constrained_dual_anchor_rrf_09205.zip`
- SHA256: `aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae`

## Corrected repository paths

- Repo root: `D:\udsc2026`
- Private input: `private_task1/input/private-official.json`
- Guarded anchor: `private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.zip`
- Team 0.92054 submissions: `private_task1/submissions/team_09205/`
- Reproduction scripts: `private_task1/scripts/analysis/team_09205/`
- Reports / official scoring result: `private_task1/reports/team_09205/`

The corrected builders auto-detect the repository root and no longer assume the old handoff layout `outputs/task1/private` or `data/task1`.

## Important scientific correction

The original handoff builders loaded `warmup.json` and `train.json` only to create an "oracle" diagnostic on Private questions with matching normalized text. That diagnostic is not required to build the submission and must not be used for model selection. The corrected builders included here do **not** load labels; they validate only the Private query IDs and output structure.

The official score file is the source of truth for the 0.92054 result. The original README's phrase `1819.92 correct queries` is not a literal count and should not be used analytically.

## What changed from 0.9198 to 0.92054

Comparing the two supplied submission files:

- 1,464 / 2,080 queries changed order.
- 1,400 of those had exactly the same Top-5 set, only reordered.
- Only 64 / 2,080 queries changed Top-5 membership.

Because Task1's primary Recall/Precision are set-based, only those 64 membership-changing queries can explain the score difference between the two supplied submissions. This is the most useful clue for the next research step: optimize **membership decisions**, not rank order inside an unchanged Top-5 set.

## Rebuilding

From the repo root:

```powershell
python private_task1/scripts/analysis/team_09205/build_constrained_dual_anchor_rrf_submission.py --v2 <PATH_TO_SUBMISSION_V2_ZIP>
```

If exactly one `submission_v2.zip` exists in the repository, `--v2` can be omitted.

The guarded anchor defaults to:

`private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.zip`

No label file is read by the corrected builder.
