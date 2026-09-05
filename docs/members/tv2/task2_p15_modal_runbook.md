# Task 2 P15 — Modal continuation runbook

P15 keeps the registered model `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2`.
It starts from the P14 LoRA adapter, trains on 781 previously untouched labeled
questions, gates on 215 disjoint questions, and fits all 996 usable questions
only if the gate improves. Retrieval and parent cross-encoder scoring are
reused from P14; they are not run again.

## Safety contract

- `Run` performs a worst-shape CUDA smoke test before training.
- A prior `Smoke` result is reused only when source/input hashes match and the
  next GPU has at least as much VRAM.
- The full fit starts only if dev METEOR improves by at least `0.003` and
  dev ROUGE-L does not fall by more than `0.015`.
- Official local scoring requires the bundled WordNet and OMW-1.4 corpora; no
  fallback metric is accepted.
- Every archive, training contract, result, and submission is hash-checked.
- A failed/rejected P15 run does not overwrite the P14 submission.

## Switch to the intended Modal account

From PowerShell at the repository root:

```powershell
.\scripts\cloud\run_task2_p15_modal.ps1 -Stage Setup
```

Complete Modal's browser login with the intended legitimate account/workspace.
Confirm the active workspace in the Modal dashboard before starting a GPU job.

## Preflight and upload

```powershell
.\scripts\cloud\run_task2_p15_modal.ps1 -Stage Check
.\scripts\cloud\run_task2_p15_modal.ps1 -Stage Prepare
```

Expected markers:

- `24 passed` from the focused local tests;
- `LOCAL_CHECK_PASS`;
- `MODAL_P15_PREPARE_COMPLETE`.

`Prepare` uploads the verified P14 archive plus the 229 MB P15 overlay and
downloads the fixed Qwen revision to the persistent Modal Volume. It uses CPU,
not a billed GPU allocation.

## Run

The direct path is:

```powershell
.\scripts\cloud\run_task2_p15_modal.ps1 -Stage Run
```

The remote order is:

1. CUDA/LoRA backward smoke;
2. one-epoch continuation on 781 questions;
3. generation and organizer-compatible scoring on 215 held-out questions;
4. promotion gate;
5. if promoted, one-epoch continuation on all 996 questions;
6. public generation, dev-selected label-free answer profile, validation;
7. result and `submission.zip` download.

Success ends with:

```text
PUBLIC_CANDIDATE_READY
TASK 2 P15 SUBMISSION: ...\artifacts\task2\modal_p15_download\submission.zip
```

If the dev improvement is insufficient, the expected safe result is
`HELDOUT_REJECTED`; no public candidate is emitted and the P14 submission stays
the submission to keep.

## Observe, resume, or stop

```powershell
.\scripts\cloud\run_task2_p15_modal.ps1 -Stage Logs
.\scripts\cloud\run_task2_p15_modal.ps1 -Stage Status
.\scripts\cloud\run_task2_p15_modal.ps1 -Stage Download
.\scripts\cloud\run_task2_p15_modal.ps1 -Stage Stop
```

Training writes an epoch checkpoint and generation checkpoints every batch.
Re-running `Run` resumes the same content-addressed contract instead of
starting an unrelated duplicate.

## Files already prepared locally

- `artifacts/task2/task2_p15_modal_input.zip`
- `artifacts/task2/task2_p15_modal_input.manifest.json`
- `artifacts/task2/training/p15_adaptation/adaptation_manifest.json`
- `artifacts/task2/evaluation/p15_answer_selector_meteor_fast/selector_report.json`

The current official 1,000-question baselines are:

- fixed P14 prefix: METEOR `0.562923`, ROUGE-L `0.432093`;
- P15 OOF answer selector: METEOR `0.565927`, ROUGE-L `0.462936`.

These are validation figures, not a promise of a particular public score.
