# Qwen teacher → BGE student salvage audit

Status: **NO_DEPLOYABLE_EXPANSION_POLICY**

## Decision

The Qwen artifact is treated as an offline teacher on F1–F4 only. Its query IDs are not mapped to the 1,000-query public population. No Qwen score or Qwen rank is present in public outputs. Because FULLDOC_TOP200_NEW candidates are outside the canonical BGE candidate pool and only a small retained BGE-score overlap exists, no candidate-expansion policy can pass the required full F1–F4 BGE-only final-ranking gate without new BGE inference. The safe deployment policy is `NO_OP_KEEP_STEP4_INCUMBENT`.

## Population and integrity

- Teacher population: F1–F4, `5600` queries; fold counts: 1,400 each.
- Public population: Step4 deployment, `1000` queries; candidate union `200000` pairs.
- Qwen merged SHA256: `25cc4b7dfb5919a47f927c545927e775c11997790984b519e2b74a71abce0415`; expected match: `YES`.
- Qwen rows/unique q-docs/queries/documents: `598192/598192/5600/8261`.
- Full-doc worklist rows: `598192`; ranked FULLDOC_TOP200_NEW q-docs: `598071`.
- Query namespace mapping attempted: **NO**. F1–F4 and public are separate populations.
- BGE/GPU/Modal/Qwen runs in this task: `0/0/0/0`.

## Canonical BGE baseline

- Baseline: `D:\udsc2026\artifacts\task1\recovery_096\baseline_093_oof\predictions.jsonl`; report: `D:\udsc2026\artifacts\task1\recovery_096\baseline_093_oof\report.json`.
- Pooled F1–F4 Recall/Precision: `0.925928571429` / `0.197392857143`.
- F1: Recall `0.936369047619`, Precision `0.198428571429`.
- F2: Recall `0.928333333333`, Precision `0.197714285714`.
- F3: Recall `0.919404761905`, Precision `0.195000000000`.
- F4: Recall `0.919607142857`, Precision `0.198428571429`.

## Qwen teacher diagnostics

Qwen scores are used only to rank the offline teacher candidates and to measure discovery coverage. They are not used to make a public ranking decision.

| Teacher depth | Qwen pairs | FULLDOC_NEW pairs | retained BGE scores | BGE score coverage |
|---|---:|---:|---:|---:|
| TOP5 | 28000 | 27921 | 151 | 0.5408% |
| TOP10 | 56000 | 55908 | 262 | 0.4686% |
| TOP20 | 112000 | 111899 | 435 | 0.3887% |

The `full_doc_new_pairs` are deliberately outside the canonical 200-candidate pool. The existing compact BGE artifact contains only retained BGE scores for a small overlap; it is insufficient to validate final BGE ranking of the expansion universe.

## Candidate policies

Four feature-only discovery rules were inspected. They use `full_doc_rank`, never Qwen score/rank at deployment:

- `POLICY_A_FULLDOC_RANK_LE_20`: `add full-document candidates with full_doc_rank <= 20; score/rank with BGE only`; teacher q-docs `18115`, affected queries `3839`, retained BGE coverage `0.3091%`; final BGE validation **NOT RUN / INCOMPLETE SCORE COVERAGE**.
- `POLICY_B_FULLDOC_RANK_LE_50`: `add full-document candidates with full_doc_rank <= 50; score/rank with BGE only`; teacher q-docs `73487`, affected queries `5331`, retained BGE coverage `0.2354%`; final BGE validation **NOT RUN / INCOMPLETE SCORE COVERAGE**.
- `POLICY_C_FULLDOC_RANK_LE_100`: `add full-document candidates with full_doc_rank <= 100; score/rank with BGE only`; teacher q-docs `214445`, affected queries `5595`, retained BGE coverage `0.2061%`; final BGE validation **NOT RUN / INCOMPLETE SCORE COVERAGE**.
- `POLICY_D_FULLDOC_RANK_LE_200`: `add full-document candidates with full_doc_rank <= 200; score/rank with BGE only`; teacher q-docs `598071`, affected queries `5600`, retained BGE coverage `0.1690%`; final BGE validation **NOT RUN / INCOMPLETE SCORE COVERAGE**.

No policy is selected. Reporting teacher gold coverage as a policy metric would be an oracle/teacher diagnostic, not a valid BGE student result, so it is not used as a deployment gate.

## Public application

The repository contains the public 4-source candidate union, but no public FULLDOC rank artifact that can instantiate the F1–F4-derived `full_doc_rank` rules. Therefore the public expansion is intentionally empty and the incumbent is preserved.

- Expansion: `D:\udsc2026\artifacts\task1\qwen_teacher_bge_student\public_bge_candidate_expansion.jsonl` — 0 rows.
- BGE worklist: `D:\udsc2026\artifacts\task1\qwen_teacher_bge_student\public_missing_bge_scores.jsonl` — 0 rows.
- Public affected queries/new pairs/unique documents/missing BGE scores: `0/0/0/0`.
- Qwen used in public scoring: **NO**.
- Final ranker: **BGE**.
- Submission created: **NO**.

## Next safe step

Provide a public full-document retrieval-rank artifact or separately authorize bounded BGE scoring for a frozen F1–F4 policy worklist. Then rerun full F1–F4 BGE-only validation before applying any public expansion.
