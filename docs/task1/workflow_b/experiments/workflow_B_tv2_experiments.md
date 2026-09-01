# Experiment Roadmaps for Workflow B – Primary

## Overview

- This file lists the **primary** experiments that belong to the **Primary Researcher** (Researcher A) and are part of the B2a branch.
- Experiments are **not** executed automatically; they are triggered according to the logical gate ordering defined in `workflow_B_b2a.md`.

## Experiment Table

| ID | Experiment | Status | Run when | Scientific question | Owner | Prerequisite | Success evidence | Kill/stop condition | Possible next branch |
|----|------------|--------|----------|---------------------|-------|--------------|------------------|----------------------|----------------------|
| P0 | B2a‑0 Zero‑shot Qwen reranker | ACTIVE | Immediately (pre‑flight passed) | Does zero‑shot semantic reranking improve top‑5 recall compared to the Workflow‑A comparator? | Primary Researcher | Model provenance audit, context audit, GPU smoke | Prediction freeze passes metric guard | Model drift / metric regression | Fine‑tuning (B2a‑1) if reviewed |
| P1 | Long‑document reranking analysis | CONDITIONAL | After P0 PASS and context audit shows truncation concerns | How does document length affect reranking performance? | Primary Researcher | P0 completed | Detailed truncation report | >1 % truncation leads to STOP | May inform retrieval adjustments |
| P2 | B2a‑1 Fine‑tuning / LoRA | DEFERRED / NOT_AUTHORIZED | Only after professor approval of B2a‑0 results | Can fine‑tuning further improve top‑5 recall? | Primary Researcher (subject to approval) | P0 PASS & scientific justification | Fine‑tuned model meets metric improvement >0.003 | Unauthorized fine‑tuning is prohibited | Integration into B2a branch |
| P3 | Alternative reranker checkpoint exploration | CONDITIONAL | If fine‑tuning is not feasible but metric improvement needed | Are alternative checkpoints beneficial without retraining? | Primary Researcher | P0 PASS | Checkpoint evaluation report | No improvement → STOP | May feed into future experiments |
| P4 | Hard‑negative neural training (research) | DEFERRED | After B2a‑0 conclusions & resources allocated | What hard‑negative samples improve reranker training? | Primary Researcher | P0 PASS | Training loss curve & validation | Resource constraints | May become future fine‑tuning branch |
| P5 | Retrieval / embedding architecture assessment | DEFERRED | When evidence suggests retrieval limits recall | Is current K77 candidate generation sufficient? | Primary Researcher | P0 PASS | Retrieval diagnostics report | Changing K77 requires professor sign‑off | May trigger new retrieval experiments |
| P6 | Multi‑stage integration planning | DEFERRED | After individual components are validated | How to integrate reranking, retrieval, and ranking stages efficiently? | Primary Researcher | All prior experiments PASS | Integration design document | Infeasibility leads to redesign | Future pipeline rollout |

---

*All experiments obey the shared Task1 governance:* official metric contract, Fold0 exclusion, public‑label exclusion, no leaderboard tuning, prediction provenance, and preregistration requirements.
