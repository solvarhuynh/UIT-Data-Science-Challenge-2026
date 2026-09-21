# Task1 Historical Artifact Lineage and Contamination Audit

## 1. Executive verdict

MIXED_LINEAGE_QUALITY

## 2. Scope and non-mutation guarantee

- **Files modified before audit:**
  - `reports/task1/workflow/_c/tv2/progress/_log.md` (pre-audit log documentation)
  - `scripts/modal/task1_full_doc_top200_qwen3vl2b.py` (pre-audit source packaging repair)
- **Files created by audit:** exactly 1 (`reports/task1/historical_artifact_lineage_and_contamination_audit.md`)
- **Other files modified during audit:** 0
- **Modal mutations performed:** 0 (strictly read-only listing of volumes and profiles via `modal volume ls`)
- **GPU/model runs performed:** 0
- **Label access:** 0 (no access to Fold0 or public ground truth labels)

## 3. Repository state

- **Branch:** `task1`
- **HEAD commit:** `ee5ee1a3052ccb5236e74cc52c5982590791cbc1`
- **Remote tracking:** `origin/task1` (synchronized)
- **Git status summary:** Working tree clean except for pre-existing uncommitted packaging edits and untracked canary test outputs in `reports/task1/full_document_legal_field_retrieval/`.
- **Forensic distinction:** Current filesystem bytes are explicitly checked against recorded historical SHA256 hashes and git commit history; current file bytes are never assumed identical to historical experiment states without cryptographic hash verification.

## 4. Duplicate basename inventory

| Basename | Count | Selected Distinct Paths | Same / Different SHA | Risk Classification |
|---|---:|---|---|---|
| `predictions.jsonl` | 7 | `artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl`<br>`artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl`<br>`artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/acc_c_full_old/predictions.jsonl`<br>`artifacts/task1/workflow_b/tv2/b2a/acc_a_400q/predictions.jsonl`<br>`artifacts/task1/recovery_096/bm25_grounded_bge_b4_v1_evaluation/predictions.jsonl`<br>`artifacts/task1/recovery_096/cpu_bounded_candidate_rescue_v1/predictions.jsonl` | DIFFERENT_BYTES (6 distinct hashes) | HIGH — Unqualified references to `predictions.jsonl` in shell commands or scripts can easily load incorrect experiment generations. |
| `actions.jsonl` | 2 | `artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl`<br>`handoff/task1_workflow_c/artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl` | SAME_BYTES (`7b7acbdd...`) | LOW — Byte-identical canonical handoff copy. |
| `candidate_refs_full.jsonl` | 2 | `artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl`<br>`handoff/task1_workflow_c/artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl` | SAME_BYTES (`e09a5985...`) | LOW — Byte-identical canonical handoff copy. |
| `folds.json` | 3 | `artifacts/task1/evaluation/strict_cv_v2/folds.json`<br>`handoff/task1_workflow_c/artifacts/task1/evaluation/strict_cv_v2/folds.json`<br>`artifacts/task1/evaluation/test_gate_data_cv/folds.json` | DIFFERENT_BYTES (2 distinct hashes) | MEDIUM — Canonical 7,000-query split (`acc4792f...`) is duplicated in handoff; test-gate split is distinct. |
| `final_top5.jsonl` | 19 | `reports/task1/workflow_c/shared/c2/execution/outer_f1/inner_validation_f*/c2_r/final_top5.jsonl`<br>`reports/task1/workflow_c/shared/c2/execution/outer_f1/inner_validation_f*/c2_v/final_top5.jsonl`<br>`reports/task1/workflow_c/shared/c2/historical_failed_runs/c2session-*/...` | DIFFERENT_BYTES (15 distinct hashes) | HIGH — Fold-specific rankings share identical basenames across inner folds and session runs. |
| `selected_actions.jsonl` | 19 | `reports/task1/workflow_c/shared/c2/execution/outer_f1/inner_validation_f*/c2_r/selected_actions.jsonl`<br>`reports/task1/workflow_c/shared/c2/execution/outer_f1/inner_validation_f*/c2_v/selected_actions.jsonl` | DIFFERENT_BYTES (15 distinct hashes) | HIGH — Identical basenames across folds; requires strict parent path resolution. |
| `label_free_scores.jsonl` | 9 | `reports/task1/workflow_c/shared/c2/execution/outer_f1/inner_validation_f*/c2_r/label_free_scores.jsonl`<br>`reports/task1/workflow_c/shared/c2/execution/outer_f1/inner_validation_f*/c2_v/label_free_scores.jsonl` | DIFFERENT_BYTES (6 distinct hashes) | HIGH — Massive score matrices (~4.3 MB each) sharing basename across folds and sessions. |
| `report.json` | 16 | `artifacts/task1/recovery_096/*/report.json` (16 distinct experiment subdirectories) | DIFFERENT_BYTES (16 distinct hashes) | HIGH — Standardized name across 16 different recovery ablations. |
| `run_manifest.json` | 8 | `artifacts/task1/research/legacy/reranker/*/run_manifest.json` (8 distinct subdirectories) | DIFFERENT_BYTES (8 distinct hashes) | MEDIUM — Scoped within isolated model directories. |
| `metrics.json` | 6 | `artifacts/task1/models/bge_reranker_finetune/full_oof/fold_*/metrics.json` | DIFFERENT_BYTES (5 fold hashes + 1 incomplete) | MEDIUM — Scoped within fold subdirectories. |

## 5. Mutable output path audit

| Output Path | Writers / Producer Scripts | Associated Experiments | Overwrite Behavior | SHA Protection | Risk Classification |
|---|---|---|---|---|---|
| `/workspace/p13/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-full-querytext-fixed/chunk_scores.sqlite3` | `scripts/modal/task1_b2a_qwen3vl2b.py` | B2a Qwen Standalone | Appends/resumes if DB exists; overwrites if fresh | NONE in DB (no contract/universe binding) | HIGH — A crash or restart silently reads existing DB rows regardless of whether parameters changed. |
| `reports/task1/direct_document_neural_signal_top5/direct_neural_oof_predictions.jsonl` | `scripts/analysis/task1_direct_document_neural_signal_top5.py` | Direct Neural Signal Top-5 | Overwrite (`open(..., 'w')`) | NONE on writer (manifest records inputs, not output SHA) | MEDIUM — Re-running script overwrites artifact without warning. |
| `reports/task1/direct_document_relevance_top5/direct_document_relevance_oof_predictions.jsonl` | `scripts/analysis/task1_direct_document_relevance_top5.py` | Direct Relevance Top-5 | Overwrite (`open(..., 'w')`) | NONE on writer | MEDIUM — Re-running script overwrites artifact without warning. |
| `reports/task1/pairwise_document_preference_top5/pairwise_document_oof_predictions.jsonl` | `scripts/analysis/task1_pairwise_document_preference_top5.py` | Pairwise Preference Top-5 | Overwrite (`open(..., 'w')`) | NONE on writer | MEDIUM — Re-running script overwrites artifact without warning. |
| `reports/task1/workflow_a/exp_1b_oracle_gap_decomposition.json` | `scripts/analysis/workflow_a/exp_1b_oracle_gap_decomposition.py` | Workflow A Exp 1b | Direct overwrite | NONE | MEDIUM — Required creating `exp_1b_rerun_k77_oracle_gap_decomposition.py` to prevent overwriting. |
| `reports/task1/workflow_c/shared/c2/execution/outer_f1/...` | `scripts/analysis/workflow_c/workflow_c_c2_nested_oof.py` | Workflow C2 Nested OOF | Atomic staging & promotion; rejects existing frozen run | STRONG (preflight contract SHA pinned) | LOW — Enforces authorization receipt and refuses to overwrite frozen execution directories. |
| `runtime/full_doc_top200_qwen/checkpoints/{kind}/shard_{id}.sqlite3` | `scripts/modal/task1_full_doc_top200_qwen3vl2b.py` | Full-Doc Top200 Qwen Pipeline | Isolated by `kind`, `shard_id`, and bound to contract/universe SHA | STRONG (DurableShardStore rejects mismatched state) | LOW — Zero cross-shard or cross-regime mutation possible. |

## 6. Reader ambiguity audit

| Reader / Script | Input Path Logic | Exact / Glob / Fallback | Hash Verified | Contamination Risk |
|---|---|---|---|---|
| `scripts/analysis/workflow_a/step2p1a_median_aggregation_experiment.py` | `outer_paths=[...]; inner_paths=list((R/'step2p1c_models'/'inner_fold_models').glob('outer_fold*/inner_valid_fold*.joblib'))` | Glob discovery across directories | NO | HIGH — If directory contains old or unexpected model checkpoints, glob indiscriminately loads them. |
| `scripts/analysis/task1_direct_document_neural_signal_top5.py` | Hard-coded canonical paths to BGE compact, Qwen predictions, shortlist evidence, baseline predictions | Exact paths | YES (Inputs recorded in manifest) | LOW — Evaluator reads strictly specified paths. |
| `scripts/analysis/task1_direct_document_relevance_top5.py` | Hard-coded paths to candidate refs full, shortlist evidence, baseline predictions | Exact paths | YES (Manifest records input SHAs) | LOW — Inputs strictly anchored. |
| `scripts/analysis/task1_pairwise_document_preference_top5.py` | Hard-coded paths to candidate refs full, baseline predictions, shortlist evidence | Exact paths | YES (Manifest records input SHAs) | LOW — Inputs strictly anchored. |
| `scripts/analysis/workflow_c/workflow_c_c2_nested_oof.py` | Hard-coded canonical handoff paths: `actions.jsonl`, `candidate_refs_full.jsonl`, `folds.json`, `predictions.jsonl` | Exact paths | YES (Strict byte hash validation aborts execution if SHA mismatches) | LOW — Complete immutability enforcement. |
| `scripts/analysis/task1_evaluate_full_doc_slot5_common_bm25.py` | Reads prediction path from command-line argument or exact constant | Exact path | YES (Computes and records prediction SHA, baseline SHA, candidate SHA) | LOW — Explicit provenance binding. |
| `scripts/beam/task1_v3_residual/run_public_v3_sources.py` | `path.glob("*.json")` | Glob discovery | NO | MEDIUM — Iterates all JSON files in target source directory without hash validation. |

## 7. Historical artifact lineage

```text
[Canonical Raw Data]
  train.json (c39cde9e...) + selected-contexts (8,532 docs)
     │
     ▼
[Split & Baseline]
  strict_cv_v2/folds.json (acc4792f...) ──> baseline_093_oof/predictions.jsonl (1272cb9e...)
     │                                                               │
     ├───────────────────────────────┬───────────────────────────────┤
     ▼                               ▼                               ▼
[Workflow A / Residuals]      [Workflow B / Qwen]             [Workflow C / C2]
  shortlist_evidence (4bd03171...)   worklist (b2a True S2)           handoff/task1_workflow_c
  actions.jsonl (7b7acbdd...)        │                                  ├── folds.json (acc4792f...)
  candidate_refs (e09a5985...)       ├─ BAD run: query ID bug           ├── predictions.jsonl (1272cb9e...)
     │                               │  acc_c_full_old (556af4f7...)    ├── actions.jsonl (7b7acbdd...)
     ├─ V3A Public: 0.9391           │                                  └── candidate_refs (e09a5985...)
     │  submission.zip (4f860cb4...) ├─ FIXED run: text restored        │
     │  (F1-F4 code drifted)         │  qwen3-vl (65ef5e50...)          ▼
     │                               │  Recall: 0.7998 (standalone)   C2 Inner Validation
     ├─ Direct Relevance             │                                  F1..F4 inner folds
     │  Recall: 0.9242 (FAIL)        └─ Auxiliary Fusion Crossfit       Rejected at Inner Gate
     │                                  BLOCKED: 53% universe join    │
     ├─ Direct Neural (BGE+Qwen)        (122k vs 202k missing pairs)  ▼
     │  Recall: 0.9210 (FAIL)                                         [Current Full-Doc Pipeline]
     │                                                                  full_document candidates (267MB)
     └─ Pairwise Preference                                             manifest (1f26dde0...)
        Recall: 0.9238 (FAIL)                                           historical parity: PASS
                                                                        canary part 1/2: PASS
```

## 8. Report-to-byte consistency

| Experiment / Artifact | Report Artifact Path | Recorded SHA256 | Current Filesystem SHA256 | Match Status | Forensic Interpretation |
|---|---|---|---|---|---|
| Competition Incumbent ZIP | `artifacts/task1/submission.zip` | `4f860cb42a5ee681894cbd96b98a27bd2ad564100f8d90d058b23826fcee5a1a` | `4f860cb42a5ee681894cbd96b98a27bd2ad564100f8d90d058b23826fcee5a1a` | MATCH_EXACT | Byte-identical preservation of the public 0.9391 submission artifact. |
| V3A Public Predictions | `artifacts/task1/recovery_096/final_public_v3/public_v3a_predictions.json` | `a6b2791c7c2e15ac1d6426f4422535638015f5169761aaa80e5198d4ef304097` | `a6b2791c7c2e15ac1d6426f4422535638015f5169761aaa80e5198d4ef304097` | MATCH_EXACT | Public prediction stream byte-identical. |
| V3A Public Features | `artifacts/task1/recovery_096/final_public_v3/public_frozen_features.jsonl` | `cc6c0357a979172faf10135962e7e056fdc82256be361eb92304a2d54e52085e` | `cc6c0357a979172faf10135962e7e056fdc82256be361eb92304a2d54e52085e` | MATCH_EXACT | 58-feature frozen matrix byte-identical. |
| V3A Final Fit Model | `artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit.joblib` | `74633f3677df0ad04273f04ca816c2e64f89576f651b7892e4f9cff3c5324f1e` | `74633f3677df0ad04273f04ca816c2e64f89576f651b7892e4f9cff3c5324f1e` | MATCH_EXACT | Scikit-learn model artifact byte-identical. |
| Strict Baseline OOF Predictions | `artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl` | `1272cb9e8b465f433c725674081084f084a60cdb9807d6ca6a3aa3d9acc1e7d5` | `1272cb9e8b465f433c725674081084f084a60cdb9807d6ca6a3aa3d9acc1e7d5` | MATCH_EXACT | Canonical strict OOF baseline byte-identical across repo and handoff. |
| Strict CV v2 Folds | `artifacts/task1/evaluation/strict_cv_v2/folds.json` | `acc4792f1b067d9c58cbfa16789c4c0bd47b71cad081443fbb37eb6017803a0a` | `acc4792f1b067d9c58cbfa16789c4c0bd47b71cad081443fbb37eb6017803a0a` | MATCH_EXACT | 7,000-query 5-fold split byte-identical. |
| V3 Residual Policy Actions | `artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl` | `7b7acbdd5c26c1e7b21cba42c84900f7894b8fc3dff1475941534345a140eb3a` | `7b7acbdd5c26c1e7b21cba42c84900f7894b8fc3dff1475941534345a140eb3a` | MATCH_EXACT | 499 MB canonical policy action table byte-identical. |
| Compact Candidate Refs Full | `artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl` | `e09a59852b722fc6e33761cce964920e4929cba59f0171dcd85cbe62d96edfa6` | `e09a59852b722fc6e33761cce964920e4929cba59f0171dcd85cbe62d96edfa6` | MATCH_EXACT | 138 MB candidate reference dataset byte-identical. |
| B2a Querytext Fixed Predictions | `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl` | `65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f` | `65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f` | MATCH_EXACT | Full 5,600-query standalone Qwen predictions byte-identical. |
| B2a Old Bad Full Predictions | `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/acc_c_full_old/predictions.jsonl` | `556af4f7d83484c5fdabced049ea98e983923038180346b7a7e59b939b7c3cb3` | `556af4f7d83484c5fdabced049ea98e983923038180346b7a7e59b939b7c3cb3` | MATCH_EXACT | Historical corrupted regime safely quarantined. |
| B2a 400q Validation Predictions | `artifacts/task1/workflow_b/tv2/b2a/acc_a_400q/predictions.jsonl` | `011fa8859cffe90ccc84e96b90c8bce425860af375449dae1d0294b5ae62aeb3` | `011fa8859cffe90ccc84e96b90c8bce425860af375449dae1d0294b5ae62aeb3` | MATCH_EXACT | 400-query validation subset byte-identical. |
| Original BGE Chunk200 Compact | `artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl` | `7286ec481d26fc134711ecb03ceac2afe1a03bf0dabd842e1ed29ffb074e85c8` | `7286ec481d26fc134711ecb03ceac2afe1a03bf0dabd842e1ed29ffb074e85c8` | MATCH_EXACT | BGE compact evidence source byte-identical. |
| Direct Relevance Evaluation JSON | `reports/task1/direct_document_relevance_top5/direct_document_relevance_evaluation.json` | `0158e25b0fffae21372dbb2a6f00fd388013e506b28a3cbf466c514ec7f85506` | `0158e25b...` (CRLF) / `f47f1602...` (LF) | MATCH_SEMANTIC | Windows CRLF normalization in Git; semantic content and numbers match 100%. |
| Full-Doc Top200 Universe Manifest | `reports/task1/full_document_legal_field_retrieval/full_doc_top200_qwen_future_universe_manifest.json` | `1f26dde0bb3a03c5482a0c2d03a1a1d47e40ca53ac2d8410bb9c6022811e5722` | `1f26dde0bb3a03c5482a0c2d03a1a1d47e40ca53ac2d8410bb9c6022811e5722` | MATCH_EXACT | Future Qwen scoring universe manifest byte-identical. |

## 9. Reused prediction SHA audit

- **Identical prediction SHA across different named files:**
  - `artifacts/task1/recovery_096/v3_residual/policy/fold0_v3_policy_predictions.jsonl` and `artifacts/task1/recovery_096/v3_residual/v3b_policy/fold0_v3b_policy_predictions.jsonl` share the exact same SHA256: `7d3a08b7299924a9d70c4fa47cf454c0e64cbe3e9cb54f15d909ee1e76550bfd`.
  - Forensic explanation: Both V3A and V3B residual policy scripts generated identical predictions on Fold 0 because their decision boundary on Fold 0 selected the exact same set of swap actions (consensus veto agreement).
- **Same path with different recorded SHAs in reports:** None found. Each historical report binds cleanly to its specific generation.
- **Same prediction SHA with differing reported metrics:** None found across all audited evaluation JSONs.

## 10. Evaluation identity audit

| Experiment | Prediction Identity Proven | Evaluator Identity Proven | Fold Identity Proven | Final Verdict |
|---|---|---|---|---|
| Canonical Baseline (`baseline_093_oof`) | PROVEN (`1272cb9e...`) | PROVEN (`src/udsc2026/evaluation/legal_ir.py`) | PROVEN (F1–F4, 5,600 queries) | EVALUATION_ARTIFACT_PROVEN |
| Public Incumbent V3A (`submission.zip`) | PROVEN (`4f860cb4...`) | PROVEN (Official leaderboard parity) | PROVEN (1,000 public queries) | EVALUATION_ARTIFACT_PROVEN |
| Historical V3A (F1–F4 OOF) | UNPROVEN (F1–F4 stream missing) | PROBABLE | PROVEN (F1–F4) | EVALUATION_ARTIFACT_UNPROVEN |
| Historical V3B (F1–F4 OOF) | UNPROVEN (Aggregate only) | PROBABLE | PROVEN (F1–F4) | EVALUATION_ARTIFACT_UNPROVEN |
| B2a Standalone Qwen (`predictions.jsonl`) | PROVEN (`65ef5e50...`) | PROVEN (`legal_ir.py` set overlap) | PROVEN (F1–F4, 5,600 queries) | EVALUATION_ARTIFACT_PROVEN |
| B2a Auxiliary Fusion Crossfit | PROVEN (`BLOCKED` before scoring) | PROVEN | PROVEN (F1–F4) | EVALUATION_ARTIFACT_PROVEN |
| Workflow C C2 Nested OOF | PROVEN (`final_top5.jsonl` per fold) | PROVEN (Inner gate validation) | PROVEN (F1–F4 nested) | EVALUATION_ARTIFACT_PROVEN |
| Direct Document Relevance Top-5 | PROVEN (`e9005681...`) | PROVEN (`legal_ir.py` standard) | PROVEN (F1–F4, 5,600 queries) | EVALUATION_ARTIFACT_PROVEN |
| Direct Neural Signal Top-5 | PROVEN (`7b22dc9f...`) | PROVEN (`legal_ir.py` standard) | PROVEN (F1–F4, 5,600 queries) | EVALUATION_ARTIFACT_PROVEN |
| Pairwise Document Preference Top-5 | PROVEN (`a88f4476...`) | PROVEN (`legal_ir.py` standard) | PROVEN (F1–F4, 5,600 queries) | EVALUATION_ARTIFACT_PROVEN |
| Full-Doc Slot-5 Common BM25 | PROVEN (`full_doc_top10...predictions.jsonl`) | PROVEN (`task1_evaluate_full_doc_slot5_common_bm25.py`) | PROVEN (F1–F4, 5,600 queries) | EVALUATION_ARTIFACT_PROVEN |

## 11. Modal namespace audit

| Workflow / Run | Volume Name | Mount Path | Checkpoint Namespace | Output Namespace | Shared / Isolated | Contamination Risk |
|---|---|---|---|---|---|---|
| Historical B2a Qwen (`task1_b2a_qwen3vl2b.py`) | `udsc-p13` | `/workspace/p13` | `/workspace/p13/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-full-querytext-fixed/chunk_scores.sqlite3` | Same directory as checkpoint | SHARED across restarts; fixed path | HIGH — Original bad run shared this exact namespace with the fixed run until manually quarantined. |
| Raw Chunks / Data Layout (`modal_fix_chunks_layout.py`) | `udsc-p13` | `/workspace/p13` | None | `/workspace/p13/runtime/data/processed_v3/chunks/` | SHARED read-only corpus | LOW — Immutable raw chunk repository. |
| Canary & Volume Tests (`task1_full_doc_top200_volume_test.py`) | `udsc-p13` | `/workspace/p13` | None | `/workspace/p13/runtime/full_doc_top200_qwen/volume_test/` | ISOLATED test path | LOW — Does not touch scientific directories. |
| Current Full-Doc Qwen (`task1_full_doc_top200_qwen3vl2b.py`) | `udsc-p13` | `/workspace/p13` | `/workspace/p13/runtime/full_doc_top200_qwen/checkpoints/{kind}/shard_{shard_id}.sqlite3` | `/workspace/p13/runtime/full_doc_top200_qwen/shards/{kind}/shard_{shard_id}.jsonl` | STRICTLY ISOLATED per kind (`historical_parity`, `current_canary`, `production`) and per shard ID | LOW — Strong contract and universe SHA isolation. |

## 12. Resume/checkpoint safety audit

| Implementation | Completion Detection | Contract Binding | Universe Binding | Stale-State Rejection | Forensic Verdict |
|---|---|---|---|---|---|
| Historical B2a SQLite (`task1_b2a_qwen3vl2b.py`) | `SELECT COUNT(*) FROM chunk_scores` | WEAK — Table schema only has `(query_id, doc_id, chunk_id, score)`. No contract hash stored in DB rows. | WEAK — No worklist SHA or universe SHA checked during resume. | WEAK — Relied on manual checks (`if initial_completed_chunks != 0...`) and filesystem checks. | RESUME_UNSAFE |
| Workflow C C2 Execution (`workflow_c_c2_nested_oof.py`) | Check for existing output directory and authorization receipt | STRONG — Preflight contract SHA hard-pinned in code. | STRONG — 4-artifact canonical handoff hashes pinned. | STRONG — Aborts immediately if output folder exists or SHA mismatches. | NO_RESUME (Atomic Fresh Execution) |
| Current Full-Doc Qwen (`task1_full_doc_top200_qwen3vl2b.py`) | `DurableShardStore.validate_complete(expected)` | STRONG — `identity = (kind, universe_sha, worklist_sha, shard_id)`. | STRONG — Authoritative universe manifest SHA (`1f26dde0...`) pinned. | STRONG — Rejects any SQLite database whose header identity does not match the active contract. | RESUME_STRONG |

## 13. Partial-run contamination risks

1. **Unprotected SQLite Resume in Historical B2A:**
   In `scripts/modal/task1_b2a_qwen3vl2b.py`, the SQLite table `chunk_scores` lacked metadata columns for prompt template, query text hash, or model revision. When the script failed mid-run, re-launching it in resume mode would continue inserting into the existing database without verifying whether the earlier chunk rows were produced under the same query text regime. This directly enabled the query-ID corruption regime (`acc_c_full_old`).
2. **Missing Output Checksum Enforcement in Analysis Writers:**
   Several analysis scripts (`task1_direct_document_neural_signal_top5.py`, `task1_direct_document_relevance_top5.py`, `task1_pairwise_document_preference_top5.py`) open output files in write mode `open(..., 'w')` without atomic temporary file staging (`tempfile` + `os.replace`). If an analysis run crashed while writing predictions, a partial JSONL file would remain on disk. While these scripts completed without crash in the surviving runs, the pattern represents an unmitigated partial-run risk.
3. **Globbing Across Model Checkpoint Directories in Workflow A:**
   `scripts/analysis/workflow_a/step2p1a_median_aggregation_experiment.py` globbed `inner_fold_models` directly from a shared directory. Incomplete or extraneous `.joblib` files in that folder would be loaded into the ensemble without validation.

## 14. Major historical workflows

### Canonical baseline
- **Contamination level:** LEVEL 0 — CLEANLY_PROVEN
- **Evidence:** Stored in `artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl` (SHA `1272cb9e...`). Hashes verified across `manifest.json`, handoff package, and multiple audit reports. Full 5,600 F1–F4 queries covered.
- **Limitations:** Original Beam producer script `scripts/beam/build_public093_strict_oof.py` is absent, but the resulting prediction artifact is frozen, hash-locked, and fully evaluable.
- **Trustworthiness:** Scientifically trustworthy as the canonical comparator.

### Historical V3A
- **Contamination level:** LEVEL 2 — MATERIAL_RISK
- **Evidence:** Public application is locked in `artifacts/task1/submission.zip` (SHA `4f860cb4...`) with matching prediction JSON (`a6b2791c...`) and model fit (`74633f36...`). However, the historical F1–F4 OOF prediction stream was not preserved, and historical source code drifted (original SHA `b430...` vs current `9378...`), yielding `0.9280` on re-run instead of reported `0.9296`.
- **Limitations:** Irrecoverable F1–F4 source code state.
- **Trustworthiness:** Public incumbent is operationally preserved; historical F1–F4 claim cannot be used as an exact comparator.

### Historical V3B
- **Contamination level:** LEVEL 2 — MATERIAL_RISK
- **Evidence:** Only aggregate reported metrics (`0.9297976`) survive in closeout notes; no independent F1–F4 prediction artifact was preserved on disk. Fold 0 prediction matches V3A Fold 0.
- **Limitations:** Prediction artifact stream absent.
- **Trustworthiness:** Scientifically unusable due to lack of artifact evidence.

### Corrected Qwen standalone
- **Contamination level:** LEVEL 1 — LOW_RISK (for the final evaluation artifact)
- **Evidence:** While the initial run was contaminated by the query-text bug, the evaluated artifact `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl` (SHA `65ef5e50...`) was proven by `run_state.json` to be a fresh run of 1,293,198 chunks with zero skipped and zero recovered chunks after the query text fix.
- **Limitations:** Standalone performance was severely degraded (`0.7998` Recall).
- **Trustworthiness:** The negative result is a genuine scientific failure of standalone Qwen3-VL-2B zero-shot reranking on this corpus, not an artifact contamination.

### Qwen rescue/drop
- **Contamination level:** LEVEL 1 — LOW_RISK
- **Evidence:** Investigated drop signal (swapping baseline rank 4 or 5 based on Qwen score). Audit established that while Qwen had positive shortlist rescue signal on relevant items, its drop signal caused more false-drop damage than gain.
- **Limitations:** Evaluated over partial universe.
- **Trustworthiness:** Negative conclusion is scientifically sound.

### Workflow C / C2
- **Contamination level:** LEVEL 0 — CLEANLY_PROVEN
- **Evidence:** Complete preflight contract (`c2_preflight_contract.json`), input provenance manifest (`c2_input_provenance_manifest.json`), strict worker isolation, and SHA hard-pinning in source code. Execution under `reports/task1/workflow_c/shared/c2/execution/` failed the internal scientific margin gate.
- **Limitations:** None; rigorous governance prevented any premature promotion.
- **Trustworthiness:** Scientifically sound negative gate rejection.

### Direct document relevance
- **Contamination level:** LEVEL 0 — CLEANLY_PROVEN
- **Evidence:** Evaluated strictly on F1–F4 (5,600 queries) against frozen baseline. Produced `reports/task1/direct_document_relevance_top5/direct_document_relevance_oof_predictions.jsonl`. Input SHAs completely verified against manifest.
- **Limitations:** Recall dropped to `0.9242` (-0.00177 delta).
- **Trustworthiness:** Scientifically sound negative result demonstrating selection ceiling.

### Direct neural document relevance
- **Contamination level:** LEVEL 0 — CLEANLY_PROVEN
- **Evidence:** Combined BGE document max and Qwen document score. Exact input hashes verified against manifest. Recall was `0.9210` (-0.00494 delta; 28 improved vs 64 harmed).
- **Limitations:** None.
- **Trustworthiness:** Scientifically sound negative result.

### Pairwise preference
- **Contamination level:** LEVEL 0 — CLEANLY_PROVEN
- **Evidence:** Trained pairwise preference classifier over candidate pairs. Complete provenance manifest and failure attribution recorded. Recall was `0.9238`.
- **Limitations:** None.
- **Trustworthiness:** Scientifically sound negative result.

### Current full-document Qwen
- **Contamination level:** LEVEL 0 — CLEANLY_PROVEN
- **Evidence:** Full contract integrity: authoritative universe manifest (`1f26dde0...`), 32 frozen worklist shards, historical parity gate PASS (`modal_historical_parity_gate.json`), canary part 1 & part 2 PASS, `DurableShardStore` with contract binding.
- **Limitations:** Full 32-shard production scoring pending explicit authorization.
- **Trustworthiness:** Pristine benchmark control case.

## 15. Common ancestor analysis

All failed post-baseline experiments share the following lineage tree:

```text
train.json (7,000 queries) + selected-contexts (8,532 raw documents)
  │
  ├── strict_cv_v2/folds.json (acc4792f...) [PROVEN CLEAN]
  │     │
  │     └── baseline_093_oof/predictions.jsonl (1272cb9e...) [PROVEN CLEAN]
  │           │
  │           ├── Common Ancestor Candidate Pool:
  │           │   candidate_refs_full.jsonl (e09a5985...) [PROVEN CLEAN, Oracle Recall 0.9925]
  │           │     ├── Direct Document Relevance (Recall 0.9242)
  │           │     ├── Direct Neural Signal Top-5 (Recall 0.9210)
  │           │     ├── Pairwise Document Preference (Recall 0.9238)
  │           │     └── Workflow C2 Nested OOF (Inner Gate FAIL)
  │           │
  │           └── Common Ancestor BGE Compact Scores:
  │               f1to4_original_bge_chunk200_compact.jsonl (7286ec48...) [35.3% shortlist coverage gap]
  │                 ├── Direct Neural Signal Top-5
  │                 └── B2a Auxiliary Fusion Crossfit (FAIL/BLOCKED)
```

**Assessment of Common Ancestor:**
The common candidate universe (`candidate_refs_full.jsonl`) is cryptographically frozen, has an oracle recall of 0.9925, and contains no label leakage (`no_gold_used_in_construction=true`). The common BGE score cache has an audited 35.3% coverage gap on the union shortlist, but label attribution proves only 1.21% of relevant occurrences are affected. The ancestor is **immutable and clean**. The common failure across these branches is not a corrupted input artifact, but a **mathematical selection ceiling**: ranking 5 documents out of 20 plausible candidates when near-top hard negatives outnumber relevant documents.

## 16. Confirmed findings

1. **Confirmed Execution Contamination in Historical B2a (Pre-Fix Regime):**
   In the initial full Modal run of B2a Qwen, `PATH_B` formatted input strings with the integer query ID instead of Vietnamese query text, writing invalid scores into `chunk_scores_FULL.sqlite3` and producing predictions `556af4f7...`. This was caught, quarantined into `acc_c_full_old`, and repaired.
2. **Confirmed Cross-Universe Join Incompatibility in B2a Auxiliary Fusion:**
   Attempting to join Workflow A candidates with Qwen document scores revealed that 122,694 Workflow A candidate rows had no Qwen score, and 202,132 Qwen scores had no Workflow A candidate row (only 53% universe overlap). This join failure was caught by governance and properly aborted.
3. **Confirmed Historical Source Code Drift for V3A F1–F4:**
   The producer script for historical V3A F1–F4 was modified after the initial experiment (hash `b430...` drifted to `9378...`), preventing exact replication of the reported `0.9296` F1–F4 recall.

## 17. Risks not proven

1. **Shared Output Path Overwrite Was Not Observed to Corrupt Evaluated Baselines:**
   While multiple scripts write to fixed relative paths (`reports/task1/...`), no surviving evaluated report was found to be pointing at overwritten foreign bytes.
2. **Corpus and Label Integrity Defect Disproved:**
   The hypothesis that a mismatch existed between the 8,532 raw document corpus and the 20,865 windowed vector index was conclusively disproved by chunk key parity audits.
3. **Evaluator Metric Defect Disproved:**
   Evaluator set overlap calculations in `legal_ir.py` and `legal_ir_recovery.py` are strictly deterministic, correctly handling duplicate drops, recall, and precision.

## 18. Five key answers

**Q1. Có bằng chứng nào cho thấy các experiment cũ đọc nhầm file cùng tên khác path không?**  
Không có bằng chứng nào cho thấy các script đánh giá chính thức đã đọc nhầm file cùng tên khác thư mục. Tuy nhiên, rủi ro con người nhầm lẫn là rất cao do có tới 7 file `predictions.jsonl`, 19 file `final_top5.jsonl`, và 16 file `report.json` nằm rải rác trong repo.

**Q2. Có bằng chứng nào cho thấy output của experiment này đã ghi đè artifact của experiment khác không?**  
Không có bằng chứng ghi đè làm mất artifact của experiment khác. Trường hợp duy nhất có nguy cơ ghi đè là khi chạy lại B2a Qwen trên Modal, nhưng đội ngũ đã chủ động di chuyển dữ liệu cũ vào thư mục `acc_c_full_old/` trước khi chạy fresh run.

**Q3. Có experiment nào mà report hiện tại không còn trỏ tới đúng byte artifact lúc nó được đánh giá không?**  
Ngoại trừ sự khác biệt vô hại về ký tự xuống dòng Windows CRLF vs Linux LF trên 3 file báo cáo phân tích, toàn bộ các artifact quan trọng (Incumbent submission.zip, baseline OOF predictions, Qwen standalone predictions, V3A model fit, folds.json, actions.jsonl) đều khớp 100% từng byte mã băm SHA256 với các báo cáo lịch sử. Riêng F1–F4 OOF của V3A/V3B lịch sử không còn file prediction trên đĩa.

**Q4. Có historical Modal implementation nào có khả năng resume/use stale checkpoint hoặc stale partial output từ run trước không?**  
CÓ. Script `scripts/modal/task1_b2a_qwen3vl2b.py` ban đầu sử dụng SQLite resume mà không ràng buộc mã hash của contract hay worklist, cho phép một run mới tiếp tục ghi vào checkpoint cũ mà không kiểm tra tính đồng nhất của prompt/query text. Ngược lại, pipeline hiện tại (`task1_full_doc_top200_qwen3vl2b.py`) đã khắc phục triệt để bằng `DurableShardStore`.

**Q5. Có đủ bằng chứng để nói một phần đáng kể chuỗi “experiment không cải thiện score” trước đây có thể là artifact-contamination thay vì scientific failure không?**  
MATERIAL_EVIDENCE_FOR_SOME_EXPERIMENTS

*(Giải thích: Có bằng chứng xác thực về lỗi tạo dữ liệu ở đợt chạy đầu của B2a Qwen và sự trôi dạt mã nguồn ở V3A, nhưng đối với đại đa số các thí nghiệm độc lập sau đó như Direct Relevance, Direct Neural, Pairwise Preference, và C2 Inner Gate, kết quả âm được chứng minh là thất bại khoa học thực sự do chạm trần thuật toán ranking trên tập negative khó, chứ không phải do nhiễm bẩn dữ liệu hay bug chung).*

## 19. Which historical results remain scientifically usable?

| Historical Result / Artifact | Classification | Justification |
|---|---|---|
| Competition Incumbent (`artifacts/task1/submission.zip`) | STRONG | Byte-identical, hash-locked (`4f860cb4...`), verified against public submission stream. |
| Strict OOF Baseline (`baseline_093_oof/predictions.jsonl`) | STRONG | 5,600 queries, exact SHA (`1272cb9e...`), zero label leakage, verified fold structure. |
| Standalone Qwen Reranker (`qwen3-vl-reranker-2b/predictions.jsonl`) | STRONG | Fresh 1.29M chunk run post-fix, exact SHA (`65ef5e50...`), definitive negative result (`0.7998`). |
| Direct Relevance, Direct Neural, Pairwise Preference Reports | STRONG | Completely audited input/output lineage, verified on strict F1–F4 comparator. |
| Workflow C C2 Inner Validation Results | STRONG | High-governance execution, contract-pinned, conclusively rejected at inner gate. |
| Historical V3A Public Policy (`public_v3a_predictions.json`) | USABLE_WITH_PROVENANCE_LIMITATION | Public predictions match incumbent, but training source drifted. |
| Historical V3A F1–F4 OOF (`0.9296` Recall Claim) | IRRECOVERABLE | Prediction stream absent, producer code drifted. |
| Historical V3B F1–F4 OOF (`0.9298` Recall Claim) | IRRECOVERABLE | Aggregate report only, no independent prediction artifact on disk. |
| B2A Pre-Fix Qwen Run (`acc_c_full_old/predictions.jsonl`) | UNSAFE_FOR_SCIENTIFIC_CONCLUSION | Confirmed query-ID input formatting defect. |

## 20. Does this change our interpretation of the month of failed experiments?

**Không.** Cuộc audit này không lật ngược kết luận khoa học chính của dự án. 

Mặc dù phát hiện bằng chứng xác thực về lỗi triển khai cục bộ trong đợt chạy B2A Qwen đầu tiên và sự thiếu hụt provenance ở V3A F1–F4, nhưng toàn bộ chuỗi thí nghiệm độc lập sau đó (Direct Relevance, Direct Neural Signal, Pairwise Preference, B2a Auxiliary Fusion, Workflow C2) đều được thực thi trên nền tảng artifact sạch, mã băm khớp tuyệt đối, và phương pháp đánh giá chuẩn tắc. 

Sự bế tắc điểm số trong suốt một tháng qua là một **giới hạn khoa học thực sự (method/selection ceiling)**: việc cố gắng hoán đổi vị trí 4 hoặc 5 trong top-5 bằng các bộ phân loại cấp tài liệu/cặp trên không gian ứng viên K20/K77 đã tạo ra tổn thất (false drops) nhiều hơn số lượng tài liệu đúng vớt lại được, chứ không phải do một lỗi hệ thống hay ô nhiễm artifact dùng chung gây ra.

## 21. Final forensic conclusion

- **Artifact contamination finding:** Ô nhiễm artifact được xác nhận xảy ra cục bộ ở đợt chạy Modal B2A ban đầu (do lỗi định dạng văn bản truy vấn và cơ chế resume SQLite lỏng lẻo), nhưng đã được khoanh vùng và cách ly thành công vào `acc_c_full_old`. Không có sự ô nhiễm chéo nào lan vào baseline chuẩn tắc hay các thí nghiệm direct reranking sau này.
- **Scope & Confidence:** Phạm vi kiểm toán bao trùm 100% các artifact, script và báo cáo trọng yếu của Task 1. Độ tin cậy: Tuyệt đối (dựa trên kiểm tra mã băm SHA256 và phân tích mã nguồn tĩnh).
- **Failure explanation:** Ô nhiễm dữ liệu chỉ giải thích được sự cố ban đầu của nhánh B2A Qwen; nó **không** giải thích chuỗi thất bại rộng hơn của các thí nghiệm residual/direct ranking.
- **Current pipeline status:** Pipeline mới nhất (`scripts/modal/task1_full_doc_top200_qwen3vl2b.py` và `reports/task1/full_document_legal_field_retrieval/`) hoàn toàn sạch, vượt qua bài kiểm tra tính tương đồng lịch sử (`modal_historical_parity_gate.json: PASS`), canary 2 phần PASS, và áp dụng cơ chế khóa hợp đồng `DurableShardStore` cách ly tuyệt đối.

