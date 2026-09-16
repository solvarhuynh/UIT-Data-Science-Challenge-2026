# Change log

## 2026-09-15 — Foundational system root-cause audit

- Added `reports/task1/foundational_system_root_cause_audit.md`.
- Performed a read-only forensic audit of query/fold identity, gold-to-corpus parity, raw-to-vector payload chunk parity, candidate ancestry, BGE and Qwen coverage boundaries, baseline evaluator behavior, direct-experiment failures, and historical reproducibility drift.
- No training, inference, model changes, submission changes, or historical-artifact modifications were performed.
- Audit verdict: `NO_SHARED_BUG_FOUND_METHOD_CEILING_MORE_LIKELY`.

## 2026-09-15 — Common representation universe parity validation

- Added `reports/task1/common_representation_universe_parity_validation.md`.
- Constructed the F1–F4 K20-plus-baseline-anchor q-doc universe in memory only, then attached persisted BGE/Qwen/source availability and retrospective labels.
- Verified the required corrected-Qwen SHA256 and exact query-identity joins.
- No training, inference, reranking, score imputation, model change, submission change, or historical-artifact modification was performed.
- Validation status: `REPRESENTATION_MISMATCH_IS_SECONDARY_LIMITATION`; safe next action: `MOVE_TO_RETRIEVAL_REPRESENTATION_OR_STOP_MODEL_SEARCH`.

## 2026-09-15 — PAIRWISE_DOCUMENT_PREFERENCE_TOP5

- Added `scripts/analysis/task1_pairwise_document_preference_top5.py` and isolated output directory `reports/task1/pairwise_document_preference_top5/`.
- Preflight found no prior scientifically evaluated document-level within-query relevance-preference ranker over the F1–F4 K20-plus-anchor universe; existing P1/action and Fold0 pairwise work was excluded as non-duplicate.
- Ran exactly one strict F1–F4 outer-OOF LogisticRegression experiment with fixed mirrored relevant-versus-nonrelevant pairs and fixed configuration.
- Result: `PAIRWISE_DOCUMENT_PREFERENCE_FAIL`; recall `0.9250803571428571` (delta `-0.0008482142857144659`), precision delta `-0.0001785714285714446`; F1 and F3 recall deltas were negative.
- High pairwise concordance (`0.9551691741697003`) did not translate to robust top-5 selection. No tuning, alternate model, rerun, retrieval, inference, submission, or historical-artifact modification was performed.

## 2026-09-15 — FULL_DOCUMENT_LEGAL_FIELD_RETRIEVAL preflight

- Added the authorized single-configuration retrieval script and isolated blocked-state artifacts under `reports/task1/full_document_legal_field_retrieval/`.
- Representation audit confirmed existing BM25 is chunk-text retrieval, while the proposed canonical full-document raw-passage representation is distinct.
- Stopped before index construction because the active environment lacks `rank_bm25`, required by the repository's existing `BM25Retriever` (`ModuleNotFoundError`).
- No package was installed and no TF-IDF, alternate retrieval engine, retrieval result, label attribution, model, or reranking fallback was used.

## 2026-09-15 — FULL_DOCUMENT_LEGAL_FIELD_RETRIEVAL authorized dependency repair

- Installed exactly the authorized package `rank-bm25==0.2.2` into `C:\\Users\\Nghia\\AppData\\Local\\Programs\\Python\\Python312\\python.exe`; `BM25_IMPORT_OK` passed.
- No other package was installed.
- The frozen repository tokenizer then failed before retrieval because `pyvi.ViTokenizer` is unavailable. `pyvi` was not authorized for installation and no alternate tokenizer was used.
- Recorded `BLOCKED_EXECUTION_CONTRACT` in a new execution-blocker artifact while preserving the earlier blocked-state artifacts. No index, retrieval output, label evaluation, model, reranking, or submission was produced.

## 2026-09-15 — HARD-60 membership resolution

- Reconstructed exact KNOWN66 from the preserved canonical pool200 artifact: 66 unique F1–F4 q-doc rows, no duplicates.
- Searched persisted reports, artifacts, and scripts for an exact hard-60 list and identity-level historical dense/BM25/word/char recovery traces; none was found.
- Verified the preserved BGE compact source explains only one known-66 occurrence, hence the prior 65-row proxy. The five other historical recoverable identities are unavailable.
- Did not create an unauditable hard-60 membership artifact, did not rerun retrieval, and did not apply the preregistered gate.
- Status: `BLOCKED_HARD60_MEMBERSHIP_UNRESOLVED`.

## 2026-09-15 — FULLDOC fixed-source integration preflight

- Analyzed only the frozen full-document top-500 artifact by exact q-doc identity; no retrieval, label-based budget selection, inference, reranking, or top-5 evaluation was run.
- Recorded structural novel-candidate volumes and reusable BGE/Qwen coverage for TOP10/TOP20/TOP50.
- Confirmed deterministic canonical document/chunk mapping, but found no preserved reproducible F1–F4 materialization-and-BGE-scoring path for new full-document candidates. The closest surviving scorer is public-only and CUDA-required.
- Status: `BLOCKED_NO_REPRODUCIBLE_NEW_CANDIDATE_SCORER`.

## 2026-09-15 — Incumbent BGE scorer parity preflight

- Traced the surviving incumbent BGE contract: byte-pinned `models/reranker`, canonical payload chunks, query/chunk pair tokenization, max length 512, CUDA fp16 one-logit chunk scores, then document `MAX(bge_score)` aggregation.
- Stopped before parity-set or new-candidate scoring because the expected local `models/reranker` files are absent; no byte-identical incumbent model/tokenizer can be verified.
- Local PyTorch is CPU-only with no CUDA, recorded as a secondary resource fact.
- Status: `BLOCKED_BGE_MODEL_PROVENANCE`; no labels, scoring, model download, or integration occurred.

## 2026-09-15 — Incumbent BGE model provenance recovery

- Recovered exact required model/tokenizer file hashes and scorer semantics from the surviving reference runner.
- Searched repository, git/LFS, artifact/checkpoint locations, and local HuggingFace/Windows caches. No matching model bytes, LFS object, public model ID, or immutable revision was found.
- Verified only 21 of 6,523 FULLDOC_TOP10 new q-docs have reusable persisted incumbent BGE scores; no broader matching score cache exists.
- Classified provenance as `BGE_MODEL_PROVENANCE_IRRECOVERABLE`; no download, inference, copying, model substitution, or integration was performed.

## 2026-09-15 — FULLDOC_TOP10_COMMON_BM25_SLOT5 runtime observation

- The single authorized label-free BM25 prediction process (PID 4988) was allowed to continue without restart or configuration change.
- It accumulated sustained CPU time and a peak observed working set above 4.8 GB, then exited before writing `full_doc_top10_common_bm25_slot5_predictions.jsonl`.
- No prediction artifact exists, therefore no SHA256 freeze boundary was reached and no relevance labels or final evaluation were opened.
- Added a dormant post-freeze evaluation script only; it was not executed. No retrieval variant, rerank, BGE/Qwen inference, final-top5 result, or submission was produced.
- Status: `BLOCKED_PREDICTION_ARTIFACT_NOT_MATERIALIZED`; a new authorization is required before any memory-safe rerun or repair.

## 2026-09-15 — FULLDOC_TOP10_COMMON_BM25_SLOT5 memory-safe repair/rerun

- Replaced only the prediction implementation: streamed the frozen 2,800,000-row full-document artifact, retained TOP10 plus incumbent rank5 state, reused 5,222 frozen incumbent scores, and recomputed only 378 missing incumbent scores one query at a time.
- Did not rerun TOP500 retrieval, alter representation/tokenizer/BM25 parameters/tie-breaks, access relevance labels before freeze, or change ranks 1–4.
- Persisted parity before predictions: 256/256 exact score matches, maximum/mean/median absolute difference 0.0, zero ordering mismatches (`PASS`).
- Atomically froze 5,600 valid unique-top5 predictions before opening labels; SHA256 `ba4e8b551674fe003a25935b2cb5e60045aab4587d80dd9bf90087e6ecf2c6d6`.
- External process observations showed peak RSS 862,924,800 bytes, compared with the prior failed attempt above 4.8 GB; the runner-local ctypes sampler was defective and its zero readings were explicitly replaced with observed process-level telemetry.
- After freeze only, canonical baseline reproduced exactly. Final status: `FULLDOC_SLOT5_FAIL`: Recall@5 `0.9155119047619048` (delta `-0.01041666666666663`), Precision@5 `0.19485714285714287` (delta `-0.00253571428571428`).
- Interpretation: `NEW_RETRIEVAL_SIGNAL_EXISTS_BUT_BM25_ONLY_SLOT5_SELECTION_IS_INSUFFICIENT`. Historical Known-66 is correctly reconstructed as 66 identities; 12 occur in full-doc TOP200 but none is selected into final slot5.

## 2026-09-16 — FULLDOC_NEW_SIGNAL_LOCALIZATION_AUDIT

- Performed read-only analysis of the SHA-verified frozen 2.8M-row full-document retrieval artifact, exact Known-66 membership, canonical pool200, frozen slot5 predictions, and retrospective F1–F4 labels. No retrieval, BM25 rebuild, inference, training, selector, or final-top5 generation was run.
- Exact Known-66 localization: 12 recovered at TOP200; only one is in TOP10 and it lost the deterministic BM25 comparison against incumbent rank5. The other 11 are outside TOP10.
- Across all genuinely new relevant q-docs, cumulative recovery is TOP5 `0`, TOP10 `1`, TOP20 `2`, TOP50 `3`, TOP100 `5`, TOP200 `12`, TOP500 `19`.
- TOP10 exposed only `1/12 = 8.33%` of new relevant q-docs available by TOP200 and utility mass `0.5/7.0333 = 7.11%`; 7 of 12 and utility mass `4.3667` lie at ranks 101–200.
- Diagnosis: `USEFUL_NEW_SIGNAL_IS_PRIMARILY_DEEPER_THAN_TOP10`; safe next action is `DESIGN_COMPUTE_FEASIBLE_DEEP_CANDIDATE_SCORING_STRATEGY`.

## 2026-09-16 — FULLDOC_TOP200_QWEN_DEEP_SCORING_FEASIBILITY

- Performed a label-free, inference-free feasibility audit of the SHA-verified full-document TOP200-new universe and the pinned corrected Qwen contract. No model call, retrieval, environment change, GPU/Modal launch, or final selection occurred.
- Target universe is 598,071 q-docs across 5,600 queries and 8,258 documents; no exact corrected Qwen score is reusable. Historical A10 throughput projects 1,794,213 units / 13.71 hours for TOP200.
- Exact Qwen model/revision, snapshot hashes, corrected-query text contract, text-only prompt, MAX_LENGTH 8192, batch size 1, top-3 within-document chunk policy, and MAX document aggregation were recovered from preserved run-state and runner source.
- Found 5 target q-docs from 3 documents with fewer than the exact required three canonical chunks. Therefore the complete target cannot be materialized under the unchanged Qwen contract.
- Status: `BLOCKED_QWEN_CANDIDATE_MATERIALIZATION`; safe next action: `CLOSE_DEEP_QWEN_SCORING_PATH`.

## 2026-09-16 — Qwen short-document materialization contract resolution

- Traced the actual historical selector and worklist validator, rather than relying on a prose “top-3” summary. `select_true_s2_prepared` slices `ordered[:topk]`; the worklist validator explicitly permits contiguous ranks `1..len(selected)` where `len(selected) <= 3`.
- Historical semantics are `TOP3_UP_TO_AVAILABLE`, not exactly three, with no padding/repetition: the 431,200-pair corrected run contained 184 one-chunk and 34 two-chunk q-doc groups, all represented in persisted document-score output.
- Identified the three current two-chunk documents and five affected target q-docs by canonical source only; no relevance labels were inspected.
- Root cause: `HISTORICAL_SCORER_ALREADY_SUPPORTS_SHORT_DOCS`. The earlier materialization failure was an audit interpretation error, not a missing-chunk or scorer-domain restriction.
- Full target is materializable under the unchanged contract: 598,071 q-docs, zero exclusions, 1,794,208 units, projected Modal A10 runtime 13.7104 hours. Status: `QWEN_TOP200_MATERIALIZATION_FULL_PASS`.

## 2026-09-16 — AUDIT_EXACT_MODAL_NEW_ACCOUNT_UPLOAD_SET

- Performed read-only source/dependency audit for the fresh Modal account. No Volume creation, upload, Modal launch, GPU, or inference occurred.
- Confirmed image-embedded inputs: 32 production worklists, universe manifest, historical parity sample, and current canary under `/opt/...`; these do not require manual Volume upload.
- Confirmed Volume-required runtime inputs: `train.json`, canonical chunk files, and historical Qwen `run_state.json`; Qwen weights are downloaded at runtime from the pinned public Hugging Face revision.
- Full production worklist references 8,261 canonical documents (1,802,587,393 bytes); canary parity/current samples require 214 documents (89,960,215 bytes). Train and run-state hashes were recorded in the audit report.
- Status: `BLOCKED_DEPENDENCY_AMBIGUOUS` because the local Modal CLI and Python module are absent, preventing verified account/profile and current upload-command syntax. No guessed commands were executed.

## 2026-09-16 — REPAIR_MODAL_LOCAL_PYTHON_SOURCE_PACKAGING

- Repaired the Modal 1.5.5 local-source packaging defect in `scripts/modal/task1_full_doc_top200_qwen3vl2b.py`: explicitly package the unchanged historical scorer and durability sibling modules at `/root`, plus `scripts/beam/task1_v2/evidence.py` at its import-compatible package path.
- Added only a CPU-only remote import smoke endpoint; it imports modules, hashes the historical scorer, and checks image-embedded artifacts. It does not mount/read the Volume, load Qwen, download weights, use GPU, or infer.
- Local import validation passed for all three required import names. Historical scorer SHA remained `e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe`; scientific constants were unchanged.
- Remote execution could not be verified: sandboxed network failed with Modal gRPC connection error 10013, and the subsequent escalated CPU-smoke request was rejected because it would export local source and embedded legal-retrieval artifacts to the Modal account. No remote import smoke or Volume test executed.
- Status remains `BLOCKED_REMOTE_IMPORT_PACKAGING` pending explicit user approval to transmit the stated local payload to Modal.

## 2026-09-16 — FINAL_MODAL_QWEN_PRELAUNCH_INTEGRITY_AUDIT

- Verified the two major frozen SHA256 values exactly and reconstructed the corrected query/chunk/prompt/model/scoring lineage without running inference or using labels.
- Recomputed the full label-free target universe: 598,071 q-docs, 5,600 queries, 8,258 documents, exact rank bands, 1,794,208 inference units, and zero query/document/chunk mapping failures. Canonical target serialization SHA256 is `432dbe5989c39a2812a53991a21122e28490fa7653475226ca88d35e545d9beb`.
- Verified all 5,600 canonical `question` values are present, non-empty, and never equal their query ID; confirmed baseline-top5 corrected-Qwen gaps are exactly 122 identities / 366 prospective units.
- Static gate blocked before any Modal canary: no executable FULLDOC_TOP200_NEW runner/worklist exists; the historical runner lacks periodic `volume.commit()` and explicitly rejects non-empty checkpoints, so interruption-safe durable resume is not established.
- Created `reports/task1/full_document_legal_field_retrieval/full_doc_top200_qwen_prelaunch_manifest.json` with source hashes, recovered contract, full audit counts, and failed persistence/resume gates. No scorer was modified and no Modal/GPU call was launched.
- Status: `BLOCKED_OUTPUT_PERSISTENCE`; decision: `DO_NOT_LAUNCH_FULL_MODAL_RUN`.

## 2026-09-16 — FULLDOC TOP200 Modal durability repair attempt

- Added a separate operational runner `scripts/modal/task1_full_doc_top200_qwen3vl2b.py`; the historical Qwen runner and frozen predictions were not modified.
- Added a generation-based SQLite durability layer with two-phase Modal Volume commits, contract/universe/shard fingerprints, fail-closed checkpoint validation, idempotent q-doc resume, unique q-doc/chunk keys, deterministic tripwires, and COMPLETE shard manifests.
- Built and verified 32 deterministic contiguous worklist shards for 598,192 future q-docs / 1,794,571 units. Universe SHA256: `28ee4cc484eccd52e0013bd9b0bd512cc4911e675d7ee8aa84413af72f83b13d`; all shard hashes/counts/order checks passed.
- Frozen canary inputs: historical parity 256 rows with exact 32/32/192 chunk-count strata; current universe 16 rows covering all four rank bands and all five short-document q-docs.
- Local interruption/resume test passed: durable completed skip 1, incomplete resume 1, duplicates 0, missing 0; wrong contract, wrong universe, and corrupt SQLite were all rejected.
- Modal CPU cross-container test was attempted but the service rejected execution before the function ran because the workspace exceeded its spend limit. Consequently A10 parity/current canaries were not launched.
- Report: `reports/task1/full_document_legal_field_retrieval/full_doc_top200_qwen_durability_repair_report.json`. Status remains `BLOCKED_OUTPUT_PERSISTENCE`; paid full campaign remains forbidden.
