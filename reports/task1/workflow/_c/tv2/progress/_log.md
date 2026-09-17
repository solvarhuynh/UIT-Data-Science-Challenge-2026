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

## 2026-09-16 — MODAL_PACKAGING_AND_VOLUME_PASS

- Exercised only the explicitly authorized CPU-only Modal import, Volume-write, and separate Volume-verify checks. No GPU/A10 allocation, Qwen inference, parity/current canary, production shard, or submission operation was invoked.
- Remote import smoke passed. Remote import paths were `/root/task1_full_doc_top200_qwen3vl2b.py`, `/root/task1_b2a_qwen3vl2b.py`, `/root/task1_full_doc_top200_durability.py`, and `/root/scripts/beam/task1_v2/evidence.py`; all frozen image-embedded inputs were present. Evidence: `modal_remote_import_smoke.json` SHA256 `d883fb9b0741423e7e25003f4645b69023bfdfe7ce656b48ab1b02d95ade8db0`.
- Fixed the operational image binding for the CPU Volume endpoints after their bare image could not import the already-authorized sibling scorer. The endpoints now use the existing packaged image but retain `cpu=1` and specify no GPU. This changed no scientific scorer/model/revision/prompt/batch/max-length/dtype setting.
- Separate successful CPU Volume invocations wrote then reloaded `/workspace/p13/runtime/full_doc_top200_qwen/volume_test/payload.json`. Write and verify SHA256 both equal `4ce190287e5bdad3855d52add00528160e9f56ba188c5bc9c884c10b3b31f3d6`; evidence artifacts: `modal_volume_write.json` SHA256 `5a209f9c9e04181cb99ca42cf59099e84d1b1a82dbd9a9339fae2e48eafd0fcd` and `modal_volume_verify.json` SHA256 `e8fdc84ccf80362f54e737e4196fa6eec8bad3f3403807973722c58956202c40`.
- Final runner SHA256: `2fff2d5a0f5cdf0626e5775cd24c87b8d6f82181b23e07cf46b07d546cdf0f99`; unchanged historical scorer SHA256: `e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe`. Qwen weights were not downloaded: no model-download path was called.
- Status: `MODAL_PACKAGING_AND_VOLUME_PASS`.

## 2026-09-16 — RUN_HISTORICAL_QWEN_PARITY_256_ON_MODAL_A10

- Pre-GPU checks passed exactly: runner SHA256 `2fff2d5a0f5cdf0626e5775cd24c87b8d6f82181b23e07cf46b07d546cdf0f99`; historical scorer `e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe`; durability `fec80d52e4fe5590b95f511b945f37a8eb200ca2a6a68ad4d8a7b5d55bfa5540`; evidence `0a57b5882c4c492050432978a48d24bd8746dca99f25a1622c68a2bcb46e17c0`; frozen historical parity sample `11cc2f98e821a0e968d1c73da90c047a350f23a796827fafd4114fd26bd7cbfa`.
- The frozen sample was not regenerated or changed: 256 q-docs with 32 one-chunk, 32 two-chunk, and 192 three-chunk rows (672 expected units). The static contract remained pinned to Qwen/Qwen3-VL-Reranker-2B revision `4bd860ac4f15ad1897a214615cccc700f8f71818`, batch 1, bfloat16, max length 8192, canonical `record["question"]`, `select_true_s2_prepared`, and MAX aggregation.
- Launched exactly one historical-parity A10 invocation in Modal profile `nghiadethuong3107`; no current canary, production shard, or retry was launched. The runner downloaded the pinned model snapshot (20 files) and began execution, but stopped at the first frozen-input validation before any q-doc was scored.
- Exact blocker from Modal app `ap-pktiglQcUaZkbUutbR9Meo`: `RuntimeError: frozen chunk-selection mismatch: 6/3689`. The failure occurred in `selected_chunks` before the first-tripwire artifact, `score_batch`, chunk-score commit, output, or parity comparison. Thus produced units/rows are zero, no fresh document scores exist, and GPU identity/revision-resolution telemetry was not returned by the current runner.
- A checkpoint namespace and `checkpoints/historical_parity/shard_00.sqlite3` appeared under the authorized Volume, but no output shard, manifest, or completed-parity result was persisted. No scientific source, model, prompt, selector, or frozen input was modified.
- Status: `BLOCKED_HISTORICAL_PARITY_INPUT_MISMATCH`. Safe next action: `FIX_ONLY_THE_IDENTIFIED_PARITY_BLOCKER`.

## 2026-09-16 — REPAIR_HISTORICAL_PARITY_FROZEN_CHUNK_REPLAY

- Confirmed the first mismatch without labels: frozen `6/3689` IDs are `3689_article_4_clause_1`, `3689_article_8_clause_3_point_b_part_2`, and `3689_article_4_clause_2`; all are canonical, document-scoped, and non-empty. The present selector instead returns `3689_article_5_clause_1`, `3689_article_8_clause_3_point_b_part_2`, and `3689_document_context_4_part_2`.
- Full frozen-input audit passed: 256 valid q-doc rows, 672 resolved units, and zero missing, duplicate, cross-document, or empty-text frozen IDs. The frozen sample remains SHA256 `11cc2f98e821a0e968d1c73da90c047a350f23a796827fafd4114fd26bd7cbfa`.
- Downloaded and audited only the failed historical-parity checkpoint. SQLite integrity is `ok`; `document_results=0`, `chunk_scores=0`, `durable_generation=0`, and completed identities are zero. It is safe to reuse the historical-parity namespace.
- Repaired only `historical_parity`: it now validates and replays `selected_chunk_ids` in frozen order from canonical files before model download. Current-canary and production continue to call `select_true_s2_prepared` unchanged. Added operational parity-report fields only (GPU/cache/path and p95 evidence); historical scorer/model/prompt/tokenizer/batch/dtype/max-length/aggregation were not changed.
- CPU preflight passed: 256 rows, 672 frozen units, 65 canonical non-empty query texts, and exact frozen chunks for `6/3689`. New runner SHA256 `e5d01ad40a036544cfc2c62c1104302eff0f711b0f11f129fb1a3122fef96c2a`; scorer/durability/evidence hashes remain `e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe`, `fec80d52e4fe5590b95f511b945f37a8eb200ca2a6a68ad4d8a7b5d55bfa5540`, and `0a57b5882c4c492050432978a48d24bd8746dca99f25a1622c68a2bcb46e17c0`.
- The requested A10 retry was not launched: the environment rejected the command under its still-active trusted CPU-only authorization constraint. No retry artifact was transmitted and no GPU was allocated. Status: `BLOCKED_HISTORICAL_PARITY`; safe next action: obtain an execution-policy authorization that supersedes the earlier CPU-only limit, then run exactly one frozen-replay parity retry.

## 2026-09-16 — HISTORICAL_QWEN_PARITY_PASS

- Under the explicit one-time A10 superseding authorization, exactly one retry ran: Modal app `ap-B8vd4dgWnP8Rnom7S36LMA`, profile/workspace `nghiadethuong3107`. No second GPU invocation, current canary, or production shard was launched.
- Runtime contract passed: actual GPU `NVIDIA A10`; CUDA was available (the unchanged loader rejects CPU fallback); model `Qwen/Qwen3-VL-Reranker-2B`; resolved revision `4bd860ac4f15ad1897a214615cccc700f8f71818`; batch 1; bfloat16; max length 8192. Snapshot downloaded to `/tmp/hf-cache/models--Qwen--Qwen3-VL-Reranker-2B/snapshots/4bd860ac4f15ad1897a214615cccc700f8f71818`.
- Frozen replay passed exactly: 256 q-docs, 672 chunk units, exact frozen IDs, 256 exact score matches, max/mean/median/p95 absolute difference all `0.0`, NaN/Inf `0`, ordering mismatches `0`. q6/doc3689 tripwire persisted the exact frozen IDs `3689_article_4_clause_1`, `3689_article_8_clause_3_point_b_part_2`, `3689_article_4_clause_2`; query text was canonical and query SHA256 was `827e1e3c5e6b0e11c49c205561504a699bb9647e1952b019a5906bf9b3254587`.
- Persistence passed under `/workspace/p13/runtime/full_doc_top200_qwen/`: checkpoint `/checkpoints/historical_parity/shard_00.sqlite3`, output `/shards/historical_parity/shard_00.jsonl`, manifest `/manifests/historical_parity/shard_00.json`; output SHA256 `90a82615b318b3b7b6a80950be8e5c0f72e0d3654dcb5a60642c1c232b21d0aa`; contract SHA256 `ae1fba276359fd809221b816a6be1c933786475be30219593805d329ac821d5e`; Volume commit completed.
- Evidence artifacts: replay result SHA256 `404a559246aad3d707a986be30565b163c10fb5b23fd6d82d2912a0fb7b5fa65`; manifest SHA256 `0cd47e72d641d5cf0550a717437b1c25fd57cd168d5c65f158aa33cc22f71fc9`; gate SHA256 `9cfc5e52c24f0b7356a7e0222f9b548f3f01dce217d36de25e6794c821142c77`; tripwire evidence persisted separately.
- Status: `HISTORICAL_QWEN_PARITY_PASS`; final decision: `SAFE_FOR_CURRENT_CANARY_REVIEW`. Stop here; current-canary execution requires separate authorization.

## 2026-09-17 — CURRENT_CANARY_PREFLIGHT

- No Modal GPU, Qwen download, inference, current-part1, or current-part2 was launched. Source hashes: runner `e5d01ad40a036544cfc2c62c1104302eff0f711b0f11f129fb1a3122fef96c2a`; scorer `e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe`; durability `fec80d52e4fe5590b95f511b945f37a8eb200ca2a6a68ad4d8a7b5d55bfa5540`; evidence `0a57b5882c4c492050432978a48d24bd8746dca99f25a1622c68a2bcb46e17c0`; current sample `d70264701534114084c972f83588411c3ad99b40e21a0014f23f5850e235348d`.
- Current sample audit passed: 16 unique q-docs, 43 expected units, chunk distribution 11×3 and 5×2. Rank-band coverage is 3 rows in 1–50, 7 in 51–100, 1 in 101–150, and 5 in 151–200. Short-document rows are `22160/125611`, `37344/176011`, `68272/198337`, `94980/176011`, and `167242/198337`, each selecting 2 chunks.
- Query gate passed for all 16 rows: missing, empty, query-id-as-text, and fallback counts are all zero. Query SHA256s were computed without printing full texts. Chunk gate passed 16/16: canonical documents present, selector counts valid, IDs unique/document-scoped, and raw texts non-empty. The current selector produced the exact IDs recorded in the preflight evidence.
- Historical frozen replay remains isolated to `kind == historical_parity`; current-canary and production paths still call `select_true_s2_prepared(question, prepare_document(rows), topk=3)` with BM25-descending/chunk-ID-ascending and TOP3_UP_TO_AVAILABLE semantics. Current/full behavior is unchanged.
- Current checkpoint namespace `/workspace/p13/runtime/full_doc_top200_qwen/checkpoints/current_canary/shard_00.sqlite3` does not exist, so it is clean before part1; no dirty checkpoint was found. Historical output remains present and its downloaded SHA256 is `90a82615b318b3b7b6a80950be8e5c0f72e0d3654dcb5a60642c1c232b21d0aa`.
- Resume semantics derived from code: part1 consumes first 8 rows (24 units), `completed_before=0`, `scored_now=8`, `completed_after=8`; separate part2 sees `completed_before=8`, skips those rows, scores 8 remaining rows (19 units), and finishes `completed_after=16` with complete output/manifest. Durable generation, upsert primary keys, incomplete-generation discard, and contract fail-closed behavior are provided by `DurableShardStore`.
- Status: `CURRENT_CANARY_PREFLIGHT_PASS`; final decision: `READY_FOR_USER_TO_RUN_CURRENT_PART1`.

## 2026-09-17 — FINAL_FULLDOC_TOP200_QWEN_PRODUCTION_LAUNCH_PREFLIGHT

- Performed read-only production preflight only; no Modal GPU, Qwen download, inference, shard launch, label access, or orchestration command was executed.
- Current source hashes: runner `e5d01ad40a036544cfc2c62c1104302eff0f711b0f11f129fb1a3122fef96c2a`; historical scorer `e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe`; durability `fec80d52e4fe5590b95f511b945f37a8eb200ca2a6a68ad4d8a7b5d55bfa5540`; evidence `0a57b5882c4c492050432978a48d24bd8746dca99f25a1622c68a2bcb46e17c0`.
- Frozen production audit passed structurally: all 32 worklists exist; 598,192 unique q-doc identities; 1,794,571 units; 8,261 unique canonical documents; zero duplicate identities; zero missing canonical chunk files; per-shard counts match the manifest. Units range 56,078–56,082, mean 56,080.34375. At measured 36.3513065193 units/sec, projected total is 13.7132 A10 hours (13.7104 hours was the prior 1,794,208-unit new-only estimate).
- Blocking identity mismatch: required manifest SHA `404d5926eb42f4a3c84b77bd96782654e33854a8eb16f2d67234821d3d37d5ce`; actual current authoritative manifest SHA `1f26dde0bb3a03c5482a0c2d03a1a1d47e40ca53ac2d8410bb9c6022811e5722`. No manifest or worklist was modified.
- Production checkpoint namespace `/workspace/p13/runtime/full_doc_top200_qwen/checkpoints/production/` is absent; no production checkpoints, outputs, or COMPLETE manifests exist. Historical parity and current-canary namespaces remain separate and present.
- The current runner has no `production-shard` Modal function/local CLI mode. Because the authoritative production manifest SHA is already blocked, no orchestration code was added and no launch command was issued. Status: `BLOCKED_PRODUCTION_UNIVERSE`; final decision: `DO_NOT_RUN_PRODUCTION`.

## 2026-09-17 — PRODUCTION_MANIFEST_FORENSICS_AND_ONE_SHARD_ORCHESTRATION

- Preserved the pre-repair manifest/worklist/builder/runner hashes and repository state. Exact old `404d5926...` manifest bytes were not recoverable; the hash appears only in the historical durability report.
- Reconstructed the distinction: manifest file SHA covers serialized metadata, source hashes, canary hashes, paths, and shard metadata; `universe_sha256` is independently derived from ordered `shard_id + worklist_sha256` pairs. The manifest file SHA is not the universe identity.
- Isolated deterministic rebuild with the unchanged builder reproduced all 32 worklists byte-for-byte, parity SHA `11cc2f98...7cbfa`, current canary SHA `d7026470...5348d`, q-docs `598192`, units `1794571`, and universe SHA `28ee4cc4...f83b13d`. The only temporary semantic differences were output path strings; canonical-path serialization reproduced current manifest SHA `1f26dde0bb3a03c5482a0c2d03a1a1d47e40ca53ac2d8410bb9c6022811e5722`. Classification: `MANIFEST_METADATA_DRIFT_ONLY`; current manifest is authoritative. Provenance note: `reports/task1/full_document_legal_field_retrieval/full_doc_top200_qwen_manifest_provenance_note.md`.
- Added only minimal one-shard production orchestration to the runner: strict `00..31` validation, current authoritative manifest SHA check, worklist hash/count/identity validation, `production_shard` Modal function, and `--mode production-shard --shard-id XX` local dispatch. It delegates to existing `score_rows(kind="production", complete=True)` and leaves scorer, selector, model, and all existing modes unchanged. New runner SHA: `abc4a47e8be6d6dcb09c9cd8486abff584a345a8dc3c259758bc433ff1297cc8`.
- CPU dispatch test passed for shard 00: 18,693 q-docs, 56,079 units, worklist SHA `ca544ce68f46b8a797542a340c972a9aebc75218525818f88ceb77df5ff8d432`; invalid IDs (`0`, `32`, `aa`, empty) rejected; manifest hash mismatch rejected. Durability self-test passed: fresh/resume/incomplete rollback, duplicate protection, contract/universe/corrupt checkpoint fail-closed.
- Production Volume namespace remains clean/absent; historical and current-canary namespaces were not modified. Temporary rebuild directory was removed after comparison. Status: `PRODUCTION_LAUNCH_PREFLIGHT_PASS`; final decision: `READY_FOR_USER_TO_RUN_PRODUCTION_SHARD_00`.

## 2026-09-17 — OPTIMIZED_QWEN_RUNNER_PREFLIGHT

- Added a separate optimized runner `scripts/modal/task1_full_doc_top200_qwen3vl2b_optimized.py` (SHA256 `82aaa1351c5955989da3a3e194ee38105d690e440b2bc00d872a035686c64917`) with a new Volume namespace `/workspace/p13/runtime/full_doc_top200_qwen_optimized/`; the existing production runner/checkpoint was not overwritten or touched.
- Added CPU-only preprocessing helper `scripts/modal/task1_full_doc_top200_qwen_preprocessing.py` (SHA256 `c9622300ed2c613978335a1c037d111fb14a8b4ad25e5d33841ed73fd795f896`) and audit harness `scripts/modal/audit_task1_full_doc_top200_optimized.py` (SHA256 `369d78ed32ca155532331df66e9c57486751b7aca4432ddc892840155c4205ca`). The historical scorer SHA remains `e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe`; model/revision/prompt/max-length/batch/selector/aggregation were not changed.
- Optimized path loads canonical questions once per container and caches one canonical document preparation per unique document; query-dependent BM25 selection remains per q-doc. Durable flushes print `PROGRESS shard=...` without per-chunk logging and persist to the new namespace only.
- CPU equivalence: historical parity `256/256`, current canary `16/16`, deterministic production sample `100/100`; preprocessing mismatches `0`, including exact query text, selected chunk IDs/order, raw chunk text, unit count, and formatted scorer inputs.
- Bounded CPU benchmark on a deterministic 100-row shard-00 spread: old preprocessing `2197.142116s`, optimized preprocessing `35.324514s`, `62.1988x` CPU preprocessing speedup. An unbounded 1,000-row attempt was stopped after exceeding 80 minutes CPU time; no GPU/model run occurred. This CPU speedup is not a claimed end-to-end GPU speedup.
- Historical throughput classification: `HISTORICAL_THROUGHPUT_PARTIALLY_COMPARABLE`. Model/revision/batch/scorer/unit semantics match, but historical timing starts after model load and uses preselected worklist chunks with raw-document caching; current production includes startup and repeats query/document preprocessing in the q-doc loop.
- Prepared manual-only canary command (not run): `.venv\\Scripts\\modal.exe run scripts/modal/task1_full_doc_top200_qwen3vl2b_optimized.py --mode optimized-current-canary`. It uses the new optimized namespace and frozen current-canary 16 rows/43 units.
- Modal mutations: `0`; GPU/model runs: `0`.

## 2026-09-17 — DEADLINE_OPTIMIZATION_READY_CHECK

- Re-read the optimization diff and stopped all long CPU benchmarking as authorized; no 1,000-row rerun, GPU run, Modal inference, or Volume mutation occurred.
- Bounded current-canary CPU equivalence rerun passed `16/16` rows with `0` mismatches, including exact query text, selected chunk IDs/order, raw chunk text, and inference-unit counts. The sample contains 11 three-unit rows and 5 two-unit rows.
- Code inspection confirms questions are loaded once per container, canonical lookup is O(1), document source/index is loaded once, invariant document preparation is cached/reused, query-dependent selection remains per q-doc, and scorer/model/prompt/max-length/batch/aggregation are unchanged.
- Existing production shard-00 checkpoint remains preserved. Optimized runner uses an isolated namespace and is not proven resume-compatible with the incumbent checkpoint (`UNPROVEN`), so the tiny benchmark must use its own namespace.
- Status: `DEADLINE_OPTIMIZATION_READY`; GPU runs `0`; Modal mutations `0`.

## 2026-09-17 — BATCHED_QWEN_OPTIMIZATION_READY

- Added inference batching only to the isolated optimized runner; the canonical scorer remains unchanged at SHA256 `e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe`. New runner SHA256: `85079590578b6be82b539a959109805949746b16745776b84e890d664966bbfc`.
- `historical.score_batch()` was audited: it accepts parallel query/document lists, applies the unchanged exact truncation policy up to 8192 tokens, uses tokenizer left-padding with dynamic `padding=True`, runs one forward over the batch, and maps one last-hidden-state score per input. The optimized runner now supports inference batches `1|4|8|16`, default `8`; no length bucketing or padding contract change was added.
- Structural unit mapping passed with zero mismatches for historical parity `256/672`, current canary `16/43`, and a fresh bounded production subset `20/60`; the previously completed deterministic production sample remains `100/100` zero-mismatch. Exact query/doc/chunk identity, query text, raw chunk text, ordering, and unit counts are preserved.
- Durability concepts remain separate: inference batch is GPU forward grouping; SQLite transaction buffer defaults to `256` q-docs or `1024` chunk rows; optional `512` q-doc mode uses `1536` chunk rows; Volume commit remains one commit per durable transaction plus final output/manifest commits. Fresh shard-00 estimates are `55` durable transactions at 256/1024 and `37` at 512/1536.
- Exact shard-00 forward-call counts are batch1 `56079`, batch4 `14020`, batch8 `7010`, batch16 `3505`.
- Tiny canary namespaces are isolated by batch (`optimized_current_canary_b1`, `optimized_current_canary_b8`); production remains unauthorized. No GPU run, Modal inference, or Volume mutation occurred.

## 2026-09-17 — SAFE_BATCH8_CANARY_TIMEOUT_GUARD

- Changed only the `optimized_current_canary` Modal function timeout from `3600` to `600` seconds. The `optimized_production_shard` timeout remains `3600` seconds.
- No scorer, model, revision, prompt, selector, max-length, batching, preprocessing/cache, aggregation, or durability code changed. No GPU run, Modal run, or Volume mutation occurred.
- Batch8 canary remains isolated under `optimized_current_canary_b8`; comparison reference is the existing old current-canary output SHA256 `fa2dca9d9753701300381955fa972b09cfcb0ff8a17dff57cc8b0e4a401b21cc`.
