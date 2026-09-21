# PRIVATE TASK1 — AI CONTEXT LOG

## Trạng thái hiện tại

- Đã chuẩn bị `PRIVATE_SAFE_BASELINE` bằng CPU-only; chưa chạy Modal/GPU/inference.
- Historical `A_BASELINE` và replay exact `A_NO_OVERLAY`: `CLOSED`.
- Không dùng artifact BGE historical chưa được chứng minh.
- F1–F4 sources: Dense, BM25, word-TFIDF KNN đều PASS; Fold0 bị loại.
- KNN OOF safety: PASS; heldout normalized exact matches sau exclusion = `0`.

## Worklist đã freeze

- Chọn `K=20` vì K10 thấp hơn K20 quá `0.002`.
- File: `private_task1/experiments/private_safe_baseline/private_safe_validation_k20.jsonl`
- SHA256: `5a927ecf67108c1df738dd02a81d92b507f59c2142696d502461e3e7a534285c`
- `112,000` q-docs / `5,600` queries / `7,716` unique documents.
- Expected inference units: `335,838`; duplicate q-docs: `0`.
- Selected chunks, local resolution và remote resolution đều PASS (`7,716/7,716`).
- Oracle Recall: `0.9664940476190478`.

## Storage và readiness

- Modal Volume: `udsc-p13`
- Chunk path: `/runtime/data/processed_v3/chunks`
- Đã upload `1,243` document còn thiếu.
- Modal inference: `0`; GPU runs: `0`; model loads: `0`.
- Readiness report: `private_task1/experiments/private_safe_baseline/readiness_report.json`
- Registry/state/runbook đã cập nhật EXP-008, trạng thái `READY_FOR_VALIDATION_GPU`.

## Scoring contract cố định

- Model: `BAAI/bge-reranker-v2-m3`
- Base revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- Selector: `true_s2_bm25_within_document_v2`
- Aggregation: `MAX`; max length: `512`
- Ranking: `FT_PRIMARY_BASE_TIEBREAK`
- Modal namespace: `runtime/task1_private_safe_validation`

## Lệnh GPU đã chuẩn bị, CHƯA chạy

```powershell
& '.venv\\Scripts\\modal.exe' run --detach --profile nan928904 scripts/modal/task1_bge_subset_gpu.py `
  --worklist private_task1/experiments/private_safe_baseline/private_safe_validation_k20.jsonl `
  --mode production `
  --batch-size 16 `
  --checkpoint-every 256 `
  --run-namespace task1_private_safe_validation
```

## Ràng buộc bắt buộc

- Không tự chạy GPU/Modal inference/Private production.
- Không dùng Qwen, Fold0, label overlay hoặc historical BGE scores chưa proven.
- Chưa tạo submission.
- Không thay scoring contract, worklist, checkpoint namespace hay artifact đã freeze.

## 2026-09-19 — PUBLIC PRODUCTION RECOVERY

- Đóng hoàn toàn A/B, F1–F4 model selection, Qwen và `PRIVATE_SAFE_BASELINE`:
  `offline_ab_research=CLOSED`, `used_for_private_production=NO`.
- Đã trace `artifacts/task1/submission.zip` qua
  `artifacts/task1/recovery_096/public_anchor_093/producer_manifest.json`.
- Exact producer: `build_legal_ir_ensemble.py` →
  `write_legal_ir_submission.py`.
- Gate label-free FAIL: producer dùng `train.json`, `warmup.json`,
  `exact_label_overlay=true`, 57 normalized matches và 5 documents added.
- Step4 calibrated không phải pipeline Private độc lập vì dùng anchor/baseline
  có provenance label-overlay.
- Private hiện: input/retrieval/candidates READY; exact production rerank
  BLOCKED; Top5/submission MISSING.
- CPU-only; GPU/Modal/upload Volume/worklist/submission: chưa thực hiện.
- Next action duy nhất: có pipeline production label-free đã xác nhận hoặc được
  ủy quyền thiết kế pipeline mới.

## 2026-09-19 — K20 FINAL RANKING DIAGNOSIS

- CPU-only trên đúng `112,000` q-doc / `5,600` F1–F4 queries; manifest,
  worklist SHA, identity coverage, finite scores và MAX aggregation đều PASS.
- FT-only tái hiện đúng: Recall `0.8423571428571428`, Precision
  `0.1787857142857143`; fold Recall F1/F2/F3/F4 =
  `0.856071/0.839762/0.836310/0.837286`.
- Candidate oracle K20 = `0.9664940476190477`; FT-only oracle-to-Top5 gap =
  `0.12413690476190486`.
- Metadata K20 chỉ có retrieval ranks `dense`, `bm25`, `knn_word`; không có
  retrieval scores. Current scores có `bge_ft_score`, `bge_base_score` và
  chunk-score arrays.
- `RETRIEVAL_RRF_NO_LABEL` valid với exact weighted-RRF
  (`dense=.2, bge=.3, knn_word=.2, bm25=.3, rrf_k=2`): Recall
  `0.924514880952381`, Precision `0.19689285714285715`; fold Recall
  `0.929821/0.928452/0.921786/0.918000`; changed queries `5568`, gained `569`,
  lost `62`, net `+507`; deployment gate PASS.
- `RRF_PLUS_CURRENT_BGE` BLOCKED vì canonical union `rrf_score/source_support`
  thiếu trên `14944` frozen K20 rows; không tự bịa metadata.
- Diagnosis: `FINAL_RANKING`; best policy `RETRIEVAL_RRF_NO_LABEL`. Không tạo
  submission và không authorize production. Chi tiết: `reports/task1/k20_final_ranking_diagnosis.json`.

## 2026-09-20 — PRIVATE RRF K20 GPU READINESS

- Đã freeze policy production label-free: `RETRIEVAL_RRF_NO_LABEL` với
  `dense=.2, bge=.3, knn_word=.2, bm25=.3, rrf_k=2`; FT-only bị loại khỏi
  full-K20 final ranking.
- Reuse dense artifact, không rerun dense/GPU. BM25 và word-TFIDF KNN được
  dựng CPU bằng implementation canonical; KNN dùng `train.json` làm labeled
  pool, không đọc answer của Private.
- Private coverage: `2080/2080` queries; exact K20 `41600/41600` q-docs;
  candidate counts min/median/max `20/20/20`; `6514` unique docs; duplicate
  q-docs `0`.
- Frozen BGE worklist: `private_task1/rerank/worklists/private_rrf_k20_bge_worklist.jsonl`;
  expected units `124798`; unresolved q-doc/chunk `0/0`; SHA256
  `cf3554e9dfa9e753896e7b5332275e12aa030269c747a59222a072ad4d6a5f86`.
- Candidate artifact SHA256:
  `f8bdbcaa319036557b7baa5f2085ca31b06fae51ec28195b774170302a79868a`.
- Remote Volume `udsc-p13`, `/runtime/data/processed_v3/chunks`: required
  `6514`, present `6514`, uploaded `119`, missing `0`. No nested staging
  entries remain.
- Runner static contract PASS: L4, timeout 21600s, batch16, checkpoint256,
  frozen selected chunks, resume-before-preprocessing, document cache and
  `torch.inference_mode()`.
- Reference validation throughput: `39.22940761943672 units/s` scorer-only;
  estimated Private pure inference `3181.24s` (~`53.0 min`), estimated total
  wall `4371.16s` (~`72.9 min`) including scaled checkpoint overhead.
- GPU/Modal inference: `0`; only storage upload was performed. Không tạo
  submission.

GPU command đã chuẩn bị, chưa chạy:

```powershell
& '.venv\\Scripts\\modal.exe' run --profile nan928904 scripts/modal/task1_bge_subset_gpu.py `
  --worklist private_task1/rerank/worklists/private_rrf_k20_bge_worklist.jsonl `
  --mode production `
  --batch-size 16 `
  --checkpoint-every 256 `
  --run-namespace task1_private_rrf_k20
```

## 2026-09-19 — VALIDATION TIMEOUT AUDIT / PATCH

- Read-only Volume audit: checkpoint
  `runtime/task1_private_safe_validation/checkpoints/production/subset.sqlite3`
  tồn tại, integrity `ok`, generation `1`, persisted `256/112000` q-doc và
  `768/335838` units; worklist SHA và selected-worklist SHA match.
- Không có `results/` hoặc `manifests/`; run timeout trước khi finalize.
- Root cause: GPU function timeout `3600s`; GPU được cấp từ function start nhưng
  code validate model + parse worklist + resolve/prepare toàn bộ 7,716 docs,
  tokenize/BM25 và materialize toàn bộ 112,000 q-doc trước model load.
- Patch runner `scripts/modal/task1_bge_subset_gpu.py`: resume đọc checkpoint
  trước expensive preprocessing; bỏ completed identities trước resolve; dùng
  selected chunk IDs đã freeze trong worklist, cache document text theo doc và
  xử lý bounded theo `checkpoint_every=256`; timeout mới `6h`.
- Scientific contract/worklist/batch16/checkpoint256 giữ nguyên. CPU parity sample
  8 q-doc / 24 units: PASS; selected IDs và text payload: PASS.
- Compile PASS; import-safe tests `5 passed`; chưa chạy GPU/Modal inference.
- Resume command chỉ được chuẩn bị, không tự chạy.

## 2026-09-20 — PRIVATE FINALIZATION / RANKING V2 STATUS

- CPU-only đã hoàn tất provenance audit và finalize Private BGE/RRF.
- Đã tạo provenance addendum:
  `private_task1/experiments/private_rrf_k20/gpu_download/manifests/production_provenance_addendum.json`.
- Đã tạo submission Private:
  `private_task1/submissions/final/submission_private.json` và
  `private_task1/submissions/final/submission_private.zip`.
- Validator JSON/ZIP PASS: `2080/2080` queries, đúng `5` documents/query.
  Submission hiện tại không bị overwrite.
- Diagnostic report: `private_task1/analysis/private_2080_diagnostic.md`;
  kết luận lỗi là `MIXED`, ưu tiên ranking trước corpus.
- Ranking attribution: `private_task1/analysis/ranking_failure_attribution.md`.
  F1–F4 current Recall `0.924514880952381`; candidate oracle
  `0.9664940476190477`; retrieval misses `136`; ranking misses `187`.
  Kết luận: `RANK_FUSION_BOTTLENECK`.
- CPU-only `CROSS_FITTED_LOGISTIC_FUSION` trên existing F1–F4 K20/scores,
  không dùng Fold0 và không áp dụng lên Private: OOF Recall `0.9273125`,
  delta `+0.002797619`, bootstrap 95% CI
  `[+0.000416667, +0.005238095]`, better/worse/same `43/23/5534`,
  rescued ranking misses `28/187`, new misses `16`.
- Ranking V2 gate FAIL: chỉ `2/4` folds không giảm; F1 và F3 giảm.
  Không tạo Ranking V2 submission; current submission vẫn là incumbent.
- Next action cố định: `RUN_PROCESSED_PV1_REPAIR`, chỉ thực hiện khi user
  yêu cầu riêng.
- GPU runs gần đây: `0`; Modal inference runs: `0`.
- Ràng buộc tiếp tục: không rerun Dense/BM25/KNN/BGE, không dùng Private
  labels/leaderboard để tune, không weight sweep/classifier search, không
  overwrite submission, không tự submit/upload.

## 2026-09-20 — PROCESSED_PV1 PHASE 1 (BLOCKED BEFORE WEB FETCH)

- `data/raw/btc` và `data/processed_v3` không bị sửa, move, rename hay
  overwrite. V3 tiếp tục là historical baseline.
- Đã tạo scaffold provenance tại `data/processed_pv1/metadata/` và script
  deterministic `scripts/data_prep/recover_pv1_empty_sources.py`.
- Local membership gate PASS: đủ đúng `20/20` IDs, mỗi raw passage rỗng và
  mỗi record có BTC-provided link. Không có mismatch danh sách.
- Web fetch chưa được chạy: terminal không thiết lập được TLS credentials với
  `thuvienphapluat.vn`; không có source snapshot/text nào bị promote.
- Phase-1 report: `data/processed_pv1/metadata/phase1_empty_recovery_report.json`:
  recovered `0`, unresolved `20`, source fetch `NOT_ATTEMPTED=20`, synthetic
  content `0`, manual legal-text edits `0`, gate `PARTIAL`.
- Theo hard gate, không chạy Phase 2, không clone/finalize corpus PV1, không
  rebuild embeddings/FAISS/retrieval/BGE, không GPU/Modal.

## 2026-09-20 — PROCESSED_PV1 PHASE 1 PUBLIC-SOURCE RECOVERY

- Updated `scripts/data_prep/recover_pv1_empty_sources.py` to use the safe
  flow: BTC identity → one TVPL probe → strict local-PDF gate → public exact
  alternate discovery → canonical cleaner/chunker only after identity PASS.
  No Cloudflare/paywall bypass, login, synthetic text, or legal-text editing.
- Final Phase-1 artifacts:
  `data/processed_pv1/metadata/phase1_recovery_manifest.jsonl`,
  `phase1_recovery_report.md`, `phase1_recovery_summary.json`, and
  per-document provenance in `data/processed_pv1/source_recovery/provenance/`.
- All `20/20` targets passed raw membership validation. All 20 TVPL requests
  returned `CLOUDFLARE_BLOCK`; no text was accepted from TVPL.
- Local PDF audit found three files: `34810.pdf`, `208668.pdf`, `181693.pdf`.
  They are image PDFs (embedded text 16/2/16 characters); all three are
  `PDF_IDENTITY_AMBIGUOUS`, so OCR/ingest was correctly not attempted.
- Public alternate discovery was attempted for all 20; no candidate met the
  exact identity + substantive-body gate. Result: recovered `0/20`, unresolved
  `20/20`, each with a reason and provenance; recovery gate `PASS` because the
  process is complete/safe rather than requiring every source to be available.
- Raw BTC and processed V3 untouched; GPU/Modal/embedding/retrieval/BGE/Phase 2
  not run. Next action remains review/provide an independently accessible exact
  source before any recovery retry; do not enter Phase 2 yet.

## 2026-09-20 — PHASE 1 RECOVERY REVIEW / PDF OCR

- Repaired standard-code parsing: PV1 now preserves separate `display_code`
  and `normalized_code` (for example `QCVN 46:2022/BTNMT` /
  `QCVN462022BTNMT`) and no longer appends title words to TCVN/QCVN identity.
  Parser regression checks PASS.
- Disabled the weak HTML-scraped Bing discovery path after it returned unrelated
  ChatGPT URLs. `SEARCH_ENGINE_VALID=NO`; unresolved rows now say
  `SEARCH_ENGINE_UNAVAILABLE`, not a false `NO_PUBLIC_SOURCE_FOUND`.
- Tested the supplied exact 67660 alternate URL. It has the expected QCVN URL
  identity, but HTTP `200` body is `PRO_PAYWALL`; no bypass and no ingest.
  `KNOWN_67660_REGRESSION=PASS` because this is explicitly recorded.
- OCR-for-identity ran on all three local image PDFs. `34810.pdf` and
  `181693.pdf` PASS via exact BTC-TVPL page-id mapping, title overlap, authority
  and issue-year evidence; full CPU OCR + canonical cleaner/chunker recovered
  both (`44,509` / `55,479` extracted chars; `177` / `156` chunks).
- `208668.pdf` FAIL: it is an unmapped 2021 draft with no final document number
  or BTC-TVPL page-id correspondence; not ingested. Wrong PDF mappings: `1`.
- Current result: recovered `2/20`, unresolved `18/20`, provenance complete,
  synthetic/manual edits `0`, raw/V3 unchanged, GPU/Modal `0`.
- Safety gate remains PASS, but `PHASE1_REVIEW_GATE=FAIL` because valid search
  infrastructure for the remaining exact documents is unavailable. Do not run
  Phase 2; next action is `FIX_PHASE1_RECOVERY`.

## 2026-09-20 — LOCAL PDF-ONLY RECOVERY FINAL RUN

- Strict local-only mode completed: `20/20` PDFs inventoried and evaluated;
  network requests `0`, GPU/Modal `0`, raw BTC and `processed_v3` unchanged.
- Recovered `18/20`: `10` via embedded PDF text and `8` via CPU OCR; all
  recovered documents passed identity and completeness gates. Synthetic legal
  content and manual legal-text edits remain `0`.
- `208668.pdf`: `IDENTITY_PASS`, `COMPLETE`; the user-verified 2021 draft
  identity was accepted correctly.
- `131890.pdf`: `IDENTITY_PASS`, `COMPLETE`; `132890.pdf` was not present, so
  `mapping_from_132890=NO`.
- Remaining unresolved PDFs: `288457.pdf` is a 2024 Law on Health Insurance
  amendment and is not proven to be the target; `191261.pdf` contains a 2023
  draft circular for public general/special schools, not the target preschool
  circular. Both remain `IDENTITY_AMBIGUOUS` and were not ingested.
- Fixed and reran two false negatives: Unicode authority normalization and
  OCR spacing in `QCVN 02-23:2017`; `10533` and `187338` now recover. Relaxed
  the completeness heuristic so repeated PDF headers do not reject complete
  `255762`.
- Final gate: `LOCAL_PDF_RECOVERY_GATE=PASS` for safe local recovery, but
  `next_action=MANUAL_REVIEW_REMAINING_PDFS`; do not start Phase 2 yet.

## 2026-09-20 — LOCAL PDF RECOVERY COMPLETED 20/20

- User confirmed `288457.pdf` is intentionally the 2024 Law amending the Law
  on Health Insurance. Its identity metadata was corrected accordingly and it
  passed identity/completeness; recovered with `314` chunks.
- User replaced the incorrect `191261.pdf`. New OCR identity is the 2023 draft
  circular on staffing/positions in public preschools; it passed identity and
  completeness with `132` chunks.
- Final local-only run: `20/20` recovered, `20/20` identity pass,
  `20/20` completeness `COMPLETE`, unresolved `0/20`.
- PDF source methods: `11` embedded-text, `9` CPU OCR. Network requests `0`,
  GPU/Modal `0`, synthetic content `0`, manual legal edits `0`.
- `LOCAL_PDF_RECOVERY_GATE=PASS`; next action is `READY_FOR_PV1_PHASE2`.

## 2026-09-21 — PV1 PHASE 2 CORPUS FREEZE

- Built canonical `data/processed_pv1` by direct-copy inheriting `8,512`
  documents from `processed_v3` and replacing exactly the `20` repaired IDs
  from `phase1_recovered`.
- Final counts: `8,532` documents, `184,633` parents, `1,272,971` chunks.
- Document membership preserved; changed document IDs are exactly the 20
  repair targets; substantive changes outside the repair set: `0`.
- Graph gate: empty documents/chunks `0/0`, duplicate IDs `0/0`, orphan
  parents/chunks `0/0`, invalid IDs `0`.
- Repaired identity/completeness: `20/20 PASS/COMPLETE`; provenance source is
  `LOCAL_USER_PROVIDED_PDF` for all repaired documents.
- `288457` has one exact duplicate chunk and several boilerplate/OCR flags in
  the quality audit; this is recorded for review and does not violate the
  requested duplicate-ID/empty-chunk corpus gate.
- `PHASE1_WEB_SEARCH_GATE=DEPRECATED_NOT_APPLICABLE`.
- `PV1_CORPUS_GATE=PASS`; no embeddings, FAISS, retrieval, BGE, Qwen, GPU, or
  Modal run. Next action: `BUILD_PV1_DENSE_EMBEDDINGS_AND_FAISS`.

## 2026-09-21 — PV1 CPU RETRIEVAL / BGE DELTA PREPARATION

- Reused the existing PASS Dense artifact: `2,080/2,080` queries; SHA256
  `17eed47d1e03bd2404e0a2056b1cd25812567065710fcfa54d3f1c2e72ec8934`.
- CPU gates PASS: canonical BM25, word-TFIDF KNN, and exact K20 candidate
  construction. PV1 K20 has `41,600` q-docs and `124,798` selected chunks.
- Label-free K20 diff versus the old Private run: `41,303` same q-docs,
  `297` additions, `297` removals, `256` changed queries.
- Corrected BGE reuse audit: `41,303` reusable, `297` rerun, reuse rate
  `99.286%`. Rerun reasons are `230 NEW_QDOC` and `67 REPAIRED_DOCUMENT`;
  selected-chunk changes, missing scores, and provenance mismatches are `0`.
- Delta worklist: `private_task1/experiments/private_pv1/bge/private_pv1_bge_delta_worklist.jsonl`;
  `297` q-docs / `891` inference units; SHA256
  `2c5b560a2d69d089e8604902e8e42711ed8b7f7c8565cb629b1e33c1a3d0ebae`.
- Remote sync is isolated to `194` PV1 chunk files (`101,725,516` bytes) in
  `private_task1/experiments/private_pv1/bge/remote_sync/`; local hardlink
  staging PASS. Modal `volume ls` confirmed the PV1 target path is absent;
  no upload, GPU inference, or Modal run was executed.
- BGE runner remains the proven L4 / batch-16 / checkpoint-256 runner with
  only backward-compatible `--chunks-root` support. Compile PASS.
- Final state: `WAITING_FOR_USER_GPU_APPROVAL`; do not launch until the staged
  PV1 chunks are manually uploaded and verified.

## 2026-09-21 — PV1 DELTA GPU ATTEMPT STOPPED

- Local delta preflight PASS: `297` q-docs, `891` units, exact worklist SHA,
  zero duplicates, zero missing selected chunks. Remote sync PASS: `194/194`
  required files under `/runtime/data/processed_pv1/chunks/`.
- The initial directory upload created an unintended nested
  `staged_chunks` directory; it was removed exactly, then the correct parent
  upload was verified. No historical namespace was touched.
- Submitted the exact L4 delta command once in namespace
  `task1_private_pv1_bge`; app `ap-3Yz2xgHE1CkBkITuAnClCQ`, call
  `fc-01M30HVQRAJEWEX7TP5KNQ3ZWK`.
- Modal call terminated before task/container allocation with empty
  `RemoteError('')`; call graph had empty `task_id`, no remote logs, and no
  checkpoint/result manifest. No GPU inference completed.
- `BGE_GPU_GATE=FAIL`; classification
  `MODAL_CALL_TERMINATED_BEFORE_TASK_ALLOCATION`. No automatic retry, merge,
  RRF, or submission was performed. Preserve the namespace and diagnose the
  Modal submission failure before any rerun.

## 2026-09-21 — PV1 DELTA GPU RETRY STOPPED

- Retry preflight PASS: profile/auth usable, remote chunks `194/194`, exact
  delta SHA, `297` q-docs / `891` units, no duplicate or missing selected
  chunks. No re-upload was needed.
- Exact same production command was retried once. App
  `ap-qo9p29dvLP92zm9JWd9Swz`; call
  `fc-01M30N6XJ8KWN46WDB2R1V52PX`.
- Unlike the first attempt, the retry allocated function/container
  `fu-1yXRfwz24hP6frQpcbCtc8` /
  `ta-01M30N6XNTW9TKFZ44ZQ8202BR`, then terminated during Modal runtime
  startup: `KeyboardInterrupt` while importing `cbor2` from
  `/__modal/deps/cbor2/_encoder.py`; logs ended with `Runner terminated`.
- No model loading, BGE scoring, checkpoint, or output artifact completed:
  `0/297` q-doc and `0/891` units.
- `BGE_GPU_GATE=FAIL`; classification
  `MODAL_CONTAINER_STARTUP_IMPORT_INTERRUPTED`. No third retry and no
  downstream merge/RRF/submission were performed.

## 2026-09-21 — PV1 MODAL RECOVERY VIA VERIFIED LOCAL CPU FALLBACK

- Frozen inputs reverified: delta worklist SHA
  `2c5b560a2d69d089e8604902e8e42711ed8b7f7c8565cb629b1e33c1a3d0ebae`,
  `297` q-docs, `891` units, and remote PV1 chunks `194/194`.
- Modal CPU preflight passed without model loading, inference, checkpoint, or
  Volume commit: `256/256` checked q-docs, `768` resolved units, exact delta
  worklist SHA. Independent L4 infrastructure smoke passed with CUDA visible
  as `NVIDIA L4`, but loaded no model and ran no inference.
- The one allowed exact production retry was submitted as app
  `ap-HEA8KCNe6g9suVFMvMTagv`, call `fc-01M30PWVR7XWB71FFEB75DZ1D6`; it
  again failed before user code during Modal runtime import of
  `/__modal/deps/cbor2/_encoder.py` with `KeyboardInterrupt`.
  Classification remains `MODAL_RUNTIME_STARTUP_FAILURE` /
  `MODAL_CONTAINER_STARTUP_IMPORT_INTERRUPTED`; no Modal BGE score was used.
- Read-only download audit found the current Modal FT volume weight hash did
  not match the frozen FT hash. The fallback therefore used the verified local
  artifact `outputs/task1/bge_ft_model/bge_m3_finetuned` with weight SHA
  `68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c` and
  config SHA `16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b`.
- Exact CPU parity gate passed on `12` unchanged historical q-docs / `36`
  units using the canonical runner's frozen-row resolver and CrossEncoder
  calls: Base max abs diff `1.430511474609375e-06`, FT max abs diff
  `9.5367431640625e-07`, `36/36` chunk scores within `1e-5`, selected chunk
  identity mismatches `0`, aggregation mismatches `0`, ordering mismatches `0`.
- Local CPU fallback scored only the frozen delta: `297/297` q-docs and
  `891/891` units; all finite, no duplicates, exact worklist SHA, output SHA
  `8811d80c8f5b0f1c00332f93e4b9ce051513f7923217f67ecb3e94678f2e177d`.
  Inference elapsed `1893.03s`, total elapsed `1924.70s`, throughput
  `0.4706748` units/s inference-only and `0.4629299` units/s wall. Route:
  `LOCAL_CPU_FALLBACK`.
- Merge gate passed: `41,303` historical reusable + `297` new local CPU =
  `41,600` q-docs / `124,798` units; selected chunk mismatches, duplicate
  identities, and non-finite scores are all `0`. Merged BGE SHA:
  `c7062cc17a9fee210a60f7881b0436c972db1e551abe9164a6c738c82ddb4437`.
- Frozen `RETRIEVAL_RRF_NO_LABEL` ran successfully with no labels/private
  answers read. Final JSON and ZIP validators both passed for `2,080` queries,
  unique Top5, zero invalid/missing/duplicate predictions. Final artifacts:
  `private_task1/submissions/pv1_final/submission_private_pv1.json` SHA
  `83db32798d1990eb86201dabc0764f7a1dbb23c1b9c67175caa9e0a86c61dfb2` and
  `private_task1/submissions/pv1_final/submission_private_pv1.zip` SHA
  `3e8fa9045a49baed5f0d007e8d52c22c25b4296f837774a756f2f3733c5402e0`.
- Final status: `READY_FOR_MANUAL_SUBMISSION`; no automatic submission was
  performed.
- Recovery helpers `scripts/evaluation/score_private_pv1_bge_delta_cpu.py` and
  `scripts/analysis/finalize_private_pv1.py` compile successfully. The failed
  Modal namespace has no `/runtime/task1_private_pv1_bge` directory. A temporary
  downloaded copy of the current Modal FT weight remains at
  `D:\udsc2026\.tmp_pv1_bge_ft` only because the environment rejected the
  destructive cleanup command; it was not used for scoring.

## 2026-09-21 — PV1 REGRESSION FORENSIC AUDIT COMPLETE

- CPU-only audit completed under
  `private_task1/experiments/pv1_regression_forensics/`; no GPU, Modal,
  Qwen rerun, Private-answer read, corpus mutation, or submission overwrite.
- Input and replay gates passed. Frozen policy remained
  `RETRIEVAL_RRF_NO_LABEL` with dense `0.2`, BGE `0.3`, word-KNN `0.2`,
  BM25 `0.3`, `rrf_k=2`.
- Exact private replay: Top5 changed `161/2080`; K20 changed `256/2080`;
  K20 additions/removals `297/297`. BGE determinism passed for `297/297`
  q-docs and `891/891` units; base and FT chunk/document score diffs were
  `0.0`, with zero selected-chunk and ordering mismatches.
- F1-F4 validation (`5,600` queries, folds 1–4, labels used only for
  evaluation) passed. V3 Recall@5 `0.924514880952381`, Precision@5
  `0.19689285714285717`; PV1 Recall@5 `0.9255863095238096`, Precision@5
  `0.19710714285714284`; deltas `+0.001071428571428612` and
  `+0.0002142857142856669`. Fold Recall deltas: F1 `+0.002142857142857113`,
  F2 `-0.0014285714285714457`, F3 `+0.0014285714285714457`, F4
  `+0.002142857142857113`. Candidate-oracle Recall@20 delta:
  `+0.00044642857142851433`.
- PV1 validation changed `554` query candidate sets (`627` q-doc additions and
  `627` removals). BGE score reuse was `111,313/112,000` (`99.3866%`); only
  `687` q-docs / `2,061` units were CPU-rerun, with `482` new-q-doc and `205`
  repaired-document reasons; no selected-chunk-change or provenance failures.
- Repaired-document audit found `0` objective structural defects. The two
  `288457` repeated chunks are legitimate repeated legal language, not a
  corpus defect. Repaired-document gold intersection: `9` validation queries
  in PV1 K20 and `9` in PV1 Top5.
- Workflow-B signal: `INSUFFICIENT_EVIDENCE` (`B1` blocked before model fit).
  Same-registered-BGE headroom: `NO`. Registration derivative status:
  `UNCLEAR`; new external reranker remains `FORBIDDEN`.
- Root-cause classification: `RETRIEVAL_SHIFT`. Recommended next action:
  `KEEP_PV1_AND_IMPROVE_SAME_BGE`.
- During the audit, validation initially stopped on a forensic-script-only
  `NameError` for `scan_chunk_hashes`; the helper was added locally and the
  script was recompiled successfully before the authoritative rerun.

## 2026-09-21 — PHASE 1 STEP4 RECOVERY / PV1 PORT COMPLETE

- CPU-only Phase 1 completed under `private_task1/experiments/sprint48_step4/`.
  No GPU, Modal, Qwen inference, model load, Private answers, Fold0 labels,
  public labels, corpus mutation, or old-submission overwrite.
- Historical Step4 handoff lineage is `PARTIALLY_RECOVERED`: all 16 present
  handoff artifacts match their manifest SHA256; the declared missing files
  remain `predictions/f1_f4_final_top5.json` and `scores/raw_pair_bge_scores.npy`.
  Exact CPU calibration replay passed: `555` raw swaps -> `77` calibrated
  swaps across `71` queries, semantically identical to the frozen public
  `submission_step4_calibrated.json`. Deployment evidence is `PARTIAL` because
  the repo has the recorded 0.9391 -> 0.9411 claim but no external receipt.
- Reconstructed the current PV1 F1-F4 BGE map: `112,000/112,000` q-docs;
  `111,313` reused historical rows + `687` PV1 delta rows; selected-chunk
  mismatches `0`. Reconstruction SHA256:
  `5e47db9d9d706660a2943656408af561504ef18f8b6cbfdb36fde99681e93bd0`.
  Frozen PV1 baseline reproduced: Recall `0.9255863095238095`, Precision
  `0.19710714285714287`.
- `STEP4_K20_PORT` (explicitly not exact historical Top-50 reproduction):
  Recall `0.9260327380952381`, Precision `0.19721428571428573`, Recall delta
  `+0.00044642857142862535`; fold deltas F1 `-0.00035714285714283367`, F2
  `+0.002142857142857224`, F3 `0`, F4 `0`. Gate `FAIL`; Top1–3 lock PASS;
  `129` queries changed, `4` improved, `1` harmed, `124` neutral; `130`
  calibrated swaps (`116` rank5, `14` rank4).
- No Step4 Private checkpoint was created because the preregistered gate
  failed. The single fallback `DIRECT_K20_PLUS_ANCHORS_DOCUMENT_TOP5` also
  failed its all-fold gate: pooled Recall delta `+0.003794642857142927`, but
  F3 delta `-0.00011904761904768524`. No fallback Private application.
- External submission status of the existing local PV1 incumbent remains
  `UNKNOWN`; no automatic submission was performed.
- Phase 2 specification prepared, not trained:
  `private_task1/experiments/sprint48_step4/same_bge_hard_negative_ft_v2_spec.md`.
  CPU-only work-size estimate:
  `private_task1/experiments/sprint48_step4/same_bge_hard_negative_ft_v2_estimate.json`;
  this is not a runtime benchmark.
  Recommended next action: `RUN_HARD_NEGATIVE_BGE_FT_V2`.
