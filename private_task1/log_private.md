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

## 2026-09-21 — PHASE 2A SAME-BGE HARD-NEGATIVE FT V2 PREFLIGHT

- CPU-only Phase 2A preflight/materialization completed under
  `private_task1/experiments/sprint48_bge_ft_v2/`; GPU runs `0`, Modal runs
  `0`, model loads `0`, and no Private answers/Fold0/public labels were used.
- Immutable gates passed: reconstructed PV1 BGE SHA
  `5e47db9d9d706660a2943656408af561504ef18f8b6cbfdb36fde99681e93bd0`,
  F1-F4 worklist SHA `5bb1f804b629a65611ee0f9e6b11c4b95026a42c3303b395b3dbc0d92ab8a2e2`,
  candidate SHA `8a56267146d6d2670da3737cbd78d68761899a0672ad2d7424e6d77ac47b30e8`,
  PV1 corpus fingerprint, strict folds, and the verified local FT weight/config
  hashes all match the Phase 1 contract.
- V2 student initialization is frozen to `CURRENT_FT` and teacher to the same
  verified local checkpoint `outputs/task1/bge_ft_model/bge_m3_finetuned`
  (weight SHA `68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c`).
  This is the only no-rewrite implementation of the reviewed V3 anchored
  trainer and the frozen current-FT teacher requirement. The mismatched Modal
  FT artifact remains forbidden.
- Training uses exactly the frozen inference-selected raw chunk that gives each
  document's current teacher MAX; the full selected-ID list is retained. No
  selector rerun, document summary, synthetic chunk, or inferred teacher logit.
  Persisted exact teacher logits are absent by design; V3 dynamically computes
  exact logits from the frozen teacher during an authorized GPU run.
- Materialized strict outer work: F1 `52,512` examples / `3,282` updates /
  `83,988` score units; F2 `52,488` / `3,281` / `83,991`; F3 `52,668` /
  `3,292` / `83,994`; F4 `52,284` / `3,268` / `83,985`. OOF total is
  `209,952` examples / `13,123` updates / `112,000` q-docs / `335,958` units;
  leakage, Fold0, Private, duplicates, invalid chunks, and non-finite cached
  teacher probabilities are all `0`.
- Final-fit worklist is `34,992` pairs / `69,984` examples / `4,374` updates.
  The isolated smoke is two pairs / four examples / one update. Safetensors
  metadata-only audit found `567,755,777` total parameters, `491,127,808`
  frozen (embeddings + layers 0--17), and `76,627,969` trainable.
- Added deterministic Phase 2A materializer, exact GPU worklist scorer, and a
  CPU-only strict-OOF gate evaluator; repaired V3 gradient accumulation,
  explicit frozen-teacher path, deterministic seed, completed-run resume
  verification, checkpoint reload verification, and a strict F1--F4 marker
  that prevents a V2 run from requesting Fold0. All compile; the V3 CPU
  preflight read `52,512` F1 training examples without loading a model. GPU
  smoke/OOF/final commands and frozen promotion gate are in
  `private_task1/experiments/sprint48_bge_ft_v2/preflight/phase2a_report.md`.

## 2026-09-21 — PHASE 2B PRE-GPU GATES / EXECUTION BLOCKER

- Final CPU-only Stage 0/1 gate passed and is recorded in
  `private_task1/experiments/sprint48_bge_ft_v2/preflight/stage2b_final_execution_gate.json`.
  It re-hashed the frozen V2 config (`09ae8694...44d7d6b`), current
  student/teacher weight and config, Phase 1 reconstructed BGE scores,
  F1--F4 worklist/candidate artifacts, strict folds, train data, PV1 corpus,
  and every materialized fold training/scoring worklist. All matched Phase 2A.
- The final-policy evaluator was repaired only at evaluator plumbing level:
  it now supports a one-fold immediate early gate and preserves the frozen
  candidate source ranks/base BGE score while replacing only `bge_ft_score`.
  CPU assertion reproduced the historical `RETRIEVAL_RRF_NO_LABEL` Top-5 for
  all 5,600 F1--F4 queries exactly (0 RRF mismatches), including baseline
  Recall `0.9255863095238096` and Precision `0.19710714285714284`.
  No model, frozen recipe, selector, aggregation, candidate, corpus, or RRF
  weight changed.
- No smoke was launched: this workspace's `.venv` has `torch 2.2.2+cpu`,
  `cuda_available=false`, zero CUDA devices, and no `nvidia-smi` executable.
  The frozen local `--device cuda` command therefore cannot supply the
  required L4. This is an execution-environment blocker, not a scientific or
  smoke failure. GPU runs `0`; Modal runs `0`; no model was loaded.
- Next action is to provide/activate the exact L4 execution route for the
  frozen smoke command; do not substitute a CPU run or alter the recipe.

## 2026-09-22 — PHASE 2B L4 GPU SMOKE PASS

- The isolated Modal route `scripts/modal/task1_same_bge_ft_v2.py --mode smoke`
  completed exactly once on Modal profile `nan928904`; no A10/other GPU and no
  retry was used. Modal app `ap-YfWcsu1lttNlHoqul07KLT`, function call
  `fc-01M32F66P5NMF5RK3P64SQ5V52`.
- Environment gate passed: actual GPU `NVIDIA L4`, CUDA `12.4`, torch
  `2.5.1+cu124`, total VRAM `23,659,151,360` bytes, peak allocated VRAM
  `5,786,695,680` bytes; no OOM/runtime failure.
- Remote model provenance was verified before model load: CURRENT_FT weight
  SHA `68bc6d16a5b898a3ff2a89d8171fc8e6b3a20d3ea7c78a15c7340b1a3f02a89c`,
  config SHA `16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b`,
  model `BAAI/bge-reranker-v2-m3`; smoke groups SHA
  `e7f776b42f7bbef0fb27fcaa53b96ea83d39412ea00fbb6e4ccb050c2acce180`.
- Frozen smoke contract passed: 2 pairs / 4 examples, exactly 1 optimizer
  step and 1 micro-step, dynamic teacher logits, finite BCE
  `0.554519534111023`, finite teacher MSE `0.000285633112071082`, finite
  total loss `0.2774025797843933`, checkpoint save/reload, and post-reload
  CrossEncoder scoring with finite scores.
- Smoke checkpoint SHA: weight
  `b925c86274c93e78c707c363225519a13646ea408756c2978738d1f66d25da74`,
  config `16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b`.
  Wall time `36.160629474` seconds. Checkpoint remains non-promotable until
  the frozen strict F1--F4 OOF gate passes.
- Evidence downloaded to
  `private_task1/experiments/sprint48_bge_ft_v2/smoke/remote_evidence/` and
  the smoke checkpoint to `.../smoke/gpu_checkpoint/`; no historical namespace
  was touched.

## 2026-09-22 — PHASE 2B REMOTE FOLD INPUT GATE PASS

- Added only Modal infrastructure plumbing for sequential fold execution; the
  canonical trainer, scorer, selected chunks, loss, recipe, and RRF evaluator
  were not changed. All modified Python files compile.
- Exact required remote inputs were synchronized into the isolated namespace
  under `/runtime/private_task1/experiments/sprint48_bge_ft_v2/remote_inputs/`
  (the uploaded directory is named `_tmp_phase2b_remote_sync`); no historical
  Modal namespace was overwritten.
- CPU-only Modal input gate passed in app
  `ap-LJZEZCfuFoAOzumpnbTZrB`, with no GPU request and no model load. It
  verified the CURRENT_FT hashes, reference SHA
  `5e47db9d9d706660a2943656408af561504ef18f8b6cbfdb36fde99681e93bd0`, all
  four train-group/worklist SHAs, 7,723 unique required documents, exactly
  7,723 remote chunk files, and zero missing required documents.
- Two earlier CPU-only input-probe attempts failed solely because of typos in
  the newly added local expected-hash constants; they did not request a GPU,
  load a model, or mutate scientific artifacts. The constants were corrected,
  and the subsequent gate passed exactly as above.

## 2026-09-22 — F1 OOF GPU EXECUTION STARTED

- F1 is the first and only fold launched after the smoke and full remote-input
  gates passed. Frozen training contract: F2+F3+F4, 52,512 examples, 3,282
  optimizer steps; held-out scoring target is 28,000 q-docs / 83,988 units.
- Modal app `ap-Z7KLb8Yaj1SmFFZSXfK89G`, function call
  `fc-01M32GFHYEPDP9SZRZAM7TEWXB`, actual requested GPU `L4`, retries `0`.
  The canonical V3 trainer is running with the unchanged frozen recipe; no
  second fold is launched until F1 completes and its CPU RRF early gate is
  evaluated.
- F1 training completed PASS: 52,512 examples, 13,128 micro-steps, exactly
  3,282 optimizer steps, checkpoint reload verified, mean loss
  `0.4917044478447598`, wall `2578.299434656` seconds. Fold-1 checkpoint
  weight SHA `4bbe295ef1a1901288230c030438663cffd775d0c1eb6aa49f1c4eed9a3ce601`,
  config SHA `16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b`.

## 2026-09-22 — F1 OOF EARLY-KILL FAIL / STOP

- F1 scoring ran once on L4 in Modal app `ap-gSvjqGE1GYbIpNWzQhehze`,
  function call `fc-01M32K0ZF6Z0PQ8D7TJTD86D7Z`, with no retry. Exact coverage
  passed: 28,000 q-docs / 83,988 units, finite scores, duplicate q-docs `0`,
  selected-chunk provenance preserved, scorer output SHA
  `2009fd527adf148bfc27d9b84aef7d4af91e6bd818190675af3bdb12bd310565`, wall
  `1411.605766494` seconds.
- CPU-only frozen RRF evaluator result is recorded at
  `private_task1/experiments/sprint48_bge_ft_v2/fold1/early_rrf_gate.json`.
  F1 baseline Recall `0.9319642857142857`, proposed Recall
  `0.9310119047619048`, delta `-0.0009523809523809268`; baseline Precision
  `0.1975714285714286`, proposed Precision `0.1972857142857143`.
- `EARLY_KILL_GATE = FAIL_F1`. This is a valid scientific negative fold, not
  an infrastructure failure. Per frozen contract, F2/F3/F4, pooled OOF,
  final-fit, Private scoring, and submission ZIP were not run. The F1 remote
  checkpoint and score artifacts are preserved; no automatic submission/upload
  occurred.

## 2026-09-22 — PHASE 2B FINAL STATUS

- `MODAL_RUNTIME_GATE = PASS`; `REMOTE_MODEL_GATE = PASS`; `REMOTE_INPUT_GATE = PASS`;
  `L4_ENV_GATE = PASS`; `SMOKE_GATE = PASS`; `EARLY_KILL_GATE = FAIL_F1`.
- `F1_DELTA = -0.0009523809523809268`; `F2_DELTA/F3_DELTA/F4_DELTA = NOT_RUN`.
  `FT_V2_OOF_GATE = NOT_REACHED`; `FINAL_CHECKPOINT_CREATED = NO`;
  `PRIVATE_V2_SUBMISSION_GATE = NOT_CREATED`.
- Final status: `FT_V2_SCIENTIFIC_FAIL`. The frozen experiment stopped at the
  required first negative outer-fold gate. The incumbent remains unchanged;
  no Fold0/Private labels, no second recipe, no F2 GPU call, no final fit, and
  no submission/upload were performed.
- Current Phase 2B execution count: `GPU_RUN_COUNT = 3` (smoke, F1 train, F1
  score); `MODAL_RUN_COUNT = 10` including the non-GPU import/input probes and
  their preserved engineering-failure evidence.


## 2026-09-22 — F1 SAME-BGE FT V2 CPU FORENSIC AUDIT

- CPU-only audit completed from frozen local artifacts; GPU runs `0`, Modal runs `0`, model/scorer not loaded or rerun.
- Baseline replay: Recall `0.9319642857142857`, Precision `0.1975714285714286`; exact expected values reproduced.
- V2: Recall `0.9310119047619048`, Precision `0.1972857142857143`; Recall delta `-0.0009523809523809268`.
- Final Top5 changed `488` queries; improved `1`, harmed `3`; relevant gained/lost `1/3`.
- BGE-only substitution gate: `PASS`; selected chunk identity and frozen source-rank inputs passed.
- Chunk-MAX audit: `1210` changed / `28000` q-docs; objective-conflict evidence: `STRONG`.
- Exact F1 checkpoint tensor integrity is unavailable because the checkpoint is not present locally; scorer runtime parity has no persisted independent sample.
- Primary classification: `INCONCLUSIVE`; do not call this `TRUE_SCIENTIFIC_FAIL` until checkpoint provenance is restored.
- Reports: `private_task1\experiments\sprint48_bge_ft_v2\fold1\forensic_audit.json`, `private_task1\experiments\sprint48_bge_ft_v2\fold1\forensic_audit.md`.


## 2026-09-22 — F1 CHECKPOINT FORENSIC CLOSURE

- CPU-only Modal Volume audit found the authoritative checkpoint at `/workspace/p13/runtime/private_task1/experiments/sprint48_bge_ft_v2/fold1/gpu_checkpoint/checkpoint`. No training, inference, or GPU was run.
- Checkpoint SHA `4bbe295ef1a1901288230c030438663cffd775d0c1eb6aa49f1c4eed9a3ce601` and config SHA `16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b` match the recorded F1 execution/score manifests.
- Tensor contract PASS: `393` tensors, missing/extra `0`, nonfinite `0`; frozen embeddings + encoder layers 0–17: `0/293` changed; trainable tensors changed `100/100`.
- Final classification: `PRACTICAL_SCIENTIFIC_FAIL_WITH_PARITY_CAVEAT`; `FT_V2_STATUS = CLOSED`; `SCORER_RUNTIME_PARITY = UNPROVEN`.
- Do not rerun SAME-BGE FT V2. Future work must use a separately named teacher-anchoring objective hypothesis.
- Tensor manifest: `private_task1\experiments\sprint48_bge_ft_v2\fold1\checkpoint_tensor_diff_manifest.json`.

## 2026-09-22 — SAME-BGE HARD-NEGATIVE FT V3 F1 EARLY-KILL

- V3 hypothesis: `CONFLICT_AWARE_TEACHER_ANCHORING`; V2 remained frozen and was
  not rerun. V3 config SHA `8523a19389f1314e64aeac81d0432adaf3e23639bad149e28331c51ceb52a6b7`.
  Training-side conflict population: positive `34704`, negative `23730`.
- CPU loss tests, legacy-formula parity, exact V2 train-group/worklist SHA
  parity, remote provenance, and L4 smoke all passed. Smoke verified both
  teacher-consistent and teacher-conflict masks, finite losses, backward,
  save, and reload.
- V3 F1 training ran once on L4: `52512` examples, `13128` micro-steps,
  `3282` optimizer steps, teacher-anchor mode `conflict-aware`, checkpoint
  reload PASS. Checkpoint weight SHA
  `5701abf3e49f3a856fe56f6565361228a0426f83055d08f8f56f1bfac99b3421`;
  config SHA `16f6e0bece36db2318601cbf5119f5f9eb616857eed068ddb73cb482f104524b`.
- V3 F1 scoring ran once on actual `NVIDIA L4`, with exact frozen coverage
  `28000` q-docs / `83988` units, finite scores, duplicate q-docs `0`, and
  selected-chunk provenance PASS. Output SHA
  `6c3923dd3fe82a243b2262eba2c6baae99d3bd7b63e14087f1d77eb635fc554f`.
- CPU-only F1 RRF comparison on `1400` held-out queries:
  CURRENT_FT `Recall 0.9319642857142857 / Precision 0.1975714285714286`;
  V2 `0.9310119047619048 / 0.1972857142857143`;
  V3 `0.9306547619047619 / 0.1972857142857143`.
  V3 delta versus CURRENT_FT: Recall `-0.0013095238095237605`, Precision
  `-0.0002857142857143058`. V3 changed `241` Top5 memberships, improved `2`
  queries, harmed `4`, neutral-changed `235`; relevant documents gained/lost
  `2/4` (membership comparison).
- `V3_F1_GATE = FAIL_F1` because Recall delta is negative. Per the frozen
  early-kill contract, F2/F3/F4, pooled OOF, final fit, Private scoring, and
  submission creation were not run. V3 branch is scientifically failed; no
  automatic upload or submission occurred. V3 GPU run count: `3` (smoke, F1
  train, F1 score).
- Post-run local checks: V3 files `py_compile` PASS; conflict-anchor unit tests
  `2 passed`; downloaded V3 score SHA matches the remote manifest exactly.

## 2026-09-22 — GUARDED_DIRECT_K20_RESIDUAL FINAL CPU EXPERIMENT

- Closed branches remained frozen: `STEP4_K20_PORT`, full `DIRECT_K20` Top5
  replacement, `BGE_FT_V2`, and `BGE_FT_V3_CONFLICT_ANCHOR`; no historical
  artifact was mutated.
- `DIRECT_SIGNAL_GATE = PASS`; exact outer-OOF replay passed. Full Direct
  Recall `0.9293809523809524`, Precision `0.1980357142857143`, delta Recall
  `+0.003794642857142927`; F3 delta `-0.00011904761904768524`. Replay counts
  matched exactly: changed `4683`, improved/harmed `68/42`, net relevant
  `+26`. Gains requiring Top1–3 change `39`; harms from Top1–3 change `26`.
- Frozen `GUARDED_DIRECT_K20_RESIDUAL_V1` contract SHA256:
  `35e4ed32467bb8015af7156a7b5c62a722c60b6fba86fb696692814dd2b286f2`.
  V1 Recall delta `+0.0036755952380952417`, but F1 delta
  `-0.001011904761904714`; `PRIMARY_GATE = FAIL`.
- Exactly one predeclared nested calibration ran because V1 had positive pooled
  signal but one negative fold. `GUARDED_DIRECT_K20_RESIDUAL_V2_NESTED` passed:
  Recall `0.9300505952380952`, delta `+0.004464285714285698`, Precision delta
  `+0.0010357142857142787`; fold Recall deltas F1/F2/F3/F4 were
  `+0.00011904761904768524 / +0.008095238095238155 /
  +0.0017857142857142794 / +0.007857142857142896`; changed/improved/harmed
  `1101/42/13`; Top1–3 changes `0`.
- Selected policy: `V2_NESTED`. Private application was label-free: `2080`
  queries, changed `36`, rank4/rank5 changes `11/25`, Top1–3 changed `0`,
  maximum one new document/query, mean Top5 Jaccard `0.9942307692307693`.
- Isolated submission created, never uploaded or overwrote incumbent:
  `private_task1/submissions/sprint48_guarded_direct/submission_private_guarded_direct.zip`.
  ZIP SHA256 `cb2b65b46068fabc5d8d6601add88ec8205dbab60a4c5536af210df3a5cf87fd`.
  Frozen final model SHA256 `4b962c56570e6315876aeb6eea26d8877a8007e636d83c70ac2e84a13c2a175c`;
  canonical validator PASS on both JSON and ZIP: `2080` questions, exactly
  `5` documents each.
- CPU-only execution: GPU runs `0`, Modal runs `0`, model loads `0`; no Fold0,
  public labels, or Private labels. Final status:
  `READY_FOR_MANUAL_SUBMISSION`; manual review only, no automatic upload.

## 2026-09-22 — FINAL PRE-SUBMISSION AUDIT

- Read-only audit confirmed the final call is
  `fit_direct_model(sorted(target_qids), feature_cache, gold)` over F1–F4.
  Exact final training population: `5600` queries, `112000` q-doc rows,
  `5832` positive rows, `106168` negative rows; feature matrix has `11`
  columns. Fold0 and Private labels were not used.
- `training_population.qdocs = 5600` is a metadata naming bug: it stores the
  query count, not the fitted-row count. It does not indicate a 5600-row fit.
  Full-matrix replay matched final StandardScaler mean/variance exactly:
  max absolute differences `0.0 / 0.0`.
- No retrain, refit, prediction change, GPU, or Modal run. JSON and ZIP
  validators both PASS (`2080` queries, exactly `5` unique documents/query).
  ZIP SHA remains
  `cb2b65b46068fabc5d8d6601add88ec8205dbab60a4c5536af210df3a5cf87fd`.
- Final decision remains `READY_FOR_MANUAL_SUBMISSION`; audit manifest:
  `private_task1/experiments/sprint48_guarded_direct/final_fit_audit_manifest.json`.

## 2026-09-22 — CORRECTED_QWEN_SINGLE_RESCUE PHASE A

- CPU-only cache/provenance audit completed and rerun after `py_compile` PASS.
- Canonical corrected-Qwen cache: `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl`;
  SHA256 `65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f`.
- Provenance PASS: Qwen3-VL-Reranker-2B, revision
  `4bd860ac4f15ad1897a214615cccc700f8f71818`, corrected scorer SHA
  `e3ed417275404c70ea75e189b5922aa9ab1fe711611391f48bb2a74d9e0cabfe`,
  selector `true_s2_bm25_within_document_v2`, aggregation `MAX`, higher score =
  more relevant. Cache has `5600` queries / `431200` q-doc rows / K77,
  duplicate keys `0`, non-finite scores `0`.
- Exact join key: `(query_id, document_id)`; no row-order join.
- Validation PV1 K20: `111816/112000` covered (`99.835714%`), missing `184`;
  full `5443`, partial `157`, zero `0`. Rank-band coverage: 1–5 `27975/28000`,
  6–10 `27958/28000`, 11–20 `55883/56000`.
- Private PV1 K20: `0/41600` covered (`0%`); `2080` zero-coverage queries;
  `PRIVATE_QWEN_CACHE_STATUS = ABSENT`.
- Structural rescue feasibility: validation `5600` queries have an eligible
  cached rank6–20 candidate; Private `0`. No labels, Recall/Precision,
  candidate selection, threshold, policy, inference, GPU, Modal, or submission
  work was performed.
- Decision: `QWEN_PROVENANCE_GATE = PASS`, `DEPLOYABILITY_CLASS = CACHE_PARTIAL`.
  Do not open Phase B; first fill the `184` missing validation identities and
  obtain a separate Private cache. Guarded Direct V2_NESTED incumbent unchanged.
- Script: `private_task1/scripts/analysis/corrected_qwen_single_rescue_phase_a.py`.
- Reports: `private_task1/reports/task1/corrected_qwen_single_rescue_phase_a.json` and
  `private_task1/reports/task1/corrected_qwen_single_rescue_phase_a.md`.

## 2026-09-22 — CORRECTED_QWEN_SINGLE_RESCUE PHASE A.5

- CPU-only Phase A.5 completed; `py_compile` PASS. No Qwen inference, GPU,
  Modal, labels, Recall/Precision, policy selection, or submission work.
- Exact validation gap: `184` q-docs across `157` queries. Root causes:
  `131` `PV1_NEW_OR_REPAIRED_DOC`, `53` `CANDIDATE_UNIVERSE_DRIFT`, `0`
  historical K77 miss, `0` identity-join issues.
- Same-contract cache search recovered `0/184`; all `184` remain for inference,
  exactly `552` chunk units (`3/3/3` min/median/max). Larger full-doc scores
  were not reused because selected-chunk provenance is unavailable/incompatible.
- Validation delta worklist:
  `private_task1/experiments/qwen_single_rescue/validation_delta_worklist.jsonl`.
- Private worklist: `41600` q-docs / `124798` chunk units; exact question-text +
  document + selected-chunk reuse is safe for `55` q-docs, leaving `41545`
  q-docs / `124633` new chunk units. No Private labels used.
- Runtime estimate only from comparable corrected Qwen NVIDIA A10 evidence:
  validation delta `15.19s`; Private new work `3428.57s` scorer-only. Excludes
  startup/checkpoint/Volume commits; no dollar cost estimated.
- `PHASE_B_READINESS = READY_FOR_VALIDATION_DELTA_GPU`. Next action is only a
  bounded validation-delta Qwen run after explicit GPU approval; do not score
  Private or open rescue policy yet. Guarded Direct V2_NESTED unchanged.
- Script: `private_task1/scripts/analysis/corrected_qwen_single_rescue_phase_a5.py`.
- Reports/worklists: `private_task1/experiments/qwen_single_rescue/`.

## 2026-09-22 — CORRECTED_QWEN_VALIDATION_DELTA_FILL PHASE A.6 PREFLIGHT

- CPU-only preflight PASS for the frozen validation delta; `py_compile` PASS.
- Delta worklist SHA256:
  `c9384fde1338e9a6b850e473876a4ff819c8f8dbfe7d90d536493a5d1d9fc428`.
  Exact Phase A.5 missing-set equality PASS: `184` q-docs / `157` queries /
  `552` selected chunk units / `64` unique documents.
- Identity/input gates PASS: duplicate q-docs `0`, duplicate selected chunk IDs
  `0`, missing query text `0`, missing chunk text `0`, non-current PV1 identities
  `0`. Model/revision/scorer/selector/MAX contract PASS.
- GPU/Modal was intentionally not run. The existing optimized runner only accepts
  its embedded 32-shard full-doc universe; the historical runner hard-codes the
  431200-pair K77 worklist. Neither safely accepts the isolated 184-row delta.
- Decision: `BLOCKED_RUNNER_INPUT_ROUTE`; no score/output/cache merge was made.
  Next action is to add only isolated delta-input route plumbing reusing the
  canonical scorer, rerun CPU preflight, then request approval for exactly one
  GPU delta job.
- Preflight script/report:
  `private_task1/scripts/analysis/corrected_qwen_validation_delta_phase_a6_preflight.py`,
  `private_task1/experiments/qwen_single_rescue/phase_a6_preflight.{json,md}`.

## 2026-09-22 — PHASE A.6 DELTA ROUTE PREPARED (GPU NOT RUN)

- Added isolated Modal wrapper
  `private_task1/scripts/modal/task1_qwen_validation_delta.py`.
- The wrapper reuses the canonical historical `download_locked_snapshot`,
  `load_model`, and `score_batch`; it does not implement a new scorer. It is
  locked to worklist SHA
  `c9384fde1338e9a6b850e473876a4ff819c8f8dbfe7d90d536493a5d1d9fc428`,
  exactly `184` q-docs / `552` units, model revision, scorer SHA, selector,
  MAX aggregation, and an isolated output namespace.
- It validates current PV1 chunk text, canonical query text, finite scores,
  duplicate identities, writes raw chunk + q-doc outputs and a manifest, then
  commits only the isolated delta namespace. It never reads Private input or
  labels.
- Both A.6 wrapper and preflight `py_compile`: PASS. CPU preflight rerun:
  `DELTA_WORKLIST_GATE = PASS`, `QWEN_MODEL_GATE = PASS`,
  `QWEN_SCORER_GATE = PASS`, `GPU_RUN_STATUS = NOT_RUN`.
- Phase status: `READY_FOR_USER_GPU_APPROVAL`. No Modal call, GPU run, model
  load, score, or validation-cache merge occurred.

## 2026-09-22 — PHASE A.6 FINAL PRE-LAUNCH REPAIR/AUDIT

- Patched `private_task1/scripts/modal/task1_qwen_validation_delta.py` to use
  canonical chunk text precedence `text -> chunk_text`; `raw_chunk_text` is no
  longer a fallback. If both raw/text exist, the wrapper counts both and fails
  the input gate on any mismatch.
- Added CPU-only `--mode cpu-preflight` over the mounted
  `/workspace/p13/runtime/data/processed_pv1/chunks`; it checks 64 documents,
  184 q-docs, 552 selected chunks, non-empty text, and cross-document IDs
  without loading Qwen or using a GPU.
- GPU function timeout changed exactly from `3600` to `900` seconds. A10,
  `cpu=8`, `memory=32768`, batch `1`, model/revision/scorer/selector/MAX and
  isolated output namespace are unchanged.
- `py_compile`: PASS. Local PV1 mirror precheck: `64` document files,
  `552` selected chunks, raw/text both-present `0`, raw/text mismatch `0`.
- Remote Modal CPU preflight was not executed because the environment approval
  review rejected the external Modal command before launch. Therefore remote
  Volume gate is **UNVERIFIED**, not silently treated as PASS; GPU runs `0`,
  Qwen loads `0`, Modal runs `0`.

## 2026-09-22 — PHASE A.6 MODAL REMOTE-PATH PACKAGING REPAIR

- Fixed the Modal packaging boundary: `REMOTE_DELTA` and `REMOTE_SCORER` are
  now absolute POSIX string literals (`/opt/qwen_validation_delta.jsonl` and
  `/root/task1_b2a_qwen3vl2b.py`), and are passed directly to `remote_path=`.
  Filesystem calls convert them back with `Path(...)` at use sites.
- `REMOTE_CHUNKS` remains a Linux-container `Path` and is not used as a
  packaging `remote_path`.
- `py_compile`: PASS. No scientific contract, worklist, selected chunks,
  batch, A10, timeout `900`, or output namespace changed.
- CPU-only Modal preflight was launched once at app
  `ap-ummlJnEwFCik2HzDJlK738`, but remote import failed before function entry.
  Exact traceback: `IndexError: 3` at
  `/root/task1_qwen_validation_delta.py:23`,
  `LOCAL_ROOT = Path(__file__).resolve().parents[3]` because the packaged
  module lives directly under `/root`.
- Per the A.6 stop rule, no second Modal route was attempted. Remote Volume
  coverage was not reached; GPU runs `0`, Modal GPU runs `0`, Qwen model loads
  `0`. This is a remaining container-import blocker, not a Qwen/scientific
  failure.

## 2026-09-22 — PHASE A.6 GUARDED IMPORT RETEST

- Repaired the remote-import boundary with guarded `_LOCAL_ROOT` detection.
  Local packaging uses repository paths only when they exist; remote `/root`
  import uses the already packaged scorer and worklist without `parents[3]`.
- `REMOTE_IMPORT_REPO_PARENT_DEPENDENCY = 0`; all `remote_path` values remain
  absolute POSIX strings. `py_compile`: PASS.
- Exactly one CPU-only Modal preflight was launched at app
  `ap-Wfvaca7SlYPvYx2RQwQzhZ`. Remote function entry succeeded; no Qwen model
  loaded and no GPU was allocated.
- Remote preflight stopped at the first input failure:
  `RuntimeError: missing PV1 chunk file: 177151` at
  `/root/task1_qwen_validation_delta.py:108`. Complete remote coverage was not
  reached; the remote input gate is FAIL, not PASS.
- No second Modal command, GPU validation-delta, Private scoring, or scientific
  evaluation was run. Final gate is blocked by missing remote PV1 file
  `177151`.

## 2026-09-22 — PHASE A.6 REMOTE PV1 INPUT SYNC

- Derived the exact set from the delta worklist; local gate PASS: `64/64`
  documents, `552/552` selected units, no missing/extra staged files, and
  source-vs-stage SHA256 parity `64/64`.
- Staging manifest:
  `private_task1/experiments/qwen_single_rescue/remote_sync/manifest.json`.
  Staging directory contains only the exact 64 document JSONL files.
- Executed exactly one bounded `modal volume put` operation. Modal reported the
  upload as successful, but its directory semantics placed the files under
  `/runtime/data/processed_pv1/chunks/staged_chunks/`, not directly under
  `/runtime/data/processed_pv1/chunks/`.
- Executed exactly one CPU-only Modal preflight after upload. Remote function
  entered, did not load Qwen, and failed at the expected target path with:
  `RuntimeError: missing PV1 chunk file: 177151`.
- No second upload, no remote move/delete, no GPU, no inference, and no
  Private/scientific evaluation was performed. Final gate:
  `BLOCKED_UPLOAD_NESTED_DESTINATION`.

## 2026-09-22 — PHASE A.6 REMOTE LAYOUT REPAIR COMPLETE

- Read-only verification found the exact manifest-required nested set:
  `64/64` files at `/runtime/data/processed_pv1/chunks/staged_chunks/`.
- Ran `modal volume cp` once using the exact 64 manifest-derived file paths;
  no second upload and no deletion of `staged_chunks/`.
- Parent layout coverage is now `64/64`, with zero missing and zero extra
  numeric document files at `/runtime/data/processed_pv1/chunks/`.
- Ran exactly one CPU-only Modal preflight. Result: `PASS`.
  Remote documents `64/64`, q-docs `184/184`, selected chunks `552/552`,
  non-empty texts `552/552`, cross-document IDs `0`, canonical text gate
  `PASS`, worklist SHA verified.
- `QWEN_MODEL_LOADED: NO`, `GPU_RUNS: 0`, volume commit `false`.
- Final launch gate: `READY_TO_RUN`. GPU `validation-delta` was not launched.

## 2026-09-22 — PHASE A.7 CORRECTED_QWEN_VALIDATION_CACHE_COMPLETE

- Downloaded only the three completed Phase A.6 artifacts from `udsc-p13`;
  the initial `volume cp` syntax was rejected because `cp` is volume-internal.
  Correct `volume get` downloads then passed all exact SHA gates.
- Frozen current PV1 F1–F4 K20 source:
  `private_task1/experiments/pv1_regression_forensics/validation_pv1_worklist_k20.jsonl`;
  `5600` queries / `112000` q-docs / exactly `20` per query.
- Historical compatible coverage: `111816` current-K20 identities. Phase A.6
  delta filled the exact remaining `184`; historical/delta overlap `0`.
- Complete cache gates PASS: duplicates `0`, missing `0`, extra `0`, nonfinite
  scores `0`, provenance contract PASS, labels/Fold0 unused.
- Complete cache:
  `private_task1/experiments/qwen_single_rescue/validation_complete/qwen_validation_k20_complete.jsonl`
  SHA256 `d66ad20546baa7641ddc39f558481c00bec56aa3ad897d8f75b82a9727c87b79`.
  Manifest: `private_task1/experiments/qwen_single_rescue/validation_complete/merge_manifest.json`.
- `GPU_RUNS_THIS_PHASE: 0`, `QWEN_INFERENCE_THIS_PHASE: 0`,
  `PRIVATE_QWEN_RUNS: 0`. Phase B readiness:
  `READY_FOR_CPU_RESCUE_EVALUATION`; stop and request Phase B separately.

## 2026-09-22 — CORRECTED_QWEN_SINGLE_RESCUE PHASE B CLOSED

- Frozen complete Qwen cache and PV1 K20 SHA gates passed. Current universe:
  `5600` F1–F4 queries / `112000` q-docs / exactly `20` per query; Fold0,
  Public labels, and Private labels were excluded.
- Reconstructed `GUARDED_DIRECT_K20_RESIDUAL_V2_NESTED` only from its exact
  OOF baseline and nested-swap artifacts. Canonical evaluator replay passed:
  Recall `0.9300505952380952`, Precision `0.19814285714285715`.
- Strict nested OOF tested only the predeclared Qwen slot-5 family:
  rank caps `{10,20}` × training-margin quantiles `{0.70,0.80,0.90,0.95}`.
  Top1–4 remained unchanged for `5600/5600` queries.
- Qwen OOF Recall `0.9296041666666667`, delta `-0.00044642857142851433`;
  Precision delta `-0.0001071428571428612`. Fold deltas: F1 `0`, F2 `0`,
  F3 `-0.0017857142857142794`, F4 `0`. Changed/improved/harmed: `76/0/3`.
- Hard scientific gate FAIL: F3 is negative, no positive folds, pooled Recall
  is below incumbent, and improved is not greater than harmed. Decision:
  `CLOSE_NO_PRIVATE_GPU`; C-PREP was not run.
- Reports: `private_task1/experiments/qwen_single_rescue/phase_b/`.
  GPU, Modal GPU, and Qwen inference runs in this phase: `0/0/0`.

## 2026-09-23 — LAST-DAY DIRECT LISTWISE TOP5 BRANCH CLOSED

- Ran the new CPU-only `DIRECT_K20_PLUS_ANCHORS_DOCUMENT_LISTWISE_TOP5`
  branch. Contract was frozen before the first fit; LightGBM `LGBMRanker`
  LambdaRank/NDCG@5 used one fixed configuration, `n_jobs=1`, with no Qwen
  features and no hyperparameter sweep.
- Technical preflight PASS: real fit/predict smoke completed with finite
  scores. F1–F4 source population was `5600` queries / `112000` K20 rows;
  incumbent anchors added outside K20: `0`; final candidate rows `112000`.
- Incumbent replay PASS: `GUARDED_DIRECT_K20_RESIDUAL_V2_NESTED` Recall
  `0.9300505952380952`, Precision `0.19814285714285715`.
- Listwise OOF: Recall `0.9305863095238095` (delta
  `+0.000535714285714306`), Precision `0.19846428571428573` (delta
  `+0.0003214285714285836`). Fold Recall deltas: F1
  `-0.00011904761904768524`, F2 `+0.004761904761904745`, F3
  `-0.002142857142857224`, F4 `-0.0003571428571429447`.
- OOF mutations: `4576` changed / `1024` unchanged; `39` improved / `30`
  harmed / `4507` neutral. Mutation counts by changed documents: 1=`688`,
  2=`1865`, 3=`1167`, 4=`686`, 5=`170`. Candidate oracle Recall ceiling:
  `0.9669404761904762`.
- Scientific gate: `FAIL` because F1, F3, F4 are negative, fewer than 3
  folds are positive, and pooled delta is below `+0.0015`. Private final fit
  and challenger were not run. GPU/Modal GPU: `0/0`.
- Decision: `DIRECT_LISTWISE_DECISION=CLOSE`. Last-day recommendation:
  `KEEP_GUARDED_DIRECT_V2_NESTED_INCUMBENT_AND_CLOSE_NEW_EXPERIMENTS`.
- 2026-09-23 FINAL DEADLINE FREEZE AUDIT: incumbent `GUARDED_DIRECT_K20_RESIDUAL / V2_NESTED` giữ nguyên; Private observed `0.9174`. Submission ZIP SHA và final model SHA đều khớp expected. Canonical JSON/ZIP validator PASS; `2080/2080` query, đúng `5` docs/query, missing/extra/duplicate/null = `0`. Deployment contract PASS (`V2_NESTED`, OOF Recall `0.9300505952380952`, Fold0/private labels không dùng). Failed branches Qwen single rescue và Direct listwise đã CLOSED, artifacts được giữ. GPU/Modal/training/refit/inference = `0`; không mutation incumbent. Final freeze: `READY_TO_KEEP_CURRENT_SUBMISSION`; next action `NO_MORE_EXPERIMENTS`. Reports: `private_task1/reports/final_deadline_freeze_audit.json` và `.md`.
- 2026-09-23 TEAM 09205 INCUMBENT FREEZE: giữ nguyên `CONSTRAINED_DUAL_ANCHOR_RRF`, Private observed `0.920544597`. Incumbent ZIP SHA gate PASS: `aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae`. Canonical validator PASS: `2080/2080` query, đúng `5` docs/query, missing/extra/duplicate/null = `0`. Exact V2 anchor `MISSING_AND_NOT_IDENTIFIABLE`; recovery đóng, không fabricated anchor. Branches `MISSING_V2_ANCHOR_RECOVERY=CLOSED`, `DUAL_ANCHOR_SELECTOR=BLOCKED_NO_EXACT_V2`, `QWEN_SINGLE_RESCUE=CLOSED`, `DIRECT_LISTWISE=CLOSED`. GPU/Modal/training = `0`; incumbent không bị thay đổi. Final freeze: `READY_TO_KEEP_09205_SUBMISSION`. Reopen chỉ khi teammate cung cấp exact V2 hoặc generator + inputs. Reports: `private_task1/reports/team_09205/final_09205_incumbent_freeze.json` và `.md`.
- 2026-09-23 GUARDED VS DIRECT TOP5 META SELECTOR: CPU-only branch trên incumbent `0.920544597`; ZIP SHA/validator PASS. Expert A (Guarded V2_NESTED) replay Recall `0.9300505952380952`; Expert B Direct Top5 replay exact `0.9293809523809524`. Complementarity: B better/A better/same = `29/32/5539`; oracle Recall `0.9337708333333333`, delta `+0.0037202380952381375`. Strict nested meta-selector (frozen Q90/Q95/Q97.5, Top1-3 lock, <=2 membership changes) OOF Recall delta = `0`, F1-F4 = `0/0/0/0`, changed/improved/harmed = `16/0/0`; hard promotion gate FAIL. Không fit/apply Private overlay, không tạo challenger, không mutation incumbent. GPU/Modal/Fold0/Public/Private labels = `0`. Decision: `CLOSE_META_SELECTOR`. Artifacts: `private_task1/experiments/guarded_vs_direct_top5_meta_selector/`.
- 2026-09-23 09205 TRIPLE-CONSENSUS LAST-CHANCE: incumbent SHA/3-anchor validator PASS. Chỉ `1891/2080` query có `09205 == 09198 == Guarded` được phép residual; `189` query còn lại immutable. Expert B exact replay PASS. One-swap oracle delta `+0.0031845238095238315`. C1 Pareto delta `0`, gate FAIL. C2 Tree2 delta `+0.00032738095238105114`, improved/harmed `4/1`, relaxed gate PASS; challenger 16 private one-swap mutations. C3 decisive-logit delta `+0.0003571428571429447`, improved/harmed `4/1`, relaxed gate PASS; challenger 10 private one-swap mutations. Hai ZIP đều PASS (`2080`, 5 unique docs/query, missing/extra/duplicate/null = 0), Top1-3 changed = 0, non-triple mutations = 0. Manual order: C3 rồi C2. GPU/Modal/Fold0/Public/Private labels = 0; incumbent không đổi, auto-submit = NO. Report: `private_task1/experiments/09205_last_chance/last_chance_report.md`.
- 2026-09-23 PRIVATE-MATCHED SEMANTIC MICROSURGERY: incumbent SHA/validator PASS; mutable/immutable `1891/189`. Label-free domain classifier AUC `0.9932851597584212`, xác nhận distribution shift mạnh. Strict HGB action OOF không chọn action: ordinary/weighted delta `0/0`, improved/harmed `0/0`, statistical gate FAIL; theo contract tiếp tục manual-evidence mode. Đã review đủ 30 action bằng hai pass đảo thứ tự, không dùng Private labels. Chọn đúng 4 `SEMANTIC_OVERRIDE` one-swap có margin `11/8/8/7`, model ratio gate và retrieval-consensus: q88908 `68024→204342`, q168090 `282223→104132`, q90972 `208105→286745`, q54780 `226240→100833`. Final ZIP SHA `228225ba197492e0f72a1feeed6b01929c340109a7ef0986237fb1a7d7b906e7`; validator PASS (`2080`, 5 unique docs/query, no missing/extra/duplicate/null), Top1-3 changed `0`, immutable mutations `0`. GPU/Modal/Private labels/auto-submit = `0/0/NO/NO`. Status `FINAL_MICROSURGERY_READY`. Report: `private_task1/experiments/09205_final_semantic_microsurgery/final_report.md`.

## 2026-09-23 — FINAL AUTOMATED MULTI-EXPERT STACK

- CPU-only `PRIVATE_ADAPTIVE_MULTI_EXPERT_STACKED_TOP5` completed; incumbent ZIP SHA/validator PASS and remained untouched (`aa8ef30147500164ce23eb2ecfa6aaa57efd393d0fdc953371315f14460f4dae`).
- Exact OOF expert reproduction: A `0.9300505952380952`, B `0.9293809523809524`, C `0.9305863095238095`, canonical R `0.924514880952381`; multi-expert union oracle `0.939485119047619` (delta vs A `+0.00943452380952381`).
- Domain AUC `0.9892423592032967`. Raw stack Recall `0.9311220238095238` (delta `+0.001071428571428612`). Nested gated Recall `0.9305565476190476` (delta `+0.0005059523809524125`), F1/F2/F3/F4 deltas `-0.00011904761904768524 / +0.0007142857142856673 / 0 / +0.0014285714285713347`; changed/improved/harmed/net `186/6/1/+5`. Gate: `LAST_SLOT`.
- Private firewall: mutable/immutable `1891/189`; automatic non-noop proposals `89`, cap/final changes `40`; immutable mutations `0`, Top1-3 changes `0`; no manual/semantic selection.
- One challenger created: `private_task1/submissions/team_09205/submission_private_09205_final_multi_expert_stack.zip`, SHA256 `2db2e1357defa4b6ea79a441c7c4c6c65c8fccffb79c47b9b7face3593b67740`, validator PASS. No auto-submit; Private score unknown. GPU/Modal/Qwen/Fold0/Public/Private labels = `0`.
- Artifacts: `private_task1/experiments/09205_final_multi_expert_stack/`. Incumbent `0.920544597` remains the known-score fallback.

## 2026-09-23 — FINAL LAST-SLOT LOCAL REGIME SELECTOR

- Stage-0 audit of the existing multi-expert challenger PASS: OOF gate `LAST_SLOT`, no gate violation; ZIP SHA `2db2e1357defa4b6ea79a441c7c4c6c65c8fccffb79c47b9b7face3593b67740`, validator PASS.
- Strict local OOF P1 selector: Recall `0.929514880952381`, delta `-0.000535714285714195`, changed/improved/harmed/net `353/2/6/-4`; F1/F2/F3/F4 deltas `0/0/-0.002142857142857224/0`. Rejected.
- Strict local OOF P2 fusion: Recall `0.9300505952380952`, delta `0`, changed/improved/harmed/net `0/0/0/0`. Rejected.
- P0 existing multi-stack remained best: Recall `0.9305565476190476`, delta `+0.0005059523809524125`, changed/improved/harmed/net `186/6/1/+5`; selected in `NORMAL_LAST_SLOT` mode.
- Final last-slot ZIP reuses P0 bytes exactly: `private_task1/submissions/team_09205/submission_private_09205_FINAL_LAST_SLOT.zip`, SHA `2db2e1357defa4b6ea79a441c7c4c6c65c8fccffb79c47b9b7face3593b67740`. Private changes `40` (`39` one-document membership, `1` order-only), protected-189 overrides `0`, Top1-3 changes `0`, validator PASS. Expected net-gain/delta proxies `1.075268817204301 / 0.0005169561621174524`; diagnostic only.
- No manual/semantic edits, hardcoded query IDs, Private/Public/Fold0 labels, leaderboard target, GPU, Modal, Qwen, or auto-submit. Artifacts: `private_task1/experiments/09205_final_last_slot_local_regime_selector/`.
