# Task Results — TV2

Đây là bản tóm tắt theo trình tự thời gian của các task TV2 đã hoàn tất hoặc bị chặn. Bằng chứng chi tiết vẫn nằm trong các report riêng của từng experiment.

## 2026-09-01 — B2a-0 — Context feasibility audit

**Mục tiêu:** Kiểm tra model provenance và khả năng xử lý context trước khi chạy Qwen zero-shot trên K77.

**Đã làm:** Xác nhận checkpoint Qwen/Qwen3-Reranker-0.6B tại revision e61197ed45024b0ed8a2d74b80b4d909f1255473. Audit toàn bộ 431.200 query–document pairs và ghi nhận phân bố độ dài.

**Kết quả chính:** Model audit PASS; context audit execution PASS; feasibility BLOCKED_LONG_DOCUMENT. p99 = 158.503 token, max = 2.145.502 token.

**Trạng thái:** BLOCKED

**Điều rút ra:** Chưa đủ điều kiện chạy full inference hợp lệ.

**Chưa được kết luận:** Qwen có cải thiện Recall/Precision hay không.

**Artifact chính:** Chưa ghi nhận trong ledger này.

**Bước tiếp theo:** Long-document root-cause audit.

## 2026-09-03 — B2a-0 — Frozen Qwen F1-F4 scientific evaluation

**Mục tiêu:** Đánh giá prediction artifact Qwen3-Reranker-0.6B đã freeze trên đúng F1–F4 bằng metric set-based macro Recall/Precision.

**Đã làm:** Xác minh SHA256 `9f73f9424de78a0824c1e7153d080890cfe599104c103a4e2e81b5763b03fcd9`, rồi dùng canonical scorer với `strict_cv_v2/folds.json` và `data/raw/btc/LegalIR/train.json`. Comparator là Workflow-A baseline tại `reports/task1/workflow_a/step2p1_oof_policy_decisions.jsonl`.

**Kết quả chính:** Qwen Recall `0.1593363095238095`, Precision `0.03410714285714286`; delta lần lượt `-0.7665922619047619` và `-0.16328571428571428`. Recall delta theo F1–F4 đều âm (`-0.7854166666666668`, `-0.7729761904761905`, `-0.7611309523809524`, `-0.7468452380952381`).

**Trạng thái:** FAIL — `REJECT_B2A_QWEN_06B`.

**Điều rút ra:** Không có tín hiệu Qwen đạt materiality hoặc fold consistency; 25/5.600 query tốt hơn, 1.165 bằng, 4.410 kém hơn theo Recall.

**Chưa được kết luận:** Không có quyết định tuning, inference mới hoặc prediction thay thế.

**Artifact chính:** Frozen prediction path ở `artifacts/task1/workflow_b/tv2/b2a/qwen06b/predictions.jsonl`; kết quả đầy đủ chỉ in terminal; chưa tạo report persistent.

**Bước tiếp theo:** Professor review kết quả B2a.

## 2026-09-05 — B2a-1 — Canonical batch-1 100-query replay

**Mục tiêu:** Xác nhận batch-1 an toàn số học trên đúng mẫu 100 query/7.700 q-doc/23.093 chunk sau khi vá runner.

**Đã làm:** Vá canonical runner sang effective model-forward batch 1 và chạy đúng một replay trong namespace cô lập; không dùng Fold0/public labels và không đụng frozen predictions.

**Kết quả chính:** PASS về hạ tầng/key coverage trên NVIDIA A10G (requested A10), nhưng Recall `0.03`, Precision `0.006`, AUC `0.9237562524511432`; 0 missing/duplicate/checkpoint errors; 714.000757828 giây, 32.343102926 chunks/s, 4.006190776824951 GiB. Prediction hash `879f66168798b2ad59170120cc05489d7f1ac7e8c550f6946b5521d4d7a52e66` thuộc namespace replay cô lập.

**Trạng thái:** FAIL — `PRODUCTION_B1_FIX_NOT_CONFIRMED`.

**Điều rút ra:** Batch-1 replay không tái tạo Recall cao của fresh B1 diagnostic dù AUC vẫn cao; nguyên nhân canonical-path divergence còn mở.

**Chưa được kết luận:** Không được suy ra fix production hay chạy full inference từ kết quả này.

**Artifact chính:** `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-100q-replay/predictions.jsonl` trên Modal Volume; historical frozen artifact không đổi.

**Bước tiếp theo:** Trace remaining full-run divergence.

## 2026-09-05 — B2a-1 — Canonical B1 post-score trace

**Mục tiêu:** Truy vết CPU-only vì sao replay B1 100 query có AUC cao nhưng Recall/Precision thấp, không chạy inference lại.

**Đã làm:** Đọc artifact replay cô lập trên Modal Volume và tái dựng MAX chunk-score → document-score → top-5 từ 7.700 document rows; dùng F1–F4 labels chỉ để kiểm tra metric.

**Kết quả chính:** `predicted_doc_ids` khớp chính xác descending top-5 cho 100/100 query; Recall/Precision tái dựng là `0.79/0.164`, AUC `0.9237562524511432`. Trái lại, metric trong replay dùng `grouped[query][:5]` chưa sort sau `work.sort(query, doc, chunk)`, tái tạo đúng `0.03/0.006`.

**Trạng thái:** PASS — `EVALUATOR_INPUT_MISMATCH` đã được chứng minh; không có post-score prediction bug.

**Điều rút ra:** Prediction artifact cô lập và thứ tự descending score là đúng; sai số nằm ở metric-input ordering.

**Chưa được kết luận:** Chưa sửa runner hoặc suy ra kết quả full inference.

**Artifact chính:** `udsc-p13:/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-100q-replay/predictions.jsonl`.

**Bước tiếp theo:** Patch proven post-score bug.

## 2026-09-05 — B2a-1 — Replay100 evaluator input fix and re-evaluation

**Mục tiêu:** Vá đúng lỗi metric-input của replay 100 query và chấm lại prediction artifact hiện có bằng CPU, không inference.

**Đã làm:** Trace call-site xác nhận lỗi chỉ ở `replay100_batch1_remote`; thay input top-5 của metric bằng cùng descending score/tie-break với prediction đã materialize. Không đổi model, score, MAX aggregation, labels, candidate pool hay prediction file.

**Kết quả chính:** Artifact SHA256 `879f66168798b2ad59170120cc05489d7f1ac7e8c550f6946b5521d4d7a52e66` có 100 query/7.700 pair/23.093 chunk; 100/100 evaluated top-5 khớp canonical predictions, missing/duplicate 0. Recall `0.79`, Precision `0.16400000000000003`, AUC `0.9237562524511432`.

**Trạng thái:** PASS — `PRODUCTION_B1_FIX_CONFIRMED` cho bounded replay.

**Điều rút ra:** Sự cố chỉ là evaluator đọc document insertion order; prediction và post-score descending ranking không lỗi.

**Chưa được kết luận:** Không suy ra lại kết quả full F1–F4 hay fresh recovery chưa có evaluator provenance.

**Artifact chính:** `udsc-p13:/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-100q-replay/predictions.jsonl`.

**Bước tiếp theo:** Run larger B1 validation.

## 2026-09-05 — B2a-1 — Disjoint 400Q batch-1 validation

**Mục tiêu:** Kiểm tra cuối cùng trên 400 query F1–F4 disjoint trước khi cân nhắc full batch-1 rerun.

**Đã làm:** Chạy đúng một FunctionCall GPU-backed A10 trong namespace mới, batch-forward 1; chọn deterministic 100 query mỗi F1–F4, loại hoàn toàn 100 query replay. K77, True-S2, prompt, model revision, MAX aggregation và metric contract giữ nguyên.

**Kết quả chính:** Hoàn thành 400/400 query, 30.800 q-doc, 92.380 chunks; overlap 0, missing/duplicate 0, 5 docs/query, top-5 consistency 400/400. Pooled Recall/Precision/AUC `0.7820833333333332/0.169/0.9139076992763887`; F1 `0.835/0.18/0.9312806121453382`, F2 `0.785/0.17/0.9147000497599813`, F3 `0.6983333333333333/0.15200000000000002/0.9038866644348269`, F4 `0.81/0.17400000000000002/0.9068348017406237`.

**Trạng thái:** PASS — `B1_LARGER_VALIDATION_PASS`.

**Điều rút ra:** Batch-1 corrected evaluation vượt toàn bộ gate pooled/fold và cho phép xem xét full F1–F4, không tự động chạy.

**Chưa được kết luận:** Không suy ra kết quả full 5.600 query.

**Artifact chính:** `udsc-p13:/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-400q-disjoint-validation/predictions.jsonl`, SHA256 `011fa8859cffe90ccc84e96b90c8bce425860af375449dae1d0294b5ae62aeb3`.

**Bước tiếp theo:** Run full F1-F4 batch1.

## 2026-09-05 — B2a-1 — Full batch-1 pre-flight

**Mục tiêu:** Kiểm tra CPU/config trước full F1–F4 batch-1 mà không launch GPU.

**Đã làm:** Xác minh worklist SHA, workload constants, scoring contract và storage Modal read-only. Profile/workspace active là `hoconlinea10`, default environment `main`, Volume `udsc-p13`; worklist và canonical chunks hiện diện.

**Kết quả chính:** Workload `5600/431200/1293198` và batch 1 đúng; model/prompt/token scoring/MAX aggregation đúng. Tuy nhiên `run_remote()` vẫn trỏ full mode vào namespace lịch sử `.../qwen3-vl-reranker-2b/chunk_scores.sqlite3`, nơi Volume có checkpoint, WAL/SHM, predictions và state. Namespace proposed clean `.../qwen3-vl-reranker-2b-batch1-full-corrected` chưa tồn tại.

**Trạng thái:** BLOCKED — full mode hiện có thể resume checkpoint Batch16 lịch sử.

**Điều rút ra:** Không được launch full trước khi tách rõ output/checkpoint provenance Batch1.

**Chưa được kết luận:** Không có kết quả full F1–F4 mới.

**Artifact chính:** Không tạo artifact; chỉ proposed namespace trên `udsc-p13`.

**Bước tiếp theo:** Fix preflight blocker.

## 2026-09-05 — B2a-1 — Full-mode checkpoint routing trace

**Mục tiêu:** Xác định read-only chính xác checkpoint mà `--mode full` thực sự mở.

**Đã làm:** Trace `main(full)` → `run.spawn` → `run` → `run_remote` → `inspect_checkpoint` → `init_db`. Kiểm tra Volume không ghi dữ liệu.

**Kết quả chính:** `OUTPUT_ROOT` line 33 hardcode `.../qwen3-vl-reranker-2b`; line 34 tạo `CHECKPOINT_DB` trong chính namespace đó, và `run_remote` truyền nguyên giá trị vào SQLite tại line 711. Volume xác nhận namespace này chứa checkpoint, WAL/SHM, predictions và state Batch16 lịch sử. Namespace corrected `...-batch1-full-corrected` không tồn tại nên checkpoint corrected không có rows/provenance để inspect.

**Trạng thái:** BLOCKED — `FULL_MODE_HARDCODED_HISTORICAL_PATH`.

**Điều rút ra:** Không có environment override, resume auto-discovery hay argument override; lỗi là route full mode cố định.

**Chưa được kết luận:** Không launch full và không tạo checkpoint mới.

**Artifact chính:** Không tạo artifact; evidence ở source lines 33–38, 691, 711 và `udsc-p13`.

**Bước tiếp theo:** Patch exact checkpoint routing bug.

## 2026-09-05 — B2a-1 — Full Batch1 corrected routing pre-flight

**Mục tiêu:** Tách hoàn toàn full Batch1 khỏi checkpoint Batch16 lịch sử và xác minh lại CPU/config trước launch.

**Đã làm:** Đổi duy nhất `OUTPUT_ROOT` của full mode trong canonical runner thành `.../qwen3-vl-reranker-2b-batch1-full-corrected`; checkpoint, predictions, errors và run state đều suy ra từ namespace này. Kiểm tra syntax CPU và storage Modal theo chế độ read-only.

**Kết quả chính:** `py_compile` PASS; namespace corrected/checkpoint chưa tồn tại trên `udsc-p13`, nên sạch trước launch. Namespace lịch sử và frozen predictions không bị chạm. Workload `5600/431200/1293198`, worklist SHA `a935e356cf8fe62a000e56c3817fd31dfc5ebbc079c4eef849803a7552c3c25a`, batch 1, profile `hoconlinea10`, environment `main` và storage inputs đều PASS.

**Trạng thái:** PASS — READY_TO_LAUNCH.

**Điều rút ra:** Full mode không còn có thể reuse historical Batch16 checkpoint qua đường dẫn hardcoded.

**Chưa được kết luận:** Chưa có full F1–F4 result mới vì không launch GPU.

**Artifact chính:** Canonical runner `scripts/modal/task1_b2a_qwen3vl2b.py`; corrected Volume namespace sẽ được tạo bởi lần launch tương lai.

**Bước tiếp theo:** Run full F1-F4 batch1.

## 2026-09-07 — B2a-1 — SQLite hot-path optimization

**Mục tiêu:** Loại bỏ chi phí checkpoint/progress tăng theo kích thước database trong full Batch1 mà không đổi semantics inference hay resume.

**Đã làm:** Giữ model-forward batch 1 và worklist order, nhưng buffer score rows trước SQLite commit: ngưỡng 256 q-doc hoặc 1.024 rows. Thay các full-table progress count mỗi flush bằng counter in-memory khởi tạo từ count authoritative lúc startup; thêm throughput rolling 60 giây.

**Kết quả chính:** Schema, primary key, WAL và synchronous FULL không đổi; checkpoint khoảng 692k rows vẫn resume trực tiếp. CPU equivalence old/buffered và interruption/resume đều PASS, duplicate/missing 0. `py_compile` và `git diff --check` PASS; không launch GPU hay ghi Volume.

**Trạng thái:** PASS — PASS_SQLITE_OPTIMIZATION.

**Điều rút ra:** Commit bị giới hạn và progress không còn chạy COUNT/GROUP BY toàn bảng trong hot loop; cumulative metric cũ vẫn giữ, có thêm rolling metric chỉ cho chunk mới commit.

**Chưa được kết luận:** Chưa đo throughput thực tế sau resume.

**Artifact chính:** `scripts/modal/task1_b2a_qwen3vl2b.py`.

**Bước tiếp theo:** Resume full Batch1 from existing checkpoint.

## 2026-09-07 — B2a-1 — Frozen full F1-F4 scientific evaluation

**Mục tiêu:** Chấm prediction Batch1 đã freeze bằng contract set-based macro Recall/Precision canonical, chỉ trên F1–F4.

**Đã làm:** Xác minh SHA256 prediction trước khi đọc labels; dùng `legal_ir_recovery.metrics`, target-only fold reader và target-only train reader. Không dùng GPU, Modal, Fold0 hoặc public labels.

**Kết quả chính:** Coverage `5600/5600`, missing/duplicate/unknown `0/0/0`. Recall/Precision pooled `0.05327380952380952/0.012071428571428573`; F1–F4 Recall lần lượt `0.05648809523809523/0.050833333333333335/0.05125/0.05452380952380952`. So với Workflow-A pooled `0.9259285714285714/0.19739285714285715`, delta là `-0.8726547619047619/-0.18532142857142858`; không fold nào nonnegative.

**Trạng thái:** PASS — evaluation completed; `REJECT_B2A_MATERIALITY`.

**Điều rút ra:** Kết quả frozen không vượt bất kỳ materiality gate đã khóa nào.

**Chưa được kết luận:** Không suy ra nguyên nhân chất lượng thấp hoặc mở run/model mới.

**Artifact chính:** `reports/task1/workflow_b/tv2/b2a/reports/b2a_qwen3vl2b_batch1_full_scientific_evaluation.json`.

**Bước tiếp theo:** Send frozen full eval to professor review.

## 2026-09-07 — B2a-1 — Frozen JSON top-5 reconstruction

**Mục tiêu:** Kiểm tra liệu top-5 frozen có bị serialization sai hay không, không dùng checkpoint SQLite lỗi integrity.

**Đã làm:** Verify SHA prediction; đọc chỉ JSON score fields, rồi tái tạo top-5 theo đúng `(score DESC, canonical_k77_rank ASC, doc_id ASC)` từ runner và evaluate F1–F4 canonical.

**Kết quả chính:** 5.600 JSON records có đủ 431.200 document scores, chính xác 77/query. Frozen và reconstructed top-5 khớp set/order `5600/5600`, overlap `5.0/5`; Recall/Precision reconstructed giữ nguyên `0.05327380952380952/0.012071428571428573`.

**Trạng thái:** PASS — `FROZEN_DOCUMENT_SCORE_FAILURE`.

**Điều rút ra:** Contradiction không phải lỗi full top-5 serialization; signal yếu đã tồn tại trong frozen document-level score ordering.

**Chưa được kết luận:** Chưa xác định nguyên nhân upstream khiến full score distribution khác bounded Batch1 validation.

**Artifact chính:** Frozen `predictions.jsonl` giữ nguyên SHA `556af4f7d83484c5fdabced049ea98e983923038180346b7a7e59b939b7c3cb3`.

**Bước tiếp theo:** Send confirmed failure to professor.

## 2026-09-07 — B2a-1 — Bounded 400Q versus full score divergence trace

**Mục tiêu:** Localize score-level divergence giữa validation Batch1 400Q tốt và full frozen thấp, không inference.

**Đã làm:** Xác định exact 400Q Volume artifact từ ledger matching đủ metric/revision; artifact không hiện diện local nên không fabricate top-5 hoặc score. Reconstruct deterministic 400 query IDs từ runner/worklist rồi evaluate full frozen trên đúng population đó.

**Kết quả chính:** Full covers `400/400` nhưng Recall/Precision trên same 400 chỉ `0.04166666666666667/0.009500000000000001`, trái với bounded `0.7820833333333332/0.169`. Static contract model/processor/prompt/tokenization/batch1/scoring/MAX aggregation là shared; bounded dùng in-memory sorted work, full dùng checkpoint/resume.

**Trạng thái:** BLOCKED — `UNRESOLVED` vì thiếu local 400Q document scores và checkpoint full hợp lệ để join same-item/prove resume membership.

**Điều rút ra:** Chênh lệch tồn tại trên đúng validation 400 query IDs, không phải khác population sample.

**Chưa được kết luận:** Không thể kết luận model-forward, scoring-contract hay resume provenance là nguyên nhân.

**Artifact chính:** Ledger-traced `udsc-p13:/runtime/artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b-batch1-400q-disjoint-validation/`.

**Bước tiếp theo:** Recover 400Q document scores and final checkpoint provenance.

## 2026-09-07 — B2a-1 — Same-item 400Q versus full checkpoint trace

**Mục tiêu:** So sánh cùng q-doc scores giữa 400Q tốt và full checkpoint hoàn chỉnh.

**Đã làm:** Verify SHA/worklist và mở `chunk_scores_FULL.sqlite3` read-only. Join toàn bộ 30.800 q-doc document scores từ 400Q với `MAX(chunk_score)` full.

**Kết quả chính:** Checkpoint integrity PASS, rows/pairs `1293198/431200`. Score correlation Pearson/Spearman chỉ `0.11502696061016578/0.0810068890816878`; median/p95 absolute delta `0.10546875/0.322265625`; top5 set match `0/400`, overlap `0.3275/5`. Full frozen top5 khớp SQLite `400/400` nhưng Recall/Precision same-400 vẫn `0.04166666666666667/0.009500000000000001`.

**Trạng thái:** BLOCKED — `FULL_VS_400Q_SCORE_DIVERGENCE_UNEXPLAINED`.

**Điều rút ra:** Divergence là score-level same-item thực tế, không phải top5 serialization hoặc population mismatch.

**Chưa được kết luận:** SQLite schema không lưu row origin nên không thể gán 693.322 recovered rows vào deterministic prefix.

**Artifact chính:** `chunk_scores_FULL.sqlite3` read-only và 400Q nested predictions/run_state.

**Bước tiếp theo:** Forensic compare pre-resume source checkpoint provenance.
