# Agent brief — repo UDSC 2026

Đọc file này trước khi sửa code cho `D:\udsc`. Đây là repo của một hệ thống LegalIR + LegalQA end-to-end, nhưng khi mục tiêu là điểm thi thì metric và evaluation protocol là ưu tiên số một. So sánh chi tiết với các repo tham khảo nằm trong [GUIDE.md](GUIDE.md) và `docs/comparisons/`.

## 1. Repo hiện tại làm gì

### Kiến trúc

- `src/udsc2026/ingestion/`: đọc BTC JSON/JSONL/TXT/PDF/DOCX, clean Unicode/OCR, parse hierarchy Document→Chapter→Section→Article→Clause/Point, chunk parent-child, validate và disk-audit.
- `src/udsc2026/retrieval/`: dense (`infrastructure/embedding` + FAISS/Qdrant), BM25 (`retrieval/sparse`), hybrid/score fusion, parent context và cross-encoder reranking.
- `src/udsc2026/evaluation/`: scorer/metrics, LegalIR candidate & document aggregation, diagnostics, strict CV, meta-fusion, submission validation, LegalQA candidate/metrics và batch rerank.
- `src/udsc2026/training/`: Task 1 chunk-training utility; các CLI train/mine nằm trong `scripts/training/` cho Task 1/2.
- `src/udsc2026/qa/`: prompt builder, citation parser, QA engine. `infrastructure/llm/` nạp model local.
- `src/udsc2026/contracts/`: schema dữ liệu giữa ingestion, retrieval, QA và evaluator. Repo công bố tập trung vào pipeline chấm thi chạy từ CLI.
- `scripts/data_prep/`: ingestion/index/BM25; `scripts/evaluation/`: scorer, candidate generation, diagnostics, CV gate; `scripts/inference/`: run pipeline và build/validate submission.
- `configs/`: `base.yaml` cho RAG; `task1_baseline.yaml` cho Task 1 và `gpu.yaml` cho inference GPU. Không trộn config thí nghiệm vào baseline đã chọn.
- `data/processed_v3` là corpus đầu vào đã được audit; không ghi đè trực tiếp. `README.md` là hướng dẫn chạy chuẩn duy nhất, còn `GUIDE.md` ghi chiến lược cải thiện điểm.

### Luồng competition hiện thiết kế

**Task 1:** đọc train/corpus → tạo candidate từ dense/lexical → union và gộp child thành document → rerank/selector → top 5 document IDs → kiểm tra scorer, ID coverage, duplicate và ZIP.

**Task 2:** query → retrieval/rerank → parent/article evidence → prompt/model sinh answer → validate `{id: {answer: string}}` và ZIP.

Metric Task 1 là macro Recall@5, Precision là tie-break. Task 2 lấy METEOR làm metric chính; ROUGE-L và kiểm tra citation/grounding là metric chẩn đoán/secondary theo scorer BTC.

## 2. Đọc kết quả có đúng phạm vi

Các artifact dưới đây có sẵn trong working tree nhưng không đồng nghĩa với điểm Private mới:

- `artifacts/task1/experiment_history_09411.md`: lịch sử ghi Recall `0,9411` trên public experiment. Không gọi là Private.
- `artifacts/final_submission_manifest.json`: submission 1.000 query và metric holdout; giữ đúng nhãn validation/holdout của nó.
- `artifacts/task2/supervised_third_parent_6000_1000_report.json`: internal strict train/eval report; METEOR `0,4709 → 0,4912`, ROUGE-L giảm `0,4120 → 0,3965`.
- `artifacts/task1/evaluation/strict_cv_v2/`: năm fold grouped theo cùng query normalization; chỉ reuse khi input SHA/corpus fingerprint còn khớp.
- `artifacts/task1/evaluation/p12_p13_prerequisites.json` và `p12_p13_candidates_gate.json`: artifact gate cũ. Xem timestamp/hash/status và đối chiếu input mới trước khi dựa vào.

Khi tạo báo cáo mới, ghi: commit, input SHA, corpus manifest/hash, fold manifest, model checkpoint hashes, config, scorer version, số query, output hash, metric scope (OOF/internal/public/private) và miss breakdown. Candidate oracle, score một fold, public score và private leaderboard không được trình bày như cùng một phép đo.

## 3. Trạng thái dữ liệu và thay đổi phải giữ nguyên

Ở lần rà soát hiện tại, các đường dẫn `data/raw/btc/LegalIR/train.json`, `data/raw/btc/LegalIR/selected-contexts/` và `data/processed_v3/metadata/manifest.json` không tồn tại. Dưới `data/` có bản sao cấu trúc processed corpus (`chunks/`, `documents/`, `parents/`, `metadata/`, `benchmarks/`), nhưng `data/metadata/processing_manifest.json` ghi `semantic_completeness_gate_passed=false` và `quality_gate_passed=false`. Không coi bản sao này là V3 canonical hoặc tự chuyển nó sang `data/processed_v3`. Một số artifact cũ còn lại nhưng không đủ input để xác minh từ đầu. Yêu cầu dữ liệu BTC đúng nguồn trước khi chạy benchmark; không tạo nhãn thay thế, không phục hồi dữ liệu đã xóa từ Git một cách tự động.

Trước khi rà soát này, `git status` đã có thay đổi của người dùng:

- sửa `.gitignore`;
- xóa `data/README.md`, `data/raw/.gitkeep`, `data/task1/warmup.json`, `data/task2/warmup.json`, `data/vector_store/.gitkeep`.

Giữ nguyên các thay đổi đó. Không reset, restore, xóa hoặc ghi đè chúng trừ khi người dùng yêu cầu rõ.

## 4. Việc cần làm và cách làm

### P0 — Khôi phục một benchmark có thể lặp lại

1. Nhận đúng `train.json`, selected contexts và test/query file theo rule BTC; ghi hash, schema/count.
2. Xác minh `strict_cv_v2` với hash nguồn. Nếu khác, tạo fold/artifact mới trong namespace mới; không sửa tay fold cũ.
3. Chạy official scorer ở baseline Task 1 và Task 2 riêng. Lưu output, log và manifest; phân loại public/internal/private.
4. Dùng `scripts/inference/validate_legal_ir_submission.py` và `validate_legal_qa_submission.py` trước khi chấm. Giữ ID set đúng, thứ tự/giới hạn top-5 đúng, ZIP đúng schema.

### P1 — Task 1 retrieval/ranking

1. Dùng fold-isolated query group; không cho câu normalized trùng nằm ở train và validation.
2. Xuất source rankings và candidate union ở nhiều K; tính candidate Recall@K, oracle ceiling, số gold ngoài pool và số gold trong pool nhưng rớt top 5.
3. Phân tích theo single/multi-gold, citation/số hiệu, query archetype và document length.
4. So BM25/dense/source union, RRF, child→document aggregation, reranker và ranker nhẹ trên cùng candidates. Giữ score/rank của mỗi nguồn để ablation không phải encode lại.
5. Tạo feature/score OOF cho mọi ranker/prior. Nếu dùng train-label frequency, nearest-query memory, centroid hoặc co-occurrence, loại validation fold khỏi chính bước tạo feature.
6. Thử top-5 selective swap từ baseline anchor; log số swap và gold cứu/mất. Không trao quyền thay top ranks cho một CE chỉ vì CE score cao.
7. Chỉ promote khi full OOF tốt hơn cùng anchor, gain không chỉ đến từ một fold, precision tie-break không xấu và official scorer/validator chấp nhận output.

### P2 — Task 2 theo METEOR

1. Đo retrieval evidence coverage trước khi chỉnh generator.
2. Tạo extractive quote-first baseline từ chunk/article, giữ nguyên legal clause/citation, rồi chấm bằng đúng BTC scorer.
3. Tách retrieval, evidence selection, answer assembly và generation thành các artifact để thấy tầng nào làm mất token overlap/grounding.
4. Huấn luyện answer selector theo METEOR của answer hoàn chỉnh; so với extractive, completion nhỏ, LoRA và candidate consensus cùng split.
5. Tính toàn bộ model active trong pipeline, không chỉ model sinh answer; giữ dưới 4B.

### P3 — Chỉ sau gain: khóa inference

Giữ model local/offline; pin dependencies; checksum model/index/corpus; resume theo query/shard; kiểm tra package CUDA/CPU thực tế; submission validator + manifest. Đặt output experiment trong `artifacts/<task>/<namespace>/`; giữ corpus V3 read-only.

## 5. Nguyên tắc kỹ thuật

- Đọc code/config/artifact trước khi sửa; không suy ra kết quả từ README badge.
- Không dùng dữ liệu public/private answer để train/tune; chỉ dùng nhãn mà BTC cho phép và tôn trọng quy định cross-task.
- Không thay scorer BTC bằng metric “đúng tiếng Việt hơn” khi chọn model.
- Giữ child passage có offset và parent ID; output Task 1 luôn là parent document ID, không phải chunk ID.
- Không chỉnh corpus/index/cache đã fingerprint bằng tay. Sửa stage owner rồi rebuild output ở namespace riêng.
- Không sửa default/baseline để thử ý tưởng; tạo config/experiment mới, ghi hypothesis, stop condition, expected cost và artifact.
- Không tải model, encode corpus toàn bộ, huấn luyện dài hoặc nộp submission nếu chưa có dữ liệu đúng, preflight và kế hoạch compute.
- Giữ contracts tách khỏi competition scoring; thay đổi schema phải theo toàn luồng consumer và validator.

## 6. Tài liệu điều hướng

- [Hướng dẫn tổng hợp](GUIDE.md)
- [Hướng dẫn chạy và cấu trúc repo](README.md)
- [Task 1 strict CV](artifacts/task1/evaluation/strict_cv_v2/README.md)
- [Task 1 history](artifacts/task1/experiment_history_09411.md)
- [Task 2 internal result](artifacts/task2/supervised_third_parent_6000_1000_report.json)
- So sánh sáu repo: `docs/comparisons/`

## 7. Quy trình làm việc trong repo này

Quy trình dưới đây được rút từ `.cursor/rules` của một dự án khác và đã điều chỉnh cho UDSC. Áp dụng cho agent sửa repo này.

1. Trước mỗi thay đổi, xác định input, output, consumer, metric, cách kiểm tra và điều kiện dừng. Tìm file cùng chức năng trước khi tạo file mới; dùng tên thể hiện trách nhiệm, không tạo bản `latest`, `final2`, `backup`.
2. Đọc `README.md`, `GUIDE.md`, file code/config liên quan và `git status` trước khi sửa. Không áp dụng đường dẫn, quy tắc về SUSEP, SQL Server, Airflow, BI hoặc cấu trúc dự án bảo hiểm từ rule cũ.
3. Khi đổi CLI, tham số, input hoặc output, cập nhật `README.md` trong cùng thay đổi. Chỉ ghi lệnh thực sự tồn tại; phân biệt kiểm tra cú pháp, smoke và chạy end-to-end.
4. Mọi Python module/hàm public mới cần docstring, type hints và trách nhiệm rõ. Dùng path tương đối hoặc config, seed rõ, tách train/validation trước khi học feature; ngăn rò rỉ theo nhóm câu hỏi đã chuẩn hóa.
5. Không bịa dữ liệu, kết quả, xác suất thắng hoặc điểm Private. Ghi input hash, checkpoint, split và scorer cùng mỗi kết quả. Dữ liệu BTC, model weights, index, artifact lớn và secrets ở ngoài Git.
6. Khi thay file, sửa toàn bộ tham chiếu và xóa bản cũ trong cùng thay đổi. Mỗi thư mục code mới phải có ít nhất hai file có chức năng thực tế; không tạo folder chỉ để dự phòng.
7. Kiểm tra nhỏ nhất có ý nghĩa, rồi xem `git diff` và `git status`. Không stage tràn lan khi working tree có thay đổi của người dùng. Báo cáo chính xác phần đã chạy, phần còn chờ data/model và bước tiếp theo.

## 8. Cổng quyết định GPU

Người dùng chấp nhận tối đa 5–8 lượt train GPU trên Modal nhưng muốn xác suất cải thiện điểm hiện tại ít nhất 70%. Đây là **ngưỡng ra quyết định** trên kết quả out-of-fold cùng input, không phải lời hứa về Private leaderboard. Trước khi chi GPU, đo baseline đúng scorer và lưu dự đoán từng query. Mỗi thử nghiệm phải có giả thuyết, chi phí dự kiến, stop condition và cùng grouped folds. Dùng `scripts/evaluation/compare_runs.py` để ước lượng `P(Δ metric > 0)` bằng paired bootstrap theo nhóm câu hỏi; cổng nội bộ yêu cầu ít nhất 100 nhóm, gain dương và xác suất bootstrap ≥0,70. Kiểm tra thêm gain theo fold và khoảng tin cậy; không dùng số liệu public/private để tune. Với Task 2, METEOR trong repo chỉ là diagnostic local, chưa có parity với scorer BTC.
