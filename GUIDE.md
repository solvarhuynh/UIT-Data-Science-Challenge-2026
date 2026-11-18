# Hướng dẫn nâng cấp repo UDSC

Tài liệu này tổng hợp so sánh repo `D:\udsc` với sáu repo trong `tra-cuu/` và nghiên cứu bổ sung từ chín poster kiến trúc của các đội. Sáu báo cáo riêng là ảnh chụp trạng thái repo **trước** đợt tái cấu trúc; các nhận xét về API/frontend/Docker/tests trong đó thuộc phiên bản cũ. Mục tiêu là chỉ ra việc nào tăng trực tiếp cơ hội đạt điểm LegalIR/LegalQA. Bản công bố hiện tập trung vào các CLI thi và đánh giá.

## Đọc nhanh

Repo hiện tại có ingestion luật, parser cấu trúc pháp lý, parent-child chunking, BM25+dense, reranker, QA có citation, evaluator, strict CV và submission validator. Bản API/frontend/Docker cũ đã được gỡ khỏi đường công bố để tập trung vào CLI thi. Điểm nghẽn là **chưa khóa được một đường benchmark có thể tái lập, cùng population với leaderboard, rồi tập trung tối ưu candidate coverage và thứ hạng cuối**.

Trong workspace hiện có:

- `artifacts/task1/experiment_history_09411.md` ghi một kết quả **0,9411** trong lịch sử thử nghiệm công khai. Đây không phải bằng chứng cho Private score.
- `artifacts/final_submission_manifest.json` mô tả submission 1.000 câu với holdout Recall của Task 1 `0,9202–0,9327` và holdout METEOR Task 2 `0,5760–0,5833`. Đây là kết quả/manifest của lần chạy được lưu; không quy đổi sang leaderboard Private.
- `artifacts/task2/supervised_third_parent_6000_1000_report.json` ghi nội bộ METEOR tăng `0,4709 → 0,4912`, trong khi ROUGE-L giảm `0,4120 → 0,3965`. Cần dùng đúng split/scorer này để theo dõi ablation, không gọi là điểm Private.
- Năm fold strict grouped CV đã được tạo cho 7.000 query: 6.984 nhóm normalized, 16 nhóm trùng gồm 32 query; mỗi fold 1.400 query validation và không có normalized query overlap. Artifact nằm ở `artifacts/task1/evaluation/strict_cv_v2/`.
- Tại thời điểm rà soát, `data/raw/btc/LegalIR/train.json`, `selected-contexts/` và manifest `data/processed_v3` **không có trong checkout**. `data/metadata/processing_manifest.json` của bản processed corpus nằm trực tiếp dưới `data/` ghi semantic completeness và quality gate đều **false**. Các báo cáo cũ và bản corpus đó không thay thế được input canonical.

Điểm Private tham khảo cho Task 1 dao động khoảng `0,9491–0,9588` ở các repo/poster. Task 2 poster/README ghi `0,6306–0,6520`. Các con số khác split nên xem là mục tiêu tham khảo, không phải so sánh A/B trực tiếp.

## Những điều đối thủ làm tốt

### 1. Tách lỗi thiếu candidate khỏi lỗi xếp hạng

Endgame ghi candidate Recall@100 `99,32%`; UIT-YQueFue36 poster ghi oracle candidate `0,9979`, nhưng còn 83 gold ở ngoài top 5 và chỉ 4 gold ngoài pool. Điều này cho thấy với candidate union tốt, nút thắt là ranking. Ngược lại, GhienAram phát hiện document-level BM25 có R@5 `42,0%`, trong khi chunk-level tăng lên `83,0%`; trong trường hợp đó cần sửa đơn vị index trước khi thêm reranker.

**Áp dụng:** mọi report Task 1 phải có candidate Recall@5/20/50/100/200, oracle ceiling, final Recall@5, gold-outside-pool và gold-rank-lost. Phân tách single-gold/multi-gold và nhóm câu hỏi. Không chỉ lưu một con số Recall cuối.

### 2. Ghép nhiều nguồn có lỗi khác nhau

Các repo top dùng BM25 cùng nhiều bi-encoder, lexical/title/character/citation channels, rồi hợp nhất ở cấp rank (RRF). Trust&Companion công bố bốn nguồn và trọng số RRF; Endgame dùng ba bi-encoder và BM25; UIT-YQueFue36 poster dùng đến bảy kênh. Lợi thế đến từ tính bổ sung giữa nguồn, không phải cứ thêm model là tốt.

**Áp dụng:** bắt đầu với nguồn đang có trong code/cache; đo source-alone và union cùng depth trên strict fold. Chỉ thêm một encoder/channel nếu cải thiện oracle hoặc final ranking qua nhiều fold và không vượt tổng ngân sách 4B.

### 3. Legal structure cần phục vụ cả retrieval và ranking

Trust&Companion kế thừa heading path; Kitchen/Artiz xét document heading; GhienAram tìm chunk rồi mở rộng nguyên Điều làm evidence. Poster VLARC của UIT-nobody dùng retrieval nhiều tầng, citation graph và LambdaRank. Đây là những cách đưa quan hệ Điều/Khoản/số hiệu luật vào scoring thay vì chỉ coi luật là văn bản thường.

**Áp dụng:** giữ child text ngắn cho candidate search; giữ parent/article, heading, số hiệu, năm và offsets để reranker/ranker nhìn đúng ngữ cảnh. Đo riêng representation có heading/parent vì nó có thể tăng index/cost. Không sửa parser chung nếu chưa có fixture/regression.

### 4. Chỉ sửa thứ hạng khi có bằng chứng đủ chắc

Endgame hiệu chỉnh logit theo query và OOF priors. Kitchen/Artiz dùng top-5 selector. Poster ctin2 báo full-text evidence tăng điểm, nhưng các lần sửa lớn và second pass không luôn giữ được gain; quy tắc đồng thuận hai comparator chỉ thay slot cuối khi threshold cao. Poster TurtleMind dùng 214 feature trong LambdaMART cho document-conditioned query memory và lịch sử điểm; tăng `+0,802` điểm phần trăm trên public trong controlled study.

**Áp dụng:** lấy baseline top 1–3 làm anchor; thử ranker/selector ở top-5 trước; thống kê số lần thay đổi, số gold được cứu/mất và delta từng fold. Query memory, source history và train label prior phải tạo OOF trong fold, không dùng label của validation.

### 5. LegalQA nên tối ưu METEOR theo dữ liệu câu trả lời

GhienAram #1 Private METEOR `0,6520` chọn chunk 256 từ rồi mở rộng Điều, sinh nhiều candidate và dùng MBR. Poster UIT-Kitchen LegalQA đạt `0,6433` với “retrieve → quote → complete” không dùng generator lớn. Poster UIT-YQueFue36 đạt `0,6417` với “quote first, generate last”. Poster UIT-HuuDuyenThienLy báo METEOR `0,6306`. Các hướng này đặt evidence và lexical overlap làm trung tâm.

**Áp dụng:** giữ extractive answer làm baseline mạnh; đánh giá bằng đúng scorer BTC. Đo retrieval coverage, quote selection, answer assembly và generation riêng. Nếu LLM chỉ giúp câu mở/kết, không bắt nó viết lại đoạn luật vốn đã có thể trích đúng. Có thể thử nhiều candidate/selector sau khi một candidate đơn đã đạt chất lượng.

## Poster của các đội — phát hiện có thể tái sử dụng

- **UIT-nobody — VLARC, Task 1:** poster ghi Private Recall@5 khoảng `0,9588`; sơ đồ có offline corpus/index/mention graph, online candidate generation, hybrid retrieval, cross-encoder/LambdaRank và lớp phục hồi cuối. Tên đội không thấy trên poster; file được ghép với UIT-nobody theo điểm Private khớp bảng xếp hạng bạn cung cấp. [Xem poster](tra-cuu/anhsodokientruc/UIT-nobody_Task1_VLARC.jpg).
- **UIT-Trust&Companion, Task 1:** biểu diễn heading giúp BM25 R@5 tăng `0,6964 → 0,7972` trên public; poster báo Private `0,9551`. [Poster](tra-cuu/anhsodokientruc/UIT-Trust-Companion_Task1_LegalIR.jpg).
- **TurtleMind — ViMIR, Task 1:** document-conditioned question memory và score history đưa vào LambdaMART 214 feature; poster báo Private R@5 `95,40%` và controlled public gain `+0,802 pp`. [Poster](tra-cuu/anhsodokientruc/TurtleMind_Task1_LegalIR.jpg).
- **ctin2, Task 1:** pairwise comparator đọc hai candidate theo cả hai thứ tự; Bradley–Terry score và agreement threshold chỉ sửa slot 5 khi model đồng ý. Poster ghi Private `0,9538`. [Poster](tra-cuu/anhsodokientruc/ctin2_Task1_LegalIR.jpg).
- **UIT-YQueFue36, Task 1:** bảy kênh lexical/semantic/citation, hai reranker và LambdaMART; oracle union `0,9979`, Private Recall@5 `0,952790`. [Poster](tra-cuu/anhsodokientruc/UIT-YQueFue36_Task1_LegalIR.jpg).
- **LegalQA:** GhienAram `0,6520`; UIT-Kitchen `0,6433`; UIT-YQueFue36 `0,6417`; UIT-HuuDuyenThienLy poster `0,6306`. Các poster nhấn mạnh chunk-level retrieval, chọn evidence tốt, trích nguyên văn, kiểm soát câu nối và/hoặc MBR. [GhienAram](tra-cuu/anhsodokientruc/GhienAram_Task2_LegalQA.jpg) · [UIT-Kitchen](tra-cuu/anhsodokientruc/UIT-Kitchen_Task2_LegalQA.jpg) · [UIT-YQueFue36](tra-cuu/anhsodokientruc/UIT-YQueFue36_Task2_LegalQA.jpg) · [UIT-HuuDuyenThienLy](tra-cuu/anhsodokientruc/UIT-HuuDuyenThienLy_Task2_LegalQA.jpg).

## Kế hoạch cải tiến theo thứ tự

### P0 — Khôi phục khả năng đo lường

1. Gắn lại đúng train/corpus BTC cho working copy, kiểm tra schema và SHA-256; không dùng dữ liệu suy đoán hoặc nhãn public/private làm train.
2. Kiểm tra fingerprint của strict folds và corpus trước khi reuse artifact `strict_cv_v2` / P12/P13. Nếu input khác, dựng artifact mới ở namespace mới.
3. Chạy lại baseline pipeline và official scorer cho Task 1 và Task 2 riêng; lưu commit, data hash, model hash, config, split, metric và submission hash cùng kết quả.
4. Ghi rõ mỗi điểm là OOF, internal holdout, public hay private. Chỉ so các kết quả cùng split/scorer.

### P1 — Tăng Task 1 Recall theo đúng lỗi

1. Với từng fold, dựng ranking BM25, dense, BGE/lexical và candidate union ở các K hiện có. Xuất curve candidate Recall@K và miss breakdown.
2. Với gold đã vào pool, so max-passage/document aggregation, RRF, score fusion và cross-encoder trên đúng pool.
3. Kiểm tra heading/article/số hiệu/năm/citation features. Ưu tiên feature nhẹ trước model mới.
4. Dùng logistic/LambdaMART nhỏ để xếp candidate doc-level từ feature OOF. Mỗi train row phải được sinh từ fold không chứa query/normalized query đó.
5. Thử gate selective top-5 swap. Giữ top 1–3 làm anchor cho thí nghiệm đầu; log số slot đổi, gold cứu/mất, precision tie-break và per-fold delta.
6. Chỉ promote phương án nếu 5-fold result tốt hơn baseline, không phụ thuộc một fold duy nhất, và scorer chính thức xác nhận package hợp lệ.

### P2 — Tăng Task 2 METEOR mà giữ grounding

1. Tạo baseline extractive: BM25+dense chunk retrieval → rerank → chọn Điều/đoạn trích → ghép câu trả lời theo cấu trúc reference.
2. Đo `candidate evidence coverage`, exact quote coverage, answer METEOR/ROUGE-L và citation consistency thành các lớp report riêng.
3. Huấn luyện selector/ranker trực tiếp theo METEOR của answer hoàn chỉnh; không chỉ dự đoán độ giống token của một passage nếu output là nhiều đoạn.
4. So extractive, extractive+small completion, generative LoRA và multi-candidate MBR trên cùng split. Tính toàn bộ tham số đang hoạt động so với trần 4B.
5. Kiểm chứng file ID/JSON/ZIP bằng validator BTC trước khi đánh giá điểm.

### P3 — Sau khi có gain, khóa vận hành

Thêm manifest nguồn, hashes, resume theo query/chunk, retry batch, lock dependencies và inference offline. Giữ output index/corpus immutable và thí nghiệm trong namespace riêng. Repo đã có nhiều thành phần này; mục tiêu là làm cho **pipeline đạt điểm** tái chạy được.

## Kế hoạch tối đa 5–8 lượt GPU trên Modal

Trước lượt GPU đầu tiên, cần có input BTC đúng nguồn, grouped folds, baseline từng query, corpus fingerprint và scorer tương ứng. Chạy `scripts/evaluation/compare_runs.py` trên hai báo cáo cùng split; ngưỡng `P(Δ>0) ≥ 0,70` là **bằng chứng nội bộ để quyết định tiếp**, không phải xác suất thắng Private đã được hiệu chuẩn. Với LegalQA, báo cáo hiện dùng METEOR diagnostic local nên phải xác minh bằng scorer BTC trước khi gọi gain là chính thức.

1. **Lượt 1–2, Task 1:** fine-tune BGE bằng `build_task1_chunk_training_v3.py` và `finetune_task1_bge_reranker_v3.py` với hai cấu hình hard negative/learning rate đã định trước. Dừng sớm nếu candidate ceiling hoặc full OOF không vượt anchor; giữ checkpoint tách riêng.
2. **Lượt 3, Task 2:** train `train_legal_qa_parent_crossencoder.py` chỉ khi evidence coverage cho thấy reranking có thể cứu gold parent. Nếu retrieval đã đủ, chuyển lượt này sang answer selector.
3. **Lượt 4–5, Task 2:** train answer selector (`train_task2_p17_answer_selector.py`, rồi `train_task2_p19_uplift_selector.py` nếu selector đầu có gain) trên answer hoàn chỉnh và split sạch. So với extractive baseline cùng scorer.
4. **Lượt 6, Task 2:** LoRA với `finetune_task2_qwen_lora.py` chỉ khi lỗi chính là diễn đạt/ghép answer sau khi evidence đã đúng. Kiểm tổng số tham số active dưới trần cuộc thi.
5. **Lượt 7–8 dự phòng:** chạy lại biến thể tốt nhất bằng seed khác hoặc một ablation đã được OOF chứng minh có cơ hội cải thiện. Không chi hai lượt dự phòng để tìm kiếm mù.

Mỗi lượt cần lưu input hash, config, checkpoint hash, seed, thời gian/chi phí, prediction theo query, metric từng fold và paired bootstrap. Không thể hứa xác suất cải thiện ≥70% trước khi khôi phục dữ liệu và đo các kết quả này; hiện tại số lượt GPU đã dùng cho kế hoạch này là **0**.

## So sánh từng repo

- [Endgame — LegalIR Top 7](docs/comparisons/01-endgame-top7-legalir.md)
- [Team Artiz / hohoangluan — LegalIR](docs/comparisons/02-team-artiz-hohoangluan.md)
- [Kietnehi — notebook Task 1/2](docs/comparisons/03-kietnehi-uit-dsc2026.md)
- [GhienAram — LegalQA #1](docs/comparisons/04-ghienaram-legalqa.md)
- [UIT-Kitchen — LegalIR](docs/comparisons/05-uit-kitchen.md)
- [UIT-Trust&Companion — LegalIR #2](docs/comparisons/06-uit-trust-companion.md)

## Kết luận

Điều cần sửa đầu tiên không phải viết lại API hay thêm agentic RAG. Hãy khôi phục input benchmark đúng, phân biệt rõ population/split, tái lập baseline, đo candidate ceiling, rồi làm ranking/evidence ablation trên strict CV. Với LegalQA, tối ưu answer theo scorer METEOR và giữ phần luật trích dẫn được nguyên văn. Chỉ sau khi gain ổn định mới đưa mô hình/feature mới vào production config.
