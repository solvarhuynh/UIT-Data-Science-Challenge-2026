# UDSC2026 Task1 LegalIR — Lịch sử Thực nghiệm và Nhật ký Quyết định

Trạng thái: Được tái hiện từ mã nguồn cục bộ, manifest, báo cáo JSON và các artifact được lưu trữ. Không có quá trình huấn luyện, truy xuất, suy luận, GPU, Beam, suy luận public hay đánh giá Fold0 mới nào được thực thi trong quá trình lập tài liệu này.

Mức độ ưu tiên của bằng chứng được sử dụng ở đây: JSON metric/đánh giá > manifest > báo cáo tổng kết > mã nguồn > ghi chú > tên thư mục. Khi một báo cáo không chứng minh được tuyên bố, nó được đánh dấu là `UNKNOWN` hoặc `LIKELY`; không suy diễn điểm số từ tên thư mục.

## 1. Sổ theo dõi điểm số và ranh giới đánh giá

Các con số này không thể thay thế cho nhau:

| Họ điểm số | Tập chia (Split) / Ý nghĩa | Recall | Diễn giải |
|---|---|---:|---|
| Strict OOF baseline | 7.000 câu hỏi train, 5 nhóm chuẩn hóa rời rạc, tổng hợp các fold | 0.9244452381 | Điểm chuẩn CV nghiêm ngặt (`baseline_093_oof/report.json`) |
| V3 residual baseline | Folds 1–4, 5.600 câu hỏi; Fold 0 được giữ lại (held out) | 0.9259285714 | Điểm tham chiếu huấn luyện/chọn mô hình, không phải điểm public |
| V3A | Tập xác thực (validation) folds 1–4 | 0.9296488095 | +0.0037202381 so với V3 residual baseline |
| V3A Fold0 | 1.400 câu hỏi đánh giá độc lập (chưa từng can thiệp) | 0.9217261905 | +0.0032142857 so với Fold0 baseline 0.9185119048 |
| V3B raw | Folds 1–4 | 0.9297976190 | +0.0038690476 so với baseline, nhưng có các fold tăng trưởng âm so với V3A |
| V3A one-swap oracle | Folds 1–4, nhãn được dùng cho mục đích chẩn đoán | 0.9676994048 | Trần lý thuyết (ceiling), không phải policy có thể triển khai |
| V3A one-swap oracle | Fold0 | 0.9646428571 | Trần lý thuyết, chỉ dùng cho đánh giá |
| Public leaderboard | Quan sát từ người dùng/bảng xếp hạng | ~0.930 → ~0.939 | Không phải bằng chứng CV; không khẳng định sự đóng góp cụ thể của từng thành phần |

Manifest CV nghiêm ngặt ghi nhận 7.000 câu hỏi, 5 fold, 1.400 câu hỏi validation mỗi fold, 5.600 câu hỏi train mỗi fold, các nhóm chuẩn hóa rời rạc và hoàn toàn không có sự trùng lặp (overlap) giữa tập train/validation đã chuẩn hóa. Fold0 được quy định tường minh chỉ dành cho đánh giá trong `v3_residual/policy/fold0_evaluation.json`; nó không được sử dụng để lựa chọn mô hình.

## 2. Dòng thời gian theo giả thuyết

### EXP-001 — Truy xuất Dense / Baseline và hoàn thiện hệ thống tính điểm (scorer plumbing)

**Giả thuyết.** Một hệ thống truy xuất dense mang tính tiền định cùng bộ chấm điểm cấp văn bản chính xác sẽ thiết lập được một baseline đáng tin cậy.

**Dữ liệu và phân chia.** Dữ liệu train LegalIR; hợp đồng đánh giá nghiêm ngặt gồm 7.000 câu hỏi và 5 fold. Các chẩn đoán P1/P2 ban đầu cũng sử dụng các mẫu thử nghiệm giới hạn 500 câu hỏi. Các dòng benchmark public không có nhãn nên không thể đánh giá dựa trên nhãn chunk giả.

**Phương pháp và kiểm soát.** Độ sâu ứng viên 5/10/20/50/100/200 (và tùy chọn 500), gộp văn bản theo lần xuất hiện đầu tiên/điểm cao nhất (first-occurrence/max document collapse), tính Recall/Precision macro chính thức, không dùng mô hình neural trong P1. Báo cáo P1 đã sửa lỗi không khớp không gian nhãn bằng cách tách biệt các dòng trung chuyển chưa gắn nhãn khỏi các tham chiếu tài liệu LegalIR.

**Điểm số.** Phân tích ứng viên P2 trên mẫu chẩn đoán 500 câu hỏi đạt CandidateDocRecall@200 là 0.9696667, với 11 truy vấn không có tài liệu gold trong tập ứng viên cache. Đây là độ bao phủ ứng viên (candidate coverage), không phải Recall chính thức cuối cùng.

**Kết quả.** ĐẠT (PASS) về mặt hạ tầng/baseline; chưa phải mô hình hoàn thiện cuối cùng. Metric điểm 0 generic trước đây là do lỗi trong luồng tính điểm (được xác nhận bởi `p1_report.json`), không phải là kết luận về hiệu năng truy xuất.

**Quyết định / Bước tiếp theo.** Giữ bộ chấm điểm nghiêm ngặt và bằng chứng baseline; chuyển sang các thực nghiệm BGE tương thích và hợp nhất ứng viên. Bằng chứng: `artifacts/task1/evaluation/p1_report.json`, `p2_candidate/summary.json`, `evaluation/strict_cv_v2/manifest.json`.

### EXP-002 — Các thực nghiệm Reranker BGE

**Giả thuyết.** Bộ tái xếp hạng cross-encoder sẽ cải thiện chất lượng của các ứng viên dense.

**Dữ liệu.** Các lần chạy BGE lịch sử trên public/train bao gồm các biến thể 200/500/all và short17. Mẫu/tập chia chính xác và các metric top-5 cuối cùng không được ghi nhận đồng nhất trong manifest của chúng.

**Kết quả.** `UNKNOWN` đối với một bảng tổng hợp ổn định duy nhất: các manifest so sánh được lưu lại chứng minh đây là các biến thể lịch sử, nhưng không tạo thành một bảng metric so sánh chung. Chúng không phải là bản V3A production.

**Quyết định.** Lưu giữ dưới dạng bằng chứng lịch sử tại `research/legacy/reranker/`; sử dụng đường dẫn reranker khớp ngữ nghĩa suy luận và hợp đồng đặc trưng đóng băng (frozen-feature) sau này. Bằng chứng: đã di chuyển `research/legacy/reranker/**`, các file `run_manifest.json` và `evaluation/comparison.json`.

### EXP-003 — Strict CV v2 (Cross-Validation nghiêm ngặt v2)

**Giả thuyết.** Đánh giá 5-fold rời rạc, chuẩn hóa theo nhóm giúp ngăn ngừa rò rỉ do diễn giải lại câu hỏi (paraphrase leakage) và giúp việc so sánh mô hình có thể tái lập được.

**Dữ liệu/Phân chia.** 7.000 câu hỏi; 6.984 nhóm chuẩn hóa; 16 nhóm trùng lặp / 32 câu hỏi trùng lặp; 5 fold; 1.400 câu validation và 5.600 câu train trên mỗi fold.

**Kết quả.** ĐẠT (PASS). Kiểm tra rò rỉ cho kết quả bằng 0 đối với sự trùng lặp giữa tập train/validation đã chuẩn hóa. Đây là quy chuẩn đánh giá, không phải cải tiến mô hình.

**Quyết định.** Giữ làm nguồn chân lý duy nhất (source of truth) cho các thực nghiệm sau này. Bằng chứng: `evaluation/strict_cv_v2/{manifest,folds,fold_stats}.json`, `evaluation/p1_report.json`.

### EXP-004 — Reranker khớp ngữ nghĩa suy luận (Inference-matched reranker)

**Giả thuyết.** Một reranker được huấn luyện/vật chất hóa với cùng ngữ nghĩa suy luận như production sẽ cải thiện thứ tự ứng viên mà không bị lệch pha giữa huấn luyện và suy luận (train/inference mismatch).

**Dữ liệu/Phân chia.** Vật chất hóa ứng viên LegalIR tập train, kèm theo kiểm toán trùng lặp Fold0 và các artifact đặc thù theo từng fold. Mức độ cải thiện tổng hợp chính xác không được cô lập trong một báo cáo tổng kết duy nhất.

**Kết quả.** `INCONCLUSIVE` (Chưa thể kết luận) nếu coi là tuyên bố đưa lên production độc lập; tập dữ liệu và các kiểm toán trùng lặp được lưu giữ, nhưng bằng chứng phần dư V3 sau đó mới là artifact mang tính quyết định.

**Quyết định.** Giữ checkpoint tái lập tại `recovery_096/inference_matched_reranker_v1/` và `models/inference_matched_reranker_v1_fold0/`; không coi đây là policy production cuối cùng. Bằng chứng: `inspection_report.json`, `dataset/audit_report*.json`, `models/.../run_manifest.json`.

### EXP-005 — Các biến thể BGE dựa trên BM25 (BM25-grounded BGE)

**Giả thuyết.** Việc tái xếp hạng các chunk được BM25 lựa chọn trong một văn bản sẽ giúp khôi phục các kết quả khớp từ vựng/pháp lý bị truy xuất dense bỏ sót.

**Dữ liệu.** 6.884 truy vấn, 161.763 chunk được chấm điểm, không có lỗi thiếu sót; suy luận GPU với độ dài tối đa 512 và định dạng fp16. `COMPLETE_NO_TOP5_EVALUATION` được ghi rõ ràng.

**Kết quả.** Không có điểm số top-5 chính thức nào được tạo ra, do đó việc đề xuất lên production là `INCONCLUSIVE`. Bằng chứng về ứng viên/danh sách xử lý ủng hộ việc tiếp tục xây dựng tập ứng viên, chứ không khẳng định về xếp hạng public.

**Quyết định.** Giữ làm bằng chứng nghiên cứu/khả năng tái lập; bước kế thừa là phương án hợp nhất đa nguồn có giới hạn và bằng chứng true-S2. Bằng chứng: `recovery_096/bm25_grounded_bge_b4_v1/report.json`, `..._evaluation/report.json`.

### EXP-006 — Các thực nghiệm Selector / Gating (Bộ chọn / Cổng phân loại)

**Giả thuyết.** Một cổng chọn lọc có thể phân bổ năng lực tính toán của BGE cho các truy vấn có khả năng cần cứu nguy (rescue), từ đó tăng recall với chi phí hữu hạn.

**Dữ liệu/Phân chia.** Chẩn đoán tập train theo từng fold và các báo cáo fold giữ lại; candidate oracle mang tính chẩn đoán rõ ràng, không phải Recall cuối cùng.

**Kết quả.** Trần ứng viên đạt khoảng 0.9767–0.9801 được ghi nhận cho các kế hoạch selector, nhưng không có điểm top-5 production nào được xác lập. Đánh giá `INCONCLUSIVE/REJECTED` dưới vai trò policy cuối cùng vì trần oracle và số lượng cứu nguy không chứng minh được lợi ích thực tế của policy.

**Dạng lỗi (Failure mode).** Độ can thiệp quá mức của cổng (gate aggressiveness) và sự tách biệt giữa ứng viên/policy; điểm oracle cao hoàn toàn có thể đi kèm với một bộ chọn tồi. Điều này được minh chứng bởi cờ `candidate_oracle_is_diagnostic_not_final_recall=true`.

**Quyết định / Bước tiếp theo.** Giữ các báo cáo chẩn đoán; chuyển hướng sang K500 thích ứng và phép hợp có giới hạn chi phí. Bằng chứng: các báo cáo `bm25_grounded_selector_gate_v1/`, `..._v2/`, `v2b_slim_preflight/`.

### EXP-007 — Hợp nhất ứng viên có giới hạn (Bounded candidate union)

**Giả thuyết.** Bổ sung các nguồn từ vựng độc lập (BM25, word-KNN, char-KNN) vào các ứng viên thích ứng sẽ giúp tăng candidate recall trong khi vẫn duy trì chi phí hữu hạn.

**Dữ liệu/Phân chia.** Chẩn đoán fold tập train với việc căn chỉnh nguồn chính xác; tập public sử dụng dữ liệu đầu vào mới lấy từ K500. Bằng chứng cắt giảm thành phần (ablation) báo cáo giá trị oracle của phép hợp thích ứng, không phải điểm top-5 cuối cùng.

**Bằng chứng.** Báo cáo cắt giảm từ vựng checkpoint cho thấy mức tăng oracle so với K200: word KNN +0.0078452, char KNN +0.0080238, BM25 +0.01235 trên giao thức chẩn đoán của nó; BM25 tốn kém tài nguyên nhất. Đây là các chẩn đoán oracle/chi phí, không phải sự đóng góp trên bảng xếp hạng public.

**Kết quả.** ĐẠT (PASS) về mặt xây dựng tập ứng viên. `public_candidate_union_manifest.json` là hợp đồng tương thích; biến thể không tương thích với dense-K500 bị loại bỏ và lưu trữ lại.

**Quyết định.** Giữ lại phép hợp tương thích; bước kế thừa là danh sách rút gọn (shortlist)/bằng chứng và đặc trưng đóng băng. Bằng chứng: `candidate_union_ablations/`, `public_candidate_union_manifest.json`, `post_k500_bounded_union_v2_audit/report.json`.

### EXP-008 — K500 thích ứng (Adaptive K500)

**Giả thuyết.** Mở rộng chính xác một tỷ lệ truy vấn được hiệu chuẩn từ K200 lên K500 thay vì chấm điểm lại toàn bộ truy vấn.

**Dữ liệu/Phân chia.** Huấn luyện mô hình final-fit từ các artifact theo từng fold; áp dụng trên public gồm 1.000 truy vấn chưa có nhãn với 250 truy vấn được mở rộng (25%).

**Phương pháp.** Hiệu chuẩn logistic/xếp hạng; quy tắc public là `ceil(query_count * selected_budget)`, căn chỉnh tiền tố K500 mới chính xác, không dùng nhãn public.

**Kết quả.** ĐẠT (PASS). Báo cáo public: 1.000 truy vấn, 250 truy vấn mở rộng, ngân sách chọn 0.25, căn chỉnh ĐẠT. Đây là sự mở rộng truy xuất bảo đảm tính đúng đắn và tương thích, không phải một thành phần có thể tách biệt đóng góp điểm số độc lập trên bảng xếp hạng.

**Quyết định.** Giữ final-fit của `adaptive_k500_v1` và `final_public_v3/public_adaptive_k500*`. Bằng chứng: `adaptive_k500_v1/{report.json,adaptive_k500_final_fit_manifest.json}`, báo cáo public.

### EXP-009 — Bộ chọn bằng chứng / true-S2 (Evidence selector / true-S2)

**Giả thuyết.** Việc chọn các chunk bằng chứng trong các văn bản ứng viên có thể tạo ra tín hiệu giai đoạn hai (Stage 2) thực sự dưới một ngân sách chunk có giới hạn.

**Dữ liệu/Phân chia.** Chẩn đoán theo fold; các báo cáo smoke và báo cáo từng phần được lưu giữ. Quy tắc chọn tránh việc chọn trên fold giữ lại một cách rõ ràng.

**Kết quả.** `INCONCLUSIVE` đối với điểm số độc lập cuối cùng: các báo cáo chứa trần candidate-oracle và kế hoạch policy nhưng không thiết lập được mức tăng top-5 production ổn định. Lưu giữ dưới dạng bằng chứng nghiên cứu.

**Quyết định / Bước tiếp theo.** Sử dụng hợp đồng shortlist/bằng chứng public đã được xác thực làm đầu vào cho các đặc trưng đóng băng, không sử dụng các báo cáo selector từng phần để khẳng định điểm số bảng xếp hạng. Bằng chứng: `evidence_selector_recovery_v1*/report*.json`, `final_public_v3/public_shortlist_report.json`.

### EXP-010 — Các nhánh V2

**Giả thuyết.** Các biến thể truy xuất/tái xếp hạng V2 và cứu nguy có giới hạn trước đây có thể cải thiện baseline.

**Dữ liệu/Phân chia.** Các artifact fold/ứng viên lịch sử; các metric so sánh chính xác mang tính đặc thù theo từng nhánh.

**Kết quả.** `REJECTED/LEGACY` (Bị loại / Di sản) nếu xét theo định hướng production. Các artifact V2 được bảo tồn rất hữu ích cho việc tái lập nhưng đã bị thay thế bởi thiết kế phần dư V3 khớp ngữ nghĩa suy luận.

**Quyết định.** Không xóa; giữ lại trong thư mục phục hồi/lưu trữ và các thư mục bằng chứng V2. Bằng chứng: `scripts/beam/task1_v2/**`, `recovery_096/v2b_slim_preflight/`, `selective_bge_*`, các báo cáo chuyển đổi.

### EXP-011 — Baseline policy phần dư V3 (V3 residual policy baseline)

**Giả thuyết.** Một policy tráo đổi tối đa 1 vị trí (one-swap) phần dư có thể cải thiện một baseline mạnh trong khi vẫn bảo vệ an toàn cho 3 thứ hạng đầu tiên.

**Dữ liệu/Phân chia.** Tập train/validation folds 1–4 dùng cho việc lựa chọn; Fold0 dùng để đánh giá độc lập. Nhãn hành động chỉ được rút ra từ baseline so với gold trong các chẩn đoán huấn luyện; áp dụng public hoàn toàn không sử dụng nhãn.

**Phương pháp.** Hợp đồng cố định 58 đặc trưng, tối đa một lần tráo đổi mỗi truy vấn, bảo vệ thứ hạng 1–3, thứ hạng 4/5 đủ điều kiện can thiệp.

**Kết quả.** Baseline folds 1–4 đạt 0.9259285714; V3A đạt 0.9296488095; độ lệch (delta) +0.0037202381; delta theo từng fold lần lượt là +0.0032143, +0.0063095, +0.0042857, +0.0010714; không có fold nào tăng trưởng âm. Baseline Fold0 là 0.9185119048, V3A đạt 0.9217261905.

**Quyết định.** ĐẠT (PASS) và là ứng viên cho production. Bằng chứng: `v3_residual/policy/policy_training_report.json`, `v3a_final_fit_manifest.json`, `fold0_evaluation.json`.

### EXP-012 — V3A final-fit và áp dụng trên Public

**Giả thuyết.** Khớp (fit) policy phần dư V3A ổn định đã chọn trên các bằng chứng huấn luyện hợp lệ và áp dụng hợp đồng đặc trưng đóng băng lên tập public.

**Dữ liệu.** Public: 1.000 truy vấn, 20.392 văn bản, 61.150 chunk; không có nhãn public, không chia fold. Báo cáo đóng băng ghi nhận suy luận CUDA fp16, batch 32, mô hình không đổi.

**Kết quả Public.** Policy ĐẠT: 328 lần tráo đổi, 672 giữ nguyên; 110 lần loại bỏ ở rank-4 và 218 lần loại bỏ ở rank-5; điều kiện tối đa một lần tráo đổi và bảo vệ top 1–3 đều đúng.

**Quyết định.** GIỮ CHO PRODUCTION (KEEP). Sự thay đổi trên bảng xếp hạng public (~0.930 lên ~0.939) chỉ là quan sát từ phía người dùng/bảng xếp hạng; không khẳng định mức tăng +0.009 thuộc về riêng một thành phần cụ thể nào. Bằng chứng: `final_public_v3/public_frozen_features_report.json`, `public_v3a_policy_report.json`, `submission.zip`.

### EXP-013 — Policy hai giai đoạn BENEFIT/HARM V3B

**Giả thuyết.** Một bộ phân loại và cổng BENEFIT/HARM hai giai đoạn có thể cải thiện việc lựa chọn hành động tốt hơn V3A.

**Dữ liệu/Phân chia.** Folds 1–4, 5.600 truy vấn; nhãn Fold0 không can thiệp. BENEFIT rất hiếm: chỉ có 301 truy vấn có bất kỳ BENEFIT nào được tổng hợp; recall của BENEFIT top 1 ở giai đoạn 1 chỉ đạt 0.2691 trong phân tích điều tra (forensic).

**Điểm số và độ ổn định.** V3B thô đạt 0.9297976190 (+0.0038690 so với baseline, +0.0001488 so với V3A), nhưng delta từng fold so với V3A lần lượt là +0.0016667, -0.0003571, -0.0025, +0.0017857: có hai fold bị giảm điểm. Không có mô hình nào đủ điều kiện được chọn.

**Nguyên nhân gốc rễ.** ĐÃ XÁC NHẬN: Xếp hạng giai đoạn 1 bỏ sót các hành động BENEFIT (220 truy vấn tổng hợp có BENEFIT nhưng top 1 giai đoạn 1 lại là non-BENEFIT); phương án chỉ dùng cổng (gate-only ablation) gây hại cho điểm số (-0.0027827 so với baseline, 4 fold đều âm). Đây không đơn thuần chỉ là vấn đề về điểm số cực đại.

**Quyết định.** BÁC BỎ (REJECT) đối với production; lưu giữ các artifact phân tích điều tra. Mô hình kế thừa: policy V3A max-one-swap ổn định. Bằng chứng: `v3b_policy/{v3b_policy_training_report.json,forensic_report.json}` và `v3_delta_report.json`.

### EXP-014 — Xếp hạng cặp BENEFIT (Pairwise BENEFIT)

**Giả thuyết.** Xếp hạng theo cặp giữa các hành động có lợi (beneficial) và có hại (harmful) sẽ cải thiện thứ tự hành động.

**Dữ liệu/Phương pháp.** Các checkpoint và báo cáo cứu nguy pairwise lịch sử; điểm số so sánh cuối cùng chính xác không có trong báo cáo chuẩn được lưu trữ.

**Kết quả.** BÁC BỎ/CHƯA THỂ KẾT LUẬN (REJECTED/INCONCLUSIVE). Sự thưa thớt của BENEFIT và mất cân bằng hành động là những rủi ro thất bại đã được ghi nhận; không có bằng chứng fold ổn định nào biện minh cho việc đưa lên production.

**Quyết định.** Giữ trong `archive/incomplete` / nghiên cứu; được thay thế bởi V3A. Bằng chứng: `archive/incomplete/bge_reranker_rescue_v1/`, các script và manifest pairwise.

### EXP-015 — LambdaMART / LTR

**Giả thuyết.** Một bộ xếp hạng dạng danh sách/cây (listwise/tree ranker) có thể học việc xếp hạng phần dư phi tuyến tốt hơn policy tuyến tính.

**Kết quả.** `UNKNOWN` dưới góc độ quyết định số liệu: không tìm thấy báo cáo metric ổn định hoàn chỉnh nào trong các artifact có thẩm quyền. Mô hình này không dùng cho production.

**Quyết định.** LOẠI BỎ/DI SẢN (REJECT/LEGACY) cho đến khi có báo cáo hoàn chỉnh theo từng fold. Không suy đoán về chất lượng thuật toán chỉ dựa vào tên file.

### EXP-016 — Delta K10

**Giả thuyết.** Giới hạn các hành động phần dư trong một cửa sổ ứng viên delta-K nhỏ sẽ giảm thiểu rủi ro gây hại.

**Kết quả.** Nhánh lịch sử đã bị loại bỏ; không tìm thấy số liệu tổng hợp ổn định đáng tin cậy nào. Rủi ro nhiều khả năng là bỏ sót cửa sổ ứng viên, nhưng bằng chứng trực tiếp là `UNKNOWN`.

**Quyết định.** Giữ nhánh phân tích điều tra; được thay thế bằng không gian hành động rank-4/5 có giới hạn của V3A. Bằng chứng: `scripts/beam/archive/v3_residual_rejected_20260823/`.

### EXP-017 — Cổng cơ hội K20 V1/V2 (Opportunity-Gated K20 V1/V2)

**Giả thuyết.** Một cổng lọc trên điểm cơ hội có thể giảm các lần tráo đổi gây hại.

**Kết quả.** Bị loại bỏ. Thực nghiệm cắt giảm chỉ dùng cổng mang lại bằng chứng tiêu cực trực tiếp: đạt 0.9231458333, delta -0.0027827381, cả 4 fold đều âm. Nguyên nhân gốc rễ được xác nhận là do việc đóng mở cổng quá mức dẫn đến nhiều lần thực thi trung tính/gây hại.

**Quyết định.** Thay thế bằng quy tắc hành động thận trọng và cơ chế bảo vệ thứ hạng đầu của V3A. Bằng chứng: các thực nghiệm cắt giảm cổng trong `v3_delta_report.json` và các script bị loại bỏ.

### EXP-018 — Thực nghiệm mẫu âm khó (Hard-negative experiment)

**Giả thuyết.** Các mẫu âm khó giúp cải thiện khả năng phân biệt hữu ích giữa các chunk suýt khớp (near-miss chunks).

**Kết quả.** `INCONCLUSIVE/REJECTED`: không tìm thấy báo cáo cải tiến ổn định và có thẩm quyền nào. Việc tăng độ phức tạp mà không có sự phân biệt hữu ích đã được xác thực là không đủ điều kiện cho production.

**Quyết định.** Chỉ lưu giữ artifact cho mục đích nghiên cứu; không bao hàm việc huấn luyện lại. Bằng chứng: các script hard-negative/manifest thực nghiệm trong recovery/archive.

### EXP-019 — Các nhánh cứu nguy (Rescue branches)

**Giả thuyết.** Tái xếp hạng cứu nguy hoặc dùng BGE chọn lọc có thể khôi phục các ứng viên bị bỏ sót với chi phí thấp.

**Bằng chứng.** BGE chọn lọc từ Beam đã chấm điểm 1.627 truy vấn được chọn / 487.914 chunk mới, không qua huấn luyện và không tạo file submission; các báo cáo cứu nguy khác ghi nhãn rõ ràng candidate oracle chỉ mang tính chẩn đoán.

**Kết quả.** Hữu ích cho việc phân tích điều tra/chi phí, không phải là một policy hoàn thiện ổn định. Cache BGE cũ không tương thích đã bị từ chối làm nguồn gốc cho K500 mới.

**Quyết định.** Giữ bằng chứng cứu nguy; production V3A sử dụng K500/BGE tương thích mới và phép hợp có giới hạn. Bằng chứng: `selective_bge_k500_v1_from_beam/{manifest,cost_report}.json`, các báo cáo `run_adaptive_k500_rescue.py`.

### EXP-020 — Kiểm toán Metric và phân tích điều tra (Metric audit and forensic analyses)

**Giả thuyết.** Bộ tính điểm, độ liên quan của việc tráo đổi, và các tín hiệu tiếp nhận/cơ hội có thể xác định xem các lỗi còn lại là do truy xuất hay do policy.

**Kết quả.** ĐẠT (PASS) dưới dạng chẩn đoán, tuyệt đối không dùng làm điểm production. Ranh giới bộ tính điểm chính thức, one-swap oracle, số lượng thất bại giai đoạn 1, và trần candidate-oracle đều được ghi nhãn rõ ràng.

**Quyết định.** Giữ lại các báo cáo; không sử dụng giá trị oracle làm tuyên bố về điểm số bảng xếp hạng. Bằng chứng: `audit_task1_metric_integrity.py`, `forensic_swap_relevance_algebra.py`, `forensic_incoming_relevance_signal.py`, `fold0_evaluation.json`.

## 3. Lý do V3A được chọn thay vì V3B

Điểm số tổng hợp cực đại của V3B chỉ cao hơn V3A +0.0001488 và không ổn định: nó thấp hơn V3A ở 2 trong số 4 fold validation, trong khi V3A cải thiện trên mọi fold so với baseline. V3A có hợp đồng bảo vệ top 1–3, tối đa một lần tráo đổi, kết quả Fold0 dương, manifest final-fit, và chuỗi mã băm (hash chain) đặc trưng/policy public có thể tái lập. V3B không có cấu hình ổn định đủ điều kiện và có bằng chứng điều tra về sự thất bại trong xếp hạng BENEFIT. Do đó, V3A được lựa chọn dựa trên độ ổn định, tính đủ điều kiện, hành vi trên Fold0 và khả năng tái lập — chứ không chỉ dựa vào điểm số cực đại đơn thuần.

## 4. Khoảng cách Oracle (Oracle gap)

Sử dụng folds 1–4, V3A đạt 0.9296488 và one-swap oracle đạt 0.9676994: khoảng cách ≈ **0.0380506**. Fold0 đạt 0.9217262 so với oracle 0.9646429: khoảng cách ≈ **0.0429167**.

Khoảng cách này chứng minh rằng không gian hành động có chứa các cơ hội có thể khôi phục được theo các chẩn đoán biết trước nhãn gold. Nó **không** chứng minh rằng một mô hình có thể triển khai thực tế sẽ chạm tới mức oracle, cũng như không khẳng định toàn bộ khoảng cách này đều do khâu truy xuất. Bằng chứng điều tra chỉ ra cả hai nguyên nhân: sự tách biệt giữa ứng viên/policy và thất bại trong việc xếp hạng hành động; sự phân bổ tỷ lệ chính xác vẫn còn đan xen.

## 5. Pipeline Public: từ ~0.930 lên ~0.939

Chuỗi xử lý production gồm:

`dense K500 public mới → tiền tố BGE K200 tương thích → K500 thích ứng (25%) → hợp nhất BM25 + word-KNN + char-KNN → danh sách rút gọn public → bằng chứng true-S2 → đặc trưng reranker đóng băng → V3A 58 đặc trưng → tráo đổi tối đa 1 lần → bảo vệ rank 1–3 → cho phép hiệu chỉnh rank 4/5`.

Phân loại các tuyên bố:

- Bằng chứng offline tồn tại cho độ bao phủ ứng viên, sự căn chỉnh thích ứng, tính đồng nhất đặc trưng, và hành vi policy trên fold/Fold0.
- Tính tương thích mới, mã hash, cam kết không dùng nhãn và các kiểm tra hợp đồng public là bằng chứng về tính đúng đắn/khả năng tái lập.
- Mức thay đổi ~+0.009 trên public là một quan sát trên bảng xếp hạng. Không có phép cắt giảm thực nghiệm nào trên public cho phép phân bổ mức tăng này riêng rẽ cho dense, thích ứng, phép hợp, bằng chứng, đặc trưng đóng băng, hay các thành phần policy.

## 6. Sổ theo dõi kết quả tiêu cực (Negative-result ledger)

| Thực nghiệm | Tập dữ liệu/Phân chia | Điểm số | Delta | Trạng thái | Dạng lỗi được xác nhận | Được thay thế bởi |
|---|---|---:|---:|---|---|---|
| Policy cơ hội chỉ dùng cổng | folds 1–4 | 0.9231458 | -0.0027827 | LOẠI BỎ (REJECT) | Cổng can thiệp quá mức; thực thi gây hại/trung tính | V3A |
| V3B raw | folds 1–4 | 0.9297976 | +0.0038690 so với baseline | LOẠI BỎ (REJECT) | Không ổn định; 2 fold thấp hơn V3A; recall top 1 BENEFIT là 0.2691 | V3A |
| Các biến thể BGE/cache cũ | Lịch sử | Điểm chung UNKNOWN | UNKNOWN | DI SẢN (LEGACY) | Nguồn gốc không tương thích / so sánh chưa hoàn chỉnh | K500/BGE tương thích mới |
| BGE dựa trên BM25 | 6.884 truy vấn | Không có điểm top-5 | N/A | CHƯA KẾT LUẬN (INCONCLUSIVE) | Chỉ có bằng chứng runtime/ứng viên | Hợp nhất có giới hạn + bằng chứng |
| Các nhánh Pairwise/LTR/delta/hard-negative | Lịch sử | Tổng hợp có thẩm quyền UNKNOWN | UNKNOWN | LOẠI BỎ/DI SẢN | Thiếu bằng chứng ổn định | V3A |

## 7. Cây quyết định cuối cùng

```text
bộ tính điểm nghiêm ngặt + baseline CV theo nhóm
  ├─ các biến thể dense/BGE cũ ──> di sản / chưa thể kết luận
  ├─ chẩn đoán ứng viên ──> hợp nhất có giới hạn
  │                           ├─ cache không tương thích ──> bị loại bỏ
  │                           └─ K500 mới + BGE tương thích ──> K500 thích ứng
  ├─ selector / gating / rescue ──> chẩn đoán; loại bỏ các nhánh không ổn định/can thiệp quá mức
  ├─ bằng chứng / true-S2 ──> hợp đồng danh sách rút gọn đã xác thực
  ├─ BENEFIT/HARM V3B ──> bị loại bỏ: không ổn định và bỏ sót BENEFIT giai đoạn 1
  └─ V3A phần dư tráo đổi tối đa 1 lần ──> các fold ổn định + Fold0 dương
                                            └─ đặc trưng đóng băng / policy public
                                                 └─ submission.zip cuối cùng
```

## 8. Bài học kinh nghiệm

1. Độ bao phủ ứng viên và xếp hạng cuối cùng là hai điểm nghẽn riêng biệt; điểm oracle không phải là điểm có thể triển khai.
2. CV nghiêm ngặt theo nhóm và Fold0 độc lập giúp ngăn ngừa việc chọn phải đỉnh điểm số ảo không thể tái lập.
3. Policy phần dư thận trọng tráo đổi tối đa 1 vị trí với cơ chế bảo vệ thứ hạng 1–3 đáng tin cậy hơn việc đóng mở cổng diện rộng.
4. BENEFIT rất thưa thớt; bộ phân loại hai giai đoạn có thể thất bại ngay trước cổng nếu thứ tự giai đoạn 1 bỏ sót hành động hữu ích.
5. Biến động điểm trên bảng xếp hạng public không thể bóc tách cụ thể nếu thiếu các thực nghiệm cắt giảm trên public.
6. Nguồn gốc dữ liệu mới và tính đồng nhất tuyệt đối về đặc trưng/schema cũng quan trọng tương đương với việc lựa chọn mô hình.
7. Xử lý trên GPU chỉ hợp lý cho suy luận đóng băng và chấm điểm lại có giới hạn khi hợp đồng đã được xác thực; bản thân việc chạy GPU không phải là bằng chứng của điểm số.

## 9. Bảng tổng hợp tổng thể (Master table)

| Thực nghiệm | Giả thuyết | Train/Public | Phân chia | Baseline | Điểm số | Delta | Ổn định? | Fold0? | Public? | GPU? | Quyết định | Thay thế | Bằng chứng |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|
| Dense / scorer plumbing | baseline đáng tin cậy | Train | 5-fold | — | 0.9244452 | — | Có | Có | Không | Không | Giữ | V3 residual | `evaluation/p1_report.json` |
| Các biến thể BGE | tái xếp hạng ứng viên | Train/Public lịch sử | hỗn hợp | UNKNOWN | UNKNOWN | UNKNOWN | Chưa rõ | Chưa rõ | Không | Một phần | Di sản | Khớp suy luận | `research/legacy/reranker/` |
| Strict CV v2 | chống rò rỉ | Train | 5-fold theo nhóm | — | Quy chuẩn ĐẠT | — | Có | Có | Không | Không | Giữ | Mọi model sau | `evaluation/strict_cv_v2/` |
| Reranker khớp suy luận | khớp train/suy luận | Train | theo fold | UNKNOWN | UNKNOWN | UNKNOWN | Chưa rõ | Đã kiểm toán | Không | Có | Tái lập | V3 residual | `recovery_096/inference_matched_reranker_v1/` |
| BGE dựa trên BM25 | cứu nguy từ vựng | Train | 6.884 truy vấn | — | Không có điểm top5 | — | Chưa rõ | Chưa rõ | Không | Có | Chưa kết luận | Phép hợp | `bm25_grounded_bge_b4_v1/report.json` |
| Selector / gating | tính toán chọn lọc | Train | theo fold | — | Chẩn đoán oracle | — | Không ổn định | — | Không | Một phần | Loại bỏ bản cuối | K500 thích ứng | `bm25_grounded_selector_gate_v*/` |
| Hợp nhất có giới hạn | bổ sung nguồn từ vựng | Train/Public | theo fold / hợp đồng public | — | Oracle/tính đúng ĐẠT | — | Có hợp đồng | — | Có | Một phần | Giữ | Danh sách rút gọn | `candidate_union_ablations/` |
| K500 thích ứng | mở rộng 25% | Train/Public | final-fit / 1000 public | — | Mở rộng 250/1000 | — | Có hợp đồng | — | Có | Không/Một phần | Giữ | Bằng chứng | `public_adaptive_k500_report.json` |
| Bằng chứng / true-S2 | chọn chunk hữu ích | Train/Public | theo fold / 1000 public | — | Hợp đồng ĐẠT; không đóng góp public độc lập | — | Chưa rõ | — | Có | Có | Giữ hợp đồng | Đặc trưng đóng băng | `public_shortlist_report.json` |
| Các nhánh V2 | cứu nguy trước đây | Train | fold lịch sử | UNKNOWN | UNKNOWN | UNKNOWN | Không có minh chứng cuối | Chưa rõ | Không | Một phần | Di sản | V3 | `recovery_096/v2*` |
| Baseline V3 | hành động phần dư | Train | folds 1–4 + Fold0 | 0.9259286 / 0.9185119 | Tương đương | — | Có | Có | Không | Không | Giữ tham chiếu | V3A | `policy_training_report.json` |
| V3A | tráo đổi 1 lần thận trọng | Train/Public | folds 1–4 + Fold0 / 1000 public | 0.9259286 | 0.9296488 / 0.9217262 | +.0037202 / +.0032143 | Có | Có | Có | GPU đóng băng | Production | — | `v3a_final_fit_manifest.json` |
| V3B | Cổng BENEFIT/HARM | Train | folds 1–4 | 0.9259286 | 0.9297976 | +.0038690 | Không | Chỉ điều tra | Không | Không | Loại bỏ | V3A | `v3b_policy_training_report.json` |
| Pairwise BENEFIT | xếp hạng hành động pairwise | Train | Lịch sử | UNKNOWN | UNKNOWN | UNKNOWN | Chưa rõ | Chưa rõ | Không | Một phần | Loại bỏ/di sản | V3A | Checkpoint cứu nguy |
| LambdaMART/LTR | ranker phi tuyến | Train | Lịch sử | UNKNOWN | UNKNOWN | UNKNOWN | Chưa rõ | Chưa rõ | Không | Chưa rõ | Loại bỏ/di sản | V3A | Không có báo cáo hoàn chỉnh |
| Delta K10 | thu hẹp cửa sổ hành động | Train | Lịch sử | UNKNOWN | UNKNOWN | UNKNOWN | Chưa rõ | Chưa rõ | Không | Chưa rõ | Loại bỏ/di sản | V3A | Script bị loại bỏ |
| Cơ hội K20 | cổng hóa cơ hội | Train | folds 1–4 | 0.9259286 | 0.9231458 | -.0027827 | Không | Không | Không | Không | Loại bỏ | V3A | Cắt giảm cổng |
| Mẫu âm khó | tăng độ phân biệt | Train | Lịch sử | UNKNOWN | UNKNOWN | UNKNOWN | Chưa rõ | Chưa rõ | Không | Một phần | Loại bỏ/di sản | V3A | Không có báo cáo hoàn chỉnh |
| Các biến thể cứu nguy | khôi phục giá rẻ | Train/Public chẩn đoán | hỗn hợp | — | Chỉ oracle/chi phí | — | Chưa rõ | — | Không | Có | Nghiên cứu | Phép hợp mới |
| Kiểm toán metric/điều tra | giải thích thất bại | Train | theo fold / Fold0 | — | Oracle/chẩn đoán | — | N/A | Có | Không | Không | Giữ chẩn đoán | — | Báo cáo điều tra |

## 10. Tính đầy đủ của bằng chứng

Bằng chứng là đầy đủ cho quyết định đưa lên production đối với baseline, CV nghiêm ngặt, hợp đồng K500 thích ứng, hợp đồng hợp nhất có giới hạn, V3A, loại bỏ V3B, các đặc trưng public đóng băng, và áp dụng public cuối cùng. Bằng chứng chưa đầy đủ hoặc không thể so sánh đối với các biến thể BGE lịch sử, Pairwise BENEFIT, LambdaMART/LTR, Delta K10, và các nhánh mẫu âm khó (hard-negative); những nhánh này vẫn được đánh dấu rõ ràng là `UNKNOWN`/di sản thay vì tự tạo ra các con số không có cơ sở.

TASK1_EXPERIMENT_HISTORY_COMPLETE
