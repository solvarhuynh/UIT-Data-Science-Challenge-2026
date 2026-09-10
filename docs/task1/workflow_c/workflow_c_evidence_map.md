# Task1 Workflow C — Bản đồ Bằng chứng và Artifact

Trạng thái hợp đồng: **GO_WITH_CONDITIONS**

Bản đồ này phân biệt bằng chứng có thể sử dụng bởi C0-A và C1-I độc lập với policy lịch sử
với bằng chứng cần thiết cho các tuyên bố cấp truy vấn V3A lịch sử chính xác trong C0-V.
Các đường dẫn dưới đây là bằng chứng hiện có; không có kết quả đầu ra dự kiến nào của Workflow C
được coi là đã hiện thực hóa.

Ủy quyền hôm nay mang tính cụ thể theo từng nhánh: TV2 C0-A là
`AUTHORIZED_AFTER_DOC_FREEZE`; TV2 C0-V là
`BOUNDED_CONDITIONAL_PROVENANCE_RECONSTRUCTION` và yêu cầu preflight dạng văn bản trước khi replay; TV4 C1-I là
`AUTHORIZED_IN_PARALLEL_AFTER_DOC_FREEZE` và không phụ thuộc C0-V. C2, C3, C4, C5, C6/triển khai công khai,
tiếp tục B1, suy luận Qwen, và GPU/Modal là `NOT_AUTHORIZED`. Fold0 và nhãn công khai không được ủy quyền
cho việc lựa chọn/đánh giá khoa học của Workflow C. Trong mọi giai đoạn áp dụng, các byte dự đoán
và danh tính đều được đóng băng và băm (hash) trước khi các nhãn đánh giá được join.
Pha cấu trúc của C1-I là không dùng nhãn; chỉ sau khi các danh tính được đóng băng,
TV4 mới có thể join nhãn F1–F4 để xác minh điểm mút oracle độc lập với policy lịch sử.

Hợp đồng C2 cố định trong tương lai, nếu được ủy quyền riêng, là
`SWAP_ONLY_EXACT_DELTA_RECALL_REGRESSION` với KEEP `ZERO_REFERENCE_SCORE`,
không có hàng/vector KEEP tổng hợp, khối lượng hoán đổi cân bằng truy vấn là 1.0/truy vấn,
và bắt buộc phải có đối chứng phong cách V3A mới trên cùng phép chia. V3A lịch sử vẫn là một
chốt chặn vô hướng/fold bất biến. Hợp đồng bootstrap tương lai vẫn là
`C3_BOOTSTRAP_PROTOCOL_PENDING_PRE_C2_FREEZE`.

Thứ tự ưu tiên của bằng chứng vẫn là:

`JSON chỉ số/đánh giá > manifest/run_state > bằng chứng thực thi thô > báo cáo chuẩn tắc > mã nguồn > nhật ký tiến độ > tên tệp`.

## Hợp đồng khoa học và các artifact cốt lõi

| Bằng chứng | Đường dẫn chuẩn tắc | Danh tính / kết quả đã xác minh | Vai trò an toàn trong Workflow C |
|---|---|---|---|
| Hợp đồng chỉ số | `reports/task1/contracts/task1_active_metric_contract.md` | Recall chính; Precision phụ; tối đa 5 tài liệu duy nhất/truy vấn | Hợp đồng đang hoạt động cho tất cả các nhánh |
| Các fold khoa học | `artifacts/task1/evaluation/strict_cv_v2/folds.json` | 7.000 ID trên F0–F4, 1.400/fold | Chỉ đọc thành viên F1–F4; loại trừ Fold0 |
| Nhãn chuẩn (Gold labels) | `data/raw/btc/LegalIR/train.json` | 7.000 truy vấn | Chỉ dùng cho join chẩn đoán/đánh giá F1–F4; không bao giờ là đặc trưng mô hình |
| Top5 baseline khoa học | `artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl` | 7.000 hàng; SHA256 `1272cb9e8b465f433c725674081084f084a60cdb9807d6ca6a3aa3d9acc1e7d5` | Baseline C0-A và danh tính không gian sau khi lọc nghiêm ngặt F1–F4; nhãn chuẩn nhúng sẵn yêu cầu cô lập |
| Hợp toàn bộ ứng viên (Full candidate union) | `artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl` | 7.000 truy vấn; giới hạn 200; SHA256 `e09a59852b722fc6e33761cce964920e4929cba59f0171dcd85cbe62d96edfa6` | Danh tính không gian toàn tập không dùng nhãn, thứ hạng và độ hỗ trợ nguồn cho C0-A/C1-I |
| Danh sách rút gọn / bằng chứng K20 | `artifacts/task1/recovery_096/v3_residual/shortlist_evidence.jsonl` | 7.000 truy vấn; tối đa 25 tài liệu; SHA256 `4bd031710f7c26c728d7b57fcf26967a5bed87539496b0c7566083fb8f28d689` | Danh tính K20, bằng chứng True-S2, và các phép join cho C0-A/C1-I |
| Các hành động hoán đổi K20 | `artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl` | 215.422 hàng hoán đổi; 58 đặc trưng; SHA256 `7b7acbdd5c26c1e7b21cba42c84900f7894b8fc3dff1475941534345a140eb3a` | Độ hữu dụng chính xác C0-A và kiểm toán schema C1-I; chỉ duy trì dạng hoán đổi (swap-only); chỉ dùng F1–F4 |
| Báo cáo khoa học V3A lịch sử | `artifacts/task1/recovery_096/v3_residual/policy/policy_training_report.json` | Recall 0.9296488095; 2.152 hoán đổi; 54 BENEFIT, 29 HARM, 2.069 NEUTRAL | Chốt chặn vô hướng tổng hợp/từng fold bất biến; không phải bằng chứng quyết định cấp hàng |
| Mô hình / manifest triển khai V3A | `artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit.joblib`; `artifacts/task1/recovery_096/v3_residual/policy/v3a_final_fit_manifest.json` | 58 đặc trưng; SHA256 mô hình `74633f3677df0ad04273f04ca816c2e64f89576f651b7892e4f9cff3c5324f1e` | Dòng dõi/nguồn gốc triển khai; không phải mô hình fit khoa học của Workflow C |
| Số đếm pháp y V3A | `artifacts/task1/recovery_096/v3_residual/v3b_policy/forensic_report.json` | 301 truy vấn cơ hội; hành động đứng đầu BENEFIT là 81; 220 lần bỏ lỡ xếp hạng; 27 lần từ chối | Chỉ là bất biến tổng hợp lịch sử trừ khi chứng minh được dòng dõi hàng C0-V chính xác |
| Đánh giá V3B | `artifacts/task1/recovery_096/v3_residual/v3b_policy/v3b_policy_training_report.json` | Recall thô 0.9297976190; +0.0001488095 so với V3A; 2/4 fold tệ hơn; chỉ dùng cổng đạt 0.9231458333 | Kết quả lịch sử tiêu cực đã được chứng minh; không replay |

## Bằng chứng Oracle hợp lệ cho C0-A

| Bằng chứng | Đường dẫn chuẩn tắc | Sự thật đã xác minh | Diễn giải theo hợp đồng |
|---|---|---|---|
| Sự tương đương chỉ số | `reports/task1/workflow_a/step0_metric_contract_report.json` | 5.600 truy vấn F1–F4; Recall cục bộ và chính thức trùng khớp cho top5 hợp lệ | Bằng chứng toàn vẹn C0-A |
| Sửa chữa toàn tập (Full-pool repair) | `reports/task1/workflow_a/exp_1a_recall_gap_report.json` | Mức tăng one-swap +0.0638154762; mức tăng không ràng buộc +0.0668779762 | Trần chẩn đoán có nhận biết nhãn, độc lập với policy lịch sử |
| Phân rã K20 | `reports/task1/workflow_a/exp_1b_oracle_gap_decomposition_report.json` | Mức tăng +0.0417708333; mở rộng rank 1–3 thêm 0; 301 truy vấn dương | Cơ hội cốt lõi K20; lỗi báo cáo liên quan đến việc so sánh với V3A lịch sử không có sẵn không làm vô hiệu các sự thật này |
| Phân rã K77 | `reports/task1/workflow_a/exp_1b_rerun_k77_oracle_gap_decomposition_report.json` | Mức tăng +0.0587857143; 426 truy vấn dương; khoảng cách đến toàn tập +0.0050297619 | Kiểm toán K77 ngay bây giờ; không ủy quyền mô hình hóa K77 |
| Kiểm toán phạm vi dữ liệu | `reports/task1/workflow_a/step3_data_scope_audit.json` | `untouched_authorized_holdout_exists=NO` | C3 tương lai vẫn mang tính chất khám phá |

Các điểm mút C0-A chính xác là:

- baseline Recall: **0.9259285714285714**;
- oracle one-swap K20: **0.9676994047619048**;
- oracle one-swap K77: **0.9847142857142857**;
- oracle one-swap toàn tập: **0.9897440476190476**.

Mỗi giá trị oracle là một **MỨC TRẦN CHẨN ĐOÁN CÓ NHẬN BIẾT NHÃN**, không phải dự báo
có thể đạt được. Khoảng cách đo được từ V3A đến K20 là xấp xỉ +0.0380505952;
mức tăng từ K20 đến K77 và K20 đến toàn tập lần lượt là xấp xỉ +0.0170148810 và
+0.0220446429. Các đại lượng này mô tả các nguồn biên độ cải thiện riêng biệt và
không phải là các mức tăng cộng dồn có thể đạt được.

## Ranh giới bằng chứng V3A lịch sử cho C0-V

Các hành động được chọn/top5 cuối cùng OOF V3A F1–F4 chính xác đã không được tuần tự hóa (serialized).
Manifest V3A ghi nhận SHA256 mã nguồn huấn luyện là
`b43068882837f96320d67ea66fd98b4c96bee702083cb1fbe82a3d2dbaf10f0d`,
trong khi tệp hiện tại được kiểm tra
`scripts/beam/task1_v3_residual/train_residual_policy.py` ghi nhận SHA256 là
`9378ec3ead1bab174bc8cec0c50d137c98090ffb5d7d7f0e8be6826383f6b570`.
Đây là một giới hạn trọng yếu về danh tính mã nguồn cho C0-V, không phải là một rào cản toàn cục
chặn C0-A hay C1-I.

Dòng dõi C0-V được chấp thuận chính xác là bắt buộc trước khi đưa ra các tuyên bố về hành động được chọn,
tài liệu đưa vào sai, tài liệu đưa ra sai, KEEP so với hoán đổi, hoặc phân loại lỗi chuyên biệt cho V3A
ở cấp độ truy vấn lịch sử; phái sinh các mục tiêu huấn luyện từ các lựa chọn lịch sử; chạy phân tích/bootstrap
theo cặp với V3A lịch sử; hoặc sử dụng các hàng lịch sử chính xác làm chốt chặn hồi quy dự đoán.

Nó không bắt buộc đối với danh tính baseline/ứng viên/nguồn, sự thật oracle K20/K77/toàn tập,
tổn thất không gian ứng viên, cơ hội oracle rank-4/rank-5, phân phối mức tăng oracle,
nguồn gốc tài liệu đưa vào mang lại lợi ích, hoặc một đối chứng mới được định nghĩa trên cùng phép chia.

Một phân loại `AGGREGATE_EQUIVALENT_RECONSTRUCTION` có thể hỗ trợ kiểm tra vô hướng/từng fold và
so sánh tuyệt đối có giới hạn trên cùng quần thể. Nó không thể xác lập danh tính hàng lịch sử
hoặc bootstrap theo cặp với lịch sử. Bất kỳ sai lệch bất biến nào không được giải thích đều là
`REPLAY_MISMATCH_BLOCKED`.

Mã SHA đặc trưng đóng băng huấn luyện được báo cáo
`cbacfd329d9ca345761e822f11707660d2e2ba4fd74c6f73a0f3552c6b12820c`
không tương ứng với một tệp đặc trưng đóng băng thô hiện có. Điều này ngăn cản việc tái tạo đặc trưng thô
từ tệp đó, nhưng C0-A và C1-I có thể kiểm toán 58 đặc trưng đã được nhúng sẵn trong `actions.jsonl`.

## Dòng dõi ứng viên và triển khai

| Giai đoạn | Bằng chứng chuẩn tắc | Vai trò đã xác minh |
|---|---|---|
| Dense K500 mới | `artifacts/task1/recovery_096/final_public_v3/public_raw_k500_manifest.json` | 1.000 truy vấn công khai; SHA256 đầu ra `c39bb73f47da54436229d0dc8651d7cb14ec36df95110ddff8666a47a671dbf1` |
| BGE K200 tương thích | `artifacts/task1/recovery_096/final_public_v3/public_bge_k200_compatible_manifest.json` | Căn chỉnh tiền tố chính xác; SHA256 đầu ra `d8d046c4e1d234f269dbb7bc3911463dbb8be43cd6f9c41f912eb87528cd59ee` |
| K500 thích ứng (Adaptive K500) | `artifacts/task1/recovery_096/final_public_v3/public_adaptive_k500_report.json` | 250/1.000 được mở rộng; không có nhãn công khai |
| RRF 4 nguồn | `artifacts/task1/recovery_096/final_public_v3/public_candidate_union_manifest.json` | Thích ứng/BM25/word-KNN/char-KNN; RRF 60; giới hạn 200; SHA256 `c23605b28d634cc4e0df6cc313eabcf4a6a9122a036e652691666570b54559ce` |
| Danh sách rút gọn công khai | `artifacts/task1/recovery_096/final_public_v3/public_shortlist_report.json` | Hợp rank <=20 kèm baseline; tối đa 25; True-S2 |
| Đặc trưng công khai | `artifacts/task1/recovery_096/final_public_v3/public_frozen_features_report.json` | 20.392 tài liệu; 61.150 đoạn văn; SHA256 đặc trưng `cc6c0357a979172faf10135962e7e056fdc82256be361eb92304a2d54e52085e` |
| Policy V3A công khai | `artifacts/task1/recovery_096/final_public_v3/public_v3a_policy_report.json` | 328 hoán đổi: rank4 110, rank5 218, KEEP 672 |
| Mô hình đương nhiệm triển khai | `artifacts/task1/submission.zip`; `reports/task1/contracts/pre_workflow_b_incumbent_forensic_report.json` | SHA256 ZIP `4f860cb42a5ee681894cbd96b98a27bd2ad564100f8d90d058b23826fcee5a1a`; quan sát được 0.9391 |

Giá trị 0.9391 chỉ là bằng chứng triển khai. Nguồn gốc chính xác của gói/tệp nhị phân bộ chấm điểm CodaBench
đang hoạt động vẫn chưa được giải quyết và chỉ chặn C6/triển khai công khai.
Nó không chặn C0-A, preflight C0-V có giới hạn, hay C1-I.

## Bằng chứng Workflow B TV2 / Qwen

| Bằng chứng | Đường dẫn chuẩn tắc | Nội dung đã xác minh | Xử lý hiện tại |
|---|---|---|---|
| Dự đoán đã sửa | `artifacts/task1/workflow_b/tv2/b2a/qwen3-vl-reranker-2b/predictions.jsonl` | 5.600 hàng; 431.200 điểm tài liệu; SHA256 `65ef5e500f6f0f3e9006da0aeaa6f07bff98e7123d7a5c2e09fd87db5eaabc2f` | Chỉ là các điểm hiện có đã đóng băng; hiện tại không có ủy quyền bổ trợ |
| Đánh giá đã sửa | `reports/task1/workflow_b/tv2/b2a/reports/b2a_qwen3vl2b_batch1_full_querytext_fixed_scientific_evaluation.json` | Recall 0.7997976190; Precision 0.1693928571 | Kết quả độc lập tiêu cực hợp lệ |
| Trạng thái chạy mới | `reports/task1/workflow_b/tv2/b2a/provenance_recovery/qwen3vl2b_querytext_fixed_full_run_state.json` | 1.293.198 đoạn văn mới; đã khôi phục/bỏ qua/lỗi đều bằng 0 | Nguồn gốc xuất xứ của lần chạy |
| Replay sai lệch đầu vào | `reports/task1/workflow_b/tv2/b2a/provenance_recovery/qwen3vl2b_micro_replay_prefix_raw.json`; `reports/task1/workflow_b/tv2/b2a/provenance_recovery/qwen3vl2b_micro_replay_postfix_raw.json` | 24 cặp truy vấn-tài liệu / 72 đoạn văn; sự tương đương chính xác sau khi sửa | Lần chạy lịch sử đạt xấp xỉ 0.0533 là do lỗi thực thi/đầu vào, không phải kết quả mô hình; khác biệt mã nguồn lịch sử vẫn không có sẵn |
| Cắt bớt / đối chiếu / chuỗi lưu ký | `reports/task1/workflow_b/tv2/b2a/reports/b2a_qwen3vl2b_true_s2_truncation_incidence_audit.json`; `reports/task1/workflow_b/tv2/b2a/reports/b2a_qwen3vl2b_official_parity_disagreement_audit.json`; `reports/task1/workflow_b/tv2/b2a/provenance_recovery/qwen3vl2b_final_prediction_chain_of_custody.json` | 0 token trên 8.192; Pearson 0.999752; Spearman 0.999244; không có đảo ngược trọng yếu; các mã hash khớp nhau | Hỗ trợ tính hợp lệ của kết quả tiêu cực đã sửa |
| Dung hợp bổ trợ lịch sử | `reports/task1/workflow_b/tv2/b2a/reports/b2a_qwen3vl2b_auxiliary_fusion_crossfit.json` | Phép join điểm liên tục Workflow-A chưa được chứng minh | Không xác lập được tính bổ trợ |

Mô hình Qwen độc lập sau khi sửa lỗi bị bác bỏ về mặt khoa học. Suy luận Qwen mới,
chạy lại toàn bộ, sử dụng GPU, và phân tích Qwen trong C1-I không được ủy quyền.
Các điểm tài liệu đã đóng băng hiện có chỉ có thể quay lại thông qua một hợp đồng riêng
sau này về tính bổ trợ của điểm đóng băng.

## Bằng chứng Workflow B TV4 / B1

| Bằng chứng | Đường dẫn chuẩn tắc | Nội dung đã xác minh | Xử lý hiện tại |
|---|---|---|---|
| Hợp đồng đăng ký trước | `reports/task1/workflow_b/tv4/b1/contracts/workflow_b_b1_preregistered_contract.json` | Hợp đồng Direct-LTR đóng băng trước khi fit | Chỉ là đặc tả lịch sử |
| Báo cáo cuối | `reports/task1/workflow_b/tv4/b1/reports/workflow_b_b1_final_report.json` | 0 mô hình và 0 truy vấn OOF | Không có kết quả khoa học |
| Khôi phục tương đương | `reports/task1/workflow_b/tv4/b1/recovery/workflow_b_b1_recovery_equivalence.json` | 5.600 truy vấn; 806.644 hàng hành động K77; 36 đặc trưng; 0 sai lệch | Chỉ là nguồn gốc xây dựng |
| Manifest khôi phục | `reports/task1/workflow_b/tv4/b1/recovery/workflow_b_b1_recovery_manifest.json` | Mã hash hàng toàn cục `d1a5695657105e56d0fbf467dd8b33091e7a61710680f38b4bff62ff946c6a80` | Các ma trận NPZ được tham chiếu vắng mặt |
| Chẩn đoán thời gian chạy | `reports/task1/workflow_b/tv4/b1/runtime/workflow_b_b1_runtime_root_cause.json` | Bị dừng từ bên ngoài trong lần fit đầy đủ đầu tiên; chẩn đoán độ tin cậy trung bình | Vấn đề thời gian chạy chưa được giải quyết |
| Kết quả điểm OOF | `reports/task1/workflow_b/tv4/b1/reports/workflow_b_b1_oof_action_scores.jsonl` | 0 byte | Điểm không thể tái sử dụng |

B1 ở trạng thái `FROZEN_OPTIONAL_C2_COMPARATOR`. Job lịch sử của nó không được tiếp tục.
Nó chỉ có thể quay lại dưới sự ủy quyền riêng sau khi tái hiện thực hóa có giới hạn chính xác,
một lần chạy khói thời gian chạy 1 fold có mã thoát tiến trình con, các giới hạn CPU/bộ nhớ
và số lần fit được đăng ký trước, và sử dụng cùng quy trình lồng nhau.

## Bằng chứng kho ngữ liệu đã xử lý

`data/processed_v3/metadata/processing_manifest.json` ghi nhận 8.532 tài liệu,
1.270.356 đoạn văn, 184.548 đoạn cha, và SHA256 của cây đã xử lý là
`80fb33ff1133ce2583097cc5a98ddc9739240a60bfe11c979892d5412dcda647`.
Nó chỉ hỗ trợ các payload ID/văn bản hiện có. Các cổng chất lượng và tính đầy đủ ngữ nghĩa
của nó mang giá trị false, vì vậy Workflow C không được diễn giải lại nó như thể đã được xác thực chất lượng mới.

## Sổ đăng ký giới hạn theo phạm vi nhánh

| Phạm vi | Giới hạn | Ảnh hưởng |
|---|---|---|
| Các tuyên bố hàng lịch sử C0-V | Thiếu các lựa chọn OOF được tuần tự hóa và lệch SHA mã nguồn | Các tuyên bố chính xác đòi hỏi `EXACT_SOURCE_REPLAY`; tương đương tổng hợp vẫn có giới hạn |
| K77 | Dòng dõi 36 đặc trưng/K77 lớn hơn khác với hợp đồng 58 đặc trưng của K20 | Kiểm toán ngay trong C1-I; chỉ mô hình hóa sau khi K20 C2 thành công và có ủy quyền riêng |
| B1 | Ma trận vắng mặt, tệp điểm trống, thời gian chạy chưa giải quyết | B1 bị đóng băng; không tiếp tục Workflow B |
| Qwen | Kết quả độc lập tiêu cực; mã nguồn lịch sử trước sửa lỗi không có sẵn | Không suy luận; chỉ dùng điểm đóng băng trong nhánh riêng được ủy quyền sau này |
| C6 | Nguồn gốc gói/nhị phân của bộ chấm điểm đang hoạt động chưa giải quyết | Chỉ chặn riêng việc triển khai |

Không có giới hạn nhánh nào ở trên tạo ra một lệnh cấm toàn cục đối với công việc C0-A hoặc C1-I sạch.
Tính toàn vẹn của K20, K77, và toàn tập phải được báo cáo riêng biệt để một sự sai lệch cục bộ ở một nhánh
vẫn được giữ nguyên tính chất cục bộ của nhánh đó.
