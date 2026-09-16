# Task1 Workflow C — Phân định Trách nhiệm TV2 / TV4

Trạng thái hợp đồng: **GO_WITH_CONDITIONS**

Workflow B vẫn bị đóng băng. Hợp đồng phân định trách nhiệm này cho phép TV2 thực hiện C0-A và
C0-V có giới hạn trong khi TV4 thực hiện độc lập C1-I. Không có phân công vai trò nào dưới đây
ủy quyền cho C2 hoặc bất kỳ giai đoạn nào sau đó.

## Cập nhật C2-PREFLIGHT cuối cùng — 2026-09-12

TV2 xác nhận C0-A `COMPLETE_PASS`; TV4 xác nhận C1-I `COMPLETE_PASS`, lineage
`C1I_COMMIT_PINNED`; C0-V `DEFERRED`. Governance C2 đã resolved và final
preflight rerun `PASS` dưới `PREFLIGHT_ONLY`; C2 training/scoring/inference
và C3 vẫn `NOT_AUTHORIZED`. TV2 là `FUTURE_C2_EXECUTION_OWNER` nếu được cấp
quyền riêng; TV4 là `INDEPENDENT_C2_REVIEWER` và không train/chọn model khác.

Môi trường pin là `.venv`: Python 3.12.6, NumPy 1.26.4, sklearn 1.7.2,
threadpoolctl 3.6.0. C2-V là fresh classifier
`C2V_FRESH_HGBC_1_7_2_K20`, seed 2026, margin 0; C2-R là exact-delta
`HistGradientBoostingRegressor`, seed 2027, margin `{0,Q75,Q90,Q95}`.
Bootstrap là 10.000 paired-query PCG64 seed 20260911. Resource limits là
1 thread/fit, 1 fit đồng thời, 32 fits, RSS 8 GiB/available 4 GiB và timeout
15 phút/fit, 8 giờ tổng. K20 core; K77/full-pool deferred; Qwen frozen signal
only; B1 frozen optional comparator.

## Phân quyền hiện tại theo người phụ trách

| Phụ trách | Công việc | Phân quyền hiện tại | Phụ thuộc |
|---|---|---|---|
| TV2 | C0-A: Giải phẫu oracle/cơ hội độc lập với policy lịch sử | `AUTHORIZED_AFTER_DOC_FREEZE` | Đóng băng hợp đồng 6 tài liệu |
| TV2 | C0-V: Tái dựng chuyên biệt cho V3A | `BOUNDED_CONDITIONAL_PROVENANCE_RECONSTRUCTION` | Preflight dạng văn bản/chỉ đọc trước khi replay CPU tất định |
| TV4 | C1-I: Kiểm toán định danh/không gian hành động/nguồn gốc xuất xứ | `AUTHORIZED_IN_PARALLEL_AFTER_DOC_FREEZE` | Đóng băng hợp đồng 6 tài liệu; không phụ thuộc C0-V |
| TV2 | C2-PREFLIGHT contract/input freeze | `PREFLIGHT_ONLY`; final rerun `PASS` | Contract/environment/provenance frozen; no training/scoring/inference |
| TV4 | Independent C2 preflight review | `PREFLIGHT_ONLY`; ready for final review | Verifies contract only; does not train/select another model |
| TV2 / TV4 | C2 training/scoring/inference and later evaluation | `NOT_AUTHORIZED` | Phân quyền riêng trong tương lai và các cổng điều kiện trước đó áp dụng |

## Trạng thái bàn giao từ Workflow B

### TV2 / Qwen

- Kết quả Recall lịch sử xấp xỉ 0.0533 là một lỗi thực thi/đầu vào,
  không phải kết quả mô hình; snapshot mã nguồn chính xác trước khi sửa lỗi vẫn
  chưa được khôi phục.
- Kết quả sau khi sửa query-text với batch-size-1 có giá trị khoa học hợp lệ trên
  5.600 truy vấn F1–F4: Recall 0.7997976190 và Precision 0.1693928571.
- Mô hình Qwen độc lập (standalone) sau khi sửa lỗi bị bác bỏ về mặt khoa học.
- Không ủy quyền chạy lại Qwen, suy luận mới, hay sử dụng GPU.
- Điểm tài liệu hiện có chỉ có thể quay lại thông qua một nhánh bổ trợ điểm đóng băng
  (frozen-score complementarity) được ký hợp đồng riêng sau này. Qwen không phải là một phần của C1-I.

### TV4 / B1

- Khôi phục tương đương đã đạt cho 5.600 truy vấn, 806.644 hàng hành động K77, 36
  đặc trưng, và không có sự sai lệch nào.
- Lần fit LightGBM đầu tiên trên toàn bộ dữ liệu dự án đã bị chấm dứt từ bên ngoài;
  số mô hình được fit và số truy vấn OOF được chấm điểm đều bằng 0.
- Tệp điểm OOF trống và các ma trận NPZ khôi phục không tồn tại.
- B1 ở trạng thái `FROZEN_OPTIONAL_C2_COMPARATOR`; không tiếp tục job Workflow B
  lịch sử.

## Phạm vi của TV2

### C0-A

TV2 sở hữu phân tích chẩn đoán và tính toàn vẹn độc lập với policy lịch sử:

- xác minh chính xác 5.600 truy vấn F1–F4 và các mã hash chuẩn tắc;
- tái lập Recall baseline 0.9259285714285714 trong sai số `1e-12`;
- tái lập trần K20, K77, và trần one-swap toàn tập 0.9676994047619048,
  0.9847142857142857, và 0.9897440476190476 trong sai số `1e-12`;
- xác minh 0 sai lệch giữa truy vấn/tài liệu/hành động/fold/schema/các phép join bắt buộc;
- đối soát phân rã từng fold với các giá trị gộp (pooled);
- báo cáo tính toàn vẹn riêng biệt cho K20, K77, và toàn tập (full-pool);
- mô tả đặc trưng số lượng cơ hội, phân phối mức tăng (gain), tổn thất không gian ứng viên,
  cơ hội ở rank-4/rank-5, và nguồn gốc xuất xứ của các tài liệu đưa vào mang lại lợi ích.

C0-A không yêu cầu các lựa chọn lịch sử của V3A trên từng truy vấn, không quy kết
các quyết định lịch sử sai, và không khẳng định tính khả học (learnability). Nó không có
thêm cổng tiếp tục nào dựa trên độ lớn oracle.

Chỉ các kết quả đầu ra được lên kế hoạch:

- `reports/task1/workflow_c/tv2/c0a/c0a_integrity_manifest.json`
- `reports/task1/workflow_c/tv2/c0a/c0a_oracle_opportunity_report.json`
- `reports/task1/workflow_c/tv2/c0a/c0a_oracle_opportunity_trace.jsonl`

### C0-V

TV2 sở hữu một nỗ lực có giới hạn nhằm tái dựng các hành động được chọn/top5 OOF V3A
lịch sử trên F1–F4. Trước khi replay, TV2 phải đóng băng một preflight chứa các đường dẫn/mã hash
ứng viên chính xác, các giá trị cấu hình/manifest lịch sử, các dependency có thể khôi phục,
seed, phạm vi chỉ gồm F1–F4, bằng chứng không kích hoạt giai đoạn Fold0 nào,
và các bất biến kỳ vọng.

TV2 chỉ được phân loại kết quả là `EXACT_SOURCE_REPLAY`,
`AGGREGATE_EQUIVALENT_RECONSTRUCTION`, hoặc `REPLAY_MISMATCH_BLOCKED`.
Tương đương tổng hợp (aggregate equivalence) là một đối chứng được tái dựng mới và không bao giờ
chứng minh được danh tính từng hàng lịch sử. TV2 không được bịa đặt hoặc tìm kiếm vô hạn
mã nguồn bị thiếu, và giới hạn của C0-V không được khái quát hóa sang công việc sạch của C0-A hoặc C1-I.

Chỉ các kết quả đầu ra được lên kế hoạch:

- `reports/task1/workflow_c/tv2/c0v/c0v_preflight.json`
- `reports/task1/workflow_c/tv2/c0v/c0v_reconstruction_manifest.json`
- `reports/task1/workflow_c/tv2/c0v/c0v_selected_actions.jsonl`
- `reports/task1/workflow_c/tv2/c0v/c0v_reconstruction_report.json`

## Phạm vi của TV4

### Pha cấu trúc C1-I

TV4 kiểm toán độc lập, không sử dụng nhãn:

- ID truy vấn, tài liệu, và hành động;
- thành viên tập K20 và K77;
- thành viên tập ứng viên/hành động toàn tập khi có chuẩn tắc;
- thứ hạng nguồn (source ranks) và độ hỗ trợ nguồn (source support);
- khả năng tương thích của schema đặc trưng;
- độ bao phủ, các mã hash, và danh tính không gian chuẩn tắc.

Sau khi các danh tính cấu trúc được đóng băng, TV4 chỉ có thể join nhãn F1–F4 để
xác minh các điểm mút oracle độc lập với policy lịch sử. Các danh tính và đầu vào điểm mút đó
phải được đóng băng trước khi truy cập nhãn. Fold0 và nhãn công khai tiếp tục bị loại trừ.

C1-I không yêu cầu C0-V. Nó loại trừ phân tích hành động được chọn của V3A lịch sử,
quy kết hành động sai so với V3A, tính bổ trợ của Qwen, và mô hình hóa challenger. Phân tích sau này
cần đến các lựa chọn lịch sử phải phụ thuộc một cách tường minh vào dòng dõi (lineage) C0-V được chấp thuận.

Chỉ các kết quả đầu ra được lên kế hoạch:

- `reports/task1/workflow_c/tv4/c1i/c1i_structural_identity_report.json`
- `reports/task1/workflow_c/tv4/c1i/c1i_action_universe_report.json`
- `reports/task1/workflow_c/tv4/c1i/c1i_feature_schema_report.json`

TV4 báo cáo riêng biệt kết quả của K20, K77, và toàn tập (full-pool). Sự cố ở K77 sẽ khép lại
K77 mà không làm vô hiệu hóa lõi K20 sạch; sự cố chỉ ở toàn tập sẽ được khoanh vùng cục bộ.

## Trình tự công việc song song

~~~text
Đóng băng tài liệu (Document freeze)
  |-- TV2: C0-A ---------------------------------------+
  |      |-- Tính toàn vẹn của lõi K20                 |
  |      |-- Trạng thái kiểm toán K77                  | công việc độc lập
  |      `-- Trạng thái chẩn đoán toàn tập (full-pool) |
  |                                                    |
  |-- TV2: C0-V preflight -> tái dựng có giới hạn      |
  |      `-- Phân loại dòng dõi (lineage)              |
  |                                                    |
  `-- TV4: C1-I đóng băng cấu trúc -> kiểm toán điểm mút-+

Không có nhánh nào ở trên tự động mở C2.
~~~

## Hợp đồng bàn giao

### TV2 sang TV4

TV2 có thể bàn giao các định nghĩa truy vấn/tài liệu/hành động chuẩn tắc của C0-A,
danh mục mã hash, schema đối soát fold, và các trạng thái không gian riêng biệt. Nó cũng có thể
bàn giao phân loại nguồn gốc C0-V, nhưng TV4 không chờ đợi hay sử dụng
các lựa chọn của C0-V trong C1-I.

TV2 phải gắn nhãn cho mọi trường được bàn giao là: thuộc về cấu trúc (structural),
chẩn đoán mang nhãn (label-bearing diagnostic), hoặc phụ thuộc hàng lịch sử (historical-row-dependent).
Không nhãn nào từ fold ngoài/mục tiêu được phép đi vào giao diện đặc trưng hoặc giao diện chấm điểm trong tương lai.

### TV4 sang TV2

TV4 bàn giao:

- các báo cáo không có sai lệch hoặc lỗi cục bộ cho các không gian K20/K77/toàn tập;
- các tên đặc trưng được cho phép và nguồn gốc xuất xứ của chúng;
- các điểm khác biệt schema giữa dòng dõi 58 đặc trưng của K20 và hợp đồng tương thích đã được chứng minh của K77;
- các phát hiện về cô lập nhãn và đóng băng dự đoán.

TV4 không bàn giao mô hình đã tinh chỉnh, ngưỡng, quyết định đưa vào/loại bỏ Qwen,
hoặc công thức được lựa chọn bằng cách sử dụng toàn bộ nhãn F1–F4.

## Ranh giới tương lai cho C2

C2 không được ủy quyền. Nếu sau này được ủy quyền riêng:

- K20 là cốt lõi (core);
- công thức chính là `SWAP_ONLY_EXACT_DELTA_RECALL_REGRESSION`;
- KEEP là `ZERO_REFERENCE_SCORE`, không có hàng/vector huấn luyện tổng hợp nào;
- bắt buộc phải có một đối chứng phong cách V3A cấu hình cố định được huấn luyện mới trên cùng phép chia (fresh same-split)
  cho đối chứng khoa học theo cặp (paired science) và không bao giờ được gọi nó là V3A lịch sử;
- mọi phần bù huấn luyện ngoài (outer-training complement) phải độc lập vượt qua cổng nội bộ của C2;
- toàn bộ 4 bundle lựa chọn phải được đóng băng trước bất kỳ phép join nhãn C3 nào.

Recall 0.9296488095238095 của V3A lịch sử và các giá trị fold được bảo toàn của nó là
các chốt chặn vô hướng/từng fold bất biến (immutable scalar/fold guards). Dòng dõi C0-V chính xác chỉ cần thiết
cho các tuyên bố hoặc so sánh đòi hỏi danh tính lịch sử trên từng truy vấn,
không bắt buộc cho đối chứng mới trên cùng phép chia.

## Ranh giới cho K77, B1, và Qwen

- K20 là `CORE`.
- K77 hiện được kiểm toán trong C1-I nhưng không được mô hình hóa. Nó chỉ có thể được mô hình hóa
  sau khi K20 vượt qua C2, không gian/schema/nguồn gốc của K77 đạt yêu cầu, và có hợp đồng riêng
  ủy quyền. Sự thất bại của K20 không đồng nghĩa với việc ủy quyền K77.
- B1 chỉ có thể quay lại dưới vai trò đối chứng C2-B1 được ủy quyền riêng sau khi
  tái hiện thực hóa chính xác, một lần chạy khói (smoke run) 1 fold có ghi nhận mã thoát tiến trình con,
  các giới hạn CPU/bộ nhớ được đăng ký trước, số lượng fit có giới hạn, và cùng quy trình đánh giá lồng nhau (nested evaluation).
  Nếu không, nó vẫn tiếp tục bị đóng băng.
- Nhánh bổ trợ điểm đóng băng của Qwen là một nhánh riêng biệt sau này. Nghiêm cấm
  suy luận mới và việc đưa Qwen vào C1-I.

## Quản trị chung

- Tất cả các tệp dự đoán và danh tính trong tương lai phải đóng băng trước khi join nhãn.
- Recall là chính, Precision là phụ, và đầu ra tối đa là 5 tài liệu duy nhất/truy vấn
  trên tập 5.600 truy vấn F1–F4.
- Điểm quan sát công khai 0.9391 chỉ là bằng chứng triển khai, không bao giờ là
  mục tiêu lựa chọn khoa học hay đối chứng sạch.
- Không còn tập kiểm tra giữ lại độc lập (holdout) nào ở cấp dự án chưa từng chạm tới;
  bằng chứng C3 trong tương lai mang tính chất khám phá (exploratory).
- Bootstrap C3 tương lai đã frozen thành 10.000 paired-query resamples có hoàn
  lại, cỡ mẫu 5.600, PCG64 seed 20260911, mean statistic và linear p10; preflight
  không được phép tính bootstrap.
- C2–C6, GPU/Modal, Fold0, nhãn công khai, và triển khai vẫn chưa được ủy quyền.
- Vấn đề bộ chấm điểm CodaBench đang hoạt động chỉ chặn C6/triển khai.
- Không ai phụ trách được phép sửa đổi các artifact của Workflow A/B hoặc các sổ cái TV2 hiện có
  như một phần của hợp đồng này.

## Hành động an toàn tiếp theo

Sau khi xem xét và đóng băng 6 tài liệu, TV2 chạy C0-A và chỉ bắt đầu bất kỳ
lần replay C0-V nào sau khi hoàn thành preflight bắt buộc; TV4 chạy C1-I song song.
Hôm nay không ủy quyền bất cứ điều gì vượt quá các giai đoạn có giới hạn này.
