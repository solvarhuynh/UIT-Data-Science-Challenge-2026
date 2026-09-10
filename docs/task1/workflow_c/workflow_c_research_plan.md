# Task1 Workflow C — Kế hoạch Nghiên cứu Chuẩn tắc

Định hướng: **WORKFLOW_C_RESIDUAL_RECOVERY_IS_BEST_NEXT_PATH**

Trạng thái hợp đồng: **GO_WITH_CONDITIONS**

Kế hoạch này tích hợp quyết nghị của giáo sư/nhóm nghiên cứu. Nó chỉ ủy quyền cho C0-A,
C0-V có giới hạn tùy thuộc vào preflight của nó, và C1-I. Nó không ủy quyền huấn luyện mô hình,
suy luận mới, GPU/Modal, Fold0, nhãn công khai, hay triển khai. Các quy tắc thực thi/vòng đời chung
được kế thừa từ `docs/task1/workflow_b/workflow_B_execution_rules.md`; không cần tài liệu quy tắc thực thi
Workflow C riêng biệt nào.

## Phạm vi khoa học

- Quần thể: chính xác 5.600 truy vấn trong F1–F4.
- Điểm mút chính: Macro Recall dựa trên tập hợp.
- Điểm mút phụ: Macro Precision dựa trên tập hợp.
- Cấu trúc đầu ra: tối đa 5 ID tài liệu duy nhất cho mỗi truy vấn.
- Fold0: bị loại trừ khỏi việc lựa chọn/đánh giá khoa học.
- Nhãn công khai và phản hồi bảng xếp hạng: bị loại trừ khỏi các lựa chọn khoa học.
- Điểm công khai 0.9391: chỉ là bằng chứng triển khai.
- Tập holdout cấp dự án chưa từng chạm tới: không còn.
- Bất kỳ kết quả C3 nào trong tương lai: `EXPLORATORY SCIENTIFIC EVIDENCE`.
- Các so sánh tạo ra dự đoán: danh tính và byte dự đoán phải được đóng băng và băm (hash)
  trước khi các nhãn đánh giá tương ứng được join.

## Sổ đăng ký ủy quyền hiện tại

| Hạng mục | Ủy quyền |
|---|---|
| TV2 C0-A | `AUTHORIZED_AFTER_DOC_FREEZE` |
| TV2 C0-V | `BOUNDED_CONDITIONAL_PROVENANCE_RECONSTRUCTION`; chỉ replay sau khi có preflight dạng văn bản |
| TV4 C1-I | `AUTHORIZED_IN_PARALLEL_AFTER_DOC_FREEZE` |
| C2 | `NOT_AUTHORIZED` |
| C3 | `NOT_AUTHORIZED` |
| C4 | `NOT_AUTHORIZED` |
| C5 | `NOT_AUTHORIZED` |
| C6 / Triển khai công khai | `NOT_AUTHORIZED` |
| Tiếp tục B1 | `NOT_AUTHORIZED` |
| Suy luận Qwen | `NOT_AUTHORIZED` |
| GPU / Modal | `NOT_AUTHORIZED` |
| Fold0 | `NOT_AUTHORIZED` |
| Nhãn công khai | `NOT_AUTHORIZED` |

Vấn đề nguồn gốc bộ chấm điểm CodaBench đang hoạt động chỉ nằm trong phạm vi C6/triển khai
và không chặn C0-A, preflight C0-V, hay C1-I.

## Câu hỏi nghiên cứu và cơ hội đo lường được

Workflow C đặt câu hỏi liệu một chính sách phần dư (residual policy) hoàn chỉnh về nguồn gốc,
phụ thuộc truy vấn có thể khôi phục một phần ổn định của cơ hội hoán đổi đơn (one-swap)
trong khi vẫn từ chối hành động an toàn hay không. Có hai nguồn biên độ cải thiện đo lường được:
lựa chọn hành động phần dư bên trong K20 và độ bao phủ không gian ứng viên/hành động vượt ra ngoài K20.
Tính khả học bên trong K20 của các đặc trưng quan sát được vẫn chưa được giải quyết.

Các mốc Recall đã được xác minh là:

- baseline: 0.9259285714285714;
- V3A lịch sử: 0.9296488095238095;
- oracle one-swap K20: 0.9676994047619048;
- oracle one-swap K77: 0.9847142857142857;
- oracle one-swap toàn tập: 0.9897440476190476.

Ba giá trị oracle là **CÁC MỨC TRẦN CHẨN ĐOÁN CÓ NHẬN BIẾT NHÃN**, không phải là
ước tính tính khả học hay dự báo có thể đạt được. Các khoảng cách xấp xỉ đo được là
V3A→K20 +0.0380505952, K20→K77 +0.0170148810, và K20→toàn tập +0.0220446429.
Chúng không được cộng dồn như các cải thiện dự đoán.

## Ranh giới bằng chứng

Tái dựng V3A lịch sử chính xác là bắt buộc đối với:

- các tuyên bố về hành động được chọn ở cấp độ truy vấn lịch sử;
- các tuyên bố về tài liệu đưa vào sai, tài liệu đưa ra sai, và KEEP so với hoán đổi trong lịch sử;
- một phân loại lỗi chuyên biệt cho V3A lịch sử;
- các mục tiêu huấn luyện phái sinh từ các lựa chọn V3A lịch sử;
- phân tích theo cặp ở cấp độ truy vấn giữa mô hình thách thức và V3A lịch sử;
- bootstrap / ý nghĩa thống kê theo cặp so với V3A lịch sử;
- các chốt chặn hồi quy ở cấp độ dự đoán đòi hỏi các hàng cũ chính xác.

Nó không bắt buộc đối với:

- top5 baseline, tư cách thành viên ứng viên, thứ hạng nguồn, hoặc độ hỗ trợ nguồn;
- các sự thật oracle K20, K77, và toàn tập;
- tổn thất không gian ứng viên và cơ hội oracle rank-4/rank-5;
- phân phối mức tăng oracle và nguồn gốc tài liệu đưa vào mang lại lợi ích;
- nghiên cứu so sánh mới được định nghĩa trên cùng phép chia.

Tái dựng tương đương tổng hợp có thể hỗ trợ đối chuẩn vô hướng tổng hợp và từng fold,
kiểm tra tính hợp lý khi replay, và so sánh tuyệt đối có giới hạn rõ ràng trên cùng quần thể 5.600 truy vấn.
Nó không được hỗ trợ danh tính hàng lịch sử, quy kết lịch sử trên từng truy vấn, huấn luyện phái sinh
từ lựa chọn lịch sử, bootstrap theo cặp với lịch sử, hoặc tuyên bố replay chính xác.

## C0-A — Giải phẫu oracle/cơ hội độc lập với policy lịch sử

**Phụ trách:** TV2

**Ủy quyền:** `AUTHORIZED_AFTER_DOC_FREEZE`

**Tính toán:** chỉ CPU có giới hạn

**Mục đích:** tính toàn vẹn kèm chẩn đoán; không đưa ra tuyên bố về tính khả học

### Đầu vào

- `artifacts/task1/evaluation/strict_cv_v2/folds.json`
- `data/raw/btc/LegalIR/train.json`, chỉ đọc cho chẩn đoán F1–F4
- `artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl`
- `artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl`
- Các báo cáo oracle Workflow-A chuẩn tắc của K20, K77, và toàn tập
- Các artifact chuẩn tắc về ứng viên, danh sách rút gọn, đặc trưng, thứ hạng nguồn, và độ hỗ trợ nguồn
  được liệt kê trong bản đồ bằng chứng

### Kết quả đầu ra được lên kế hoạch

- `reports/task1/workflow_c/tv2/c0a/c0a_integrity_manifest.json`
- `reports/task1/workflow_c/tv2/c0a/c0a_oracle_opportunity_report.json`
- `reports/task1/workflow_c/tv2/c0a/c0a_oracle_opportunity_trace.jsonl`

Các đường dẫn này được lên kế hoạch; nhiệm vụ tài liệu hóa này không tạo ra chúng.

### Phân tích bắt buộc

C0-A tái lập baseline F1–F4, các không gian K20, K77, và one-swap toàn tập; số đếm cơ hội;
phân phối mức tăng từng truy vấn/từng fold; tổn thất không gian ứng viên; cơ hội rank-4/rank-5;
nguồn gốc tài liệu đưa vào mang lại lợi ích; và tính toàn vẹn của truy vấn/tài liệu/hành động/fold/schema/join.
Nó không được tuyên bố rằng V3A lịch sử đã chọn một hành động sai cụ thể cho bất kỳ truy vấn nào.

### Cổng toàn vẹn

Đạt yêu cầu (PASS) đòi hỏi:

- chính xác 5.600 truy vấn F1–F4;
- 0 sai lệch danh tính truy vấn;
- 0 sai lệch danh tính tài liệu;
- 0 sai lệch danh tính hành động;
- 0 sai lệch fold;
- 0 sai lệch schema;
- 0 sai lệch phép join bắt buộc;
- các mã hash chuẩn tắc liên quan được xác minh;
- baseline Recall 0.9259285714285714 trong sai số `1e-12`;
- trần K20 0.9676994047619048 trong sai số `1e-12`;
- trần K77 0.9847142857142857 trong sai số `1e-12`;
- trần one-swap toàn tập 0.9897440476190476 trong sai số `1e-12`;
- các tổng/phân rã từng fold được đối soát khớp với các giá trị gộp.

Không có thêm cổng tiếp tục dựa trên độ lớn oracle. Cụ thể, C0-A không sử dụng
ngưỡng biên độ gộp hoặc ngưỡng biên độ số lượng fold.

### Cô lập nhánh

Kết quả K20, K77, và toàn tập nhận các trạng thái toàn vẹn riêng biệt. Thất bại chỉ ở K77
sẽ khép lại K77 nhưng không làm vô hiệu hóa lõi K20 sạch. Thất bại nguồn gốc chỉ ở toàn tập
sẽ khép lại hoặc giới hạn kết luận toàn tập mà không làm ô nhiễm các sự thật sạch của K20/K77.

### Quy tắc dừng

Dừng lại và khoanh vùng cục bộ bất kỳ sai lệch nào đối với không gian bị ảnh hưởng.
Không tinh chỉnh bằng chứng cho đến khi chúng khớp nhau. Trạng thái K20 sạch có thể vẫn sử dụng được
ngay cả khi K77 hoặc toàn tập bị khép lại.

## C0-V — Tái dựng chuyên biệt cho V3A có giới hạn

**Phụ trách:** TV2

**Ủy quyền:** `BOUNDED_CONDITIONAL_PROVENANCE_RECONSTRUCTION`

**Tính toán:** chỉ replay CPU tất định có giới hạn sau preflight

C0-V cố gắng tái dựng các hành động được chọn/top5 OOF V3A F1–F4 lịch sử.
Đây là công việc xác minh nguồn gốc có giới hạn, không phải việc khảo cổ mã nguồn vô hạn.

### Preflight bắt buộc trước khi replay

Preflight dạng văn bản/chỉ đọc phải ghi nhận:

- đường dẫn và mã hash đầu vào ứng viên chính xác;
- các giá trị cấu hình và manifest lịch sử;
- danh tính gói/thư viện phụ thuộc có thể khôi phục;
- seed tất định;
- phạm vi rõ ràng chỉ gồm F1–F4;
- bằng chứng không kích hoạt giai đoạn Fold0 nào;
- các bất biến kỳ vọng về tổng hợp, từng fold, số lượng hành động, danh tính, và phép join.

Nếu danh tính mã nguồn lịch sử chính xác không thể khôi phục trong nỗ lực có giới hạn,
hãy giữ lại giới hạn nguồn gốc đó. Không bịa đặt hoặc liên tục tìm kiếm mã nguồn cũ vô hạn,
và không tự động chặn C0-A hoặc C1-I.

### Kết quả đầu ra được lên kế hoạch

- `reports/task1/workflow_c/tv2/c0v/c0v_preflight.json`
- `reports/task1/workflow_c/tv2/c0v/c0v_reconstruction_manifest.json`
- `reports/task1/workflow_c/tv2/c0v/c0v_selected_actions.jsonl`
- `reports/task1/workflow_c/tv2/c0v/c0v_reconstruction_report.json`

Các đường dẫn này được lên kế hoạch; nhiệm vụ tài liệu hóa này không tạo ra chúng.

### Các phân loại được phép

- `EXACT_SOURCE_REPLAY`: danh tính mã nguồn lịch sử, đầu vào, dependency, cấu hình, seed,
  và hành vi tất định được chứng minh và tất cả các bất biến đã đăng ký trước đều khớp.
- `AGGREGATE_EQUIVALENT_RECONSTRUCTION`: danh tính mã nguồn lịch sử không được chứng minh,
  nhưng mọi bất biến đã đăng ký trước về tổng hợp/từng fold/số lượng hành động/phép join đều tái lập được.
  Đây là một đối chứng được tái dựng mới, không phải V3A lịch sử chính xác ở cấp độ truy vấn.
- `REPLAY_MISMATCH_BLOCKED`: bất kỳ bất biến liên quan nào không được giải thích có sự khác biệt.

Phân loại này chỉ chi phối các tuyên bố cần dòng dõi hàng chuyên biệt của V3A.
Nó không làm thay đổi các sự thật C0-A sạch độc lập với policy lịch sử.

## C1-I — Kiểm toán định danh/không gian hành động/nguồn gốc xuất xứ

**Phụ trách:** TV4

**Ủy quyền:** `AUTHORIZED_IN_PARALLEL_AFTER_DOC_FREEZE`

**Phụ thuộc vào C0-V:** không có đối với phần kiểm toán cấu trúc

### Pha cấu trúc

Pha này không sử dụng nhãn. Nó kiểm toán:

- ID truy vấn, tài liệu, và hành động;
- tư cách thành viên K20 và K77;
- tư cách thành viên ứng viên/hành động toàn tập khi có chuẩn tắc;
- thứ hạng nguồn và độ hỗ trợ nguồn;
- tính tương thích của schema đặc trưng;
- độ bao phủ và các mã hash;
- danh tính không gian chuẩn tắc.

Sau khi các danh tính cấu trúc được đóng băng, C1-I chỉ có thể join nhãn F1–F4 để xác minh
điểm mút oracle độc lập với policy lịch sử. Các dự đoán/danh tính điểm mút phải được đóng băng trước.

C1-I loại trừ phân tích hành động được chọn của V3A lịch sử, quy kết hành động sai so với V3A,
tính bổ trợ của Qwen, và mô hình hóa thách thức. Phân tích trong tương lai đòi hỏi các lựa chọn được tái dựng
phải phụ thuộc một cách tường minh vào dòng dõi C0-V được chấp thuận.

### Kết quả đầu ra được lên kế hoạch

- `reports/task1/workflow_c/tv4/c1i/c1i_structural_identity_report.json`
- `reports/task1/workflow_c/tv4/c1i/c1i_action_universe_report.json`
- `reports/task1/workflow_c/tv4/c1i/c1i_feature_schema_report.json`

Các đường dẫn này được lên kế hoạch; nhiệm vụ tài liệu hóa này không tạo ra chúng.

### Cổng và cô lập nhánh

Mỗi khối không gian/schema/nguồn nhận trạng thái PASS/LIMITED/FAIL riêng.
Sự sai lệch ở K77 hoặc toàn tập không chặn bằng chứng K20 C0-A sạch. C1-I không lựa chọn
mô hình, ngưỡng, hay khối đặc trưng cho C2.

## Chính sách K20 và K77

K20 là `CORE`: nó có schema hành động 58 đặc trưng đã được hiện thực hóa, số lượng hành động ít hơn,
dòng dõi mô hình đương nhiệm mạnh hơn, và là phép thử thất bại ít tốn kém hơn.

K77 hiện được kiểm toán trong C1-I nhưng không được mô hình hóa. Việc mô hình hóa K77 chỉ có thể mở ra
khi thỏa mãn cả ba điều kiện:

1. Công thức chính K20 vượt qua C2;
2. Không gian/schema/nguồn gốc K77 đạt yêu cầu;
3. Có hợp đồng riêng ủy quyền cho mô hình thách thức K77.

Sự thất bại trong việc học của K20 không ủy quyền cho K77.

## Hợp đồng KEEP tương lai cố định

Hợp đồng KEEP là `ZERO_REFERENCE_SCORE`:

- `actions.jsonl` tiếp tục chỉ chứa các hành động hoán đổi;
- không có hàng huấn luyện KEEP nào được tổng hợp;
- không có vector đặc trưng KEEP nào được tạo dựng;
- độ hữu dụng của KEEP chính xác bằng 0;
- chỉ huấn luyện trên độ hữu dụng của các hành động hoán đổi;
- mỗi truy vấn đóng góp tổng trọng số huấn luyện hoán đổi chính xác là 1.0, chia đều trên các hàng hoán đổi hợp lệ của truy vấn đó;
- không được phép đánh lại trọng số theo lớp BENEFIT/HARM/NEUTRAL làm méo mó độ hữu dụng kỳ vọng;
- khi suy luận, xác định độ hữu dụng hoán đổi dự đoán tốt nhất;
- chỉ thực thi nó khi nó vượt qua một cách nghiêm ngặt biên độ được chọn từ OOF nội bộ;
- nếu không thì chọn KEEP; trường hợp hòa điểm chính xác sẽ chọn KEEP.

KEEP là một lựa chọn từ chối/tham chiếu, không phải là một hành động tổng hợp được học.

## C2 — Công thức tương lai cố định và đối chứng trên cùng phép chia

**Ủy quyền:** `NOT_AUTHORIZED`

### Mô hình thách thức chính C2-R

Công thức chính là `SWAP_ONLY_EXACT_DELTA_RECALL_REGRESSION`:

- mục tiêu: delta Recall chính xác trên từng truy vấn cho mỗi hoán đổi hợp lệ;
- một cấu hình hồi quy nông đóng băng;
- hàm mục tiêu squared-error;
- đánh trọng số cân bằng truy vấn với tổng trọng số 1.0/truy vấn;
- không đánh lại trọng số theo lớp kết quả;
- không quét diện rộng siêu tham số, pairwise, listwise, hay LambdaMART;
- áp dụng quy tắc KEEP `ZERO_REFERENCE_SCORE` ở trên.

Delta Recall chính xác nhận các giá trị rời rạc được xác định bởi lực lượng nhãn chuẩn của truy vấn.
Mô hình là một đại lượng thay thế squared-error; macro Recall đầu-cuối, chứ không phải riêng MSE/AUC/độ chính xác hành động,
sẽ chi phối việc tiếp tục.

### Đối chứng bắt buộc C2-V

Nghiên cứu khoa học theo cặp C2/C3 đòi hỏi một đối chứng 3 lớp phong cách V3A cấu hình cố định mới
trong cùng quy trình chia tách nội bộ/ngoài và đóng băng dự đoán như C2-R. Đối chứng mới này không phải là V3A
lịch sử chỉ vì các chỉ số của nó trùng khớp.

Recall gộp 0.9296488095238095 của V3A lịch sử và các chỉ số fold được bảo tồn là các chốt chặn hồi quy vô hướng/fold
bất biến. Nếu đối chứng mới khác biệt với một chốt chặn áp dụng lớn hơn `1e-12`, cách diễn đạt tương đương lịch sử
bị chặn và việc rà soát dòng dõi bắt đầu. Nó chỉ có thể tiếp tục dưới một tên đối chứng rõ ràng mới nếu ban quản trị/giáo sư
cho phép; nó không bao giờ được dán nhãn lại thành V3A lịch sử.

### Đối chứng tùy chọn C2-B1

B1 ở trạng thái `FROZEN_OPTIONAL_C2_COMPARATOR`, đối chứng học được tùy chọn duy nhất sau này.
Job Workflow B không được tiếp tục. Việc tham gia C2-B1 trong tương lai đòi hỏi ủy quyền riêng,
tái hiện thực hóa ma trận/hành động chính xác, tái lập các ID/đặc trưng/mã hash, một lần chạy khói thời gian chạy
1 fold thành công có ghi nhận mã thoát tiến trình con, các giới hạn CPU/bộ nhớ và số lần fit được đăng ký trước,
và cùng quy trình đánh giá lồng nhau. Không cho phép quét LambdaMART/listwise chung chung.

### Cổng OOF nội bộ

Đối với mọi phần bù huấn luyện ngoài, C2-R so với đối chứng mới trên cùng phép chia phải thỏa mãn:

- delta Recall OOF nội bộ gộp ít nhất +0.002;
- delta Recall của cả 3 fold thẩm định nội bộ đều không âm;
- delta Precision gộp ít nhất -0.001;
- tính toàn vẹn dự đoán/nguồn gốc xuất xứ PASS.

Toàn bộ 4 bundle lựa chọn ngoài phải được đóng băng trước bất kỳ phép join nhãn ngoài C3 nào trong tương lai.
Nếu bất kỳ phần bù nào không có mô hình thách thức đạt chuẩn, hãy dừng lại trước C3.
Không chọn một công thức toàn cục bằng cách sử dụng các kết quả F1–F4 ngoài gộp lại.

## C3 — Đánh giá khoa học ngoài tương lai

**Ủy quyền:** `NOT_AUTHORIZED`

Đối với mỗi fold ngoài, quy trình C2 đã đóng băng huấn luyện/lựa chọn chỉ sử dụng 3 fold còn lại,
chấm điểm fold được giữ lại mà không nạp nhãn của nó, ghi và xác thực cấu trúc các dự đoán của đối chứng mới
và mô hình thách thức, rồi đóng băng danh tính/byte/mã hash. Chỉ sau đó các nhãn ngoài mới được phép join.
Việc gộp (pooling) chỉ diễn ra sau khi toàn bộ 4 bundle ngoài đã được đóng băng.

Đối chiếu với đối chứng mới trên cùng phép chia đã đóng băng dự đoán, cổng C3 trong tương lai là:

- delta Recall theo cặp gộp ít nhất +0.003;
- delta Recall của cả 4 fold đều không âm;
- delta Precision gộp ít nhất -0.001;
- phân vị thứ 10 của bootstrap truy vấn theo cặp về delta Recall lớn hơn 0;
- Recall gộp tuyệt đối ít nhất 0.9326488095238095;
- không có fold nào thấp hơn giá trị vô hướng V3A lịch sử bất biến cho fold đó;
- tính toàn vẹn và nguồn gốc xuất xứ đầy đủ đạt PASS.

Kết quả phải được dán nhãn `EXPLORATORY SCIENTIFIC EVIDENCE` vì không còn tập holdout cấp dự án độc lập nào chưa từng chạm tới.

Phương pháp bootstrap, seed, số lần lấy mẫu lại, và đầu vào theo cặp từng truy vấn là:

`C3_BOOTSTRAP_PROTOCOL_PENDING_PRE_C2_FREEZE`

Không tìm thấy giao thức bootstrap ở cấp độ truy vấn chuẩn tắc tương thích nào cho Task1 trong các quy ước Workflow-A đã kiểm tra.
Giáo sư/nhóm phải ấn định giao thức trước khi đóng băng lựa chọn C2 và trước khi tạo dự đoán C3.
Hợp đồng đang chờ xử lý này không chặn C0-A hay C1-I.

## Các giai đoạn sau

- **C4 — Độ ổn định / độ bền vững:** `NOT_AUTHORIZED`; chỉ kiểm tra được đăng ký trước trên các dự đoán C3 đã đóng băng,
  không bao giờ lựa chọn mô hình chiến thắng sau sự thật (post-hoc).
- **C5 — Fit cuối cùng:** `NOT_AUTHORIZED`; một cấu hình đã đóng băng được bình duyệt, chỉ trên F1–F4,
  không có tuyên bố hiệu năng không thiên lệch trên tập huấn luyện.
- **C6 — Triển khai:** `NOT_AUTHORIZED`; chỉ suy luận công khai không nhãn sau khi có ủy quyền riêng
  và giải quyết nguồn gốc bộ chấm điểm đang hoạt động. Không phản hồi từ bảng xếp hạng nào được phép chọn hoặc tinh chỉnh hệ thống.

## Chính sách Qwen

Mô hình Qwen độc lập sau khi sửa lỗi bị bác bỏ về mặt khoa học. Không ủy quyền chạy lại toàn bộ,
suy luận Qwen mới, hay sử dụng GPU. Điểm tài liệu hiện có chỉ có thể quay lại thông qua một nhánh bổ trợ
điểm đóng băng được đặc tả và ủy quyền riêng sau này. Tính bổ trợ của Qwen không phải là một phần của C1-I.

## Lộ trình quyết định

~~~text
ĐÓNG BĂNG TÀI LIỆU (DOC FREEZE)
  |-- TV2 C0-A toàn vẹn/giải phẫu ------------------------+
  |      |-- Trạng thái K20 (lõi - core)                  |
  |      |-- Trạng thái K77 (chỉ kiểm toán)               |
  |      `-- Trạng thái toàn tập (chẩn đoán)              |
  |                                                       |
  |-- TV2 C0-V preflight văn bản -> nỗ lực có giới hạn    | độc lập
  |      `-- chính xác / tương đương tổng hợp / sai lệch  |
  |                                                       |
  `-- TV4 C1-I đóng băng cấu trúc -> xác minh điểm mút----+

Không có điều nào ở trên tự động ủy quyền C2.
Ủy quyền riêng trong tương lai -> C2 trên toàn bộ 4 phần bù
  |-- bất kỳ phần bù nào thất bại -> DỪNG lại trước C3
  `-- tất cả đạt và tất cả bundle đều đóng băng -> C3 có thể được xem xét riêng
        |-- cổng C3 tương lai thất bại -> giữ nguyên mô hình đương nhiệm
        `-- cổng đạt -> chỉ là bằng chứng khám phá; C4 tiếp tục có cổng riêng
~~~

## Kỷ luật dừng

- Bất kỳ sai lệch C0-A nào đều được khoanh vùng cục bộ cho K20, K77, hoặc toàn tập tương ứng.
- Sai lệch C0-V chặn các tuyên bố hàng lịch sử chuyên biệt của V3A, không chặn C0-A/C1-I.
- Thất bại trong việc khôi phục mã nguồn chính xác không bao giờ cho phép bịa đặt mã nguồn lịch sử.
- Việc học thất bại của K20 không ủy quyền cho K77, một mô hình lớn hơn, hay GPU.
- Kiểm toán K77 thất bại sẽ khép lại K77 mà không làm vô hiệu hóa công việc K20 sạch.
- Ma trận B1 bị thiếu sẽ khép lại C2-B1; chúng không ủy quyền cho việc khôi phục Workflow B.
- Tài liệu Qwen bị thiếu không ủy quyền cho suy luận mới.
- Bất kỳ thất bại phần bù C2 nào trong tương lai đều phải dừng lại trước C3.
- Bất kỳ thất bại C3/C4 nào trong tương lai đều giữ nguyên mô hình đương nhiệm và nghiêm cấm tinh chỉnh sau sự thật.
- Fold0, nhãn công khai, và phản hồi bảng xếp hạng không bao giờ tạo thành ngoại lệ.

## Công việc khả thi đầu tiên sau khi đóng băng tài liệu

Chỉ những điều sau đây là việc tiếp theo:

1. TV2 chạy C0-A.
2. TV2 có thể bắt đầu công việc C0-V có giới hạn chỉ bằng cách viết preflight bắt buộc,
   sau đó chạy replay CPU có giới hạn được mô tả ở đó.
3. TV4 chạy C1-I song song.

Huấn luyện C2, đánh giá C3, Qwen, B1, GPU/Modal, Fold0, nhãn công khai, và triển khai
rõ ràng không phải là bước tiếp theo và vẫn chưa được ủy quyền.
