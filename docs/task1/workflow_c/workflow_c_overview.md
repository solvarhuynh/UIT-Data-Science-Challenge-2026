# Task1 Workflow C — Kiểm toán Hệ thống và Tổng quan Nghiên cứu

Trạng thái hợp đồng: **GO_WITH_CONDITIONS**

Định hướng nghiên cứu: **WORKFLOW_C_RESIDUAL_RECOVERY_IS_BEST_NEXT_PATH**

## Cập nhật C2-PREFLIGHT cuối cùng — 2026-09-12

`C0-A` và `C1-I` là `COMPLETE_PASS`, lineage C1-I là `C1I_COMMIT_PINNED`,
còn C0-V `DEFERRED`. Governance đã giải quyết toàn bộ lựa chọn C2 và lần
preflight cuối đạt `PASS` với ủy quyền chỉ là `PREFLIGHT_ONLY`; C2 training,
scoring, inference và C3 vẫn `NOT_AUTHORIZED`. Môi trường hiện hữu `.venv` đã
được pin và constructor-smoke thành công, không fit, tại Python 3.12.6,
NumPy 1.26.4, scikit-learn 1.7.2 và threadpoolctl 3.6.0.

C2-V được đóng băng dưới danh tính mới `C2V_FRESH_HGBC_1_7_2_K20`, là
`HistGradientBoostingClassifier` ba lớp với utility `P(B)-1.5P(H)`, seed 2026
và margin cố định 0; nó không phải historical V3A. C2-R là
`HistGradientBoostingRegressor` squared-error cho exact delta Recall, seed
2027, với margin inner-OOF tối đa `{0,Q75,Q90,Q95}`. Bootstrap C3 tương lai
đã đóng băng ở 10.000 paired-query replicates, PCG64 seed 20260911. CPU là
1 thread/fit, 1 fit đồng thời, tối đa 32 fits, RSS 8 GiB/available 4 GiB,
timeout 15 phút/fit và 8 giờ tổng. TV2 là future execution owner nếu được cấp
quyền riêng; TV4 là independent reviewer. K20 `CORE`; K77/full-pool deferred;
Qwen `FROZEN_SIGNAL_ONLY`; B1 `FROZEN_OPTIONAL_COMPARATOR`.

Đây là điểm truy cập chuẩn tắc cho Workflow C sau khi được giáo sư/nhóm bình duyệt.
Đây là một hợp đồng tài liệu và ủy quyền, không phải là sự ủy quyền cho bất kỳ thực nghiệm nào
vượt quá các giai đoạn được đánh dấu rõ ràng là đã được ủy quyền dưới đây. Các quy tắc thực thi
và vòng đời chung vẫn được kế thừa từ `docs/task1/workflow_b/workflow_B_execution_rules.md`.

## Phân quyền hôm nay

| Nhánh hoặc tài nguyên | Trạng thái hiện tại |
|---|---|
| TV2 C0-A | `AUTHORIZED_AFTER_DOC_FREEZE` |
| TV2 C0-V | `BOUNDED_CONDITIONAL_PROVENANCE_RECONSTRUCTION`; replay CPU tất định yêu cầu preflight dạng văn bản trước tiên |
| TV4 C1-I | `AUTHORIZED_IN_PARALLEL_AFTER_DOC_FREEZE` |
| C2-PREFLIGHT contract/provenance | `PREFLIGHT_ONLY`; final governance rerun `PASS` |
| C2 training/scoring/inference, C3, C4, C5 | `NOT_AUTHORIZED` |
| C6 / Triển khai công khai | `NOT_AUTHORIZED` |
| Tiếp tục Workflow B B1 | `NOT_AUTHORIZED` |
| Suy luận Qwen mới | `NOT_AUTHORIZED` |
| GPU hoặc Modal | `NOT_AUTHORIZED` |
| Fold0 cho lựa chọn/đánh giá khoa học của Workflow C | `NO` |
| Nhãn công khai hoặc phản hồi từ bảng xếp hạng cho việc lựa chọn | `NO` |

Nguồn gốc bộ chấm điểm CodaBench đang hoạt động chưa được giải quyết chỉ chặn C6/triển khai.
Nó không chặn C0-A hay C1-I.

## Hợp đồng khoa học chuẩn tắc

- Quần thể khoa học: chính xác 5.600 truy vấn thẩm định trong F1–F4.
- Chỉ số chính: Macro Recall dựa trên tập hợp.
- Chỉ số phụ: Macro Precision dựa trên tập hợp.
- Ràng buộc đầu ra: không quá 5 ID tài liệu duy nhất cho mỗi truy vấn.
- Fold0 bị loại trừ khỏi việc lựa chọn và đánh giá khoa học trong Workflow C.
- Nhãn công khai và phản hồi từ bảng xếp hạng bị loại trừ khỏi các lựa chọn khoa học.
- Điểm công khai quan sát được 0.9391 chỉ là bằng chứng triển khai, không phải là đối chứng
  khoa học sạch hay mục tiêu.
- Không còn tập kiểm tra giữ lại độc lập (holdout) cấp dự án nào chưa từng chạm tới. Kết quả C3
  trong tương lai phải được dán nhãn `EXPLORATORY SCIENTIFIC EVIDENCE`.
- Mọi đánh giá tạo ra dự đoán trong tương lai đều phải đóng băng byte dự đoán, danh tính,
  và mã hash trước khi join các nhãn tương ứng.

## Các mốc số liệu đã được xác minh

| Đại lượng | Giá trị F1–F4 | Diễn giải theo hợp đồng |
|---|---:|---|
| Recall baseline phần dư V3 khoa học | 0.9259285714285714 | Baseline độc lập với policy lịch sử |
| Recall V3A lịch sử | 0.9296488095238095 | Chốt chặn hồi quy vô hướng gộp bất biến |
| Oracle one-swap K20 | 0.9676994047619048 | Trần chẩn đoán có nhận biết nhãn |
| Oracle one-swap K77 | 0.9847142857142857 | Trần chẩn đoán có nhận biết nhãn |
| Oracle one-swap toàn tập (full-pool) | 0.9897440476190476 | Trần chẩn đoán có nhận biết nhãn |
| Mô hình đương nhiệm triển khai công khai quan sát được | 0.9391 | Chỉ là bằng chứng triển khai |

Các giá trị oracle là **CÁC MỨC TRẦN CHẨN ĐOÁN CÓ NHẬN BIẾT NHÃN**, không bao giờ là
dự báo có thể đạt được. Có hai nguồn biên độ cải thiện đo lường được: lựa chọn hành động
phần dư bên trong K20 và độ bao phủ không gian ứng viên/hành động vượt ra ngoài K20.
Tính khả học bên trong K20 của các đặc trưng quan sát được vẫn chưa được giải quyết.

- Khoảng cách từ V3A lịch sử tới oracle K20: xấp xỉ +0.0380505952.
- Mức tăng từ K20 tới oracle K77: xấp xỉ +0.0170148810.
- Mức tăng từ K20 tới one-swap toàn tập: xấp xỉ +0.0220446429.

Những chênh lệch đo được này không phải là các mức tăng cộng dồn có thể đạt được.

## C0 gồm hai lộ trình riêng biệt

### C0-A — Giải phẫu oracle/cơ hội độc lập với policy lịch sử

C0-A là giai đoạn chẩn đoán và tính toàn vẹn. Nó không yêu cầu các lựa chọn lịch sử của V3A
trên từng truy vấn và không đưa ra tuyên bố về tính khả học. TV2 có thể chạy nó sau khi
hợp đồng 6 tài liệu này được đóng băng.

Nó tái lập và mô tả đặc trưng baseline F1–F4, các không gian K20, K77, và one-swap toàn tập;
số lượng cơ hội và phân phối mức tăng; tổn thất không gian ứng viên; cơ hội rank-4/rank-5;
nguồn gốc xuất xứ của tài liệu đưa vào mang lại lợi ích; và tính toàn vẹn định danh/phép join.

C0-A chỉ đạt (PASS) khi đáp ứng tất cả các điều kiện sau:

- chính xác 5.600 truy vấn F1–F4;
- 0 sai lệch về truy vấn, tài liệu, hành động, fold, schema, và các phép join bắt buộc;
- tất cả các mã hash chuẩn tắc liên quan đều được xác minh;
- tái lập Recall baseline 0.9259285714285714 trong sai số `1e-12`;
- tái lập trần K20 0.9676994047619048 trong sai số `1e-12`;
- tái lập trần K77 0.9847142857142857 trong sai số `1e-12`;
- tái lập trần one-swap toàn tập 0.9897440476190476 trong sai số `1e-12`;
- các tổng và phân rã theo từng fold khớp hoàn toàn với các giá trị gộp.

Tính toàn vẹn của K20, K77, và toàn tập nhận các trạng thái riêng biệt. Lỗi chỉ xảy ra ở K77
sẽ khép lại K77 mà không làm vô hiệu hóa lõi K20 sạch; tương tự, lỗi chỉ ở toàn tập sẽ được
khoanh vùng cục bộ. Không có ngưỡng tiếp tục dựa trên độ lớn oracle riêng biệt nào trong C0-A.

### C0-V — Tái dựng chuyên biệt cho V3A

C0-V là một nhiệm vụ tái dựng/nguồn gốc xuất xứ có giới hạn, không phải việc khảo cổ mã nguồn vô hạn.
Nó cố gắng tái dựng tất định các hành động được chọn/top5 OOF V3A lịch sử trên F1–F4. Trước bất kỳ
lần replay CPU nào, TV2 phải viết một bản preflight chỉ đọc ghi lại các đường dẫn và mã hash của dữ liệu
đầu vào ứng viên, các giá trị cấu hình/manifest lịch sử, danh tính các gói thư viện có thể khôi phục,
seed tất định, phạm vi chỉ gồm F1–F4, bằng chứng không kích hoạt giai đoạn Fold0 nào, và các bất biến kỳ vọng.

Các phân loại duy nhất được phép là:

- `EXACT_SOURCE_REPLAY`: danh tính mã nguồn, đầu vào, dependency, cấu hình, seed, hành vi tất định,
  và tất cả các bất biến đều được chứng minh.
- `AGGREGATE_EQUIVALENT_RECONSTRUCTION`: danh tính mã nguồn chưa được chứng minh nhưng các bất biến
  đã đăng ký trước về tổng hợp, từng fold, số lượng hành động, và phép join đều tái lập được.
  Đây là một đối chứng được tái dựng mới, không phải V3A lịch sử chính xác ở cấp độ truy vấn.
- `REPLAY_MISMATCH_BLOCKED`: một bất biến liên quan bị sai lệch mà không có lời giải thích.

Nếu danh tính mã nguồn chính xác không được khôi phục trong nỗ lực có giới hạn, hãy giữ lại
giới hạn đó và không bịa đặt mã nguồn. Kết quả đó không tự động chặn C0-A hoặc C1-I.

## Ranh giới bằng chứng C0-V chính xác

Tái dựng chính xác là bắt buộc đối với các tuyên bố về hành động được chọn, tài liệu đưa vào sai,
tài liệu đưa ra sai, và KEEP so với hoán đổi ở cấp độ truy vấn lịch sử; phân loại lỗi chuyên biệt
cho V3A lịch sử; các mục tiêu huấn luyện phái sinh từ các lựa chọn lịch sử; phân tích hoặc bootstrap
theo cặp giữa mô hình thách thức và V3A lịch sử; và các chốt chặn hồi quy ở cấp độ hàng đòi hỏi các hàng cũ chính xác.

Nó không bắt buộc đối với top5 baseline, tư cách thành viên ứng viên, thứ hạng/độ hỗ trợ nguồn,
các sự thật oracle K20/K77/toàn tập, tổn thất không gian ứng viên, cơ hội oracle rank-4/rank-5,
phân phối mức tăng oracle, nguồn gốc tài liệu đưa vào mang lại lợi ích, hoặc nghiên cứu so sánh mới
được định nghĩa trên cùng phép chia.

Tái dựng tương đương tổng hợp chỉ hỗ trợ đối chuẩn vô hướng tổng hợp/từng fold, kiểm tra tính hợp lý khi replay,
và so sánh tuyệt đối có giới hạn rõ ràng trên cùng quần thể 5.600 truy vấn. Nó không hỗ trợ danh tính hàng lịch sử,
quy kết ở cấp độ truy vấn lịch sử, huấn luyện từ các quyết định được cho là của lịch sử, hoặc bootstrap theo cặp với lịch sử.

## C1-I — Kiểm toán định danh/không gian hành động/nguồn gốc xuất xứ

TV4 có thể chạy C1-I song song với C0-A sau khi đóng băng tài liệu. Pha cấu trúc của nó không dùng nhãn
và kiểm tra ID truy vấn, ID tài liệu, ID hành động, tư cách thành viên chuẩn tắc K20/K77/toàn tập,
thứ hạng/độ hỗ trợ nguồn, mã hash, độ bao phủ, tính tương thích của schema đặc trưng, và danh tính không gian.
Sau khi các danh tính đó được đóng băng, nhãn F1–F4 chỉ có thể được join để xác minh điểm mút oracle
độc lập với policy lịch sử.

C1-I không phụ thuộc vào C0-V. Nó loại trừ việc quy kết hành động được chọn của V3A lịch sử,
mô hình hóa thách thức, và tính bổ trợ của Qwen. Bất kỳ phân tích nào sau này thực sự phụ thuộc
vào các lựa chọn lịch sử được tái dựng đều phải khai báo và sử dụng dòng dõi C0-V được chấp thuận.

## Hợp đồng C2 tương lai được đóng băng

C2 chưa được ủy quyền. Hợp đồng KEEP của nó là `ZERO_REFERENCE_SCORE`, và công thức chính là
`SWAP_ONLY_EXACT_DELTA_RECALL_REGRESSION`:

- `actions.jsonl` tiếp tục chỉ chứa các hành động hoán đổi; không có hàng hay vector đặc trưng KEEP nào được tổng hợp;
- KEEP là mốc tham chiếu độ hữu dụng bằng 0 chính xác và là lựa chọn từ chối hành động;
- huấn luyện một cấu hình hồi quy nông đóng băng với hàm mất mát squared error trên delta Recall chính xác từng truy vấn cho các hoán đổi hợp lệ;
- đánh trọng số các hàng sao cho mỗi truy vấn đóng góp tổng trọng số huấn luyện hoán đổi chính xác là 1.0; không đánh lại trọng số cho các lớp kết quả BENEFIT/HARM/NEUTRAL;
- không thực hiện quét diện rộng siêu tham số, pairwise, listwise, hay LambdaMART;
- chỉ thực thi hoán đổi dự đoán tốt nhất khi độ hữu dụng dự đoán của nó vượt qua một cách nghiêm ngặt biên độ được chọn từ OOF nội bộ; nếu không thì chọn KEEP, trường hợp hòa điểm chính xác sẽ chọn KEEP;
- đánh giá việc tiếp tục bằng macro Recall đầu-cuối, với Precision là chốt chặn rủi ro—không chỉ dựa vào riêng MSE, AUC, hay độ chính xác hành động.

Delta Recall chính xác mang tính rời rạc tùy thuộc vào lực lượng tập nhãn chuẩn (gold-cardinality) của truy vấn;
công thức không được thổi phồng độ chính xác của hồi quy liên tục.

Bắt buộc phải có một đối chứng 3 lớp phong cách V3A cấu hình cố định mới dưới cùng quy trình chia tách nội bộ/ngoài
và đóng băng dự đoán cho nghiên cứu khoa học theo cặp C2/C3. Nó không phải là V3A lịch sử. Giá trị gộp 0.9296488095238095
của V3A lịch sử và các chỉ số fold được bảo tồn của nó vẫn là các chốt chặn hồi quy vô hướng/fold bất biến.
Nếu đối chứng mới khác biệt với các chốt chặn đó lớn hơn `1e-12`, tuyên bố tương đương lịch sử bị chặn
và việc rà soát dòng dõi bắt đầu. Đối chứng chỉ có thể tiếp tục dưới một tên đối chứng rõ ràng mới nếu ban quản trị cho phép.

Đối với mọi phần bù huấn luyện ngoài, mô hình thách thức phải đánh bại đối chứng mới ít nhất +0.002
Recall OOF nội bộ gộp, không âm trên cả 3 fold thẩm định nội bộ, có delta Precision gộp ít nhất -0.001,
và vượt qua tính toàn vẹn dự đoán/nguồn gốc xuất xứ. Toàn bộ 4 bundle lựa chọn ngoài phải được đóng băng
trước bất kỳ phép join nhãn ngoài C3 nào. Thất bại ở bất kỳ phần bù nào đều phải dừng lại trước C3;
không có công thức toàn cục nào được chọn từ các kết quả F1–F4 ngoài gộp lại.

## Cổng C3 tương lai được đóng băng

C3 chưa được ủy quyền. Đối chiếu với đối chứng mới trên cùng phép chia đã được đóng băng dự đoán,
việc thăng cấp trong tương lai đòi hỏi tất cả các điều kiện sau:

- delta Recall theo cặp gộp ít nhất +0.003;
- delta Recall của cả 4 fold ngoài đều không âm;
- delta Precision gộp ít nhất -0.001;
- phân vị thứ 10 của bootstrap truy vấn theo cặp về delta Recall lớn hơn 0;
- Recall gộp tuyệt đối ít nhất 0.9326488095238095;
- không có fold nào thấp hơn giá trị vô hướng fold V3A lịch sử bất biến của nó;
- tính toàn vẹn và nguồn gốc xuất xứ đầy đủ đạt PASS.

Bootstrap C3 tương lai đã frozen trước scoring: một paired outer-OOF query
delta C2-R trừ fresh C2-V là resampling unit; 10.000 replicate có hoàn lại,
mỗi replicate 5.600 query, PCG64 seed 20260911, statistic là arithmetic mean,
và p10 dùng `numpy.quantile(..., 0.10, method="linear")`. Không tính bootstrap
trong preflight này.

## Chính sách nhánh

- **K20:** `CORE`. Nó có schema hành động 58 đặc trưng đã được hiện thực hóa, số lượng hành động ít hơn,
  dòng dõi mô hình đương nhiệm mạnh mẽ hơn, và là phép thử thất bại ít tốn kém hơn.
- **K77:** kiểm toán ngay bây giờ trong C1-I; không xây dựng mô hình. Việc xây dựng mô hình chỉ có thể mở ra
  sau khi K20 vượt qua C2, không gian/schema/nguồn gốc K77 đạt yêu cầu, và có hợp đồng riêng ủy quyền.
  Thất bại trong việc học của K20 không ủy quyền cho K77.
- **B1:** `FROZEN_OPTIONAL_C2_COMPARATOR`. Không bao giờ tiếp tục job Workflow B. Việc tham gia trong tương lai
  yêu cầu ủy quyền riêng, tái hiện thực hóa và mã hash chính xác, chạy khói thời gian chạy trên 1 fold
  với mã thoát tiến trình con, các giới hạn tài nguyên và số lần fit được đăng ký trước, và cùng quy trình đánh giá lồng nhau.
- **Qwen:** mô hình Qwen độc lập đã sửa lỗi bị bác bỏ về mặt khoa học. Không ủy quyền suy luận mới,
  chạy lại toàn bộ, hay sử dụng GPU. Điểm tài liệu hiện có chỉ có thể quay lại thông qua một nhánh
  bổ trợ điểm đóng băng được ký hợp đồng riêng sau này, không phải C1-I.

## Chỉ các kết quả đầu ra được lên kế hoạch

Không có kết quả đầu ra nào được tạo ra bởi nhiệm vụ tài liệu hóa này. Công việc được ủy quyền
sau này chỉ có thể ghi các kết quả đầu ra C0-A, C0-V, và C1-I tối thiểu được lên kế hoạch,
được liệt kê trong kế hoạch nghiên cứu và tài liệu phân định trách nhiệm tại:
`reports/task1/workflow_c/tv2/c0a/`,
`reports/task1/workflow_c/tv2/c0v/`, và
`reports/task1/workflow_c/tv4/c1i/`.

## Hành động an toàn tiếp theo

Xem xét và đóng băng 6 tài liệu này. Sau đó TV2 chạy C0-A và có thể thực hiện C0-V có giới hạn
chỉ sau khi hoàn thành preflight; TV4 chạy C1-I song song. Việc huấn luyện C2, đánh giá C3,
Qwen, B1, GPU/Modal, Fold0, nhãn công khai, và triển khai không phải là các hành động tiếp theo
và vẫn chưa được ủy quyền.
