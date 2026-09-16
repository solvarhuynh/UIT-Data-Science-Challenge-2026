# Task1 Workflow C — Bình duyệt Phản biện Độc lập và Quyết nghị Hợp đồng

## Phán quyết hiện tại

**GO_WITH_CONDITIONS**

Phán quyết hướng đi độc lập:
**WORKFLOW_C_RESIDUAL_RECOVERY_IS_BEST_NEXT_PATH**.

Tệp này lưu giữ lại bản bình duyệt đối kháng trước đó và ghi nhận riêng quyết nghị của
giáo sư/nhóm nghiên cứu. Quyết nghị này là hợp đồng chuẩn tắc hiện tại ở bất kỳ điểm nào
nó khác với khuyến nghị tạm thời ban đầu của bản bình duyệt.

## Cập nhật C2-PREFLIGHT cuối cùng — 2026-09-12

C0-A/C1-I `COMPLETE_PASS`, C1-I `C1I_COMMIT_PINNED`, C0-V `DEFERRED`.
Professor governance đã giải quyết exact configuration và final C2 preflight
rerun đạt `PASS` dưới `PREFLIGHT_ONLY`; không có C2 training, inference,
score, prediction hay C3 evaluation. Môi trường `.venv` được pin ở Python
3.12.6, NumPy 1.26.4, sklearn 1.7.2 và threadpoolctl 3.6.0; constructor-only
smoke đạt `PASS` mà không gọi fit/predict.

C2-V là control mới `C2V_FRESH_HGBC_1_7_2_K20`, classifier ba lớp seed 2026,
utility `P(B)-1.5P(H)` và margin 0; nó không phải historical V3A. C2-R là
`HistGradientBoostingRegressor` squared-error exact-delta seed 2027 với margin
inner-OOF `{0,Q75,Q90,Q95}`. Bootstrap được frozen ở 10.000 paired-query
replicates/PCG64 seed 20260911. Resource contract là 1 thread/fit, 1 fit đồng
thời, 32 fits, RSS 8 GiB/available 4 GiB, timeout 15 phút/fit và 8 giờ tổng.
TV2 là future execution owner nếu được cấp quyền riêng; TV4 là independent
reviewer. K20 core, K77/full-pool deferred, Qwen/B1 giữ frozen status.

Quần thể khoa học đã được quyết nghị là chính xác 5.600 truy vấn F1–F4. Recall là chính,
Precision là phụ, và mỗi kết quả đầu ra có tối đa 5 tài liệu duy nhất/truy vấn.
Fold0 và nhãn công khai bị loại trừ; điểm công khai 0.9391 chỉ là bằng chứng triển khai.
Các dự đoán và danh tính phải đóng băng trước các phép join nhãn áp dụng. Do không còn
tập kiểm tra giữ lại (holdout) ở cấp dự án chưa từng chạm tới, các kết quả C3 trong tương lai
mang tính chất khám phá (exploratory).

## 1. Các phát hiện bình duyệt độc lập ban đầu

Phần này mang tính lịch sử. Nó ghi lại lập luận và các rủi ro được nêu ra trước
quyết định của giáo sư/nhóm; đây không phải là bảng ủy quyền.

### Tại sao phục hồi phần dư (residual recovery) được ưu tiên

Bản bình duyệt nhận thấy bằng chứng ủng hộ việc nghiên cứu bài toán chính sách quyết định phần dư (residual decision-policy)
trước khi mở rộng biểu diễn hoặc năng lực tính toán:

- K20 cho phép mức trần one-swap có nhận biết nhãn (label-aware) là 0.9676994048,
  cao hơn xấp xỉ +0.0380505952 so với V3A lịch sử.
- Bằng chứng pháp y tổng hợp ghi nhận hành động đứng đầu là BENEFIT chỉ ở 81 trong số 301
  truy vấn có cơ hội, 220 lần bỏ lỡ xếp hạng, và 27 lần từ chối hành động BENEFIT đứng đầu.
- Phân tích cắt giảm (ablation) cổng diện rộng của V3B mang lại kết quả tiêu cực trên cả 4 fold;
  recall BENEFIT ở giai đoạn 1 của nó vẫn ở mức thấp.
- Mở rộng các rank 1–3 không tạo thêm biên độ cải thiện oracle nào.
- Mô hình Qwen zero-shot độc lập sau khi sửa lỗi đạt điểm thấp hơn nhiều so với mô hình đương nhiệm (incumbent).
- B1 không tạo ra mô hình đã fit hay điểm khoa học nào và rất tốn kém để tiếp tục chạy lại.

Bản bình duyệt cảnh báo rõ ràng rằng một oracle có nhận biết nhãn chỉ chứng minh sự tồn tại của cơ hội,
chứ không chứng minh tính khả học (learnability). Nó cũng cảnh báo rằng việc không có sẵn các lựa chọn OOF V3A
lịch sử và sự sai lệch SHA mã nguồn có thể khiến việc quy kết lỗi ở cấp độ truy vấn trở nên không an toàn.

### Phê bình công thức ban đầu

Bản bình duyệt nhận định rằng delta Recall chính xác là phù hợp về mặt toán học với
chỉ số tập hợp chính thức, đồng thời lưu ý rằng nó rời rạc theo lực lượng tập nhãn chuẩn (gold-cardinality)
và hàm mất mát cây chỉ là một đại lượng thay thế (surrogate). Nó khuyến nghị đánh trọng số cân bằng truy vấn
bởi vì các hành động BENEFIT rất thưa thớt và các truy vấn có số lượng hành động khác nhau.
Recall đầu-cuối (end-to-end Recall), chứ không phải AUC hành động hay độ chính xác hành động,
được xác định là điểm mút chi phối, với Precision đóng vai trò chốt chặn rủi ro.

Bản bình duyệt ban đầu mô tả KEEP như một hành động được biểu diễn rõ ràng cùng nhau
và lập luận rằng cần một biên độ từ chối lồng nhau (nested abstention margin) để tránh lặp lại
cổng diện rộng gây hại của V3B. Chi tiết biểu diễn đó chỉ mang tính tạm thời và không được chấp thuận nguyên văn;
quyết nghị cuối cùng thay vào đó cố định KEEP là một mốc tham chiếu không cần học với độ hữu dụng bằng 0,
không có hàng hay vector tổng hợp nào.

Bản bình duyệt giữ nguyên các hạn chế tối đa một lần hoán đổi (max-one-swap) và rank-4/rank-5 cho
nhánh đầu tiên: mức tăng sửa đổi one-swap trên toàn tập là +0.0638154762 so với +0.0668779762
của sửa đổi không ràng buộc, trong khi việc mở rộng rank 1–3 không mang lại mức tăng đo lường được nào.

### Các phát hiện về tính không chắc chắn ban đầu

- Việc liệu 58 đặc trưng K20 có thể sắp xếp thứ tự độ hữu dụng chính xác một cách đáng tin cậy hay không
  vẫn chưa rõ. Các chẩn đoán từ V3A và Workflow-A cho thấy có tín hiệu, nhưng không tạo ra mức tăng chính sách
  đáng kể và ổn định.
- Sự đồng thuận giữa các nguồn ứng viên có thể giúp phân biệt hành động, nhưng độ bao phủ ứng viên
  đơn thuần không chứng minh được tín hiệu chính sách.
- Việc quy kết giữa tài liệu đưa vào sai (wrong-incoming) so với tài liệu đưa ra sai (wrong-outgoing)
  là không hợp lệ nếu không có dòng dõi lịch sử được chấp thuận trên từng truy vấn.
- Các phương pháp theo cặp (pairwise) hoặc theo danh sách (listwise) có nguy cơ lặp lại thất bại
  về nhãn thưa và nguồn gốc xuất xứ. Không có căn cứ để tìm kiếm diện rộng bằng LambdaMART.
- Khả năng bổ trợ của Qwen được coi là chưa xác định dù Recall độc lập kém;
  bản bình duyệt ban đầu đề xuất một thử nghiệm join có giới hạn các điểm còn lại.
  Quyết nghị cuối cùng không ủy quyền thử nghiệm đó như một phần của C1-I.
- True-S2/MAX chưa từng có thử nghiệm cắt giảm nguyên nhân độc lập, nhưng cả bằng chứng Qwen đã sửa
  lẫn dòng dõi đoạn văn (chunk) bị thiếu đều không đủ biện minh cho việc thay đổi nó trước tiên.

### Các dạng thức thất bại ban đầu

1. **Thất bại về nguồn gốc xuất xứ (Provenance failure).** Tái dựng một chính sách khác trong khi vẫn gọi nó là
   V3A lịch sử có thể làm vô hiệu hóa các tuyên bố ở cấp độ hàng. Chỉ sự ngang bằng về mặt tổng hợp không đủ để
   chứng minh danh tính trên từng truy vấn.
2. **Lạm dụng oracle (Oracle misuse).** Các giá trị 0.9677, 0.9847, và 0.9897 là các mức trần có nhận biết nhãn,
   không phải điểm số mô hình kỳ vọng đạt được.
3. **Quá khớp do lựa chọn (Selection overfitting).** F1–F4 đã được dùng để cung cấp thông tin cho các công việc trước đó. OOF
   lồng nhau giúp giảm nhẹ rò rỉ tức thì nhưng không thể khôi phục lại một tập holdout cấp dự án chưa từng chạm tới.
4. **Không khớp chỉ số (Metric mismatch).** Một đại lượng thay thế hồi quy/phân loại không thể thay thế
   cho việc đánh giá macro Recall và Precision đầu-cuối đã đóng băng.
5. **Không khớp ứng viên/schema (Candidate/schema mismatch).** Dòng dõi 58 đặc trưng của K20 và dòng dõi 36 đặc trưng
   rộng hơn của K77 không thể bị coi một cách thầm lặng là giống nhau.
6. **Từ chối/hiệu chuẩn gây hại (Harmful abstention/calibration).** Một ngưỡng được chọn bằng cách sử dụng các nhãn ngoài
   có thể lặp lại sự can thiệp gây hại của V3B.
7. **Thất bại khi chạy (Runtime failure).** B1 đã chứng minh sự cần thiết của số lần fit có giới hạn,
   kế hoạch bộ nhớ/thời gian chạy, bằng chứng gia tăng, và mã thoát tiến trình con được ghi nhận.

### Các cổng và khuyến nghị tạm thời ban đầu

Trước khi có quyết nghị của giáo sư/nhóm, bản bình duyệt đã khuyến nghị tạm thời một giai đoạn
kết hợp replay/giải phẫu, bao gồm một bước sàng lọc biên độ lựa chọn gộp ít nhất +0.010
và ít nhất 3 trong 4 fold đạt +0.005. Nó cũng đề xuất các cổng bổ trợ Qwen và thử nghiệm ngữ nghĩa sau này.
Những đề xuất đó là các phát hiện tư vấn của bản bình duyệt, không phải sự phê duyệt.

Khuyến nghị cuối cùng ban đầu là chỉ thực hiện replay/giải phẫu mô hình đương nhiệm trên CPU trước tiên
và tránh huấn luyện mô hình thách thức (challenger), tránh tiếp tục B1, chạy lại Qwen, Fold0, nhãn công khai, và GPU.
Nó đánh giá điểm công khai gần 0.96 là một ước vọng chứ không phải một dự báo có thể bảo vệ được,
vì biên độ cải thiện là có nhận biết nhãn, các mức tăng học được từ trước đến nay đều nhỏ,
không còn tập holdout chưa chạm tới, và nguồn gốc bộ chấm điểm vẫn chưa được giải quyết.

## 2. Quyết nghị hợp đồng của giáo sư/nhóm

Giáo sư/nhóm nghiên cứu chấp thuận hướng nghiên cứu nhưng sửa đổi hợp đồng.
Phần này thay thế mọi khuyến nghị tạm thời ở trên nếu không được chấp thuận nguyên văn.

### Ủy quyền cụ thể theo từng nhánh cuối cùng

| Nhánh / Tài nguyên | Trạng thái chuẩn tắc |
|---|---|
| TV2 C0-A | `AUTHORIZED_AFTER_DOC_FREEZE` |
| TV2 C0-V | `BOUNDED_CONDITIONAL_PROVENANCE_RECONSTRUCTION`; việc replay yêu cầu preflight dạng văn bản |
| TV4 C1-I | `AUTHORIZED_IN_PARALLEL_AFTER_DOC_FREEZE` |
| C2, C3, C4, C5 | `NOT_AUTHORIZED` |
| C6 / Triển khai công khai | `NOT_AUTHORIZED` |
| Tiếp tục B1 | `NOT_AUTHORIZED` |
| Suy luận Qwen | `NOT_AUTHORIZED` |
| GPU / Modal | `NOT_AUTHORIZED` |
| Fold0 cho lựa chọn/đánh giá khoa học | `NO` |
| Nhãn công khai / Lựa chọn theo bảng xếp hạng | `NO` |

Vấn đề nguồn gốc bộ chấm điểm CodaBench chỉ chặn C6/triển khai, không chặn C0-A hay C1-I.

### Quyết nghị về giai đoạn ban đầu kết hợp

Giai đoạn kết hợp ban đầu được tách làm hai:

- **C0-A** là giải phẫu oracle/cơ hội độc lập với policy lịch sử. Nó tái lập
  baseline, K20, K77, và không gian one-swap toàn tập cùng các sự thật về cơ hội/mức tăng/nguồn
  mà không cần các lựa chọn của V3A lịch sử. Đây là giai đoạn chẩn đoán/toàn vẹn và không đưa ra tuyên bố về tính khả học.
- **C0-V** là tái dựng chuyên biệt cho V3A có giới hạn. Nó bắt đầu bằng một preflight
  dạng văn bản và kết thúc với phân loại `EXACT_SOURCE_REPLAY`,
  `AGGREGATE_EQUIVALENT_RECONSTRUCTION`, hoặc `REPLAY_MISMATCH_BLOCKED`.
  Danh tính mã nguồn bị thiếu được giữ lại dưới dạng một giới hạn; nó không dẫn tới việc khảo cổ vô hạn
  hoặc chặn đứng C0-A/C1-I trên diện rộng.

Cổng của C0-A là chính xác 5.600 truy vấn F1–F4; 0 sai lệch về truy vấn/tài liệu/hành động/fold/schema/join bắt buộc;
các mã hash được xác minh; baseline 0.9259285714285714, K20 0.9676994047619048, K77 0.9847142857142857,
và toàn tập 0.9897440476190476 đều nằm trong sai số `1e-12`; và đối soát giữa từng fold với giá trị gộp.
Trạng thái của K20, K77, và toàn tập là riêng biệt. Bước sàng lọc biên độ oracle tạm thời ban đầu
được loại bỏ khỏi cổng C0-A cuối cùng.

### Quyết nghị về ranh giới bằng chứng

Dòng dõi C0-V chính xác chỉ bắt buộc đối với các tuyên bố về hành động được chọn trên từng truy vấn lịch sử,
tài liệu đưa vào sai, tài liệu đưa ra sai, KEEP so với hoán đổi, và phân loại lỗi chuyên biệt cho V3A;
các mục tiêu phái sinh từ lựa chọn lịch sử; phân tích hoặc bootstrap theo cặp với V3A lịch sử;
và các chốt chặn hồi quy ở cấp độ hàng đòi hỏi các hàng cũ chính xác.

Nó không bắt buộc đối với các sự thật về baseline/ứng viên/nguồn, sự thật oracle K20/K77/toàn tập,
tổn thất không gian ứng viên, cơ hội rank-4/rank-5, phân phối mức tăng, nguồn gốc tài liệu đưa vào có lợi,
hoặc nghiên cứu so sánh mới trên cùng phép chia. Tương đương tổng hợp chỉ hỗ trợ kiểm tra tính hợp lý
ở cấp độ vô hướng/từng fold, không chứng minh danh tính hàng lịch sử hay bootstrap theo cặp với lịch sử.

### Quyết nghị về việc kiểm toán của TV4

C1-I là kiểm toán cấu trúc không nhãn về danh tính truy vấn/tài liệu/hành động,
tư cách thành viên K20/K77/toàn tập, thứ hạng/độ hỗ trợ nguồn, các mã hash, độ bao phủ, và schema đặc trưng.
Sau khi các danh tính được đóng băng, nó chỉ có thể join nhãn F1–F4 để xác minh điểm mút oracle
độc lập với policy lịch sử. Nó không phụ thuộc vào C0-V và loại trừ quy kết V3A lịch sử, tính bổ trợ của Qwen,
và việc xây dựng mô hình.

### Quyết nghị về KEEP và công thức chính trong tương lai

Hợp đồng KEEP cố định là `ZERO_REFERENCE_SCORE`:

- các hành động hoán đổi vẫn là các hàng duy nhất trong `actions.jsonl`;
- không có hàng huấn luyện hay vector đặc trưng KEEP nào được tổng hợp;
- KEEP có độ hữu dụng chính xác bằng 0;
- mỗi truy vấn đóng góp tổng trọng số huấn luyện hoán đổi là 1.0;
- không cho phép đánh lại trọng số cho các lớp kết quả BENEFIT/HARM/NEUTRAL;
- chỉ thực thi hoán đổi dự đoán tốt nhất nếu nó vượt qua một cách nghiêm ngặt biên độ được chọn từ OOF nội bộ;
  nếu không thì chọn KEEP; trường hợp hòa điểm chính xác sẽ chọn KEEP.

Công thức chính trong tương lai là `SWAP_ONLY_EXACT_DELTA_RECALL_REGRESSION`: một mô hình hồi quy nông đóng băng,
hàm mất mát squared error, trọng số cân bằng truy vấn, không quét siêu tham số, và macro Recall đầu-cuối
là điểm mút quyết định việc tiếp tục. Hiện tại C2 chưa được ủy quyền.

### Quyết nghị về đối chứng và các cổng trong tương lai

Nghiên cứu so sánh theo cặp trong tương lai yêu cầu một đối chứng phong cách V3A cấu hình cố định mới
dưới cùng quy trình chia tách/đóng băng. Nó không phải là V3A lịch sử. Recall gộp lịch sử 0.9296488095238095
và các chỉ số fold được bảo tồn là các chốt chặn hồi quy vô hướng/fold bất biến. Sai lệch lớn hơn `1e-12`
sẽ chặn cách diễn đạt tương đương lịch sử và kích hoạt việc rà soát dòng dõi; một đối chứng mới được đổi tên
chỉ có thể tiếp tục nếu ban quản trị cho phép.

Đối với mọi phần bù huấn luyện ngoài, C2 tương lai yêu cầu mô hình thách thức trừ đi đối chứng mới
phải đạt Recall OOF nội bộ gộp ít nhất +0.002, cả 3 fold nội bộ không âm, delta Precision gộp ít nhất -0.001,
và tính toàn vẹn dự đoán/nguồn gốc PASS. Toàn bộ 4 bundle lựa chọn phải đóng băng trước bất kỳ phép join nhãn C3 nào.
Bất kỳ phần bù nào thất bại đều phải dừng lại trước C3.

C3 trong tương lai yêu cầu delta Recall theo cặp gộp ít nhất +0.003, cả 4 fold không âm, delta Precision gộp
ít nhất -0.001, phân vị thứ 10 của bootstrap truy vấn theo cặp lớn hơn 0, Recall gộp tuyệt đối ít nhất 0.9326488095238095,
không có fold nào thấp hơn giá trị vô hướng fold V3A lịch sử, và toàn bộ tính toàn vẹn PASS.
Kết quả vẫn là `EXPLORATORY SCIENTIFIC EVIDENCE`.

Bootstrap C3 tương lai đã frozen trước scoring: resampling unit là một paired
outer-OOF query delta C2-R trừ fresh C2-V; lấy 10.000 mẫu có hoàn lại, mỗi mẫu
5.600 query, PCG64 seed 20260911, statistic là mean và p10 là
`numpy.quantile(..., 0.10, method="linear")`. Protocol chưa được phép tính lúc preflight.

### Quyết nghị về các nhánh tùy chọn

- K20 là `CORE`.
- K77 hiện được kiểm toán, nhưng chỉ được mô hình hóa sau khi K20 vượt qua C2, tính toàn vẹn K77 đạt yêu cầu,
  và có hợp đồng riêng ủy quyền. Sự thất bại của K20 không ủy quyền cho K77.
- B1 là `FROZEN_OPTIONAL_C2_COMPARATOR`; job Workflow B không thể tiếp tục. Việc quay lại yêu cầu
  ủy quyền riêng, tái hiện thực hóa chính xác, chạy khói 1 fold có mã thoát tiến trình con,
  các giới hạn tài nguyên/lần fit có giới hạn, và cùng quy trình đánh giá lồng nhau.
- Qwen độc lập bị bác bỏ. Không ủy quyền suy luận hay sử dụng GPU. Điểm hiện có chỉ có thể quay lại
  trong một nhánh bổ trợ điểm đóng băng được ký hợp đồng riêng sau này.

## Quyết định bình duyệt chuẩn tắc cuối cùng

Hợp đồng sửa đổi đã được chứng minh và giới hạn nội bộ. Sau khi 6 tài liệu được xem xét và đóng băng,
TV2 có thể chạy C0-A và chỉ có thể bắt đầu C0-V với bản preflight của nó; TV4 có thể chạy C1-I song song.
Việc huấn luyện C2, đánh giá C3, Qwen, B1, GPU/Modal, Fold0, nhãn công khai, và triển khai vẫn chưa được ủy quyền.

Các mức trần có nhận biết nhãn chỉ ra cơ hội nhưng không dự báo điểm công khai.
Không có tuyên bố nào gần mức 0.96 được bảo đảm bởi hợp đồng này.
