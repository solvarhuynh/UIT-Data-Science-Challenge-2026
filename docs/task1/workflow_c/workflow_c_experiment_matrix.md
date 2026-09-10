# Task1 Workflow C — Ma trận Thực nghiệm Tinh gọn

Định hướng: **GO_WITH_CONDITIONS**

Ma trận này là một danh mục ủy quyền, không phải một yêu cầu thực thi. Các quy tắc chung
vẫn được kế thừa từ `docs/task1/workflow_b/workflow_B_execution_rules.md`. Các kết quả đầu ra
được lên kế hoạch được liệt kê trong kế hoạch nghiên cứu và tài liệu phân định trách nhiệm;
không có tệp nào tự động tồn tại chỉ vì chúng được đặt tên ở đó.

| ID | Mục đích / Công thức | Phụ trách | Tính toán | Phụ thuộc bắt buộc hoặc cổng tương lai | Ủy quyền hiện tại |
|---|---|---|---|---|---|
| C0-A | Giải phẫu oracle/cơ hội độc lập với policy lịch sử; trạng thái toàn vẹn riêng biệt cho K20, K77, toàn tập | TV2 | CPU | Đóng băng 6 tài liệu; cổng 5.600 truy vấn/không sai lệch/mã hash/đối soát; tái lập baseline và cả 3 mức trần trong sai số `1e-12` | `AUTHORIZED_AFTER_DOC_FREEZE` |
| C0-V | Tái dựng nguồn gốc xuất xứ chuyên biệt cho V3A có giới hạn | TV2 | CPU | Preflight dạng văn bản trước khi replay; chỉ phân loại là replay nguồn chính xác, tái dựng tương đương tổng hợp, hoặc sai lệch replay | `BOUNDED_CONDITIONAL_PROVENANCE_RECONSTRUCTION` |
| C1-I | Kiểm toán định danh/không gian/nguồn/schema không nhãn, sau đó xác minh điểm mút F1–F4 với danh tính đã đóng băng | TV4 | CPU | Đóng băng 6 tài liệu; không phụ thuộc C0-V | `AUTHORIZED_IN_PARALLEL_AFTER_DOC_FREEZE` |
| C2-R | `SWAP_ONLY_EXACT_DELTA_RECALL_REGRESSION`; squared error, cân bằng truy vấn, chỉ hoán đổi (swap-only), KEEP làm mốc tham chiếu 0 | TV2 | CPU | Lõi K20 và ủy quyền tương lai; trong mọi phần bù ngoài: delta Recall >= +0.002, 3/3 fold trong không âm, delta Precision >= -0.001, tính toàn vẹn PASS | `NOT_AUTHORIZED` |
| C2-V | Đối chứng phong cách V3A cấu hình cố định mới trên cùng phép chia/đóng băng; bắt buộc cho đối chứng so sánh theo cặp tương lai | TV2 cùng TV4 kiểm toán | CPU | Kiểm tra các chốt chặn vô hướng/fold lịch sử trong sai số `1e-12`; không bao giờ đổi nhãn thành V3A lịch sử | `NOT_AUTHORIZED` |
| C2-B1 | Đối chứng học được tùy chọn có giới hạn; job Workflow B vẫn tiếp tục đóng băng | TV4 kiểm toán / TV2 đánh giá | CPU | Ủy quyền riêng, tái hiện thực hóa chính xác, chạy khói 1 fold có mã thoát, giới hạn tài nguyên/lần fit cố định, cùng quy trình lồng nhau | `NOT_AUTHORIZED` |
| C3 | So sánh khoa học lồng nhau ngoài có đóng băng dự đoán đối chiếu với đối chứng C2-V mới | TV2 cùng TV4 bình duyệt | CPU | Cả 4 bundle C2 đều đóng băng; delta Recall >= +0.003; 4/4 fold không âm; delta Precision >= -0.001; bootstrap theo cặp p10 > 0; Recall tuyệt đối >= 0.9326488095238095; không fold nào thấp hơn chốt chặn fold V3A lịch sử; tính toàn vẹn PASS | `NOT_AUTHORIZED` |
| C4 | Kiểm tra độ ổn định được đăng ký trước trên các kết quả C3 đã đóng băng | TV4 bình duyệt | CPU | Ủy quyền riêng sau C3; không lựa chọn sau sự thật (post-hoc) | `NOT_AUTHORIZED` |
| C5 | Một lần fit cuối cùng được đóng băng trên F1–F4 | TV2 | CPU | Ủy quyền riêng sau khi bằng chứng khám phá được bình duyệt | `NOT_AUTHORIZED` |
| C6 | Bình duyệt suy luận/triển khai công khai không nhãn | TV2 cùng ban quản trị | CPU trừ khi có thay đổi riêng | Ủy quyền riêng; giải quyết nguồn gốc bộ chấm điểm CodaBench đang hoạt động; các cổng cấu trúc/nguồn gốc công khai | `NOT_AUTHORIZED` |

## Quy tắc cố định trên toàn ma trận

- C0-A chỉ phục vụ mục đích toàn vẹn/chẩn đoán. Nó không có thêm cổng ngưỡng độ lớn oracle
  nào và không đưa ra tuyên bố về tính khả học.
- K20 là `CORE`. K77 hiện được kiểm toán nhưng chỉ được mô hình hóa sau khi K20 vượt qua C2,
  hợp đồng K77 đạt yêu cầu, và có hợp đồng riêng ủy quyền.
- KEEP sử dụng `ZERO_REFERENCE_SCORE`: không có hàng/vector KEEP tổng hợp; các trường hợp hòa
  hoặc không vượt biên độ nghiêm ngặt sẽ chọn KEEP.
- Mỗi truy vấn đóng góp tổng trọng số huấn luyện hoán đổi là 1.0; các lớp kết quả không được đánh lại trọng số.
- Khoa học theo cặp C2/C3 sử dụng đối chứng mới trên cùng phép chia. Recall 0.9296488095238095
  của V3A lịch sử và các chỉ số fold của nó là các chốt chặn vô hướng/fold bất biến, không phải dữ liệu theo cặp ở cấp độ hàng.
- `C3_BOOTSTRAP_PROTOCOL_PENDING_PRE_C2_FREEZE` phải được giáo sư/nhóm giải quyết
  trước khi đóng băng lựa chọn C2 và trước khi tạo dự đoán C3.
- Tất cả các byte dự đoán và danh tính tương lai phải đóng băng trước khi join nhãn.
- F1–F4 chứa 5.600 truy vấn khoa học; Recall là chính, Precision là phụ, và kết quả đầu ra
  chứa tối đa 5 tài liệu duy nhất/truy vấn.
- Fold0 và nhãn công khai bị loại trừ. Điểm công khai 0.9391 chỉ là bằng chứng triển khai.
  C3 vẫn là `EXPLORATORY SCIENTIFIC EVIDENCE` vì không còn tập kiểm tra giữ lại (holdout) ở cấp dự án chưa từng chạm tới.
- Suy luận Qwen, tiếp tục B1, GPU/Modal, và triển khai công khai không được ủy quyền.
  Điểm Qwen hiện có yêu cầu một hợp đồng bổ trợ điểm đóng băng riêng sau này và không phải là một phần của C1-I.
- Nguồn gốc bộ chấm điểm CodaBench chưa được giải quyết chỉ chặn C6/triển khai.
