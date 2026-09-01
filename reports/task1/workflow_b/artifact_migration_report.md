# Workflow-B artifact migration report

## Phạm vi

Đây là migration chỉ về đường dẫn và ownership. Không chạy experiment, không chạy inference/training, không tính metric và không thay đổi scientific result.

## Kết quả

- B2a đã được chuẩn hóa từ reports/task1/workflow_b/b2a/ sang reports/task1/workflow_b/tv2/b2a/ vì B2a thuộc TV2.
- B1 đã được chuẩn hóa từ reports/task1/workflow_b/b1/ sang reports/task1/workflow_b/tv4/b1/ vì B1 thuộc TV4.
- Hai ledger trực tiếp cũ đã được chuyển vào reports/task1/workflow_b/tv2/task_results.md và reports/task1/workflow_b/tv4/task_results.md.
- Đã tạo các root shared/, tv2/ và tv4/ cùng các nhánh cấu trúc cần thiết cho Workflow-B.
- reports/task1/progress_log.md được giữ nguyên vì đây là progress log append-only toàn Task1.

## Bảo toàn nội dung

Các thao tác là move cùng volume, không phải copy rồi sửa nội dung. Vì vậy file size và SHA256 của artifact được bảo toàn qua migration; không có bản sao B1/B2a canonical thứ hai được tạo. Manifest đi kèm ghi lại old path, new path, owner và trạng thái content_changed = false.

## Quy tắc sau migration

Artifact Workflow-B mới phải nằm dưới reports/task1/workflow_b/. Artifact B2a đặt dưới tv2/b2a/, B1 đặt dưới tv4/b1/, định nghĩa dùng chung đặt dưới shared/, còn lịch sử Task1 chung tiếp tục ở reports/task1/progress_log.md.
