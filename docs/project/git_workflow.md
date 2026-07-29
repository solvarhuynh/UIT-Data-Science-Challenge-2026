# 04. Git Workflow Nhanh

Tài liệu này là bản rút gọn để cả thành viên và AI có thể dùng nhanh khi làm việc hằng ngày.

## 1. Quy tắc chung

- Tuyệt đối không commit thẳng vào `main`.
- Mỗi người làm trên nhánh riêng: `tv1`, `tv2`, `tv3`, `tv4`, `tv5`.
- Trước khi tạo PR, luôn sync lại với `main` mới nhất.
- Commit ngắn, rõ, đúng kiểu: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf` và không dùng `git add .`
- Nếu sửa `contracts/`, `api/`, `configs/` thì phải báo leader và review kỹ.


## 2. Quy trình làm việc mỗi ngày

### 2.1 giai đoạn 1

- 1. Cất code đang làm dở

    git stash

- 2. Lấy code mới nhất của dự án từ main

    git checkout main
    git pull origin main

- 3. Quay lại nhánh cá nhân và cập nhật code main vào nhánh

    git checkout tvX
    git merge main

- 4. Lấy lại code dở ra làm tiếp

    git stash pop

### 2.2 giai đoạn 2

- Tạo các bộ git push theo từng nhóm công việc đã hoàn thành
- 1 bộ git đẩy code lên gồm:

    git add <tên các file đã làm việc trong nhóm đó>
    git commit -m "feat(scope): short message"
    git push origin tvX
```
