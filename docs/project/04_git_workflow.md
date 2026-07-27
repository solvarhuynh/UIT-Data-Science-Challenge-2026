# 04. Git Workflow Nhanh

Tài liệu này là bản rút gọn để cả thành viên và AI có thể dùng nhanh khi làm việc hằng ngày.

## 1. Quy tắc chung

- Không commit thẳng vào `main`.
- Mỗi người làm trên nhánh riêng: `tv1`, `tv2`, `tv3`, `tv4`, `tv5`.
- Trước khi tạo PR, luôn sync lại với `main` mới nhất.
- Commit ngắn, rõ, đúng kiểu: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`.
- Nếu sửa `contracts/`, `api/`, `configs/` thì phải báo leader và review kỹ.

## 2. Quy trình làm việc mỗi ngày

```bash
git checkout main
git pull origin main
git checkout -b feat/tvX-ten-viec

# code xong
git status
git add .
git commit -m "feat(scope): short message"

# trước khi tạo PR
git checkout main
git pull origin main
git checkout feat/tvX-ten-viec
git merge main

# nếu không lỗi thì push
git push origin feat/tvX-ten-viec
```

Mẫu cho AI hiểu nhanh:

- Nếu user nói `pull main` thì chạy: `git checkout main && git pull origin main`
- Nếu user nói `tạo branch mới` thì chạy: `git checkout -b feat/tvX-ten-viec`
- Nếu user nói `commit code` thì chạy: `git add . && git commit -m "feat(scope): short message"`
- Nếu user nói `đồng bộ trước PR` thì chạy: `git checkout main && git pull origin main && git checkout feat/tvX-ten-viec && git merge main`
- Nếu user nói `push nhánh` thì chạy: `git push origin feat/tvX-ten-viec`

## 3. Khi có sự cố

### 3.1 Lỡ commit file nặng hoặc `.env`

```bash
git reset HEAD~1
echo ".env" >> .gitignore
echo "data/raw/*" >> .gitignore
git add .
git commit -m "fix: remove sensitive or heavy files"
git push origin feat/tvX-ten-viec --force
```

### 3.2 Sửa contract làm gãy API người khác

```bash
git checkout origin/main -- src/dsc2026_legal/contracts/models.py
git add src/dsc2026_legal/contracts/models.py
git commit -m "fix: restore contract compatibility"
git push origin feat/tvX-ten-viec
```

### 3.3 Nhánh bị lệch quá xa `main`

```bash
git fetch origin
git checkout feat/tvX-ten-viec
git merge origin/main
pytest
git push origin feat/tvX-ten-viec
```

## 4. Trước khi mở PR

- Đã sync với `main` mới nhất.
- Đã chạy test cần thiết.
- Không có file nhạy cảm, file tạm, file nặng.
- PR ghi rõ: làm gì, đổi file nào, có ảnh hưởng ai không.
