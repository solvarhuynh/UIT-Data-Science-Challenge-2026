# 04. Quy chuẩn Git Workflow

Tài liệu này quy định luồng làm việc Git bắt buộc cho toàn bộ đội (TV1-TV5) trong dự án `udsc2026`. Mục tiêu là đảm bảo `main` luôn ở trạng thái chạy được (`uvicorn udsc2026.api.app:app --reload` không bị vỡ), lịch sử commit rõ ràng và các thành viên có thể làm việc song song mà không giẫm code lên nhau.

## 1. Nguyên tắc bắt buộc

- **Không ai được commit trực tiếp vào `main`.** Mọi thay đổi phải đi qua Pull Request (PR) đã được review và merge.
- `main` chỉ nhận code thông qua merge PR đã pass CI (`pre-commit`, test, build).
- Mỗi thành viên làm việc trên nhánh cá nhân/tính năng riêng, tương ứng với worktree đã cấu hình sẵn (xem `scripts/setup_worktrees.sh`).
- Không merge PR nếu vi phạm ranh giới module đã quy định tại `docs/project/khung_repo.md` (ví dụ: TV2 sửa logic trong `qa/`, TV3 định nghĩa lại `RetrievalHit`, v.v.).
- Leader TV1 là người review bắt buộc cho mọi PR chạm vào `src/udsc2026/api/`, `src/udsc2026/contracts/`, `configs/` hoặc ảnh hưởng luồng chung.

## 2. Luồng làm việc: Issue → Branch → Code → PR → Review → Merge

```text
┌────────┐    ┌──────────┐    ┌────────┐    ┌──────┐    ┌────────┐    ┌────────┐
│ Issue  │ →  │  Branch  │ →  │  Code  │ →  │  PR  │ →  │ Review │ →  │ Merge  │
└────────┘    └──────────┘    └────────┘    └──────┘    └────────┘    └────────┘
```

### Bước 1 — Issue

- Mọi thay đổi (feature, bugfix, refactor, doc) đều xuất phát từ một Issue trên GitHub/tracker chung.
- Issue phải nêu rõ: mục tiêu, module liên quan (TV1-TV5), Definition of Done tham chiếu tới `docs/member/tvX.md` nếu có.
- Không tạo branch khi chưa có Issue tương ứng, trừ hotfix khẩn cấp (phải gắn nhãn `hotfix` và thông báo trong kênh chung).

### Bước 2 — Branch

- Branch tạo từ `main` mới nhất (`git pull origin main` trước khi tạo branch).
- Đặt tên branch theo cấu trúc:

```text
feature/<khu-vuc>-<mo-ta-ngan>
fix/<khu-vuc>-<mo-ta-ngan>
docs/<mo-ta-ngan>
hotfix/<mo-ta-ngan>
```

Ví dụ theo phân công hiện tại của dự án:

```text
feature/tv1-api
feature/tv2-dense-frontend
feature/tv3-qa-hybrid
feature/tv4-data-ui
feature/tv5-rerank-devops
```

- Không code trực tiếp trên `main`, kể cả để "sửa nhanh".

### Bước 3 — Code

- Tuân thủ `docs/project/05_coding_convention.md` (Black, Ruff, Type Hint, Docstring).
- Chạy `pre-commit run --all-files` trước khi commit (hook đã cấu hình ở `.pre-commit-config.yaml`: black, isort, flake8, bandit, mypy, pydocstyle).
- Chỉ sửa file trong phạm vi module được phân công (`experiments/tvX/` để thử nghiệm cá nhân, `src/udsc2026/` chỉ chứa code đã chín, dùng chung).
- Commit thường xuyên với message rõ ràng theo chuẩn ở mục 3.

### Bước 4 — Pull Request (PR)

- Mở PR từ branch cá nhân vào `main`, không merge trực tiếp giữa hai nhánh feature với nhau trừ khi đã thống nhất.
- Mô tả PR bắt buộc gồm:
  - **Scope**: module/chức năng thay đổi.
  - **File đã sửa**: danh sách file/thư mục chính.
  - **Contract thay đổi**: có đụng tới `src/udsc2026/contracts/` không, nếu có phải nêu rõ ai bị ảnh hưởng (TV2/TV3/TV5).
  - **Test đã chạy**: lệnh test đã chạy và kết quả (`pytest`, `scripts/run_working_tests.sh`, `scripts/check-ci-local.sh`).
- PR phải liên kết tới Issue gốc (`Closes #<issue-number>`).
- Không để PR quá lớn (nhiều module không liên quan trong 1 PR); tách nhỏ theo từng nhiệm vụ trong `docs/member/tvX.md`.

### Bước 5 — Review

- Tối thiểu 1 người review khác thành viên tác giả PR.
- Leader TV1 bắt buộc review nếu PR chạm `api/`, `contracts/`, `configs/`.
- Reviewer kiểm tra:
  - CI xanh (lint, type-check, test).
  - Không phá vỡ contract chung (`RetrievalHit`, schema request/response).
  - Không mở rộng phạm vi trách nhiệm sang module của người khác (xem ranh giới trong `docs/project/khung_repo.md`).
- Tác giả PR chỉnh sửa theo góp ý, không tự ý merge khi còn comment "changes requested".

### Bước 6 — Merge

- Merge vào `main` chỉ khi: CI pass, đủ review, không còn comment chưa giải quyết.
- Ưu tiên **Squash and merge** để giữ lịch sử `main` gọn, mỗi PR tương ứng một commit rõ ràng trên `main`.
- Xóa branch sau khi merge để tránh rác branch.
- Sau merge, các thành viên khác chạy `git pull origin main` trên worktree của mình trước khi tiếp tục code để tránh conflict tích lũy.

## 3. Chuẩn commit message

Dự án dùng chuẩn commit dạng rút gọn của [Conventional Commits](https://www.conventionalcommits.org/), bắt buộc dùng một trong các prefix sau:

| Prefix | Ý nghĩa | Ví dụ |
| :--- | :--- | :--- |
| `feat` | Thêm tính năng/chức năng mới | `feat(retrieval): add DenseRetriever với Qdrant adapter` |
| `fix` | Sửa lỗi | `fix(qa): sửa lỗi parser citation trả sai chunk_id` |
| `docs` | Thay đổi tài liệu, không đổi code | `docs: cập nhật quy trình git workflow` |
| `refactor` | Tái cấu trúc code, không đổi hành vi | `refactor(contracts): tách RetrievalHit khỏi module retrieval` |
| `test` | Thêm/sửa test | `test(hybrid): thêm test cho HybridRetriever normalize score` |
| `chore` | Việc vặt, cấu hình, dependency, CI | `chore: nâng version black trong pre-commit` |
| `perf` | Cải thiện hiệu năng | `perf(api): thêm cache Redis cho endpoint /query` |

Quy tắc format:

```text
<prefix>(<phạm-vi-tuỳ-chọn>): <mô tả ngắn gọn, thì hiện tại, không viết hoa đầu câu, không dấu chấm cuối>
```

- Phạm vi (`scope`) nên là tên module: `api`, `retrieval`, `qa`, `contracts`, `ingestion`, `evaluation`, `frontend`, `docker`.
- Mô tả ngắn gọn (dưới ~72 ký tự), có thể viết tiếng Việt hoặc tiếng Anh nhưng phải nhất quán trong cùng một PR.
- Với thay đổi phức tạp, thêm phần body sau dòng trống để giải thích lý do (why), không chỉ mô tả cái gì (what).
- Commit vi phạm contract chung (ví dụ đổi field trong `RetrievalHit`) bắt buộc phải nêu rõ trong message và trong mô tả PR, đồng thời tag người bị ảnh hưởng.

## 4. Xử lý xung đột và đồng bộ

- Rebase branch cá nhân lên `main` mới nhất trước khi mở PR (`git fetch origin && git rebase origin/main`), tránh merge commit rác.
- Nếu conflict phát sinh ở `contracts/retrieval.py`, dừng lại và thống nhất với người liên quan trước khi tự ý resolve.
- Không merge thay đổi làm đổi output chunk/schema nếu chưa thông báo cho các thành viên tiêu thụ dữ liệu đó (theo nguyên tắc merge tại `docs/project/khung_repo.md`).
