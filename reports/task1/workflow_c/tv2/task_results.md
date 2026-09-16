## 2026-09-12 — C2-PREFLIGHT — Final governance rerun

**Mục tiêu:** Hoàn tất freeze contract/environment/provenance C2 theo đúng
quyết nghị giáo sư, không train, score, predict hoặc evaluate.

**Đã làm:** Xác minh C1-I commit pinned, rehash 18 provenance entries, dùng
`.venv` hiện hữu đúng Python 3.12.6/NumPy 1.26.4/sklearn 1.7.2/threadpoolctl
3.6.0 và chạy constructor-only API smoke. Đã update in place hai C2 JSON cùng
sáu tài liệu Workflow C.

**Kết quả chính:** C2-V fresh classifier và C2-R exact-delta regressor, seeds,
margin `{0,Q75,Q90,Q95}`, tie-break, splits, inner gate, historical guards,
paired bootstrap và resource limits đều `FROZEN`. JSON validation, full
preflight assertions và `git diff --check` đều PASS.

**Trạng thái:** PASS (`PREFLIGHT_ONLY`).

**Điều rút ra:** Môi trường exact đã tồn tại nên không cần tạo/sửa venv.

**Chưa được kết luận:** Không có kết quả C2/C3; training và inference vẫn
`NOT_AUTHORIZED`.

**Artifact chính:** `reports/task1/workflow_c/shared/c2/c2_preflight_contract.json`;
`reports/task1/workflow_c/shared/c2/c2_input_provenance_manifest.json`.

**Bước tiếp theo:** `SEND_FINAL_C2_PREFLIGHT_FOR_EXECUTION_AUTHORIZATION`.
