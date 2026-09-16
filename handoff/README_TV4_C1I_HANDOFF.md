# Task1 Workflow C1-I — Handoff Guide for TV4

## Mục đích

Thư mục này là gói handoff để chuyển 4 artifact canonical từ
workspace nguồn sang workspace TV4.

Các artifact KHÔNG được regenerate.
Đây là bản copy byte-identical của artifact Task1 canonical.

## 1. Đặt file vào đâu?

Copy thư mục `artifacts` trong handoff vào repository root của TV4.

Sau khi copy, TV4 phải có đúng:

artifacts/task1/evaluation/strict_cv_v2/folds.json

artifacts/task1/recovery_096/baseline_093_oof/predictions.jsonl

artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl

artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/
candidate_refs_full.jsonl

Không đổi tên.
Không sửa nội dung.
Không regenerate.

## 2. Hash bắt buộc

folds.json:

acc4792f1b067d9c58cbfa16789c4c0bd47b71cad081443fbb37eb6017803a0a

Structural contract:
- 7,000 unique query IDs
- F0-F4
- 1,400 IDs/fold

baseline predictions:

1272cb9e8b465f433c725674081084f084a60cdb9807d6ca6a3aa3d9acc1e7d5

actions.jsonl:

7b7acbdd5c26c1e7b21cba42c84900f7894b8fc3dff1475941534345a140eb3a

candidate_refs_full.jsonl:

e09a59852b722fc6e33761cce964920e4929cba59f0171dcd85cbe62d96edfa6

Kiểm tra bằng PowerShell:

Get-FileHash -Algorithm SHA256 "<PATH>"

Tất cả SHA phải đúng trước khi rerun C1-I.

## 3. Manifest

`c1i_canonical_handoff_manifest.json` là provenance record của gói handoff.

Không copy file manifest vào `artifacts/task1/...`.

Giữ nó ở thư mục handoff.

## 4. Sau khi copy

Xác nhận:

- 4/4 path tồn tại
- 4/4 SHA đúng
- không artifact nào bị sửa

Sau đó rerun:

scripts/analysis/workflow_c/
workflow_c_c1i_identifier_universe_audit.py

Không sửa scientific logic của script để ép PASS.

Nếu path/SHA/schema mismatch: STOP.

## 5. C1-I hiện không được phép làm gì?

- Không historical V3A selected-action analysis
- Không Qwen
- Không Qwen complementarity
- Không resume B1
- Không training
- Không model inference
- Không Fold0/public labels
- Không GPU
- Không Modal

## 6. Output cần gửi lại

reports/task1/workflow_c/tv4/c1i/
c1i_structural_identity_report.json

reports/task1/workflow_c/tv4/c1i/
c1i_action_universe_report.json

reports/task1/workflow_c/tv4/c1i/
c1i_feature_schema_report.json

C1-I PASS không tự động authorize C2.

Cần review C1-I cùng kết quả TV2 C0-A trước.