# Hướng Dẫn Vị Trí và Khôi Phục File Lớn (Task 1 / UDSC 2026)

Tài liệu này hướng dẫn vị trí đặt các file dữ liệu và artifact lớn (> 50 MB / > 100 MB).
Do GitHub giới hạn dung lượng 100 MB/file, các file này được đóng gói ngoại vi trong file zip này.

## 1. Cách giải nén vào Workspace

Giải nén toàn bộ nội dung file zip này trực tiếp vào thư mục gốc của repository (`d:/udsc2026` hoặc repo root).
Cấu trúc thư mục trong file zip đã được tạo tương ứng với cấu trúc thư mục của repository.

## 2. Bảng Danh Mục File & Vị Trí Chi Tiết

| STT | File / Đường dẫn đích | Kích thước | SHA256 Checksum | Mục đích |
|---|---|---|---|---|
| 1 | `reports/task1/full_document_legal_field_retrieval/full_document_legal_field_retrieval_candidates.jsonl` | 267.37 MB | `3519a417cbeed46491e270f11a2b63707856cad594cda6c239e37a4a78204ab0` | Ứng viên tìm kiếm văn bản pháp luật đầy đủ (Full Document Retrieval Candidates) |
| 2 | `handoff/task1_workflow_c/artifacts/task1/recovery_096/v3_residual/policy/actions.jsonl` | 476.25 MB | `7b7acbdd5c26c1e7b21cba42c84900f7894b8fc3dff1475941534345a140eb3a` | Canonical Handoff C1-I: Policy actions dataset |
| 3 | `handoff/task1_workflow_c/artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl` | 132.06 MB | `e09a59852b722fc6e33761cce964920e4929cba59f0171dcd85cbe62d96edfa6` | Canonical Handoff C1-I: Candidate references compact dataset |
| 4 | `reports/task1/workflow_c/shared/c2/execution/.worker_jobs/inner_o1_i4/c2_r/OdS45tX7Q86w0eLCCthrvUlTxgdpQx89.json` | 68.98 MB | `5fe1847184f25f215448ebddee416bf77ab0fb83fea48efaa9aa690c05ee378b` | Workflow C2 nested evaluation: Ma trận huấn luyện worker cache |
| 5 | `reports/task1/workflow_c/shared/c2/execution/.worker_jobs/inner_o1_i2/c2_r/XzmFz2e-slCQbsDCWzcrb8sE9rhTZk3Q.json` | 68.98 MB | `6ce1e5c3449cfdaddff58816449b61b7765297a08a425ae8e1710c3aed12af87` | Workflow C2 nested evaluation: Ma trận huấn luyện worker cache |
| 6 | `reports/task1/workflow_c/shared/c2/execution/.worker_jobs/inner_o1_i3/c2_r/kjkJR5njnRBDm3K19_nLuoFt03_rBH3C.json` | 68.97 MB | `a0d64623006a23035310a7319dc2c79959a164b61ba9df51eec0eacea7bbae35` | Workflow C2 nested evaluation: Ma trận huấn luyện worker cache |
| 7 | `reports/task1/workflow_c/shared/c2/execution/.worker_jobs/inner_o1_i4/c2_v/8A1bCYBcI1uqwprxvA5MURpsv8GgO9xk.json` | 68.81 MB | `8741a5e003c9743f3ca7e3a307f4c202f007d7c637249c393af68da301a625b4` | Workflow C2 nested evaluation: Ma trận huấn luyện worker cache |
| 8 | `reports/task1/workflow_c/shared/c2/execution/.worker_jobs/inner_o1_i4/c2_v/M1mXa2bghUajcQUDKf5PU1GXkr7qv1Nv.json` | 68.81 MB | `5964976eff9b758e46b066b53feb0b67cb91e7539a0488cbf6b7f2253654f851` | Workflow C2 nested evaluation: Ma trận huấn luyện worker cache |
| 9 | `reports/task1/workflow_c/shared/c2/execution/.worker_jobs/inner_o1_i2/c2_v/APivLeNP-hH8CaPiss6Ji_UlX5DecXwv.json` | 68.81 MB | `5cbc60c8e04b293cd9dc1c4ae12b2bffe369b405fd97852570373855fc36a657` | Workflow C2 nested evaluation: Ma trận huấn luyện worker cache |
| 10 | `reports/task1/workflow_c/shared/c2/execution/.worker_jobs/inner_o1_i2/c2_v/KI0OMhDyNcVFINUfX_OKTD-Rhaumedi-.json` | 68.81 MB | `6579b484c06e6483e1c2d6df1af1c0928d556572204c1b1ebe3e01ebe7466906` | Workflow C2 nested evaluation: Ma trận huấn luyện worker cache |
| 11 | `reports/task1/workflow_c/shared/c2/execution/.worker_jobs/inner_o1_i2/c2_v/RL6MFPXOQMBymAWoQSw9gCEhHz8nC8pD.json` | 68.81 MB | `b44862a365306d1d0eb253e90e44e04e77814116170197bd46d071e4f8361075` | Workflow C2 nested evaluation: Ma trận huấn luyện worker cache |
| 12 | `reports/task1/workflow_c/shared/c2/execution/.worker_jobs/inner_o1_i3/c2_v/-Fa9YE50-mAKjZEflRA4gCfUHT5IMZPL.json` | 68.80 MB | `fe7c3b7b8553b74aecb065fc8ae1245aaf47402eb96285ffd90e5ca6d6db4f02` | Workflow C2 nested evaluation: Ma trận huấn luyện worker cache |
| 13 | `reports/task1/workflow_c/shared/c2/execution/.worker_jobs/inner_o1_i3/c2_v/ZN_z2fLNgUO8FJPNr138k9hwhCnyB-9n.json` | 68.80 MB | `83f853d37c555331bc202cc14b213c4babee825b53243bd911a9a89447805678` | Workflow C2 nested evaluation: Ma trận huấn luyện worker cache |

## 3. Kiểm tra tính toàn vẹn sau khi giải nén

Sau khi đặt các file vào đúng vị trí, kiểm tra mã SHA256 bằng PowerShell:
```powershell
Get-FileHash -Algorithm SHA256 '<duong_dan_file>'
```
Đảm bảo mã hash khớp hoàn toàn với bảng trên.
