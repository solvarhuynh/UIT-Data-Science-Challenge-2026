# Arrangement Log

The following Python files were reorganized in the `scripts` directory.

## Created folders
- `warmup`
- `validation`
- `evaluation`
- `packaging`
- `utils`
- `submission`
- `misc`

## File movements
| Original location | New location |
|-------------------|--------------|
| `audit_legal_ir_warmup.py` | `warmup/` |
| `audit_legal_qa_warmup.py` | `warmup/` |
| `make_warmup_smoke_submission.py` | `warmup/` |
| `make_legal_qa_warmup_smoke_submission.py` | `warmup/` |
| `validate_chunk_mapping.py` | `validation/` |
| `validate_legal_ir_submission.py` | `validation/` |
| `validate_legal_qa_submission.py` | `validation/` |
| `evaluate.py` | `evaluation/` |
| `evaluate_legal_ir.py` | `evaluation/` |
| `evaluate_legal_qa.py` | `evaluation/` |
| `package_codabench_legal_ir.py` | `packaging/` |
| `package_kaggle.py` | `packaging/` |
| `package_kaggle_task1.py` | `packaging/` |
| `_run_compat.py` | `utils/` |
| `__init__.py` (original) | `utils/` |
| `write_legal_ir_submission.py` | `submission/` |
| `write_legal_qa_submission.py` | `submission/` |
| `write_submission.py` | `submission/` |
| `benchmark_retrieval_internal.py` | `misc/` |

All remaining files are now located within these folders, leaving the top‑level `scripts` directory free of solitary `.py` files.
