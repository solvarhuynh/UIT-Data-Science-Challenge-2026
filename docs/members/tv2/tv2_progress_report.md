# Báo cáo tiến độ TV2 — Task 1 LegalIR

Báo cáo này dựa trên source, test, manifest, config, model registry,
`git status/diff` và official scorer hiện có. Trạng thái “đã cài đặt code” và
“đã hoàn thành experiment GPU đầy đủ” được phân biệt riêng.

## Tóm tắt trạng thái

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Official scorer parity | DONE | `artifacts/task1/evaluation/p1_report.json`; golden tests |
| Strict grouped CV | DONE | `artifacts/task1/evaluation/strict_cv_v2/`; 5 folds/7,000 questions |
| Dense baseline | DONE | `artifacts/task1/train500b_dense200_manifest.json` |
| BGE GPU reranker artifact | DONE | `artifacts/task1/train500-bge-reranker-200-m512/run_manifest.json` |
| Candidate diagnostics | DONE | P2 summary/smoke artifacts |
| P3 lexical | EXPERIMENTAL | implementation/tests; full CPU ablation timed out |
| P4 parent retrieval | EXPERIMENTAL | implementation/tests; bounded plan only |
| P5 document candidates | EXPERIMENTAL | implementation/tests; no full GPU score |
| P6 operational runner | DONE | dry-run and runner tests |
| Public Top-1 improvement | PENDING GPU | no promotion manifest |
| P11 BGE-M3 rescue experiment | DEFERRED | no model download/index; trigger only after P12/P13/P15 plateau |
| P12 hard/semi-hard negatives | P12_FULL_PENDING | code-ready miner with canonical positive evidence; full 7,000-query Kaggle run pending |
| P13 BGE reranker fine-tuning | KAGGLE_GPU_SMOKE_PENDING | fixed-epoch strict OOF trainer and prerequisite gate; real GPU smoke/full OOF pending |

## Nhật ký các task đã thực hiện

| Task | Trạng thái | File/artifact | Mục đích và ảnh hưởng runtime | Ảnh hưởng score / rủi ro | Kiểm tra |
|---|---|---|---|---|---|
| Contract/retrieval foundation | DONE | `src/udsc2026/contracts`, `src/udsc2026/retrieval` | typed chunks/hits and retrieval adapters | enables retrieval; contract drift remains a risk | retrieval/adapter tests |
| Index versioning | DONE | `scripts/data_prep/index_chunks.py`, dense manifests | corpus/model hash, dimension, collection, commit | prevents stale index reuse | `test_index_versioning.py` |
| Resilient embedding | DONE | embedding client and index script | batch/windowed embedding and manifests | heavy GPU indexing; no new score claim | embedding/index smoke |
| BM25/tokenizer regression | DONE | sparse BM25/tokenizer modules | deterministic Vietnamese lexical retrieval | sparse index must match corpus | BM25/tokenizer tests |
| Top-k guards | DONE | rerank/evaluation CLIs | validates candidate depth/top-n | prevents invalid scoring shapes | reranker tests |
| Data audit | DONE | audit/mapping scripts | verifies 8,532 docs, 184,548 parents, 1,270,356 chunks | audit is not relevance score | `artifacts/tv2/data_audit/*` |
| Dense artifacts | DONE | dense candidate CLI/manifests | cached DEk21 candidates at depth 200 | reuse only matching hashes | manifest validation |
| Ensemble/CV | DONE | ensemble builder and strict CV builder | word KNN/BM25/dense/BGE/RRF and OOF structure | leakage risk controlled by grouped folds | strict CV/ensemble tests |
| P1 | DONE | evaluator, CLI, diagnostics | official semantics and zero-metric plumbing | no model change | P1 report and unit gate |
| P2 | DONE | candidate analyzer/diagnostics | document coverage and early collapse | cached 500-question diagnostic | Recall@200 0.965; unit/data gate |
| P3 | EXPERIMENTAL | lexical module and ablation CLI | char KNN, BM25F-like fields, citation parser/cache | no validated full-CV score; full corpus CPU run timed out | synthetic tests/unit gate |
| P4 | EXPERIMENTAL | multigranularity module and plan CLI | versioned child/parent text and parent→doc fusion | no full parent GPU index/OOF score | 10-record plan/unit gate |
| P5 | EXPERIMENTAL | document candidate module | union, evidence selection, reranker interface | no full GPU score | 200-chunk/3-doc tests |
| P6 | DONE | `scripts/task1/run_legal_ir_pipeline.py`, baseline config | resumable orchestration/manifests/submission hand-off | missing neural cache is reported, not hidden | dry-run/unit gate |
| P12 | P12_FULL_PENDING | `scripts/training/mine_task1_negatives.py` | canonical-evidence, leakage-safe hard/semi-hard dataset | full GPU candidate cache/mining pending | miner unit tests + manifest |
| P13 | KAGGLE_GPU_SMOKE_PENDING | `scripts/training/finetune_task1_bge_reranker.py` | fixed-epoch strict five-fold document-evidence trainer | GPU/Kaggle only; no production change before OOF | mock trainer, fold-leakage, serialization tests |
| P16 | DONE | [`tv2_task1_kaggle_p12_p13.md`](tv2_task1_kaggle_p12_p13.md) | temporary P12/P13-only GPU runbook | public/submission stage deferred | doc/CLI checks |

## Current model registry

| Role | Model | Local path | Actual state |
|---|---|---|---|
| Dense | `huyydangg/DEk21_hcmute_embedding_v2` | `models/dek21-v2` | validated; revision `99a2963b2f51fa7a570a3e7f550d7993b9de90a8` |
| Task1 reranker | `BAAI/bge-reranker-v2-m3` | `models/reranker` | validated GPU artifact; revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Legal generator | `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2` | `models/qwen3-legal` | registered QA model; not promoted as Task1 retrieval/reranker |

Source: [`model_registry.md`](../../models/model_registry.md).

## Artifact map

| Family | Producer/input | Purpose | Rebuild/delete | Submission impact |
|---|---|---|---|---|
| `artifacts/tv2/data_audit/*` | TV2 audit over processed corpus | integrity/mapping | safe to rebuild/delete | none |
| `artifacts/task1/*dense*` | DEk21 + indexed chunks | cached candidates | hash-dependent rebuild | indirect |
| `artifacts/task1/*reranker*` | TV5 BGE + candidates | rerank diagnostics | rebuildable | indirect |
| `artifacts/task1/cv_strict/*` | strict CV + train labels | OOF folds/reports | rebuildable | none until promoted |
| `artifacts/task1/evaluation/*` | P1–P6 scripts/tests | diagnostics/contracts | rebuildable | none |
| `submission.zip` | existing submission writer | package with only `submission.json` | regenerate | direct |

## Measured results

P2 cached smoke (`train500_dense200`, 500 questions) measured:

| Metric | Value |
|---|---:|
| CandidateDocRecall@5 | 0.769833 |
| CandidateDocRecall@100 | 0.946000 |
| CandidateDocRecall@200 | 0.965000 |
| Mean unique docs at depth 200 | 61.258 |
| Retrieval misses | 11 |

These are validation diagnostics, not a leaderboard claim. Public transport
rows containing `LABEL_NOT_AVAILABLE` are not valid gold evaluation.

## Validation commands

```powershell
python -m compileall -q src scripts
git diff --check
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier unit
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-tv2-task1.ps1 -Tier data
powershell -ExecutionPolicy Bypass -File scripts/ci_cd/check-all.ps1 -SkipFrontend -SkipCompose
```

The first four are local light/medium checks. The final command is the full
project pre-push check and may require additional dependencies/runtime.

## Sửa lỗi Windows trong pytest

`check-all.ps1` trước đây dùng cố định `.pytest_runtime` làm thư mục tạm.
Trên Windows, handle hoặc quyền còn sót từ lần chạy trước có thể khiến pytest
không xóa được thư mục và tạo hàng loạt lỗi `PermissionError: [WinError 5]`.

Script hiện tạo một thư mục `basetemp` riêng theo GUID cho mỗi lần chạy. Cách
này không đụng vào corpus hay artifact, tránh xung đột giữa các lần chạy và đã
được xác nhận bằng kết quả `854 passed, 10 skipped` cùng coverage `85.63%`.

## Boundary for future promotion

P3/P4/P5 must not be described as score improvements until a robust strict-CV
before/after report and matching manifest exist. Neural/index changes also need
a GPU run. P16 now has a cell-by-cell Task1 runbook; it does not itself claim a
GPU score or change production configuration.
