# B2a-0 Long-document root-cause audit

Status: **PASS**

Canonical path: `reports/task1/workflow_b/b2a/audit`

Unlabeled CPU/tokenizer forensic audit. The frozen Qwen format and revision are unchanged; no model inference, labels, Fold0, public data, truncation, chunking, or max-length change was used.

Pairs audited: **431200**; unique K77 documents: **8405**

## Extreme pair diagnostics

| marker | fold | query | document | K77 rank | total tokens | query tokens | document tokens | overhead | query chars | document chars |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| p50 | 1 | 100844 | 288065 | 67 | 18455 | 49 | 18358 | 48 | 85 | 60674 |
| p90 | 3 | 11002 | 141634 | 6 | 64539 | 42 | 64449 | 48 | 59 | 218000 |
| p95 | 1 | 100268 | 21398 | 19 | 95289 | 55 | 95186 | 48 | 108 | 331410 |
| p99 | 4 | 100306 | 102434 | 55 | 158503 | 59 | 158396 | 48 | 132 | 538064 |

Top-100 longest pairs are serialized in JSON (diagnostics only; no full legal text). Extreme maximum pair: **2145502 tokens**.

## Distribution

Document-only distribution (tokens): min **3**; p50 **7388.0**; p90 **28392.6**; p95 **42347.2**; p99 **95378.84**; p99.5 **116020.84**; p99.9 **204524.804**; max **2145387**.

Document-only thresholds (count; percentage): >8192 **3859; 45.913147%**; >16384 **1829; 21.760857%**; >32768 **668; 7.947650%**; >65536 **184; 2.189173%**; >131072 **31; 0.368828%**.

Pair thresholds (count; percentage): >8192 **338764; 78.563080%**; >16384 **236799; 54.916280%**; >32768 **125568; 29.120594%**; >65536 **42561; 9.870362%**; >131072 **9278; 2.151670%**.

Pairs >32768: **125568 (29.120594%)**
Unique docs >32768: **668 (7.947650%)**
Docs covering 50/80/95% of >32768 pairs: **131/323/520**

## Root-cause checks

Duplicate large blocks in top100: **100**; repeated-ratio ≥25%: **100**; same text under multiple IDs touching top100: **0**.
Primary classification: **D_MIXED_REAL_AND_PIPELINE_PROBLEM**.

The top-100 pairs involve two unique documents (`68843`, `4644`); both have exact duplicate large blocks and repeated-text ratio ≥25%. The read-only offending source is `data/raw/btc/LegalIR/selected-contexts/context_<document_id>.json`, field `passage`; the construction function was not identified. No join-key mismatch was observed. This is recorded for professor review; no repair was made.

## Scientific consequence (Tiếng Việt)

Toàn bộ tài liệu không khả thi với Qwen 32K theo hợp đồng hiện tại vì p99 vượt 32K rất xa. Audit CPU đã PASS nhưng cổng khả thi bị BLOCKED_LONG_DOCUMENT; cần một thí nghiệm/hợp đồng preregister mới để xử lý tài liệu dài. Chưa có đánh giá ngữ nghĩa, nên không được kết luận reranker thất bại. GPU smoke tiếp tục bị chặn.

No repair or redesign was performed.
