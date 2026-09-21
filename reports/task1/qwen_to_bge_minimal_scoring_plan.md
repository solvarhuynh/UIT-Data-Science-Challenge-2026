# Qwen → BGE minimal scoring plan

## Decision

`QWEN_TO_BGE_MINIMAL_WORKLIST_READY`. This is a CPU/read-only planning audit. No BGE inference, GPU, Modal, Qwen inference, or submission was run.

The Qwen teacher is used only to define bounded TOP5/TOP10/TOP20 discovery worklists. Final ranking must use BGE scores only; `teacher_qwen_score` is retained solely for provenance.

## Population and namespace

- F1–F4 union: **5600 queries**, **1120000 pairs**, candidate count min/median/max = **200/200.0/200**.
- Union row fields: `candidates, query_id`; candidate fields: `doc_id, rrf_score, source_ranks, source_support, union_rank`. The actual document field is `doc_id`, not `document_id`.
- Duplicate `(query_id, doc_id)` identities: **0**.
- Fold0 IDs in union: **NO** (0); Fold0 was excluded completely.
- Qwen SHA256: `25cc4b7dfb5919a47f927c545927e775c11997790984b519e2b74a71abce0415`; expected match: **YES**.
- Qwen rows/unique q-docs/queries/non-finite/duplicates: **598192/598192/5600/0/0**.
- Qwen ↔ F1–F4 query intersection: **5600/5600**.

## Offline Qwen discovery accounting

| Depth | Qwen pairs | In Step4 Top200 | QWEN_DISCOVERED_NEW | Affected queries | Unique new docs | Reusable BGE | Missing BGE | Final missing worklist | % of 1,120,000 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TOP5 | 28000 | 2504 | 25496 | 5594 | 6021 | 57 | 25439 | 25902 | 2.3127% |
| TOP10 | 56000 | 4622 | 51378 | 5600 | 7043 | 96 | 51282 | 51745 | 4.6201% |
| TOP20 | 112000 | 8373 | 103627 | 5600 | 7656 | 171 | 103456 | 103919 | 9.2785% |

Baseline slot4/5 pair set: **11200**; reusable BGE scores: **10737**; missing: **463**.

## Reusable BGE score inventory

The only source counted as reusable for this F1–F4 plan is `D:\udsc2026\artifacts\task1\recovery_096\baseline_093_oof\sources\f1to4_original_bge_chunk200_compact.jsonl`. It has **1120000** chunk-score rows and **351762** unique `(query_id, doc_id)` pairs. Document-level reuse is `MAX(hits[].bge_score)` over available chunk hits.

The source schema is `query_id`, `fold`, `gold_documents`, `hits`; each hit has `doc_id`, `chunk_id`, `dense_rank`, `dense_score`, `bge_score`. This audit did not read `gold_documents` and did not use labels. The persisted artifact records model family/id and local path, but does not connect its scores to the requested immutable revision or exact weight hash; this is marked `MODEL_REVISION_UNPROVEN_FOR_PERSISTED_SCORE_ARTIFACT`.

Union-wide reusable q-doc coverage: **347056/1120000 (30.9871%)**. Relevant TOP20-new + baseline slot4/5 pairs: **114824**, reusable **10905**, missing **103919**.

Other BGE-looking artifacts were not mixed into the inventory: Fold0 compact scores are the wrong population; public BGE scores are the wrong namespace; selective-k500/BM25-grounded scores use different worklists and recorded model hashes; Step4 swap reports contain only public normalized evidence, not a complete F1–F4 raw q-doc score table.

## Step4 Modal scorer audit

`modal_step4_ensemble.py` is **not** minimal-subset capable as written. It hard-codes public question, baseline, union, and evidence locations, iterates all public queries, and only exposes `shortlist_top_n`/fusion parameters. It scores both FT and Base BGE on an L40S at batch 64, then performs per-query min-max normalization over the selected shortlist. It does not accept these generated worklists or persist a complete raw score table.

Required future patch (not made): accept a frozen `(query_id, document_id, evidence)` worklist, bypass public hard-coded discovery, and persist model-specific q-doc scores before fusion.

## Public reproducibility audit

`PUBLIC_FULLDOC_CANDIDATE_GENERATION = NO`. The current FULLDOC runner consumes frozen train/F1–F4 worklists and a future-universe manifest; `build_bounded_union_200_candidate_refs.py` only combines existing public retrieval sources. No deterministic public FULLDOC rank/worklist generator for `data/raw/btc/LegalIR/public-official.json` is present. Therefore no public deployment policy can be instantiated from this audit.

## Execution plan

1. Run **TOP5** first on the generated missing-BGE worklist only.
2. Lock baseline ranks 1–3; compare baseline slots 4/5 against Qwen-discovered candidates using BGE features only.
3. Permit at most one replacement/query initially; validate `candidate BGE score > weakest current slot` and a margin selected on F1–F4 only.
4. Open TOP10 only if TOP5 is scientifically useful; open TOP20 only if TOP10 remains useful. Qwen score/rank is never a deployed ranking feature.

## Generated artifacts

- `D:\udsc2026\artifacts\task1\qwen_to_bge_minimal\summary.json`
- `D:\udsc2026\artifacts\task1\qwen_to_bge_minimal\reusable_bge_score_inventory.json`
- `D:\udsc2026\artifacts\task1\qwen_to_bge_minimal\top5_missing_bge.jsonl`
- `D:\udsc2026\artifacts\task1\qwen_to_bge_minimal\top10_missing_bge.jsonl`
- `D:\udsc2026\artifacts\task1\qwen_to_bge_minimal\top20_missing_bge.jsonl`
