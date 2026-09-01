# B2a-0 Passage provenance and long-document corruption map

Status: **BLOCKED** — `BLOCKED_PASSAGE_PRODUCER_UNRESOLVED`

Canonical output directory: `reports/task1/workflow_b/tv2/b2a/audit`

## Boundary reconciliation

The canonical operator is **STRICT_GT_32768**. Document-only count is **668**; `==32768` count is **0**. Pair total length has **125568** rows and touches **669** documents above 32768. The historical 668/669 difference is resolved: document-only length excludes query/template content, while pair length includes it; document `135644` has document-only length 32669 but reaches 32796 tokens in at least one formatted pair.

## Passage provenance

Repository code consumes `data/raw/btc/LegalIR/selected-contexts/context_<id>.json`, field `passage`; it does not contain a producer that generates, appends, paginates, deduplicates, or concatenates those files. The available documentation identifies `selected-contexts.zip` as BTC-distributed source data. Therefore source-vs-generator attribution is **UNRESOLVED** and no data repair is authorized.

## Corruption map

Long documents audited: **668**. Likely genuine long documents: **588**. Mixed/ambiguous: **80**. Confirmed pipeline/upstream corruption: **0 / 0**.

A long document is not automatically faulty: legal annexes, schedules, tables, and consolidated instruments can be legitimately long. `68843` and `4644` retain strong duplicate-block/boilerplate signals, but the available evidence only proves that they are present in the released `passage` field—not whether they were introduced upstream or by an unavailable generator.

Supplemental verification with the historical normalized-paragraph rule (minimum 120 characters) reproduces **296** duplicate large blocks for `68843` and **609** for `4644`. The population map uses the stricter >=512-character rule stated in JSON to avoid treating short repeated legal phrases as corruption.

Safe deterministic repair candidates: **0**. Dry-run projection without any repair: **125568 pairs (29.120594%)** and **668 unique documents** remain above 32K. GPU and chunking remain out of scope; no Qwen inference or scientific metric was run.

Next: **NEEDS_MORE_PROVENANCE**.
