# HARD-60 historical membership resolution

## Status

`BLOCKED_HARD60_MEMBERSHIP_UNRESOLVED`

## Known-66 provenance

- **source:** `artifacts/task1/recovery_096/inference_matched_reranker_v1/compact/candidate_refs_full.jsonl`
- **SHA256:** `e09a59852b722fc6e33761cce964920e4929cba59f0171dcd85cbe62d96edfa6`
- **construction:** F1–F4 gold `(query_id, document_id)` occurrences absent from each query's canonical candidate pool of at most 200 documents.
- **count:** 66
- **duplicate q-doc count:** 0
- **fold distribution:** F1 10; F2 12; F3 26; F4 18.

This is an exact deterministic reconstruction of the known-66 set from the preserved canonical pool artifact. No approximate membership was used.

## Historical recoverable-six

No persisted `hard60`/`hard_60` membership list, no identity-level dense-K500 recovery list, and no identity-level historical BM25/word-KNN/char-KNN recovery list was found in the repository artifacts, reports, or scripts searched.

The only surviving identity-level compatible source is the original BGE compact artifact:

- `artifacts/task1/recovery_096/baseline_093_oof/sources/f1to4_original_bge_chunk200_compact.jsonl`
- SHA256 `7286ec481d26fc134711ecb03ceac2afe1a03bf0dabd842e1ed29ffb074e85c8`
- It contains exactly **one** of the known-66 q-doc identities.

The historical assertion requires six unique prior recoveries (three raw-dense K500 and three different BM25; word/char zero). The remaining five exact q-doc identities cannot be recovered from the surviving evidence. Therefore the required check `|dense ∪ BM25 ∪ word ∪ char| = 6` cannot be verified.

## Why the preliminary set had 65

- **proxy definition used:** `KNOWN66 − q-doc identities present in preserved BGE compact source`
- **result:** 65 rows, because exactly one known-66 q-doc is BGE-present.
- **five-row discrepancy:** the five additional historical recoveries asserted by the prior dense/BM25 analysis are not represented by a persisted identity-level source trace. This is a **different-definition/provenance-loss** issue, not a duplicate or a retrieval result.
- **resolved:** no. The 65-row proxy is invalid as HARD60.

## Gate application

The already-frozen full-document retrieval output was not rerun. Its known-66 result remains 12 recovered at rank ≤200. The original hard-60 gate cannot be applied because the exact six rows to subtract are unrecoverable.

No `hard60_exact_membership.jsonl` was created: producing one would falsely claim a set equation that cannot be evidenced.

## Safe next action

`RESOLVE_EXACT_HISTORICAL_MEMBERSHIP_BEFORE_ANY_NEW_EXPERIMENT`
