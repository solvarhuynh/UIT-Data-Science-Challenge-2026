# TASK1_DATA_PROVENANCE_AUDIT

Audit này chỉ đọc dữ liệu/code cục bộ trong `D:\udsc2026`. Không chạy GPU,
Modal, inference, web fetch, label-based model selection, hoặc sửa ingestion/data.

## STATUS

`DATA_PIPELINE_HAS_ISSUES`

Integrity của corpus hiện tại đạt, nhưng semantic/quality gate chưa đạt vì có 20
BTC context rỗng và 1.359 document cần manual review. Không có bằng chứng về
document ngoài lineage BTC hoặc legal text bị tự bịa.

## RAW

Canonical raw root: `data/raw/btc`.

| Hạng mục | Kết quả |
|---|---:|
| Tổng physical files dưới raw root | 17.070 |
| Physical `context_*.json` | 17.064 |
| Unique context/document IDs | 8.532 |
| BTC QA fixture JSON files | 4 |
| BTC overview DOCX files | 2 |
| LegalIR contexts | 8.532 |
| LegalQA contexts | 8.532 |
| Exact duplicate ID groups | 8.532 |
| Conflicting duplicate groups | 0 |
| Exact duplicate physical copies | 8.532 |
| Context files có link | 17.064/17.064 |
| Context files thiếu link | 0 |
| Duplicate source URLs | 8.532 cross-task copies; không có URL trùng giữa các unique IDs |
| Zero-byte context files | 0 |
| Malformed JSON | 0 |
| Missing required `id`/`passage`/`link` field | 0 key-level; 40 physical rows có passage rỗng |

Tất cả 17.064 link trỏ tới domain `thuvienphapluat.vn`. Raw manifest:
`data/processed_v3/metadata/manifest.json` ghi `physical_context_file_count=17064`,
`accepted_context_count=8532`, `exact_duplicate_context_count=8532`,
`conflicting_duplicate_context_count=0`, `orphan_context_count=0`.

Các duplicate là bản LegalIR/LegalQA cùng context, nội dung giống hệt; extractor
giữ bản canonical LegalIR và không đưa bản LegalQA thứ hai vào processed corpus.

## SOURCE_FETCH

`EXTERNAL_FETCH_USED = NO` trong code/artifact evidence hiện có.

- Không tìm thấy downloader/crawler/source-reconstruction script cho context.
- Không tìm thấy `requests`, `httpx`, `urllib` fetch, Selenium, Playwright,
  `wget` hoặc `curl` trong ingestion/data-prep path.
- `source_link` chỉ được đọc vào metadata; không được dùng để tải passage.
- Downloaded documents có provenance chứng minh được: `0`.
- Source domain quan sát được: `thuvienphapluat.vn`.
- Mapping raw context → BTC link: `17.064/17.064`.
- Document không có original BTC linkage: `0`.

Vì vậy audit này không thể chứng minh một thao tác tải ngoài repo trong lịch sử
nếu thao tác đó từng được thực hiện thủ công. Evidence trong repository chỉ chứng
minh passage hiện có đến từ raw BTC record; 20 passage rỗng chưa được supplement.

## SYNTHETIC_OR_MANUAL

Không thấy LLM generation, hallucinated legal text, hard-coded legal passage,
manual replacement map, hoặc text augmentation trong ingestion path.

Phân loại text không nguyên văn raw:

- `SOURCE_TEXT`: passage raw BTC; các supplemental chunks chỉ cắt lại vùng text
  chưa nằm trong article, vẫn lấy từ cùng `cleaned_text`.
- `NORMALIZATION_ONLY`: NFC/OCR-safe replacements, whitespace và line-wrap.
- `STRUCTURAL_RECONSTRUCTION`: parser state machine, repaired split headings,
  parent/child metadata và fallback segmentation; không thêm câu luật.
- `SYNTHETIC_CONTENT`: 20 placeholder chunks/parents cho 20 context rỗng,
  chứa nhãn `Tên văn bản: ...`, được đánh dấu `synthetic_placeholder=true` và
  không được coi là căn cứ pháp luật.
- Synthetic benchmark 100 rows nằm riêng trong
  `data/processed_v3/benchmarks/synthetic_qa.jsonl`, có provenance tới source
  chunk corpus; không phải corpus legal content.
- `MANUAL_CONTENT`: không có evidence.

## HISTORICAL_FAILURES

Artifact membership chính xác mang tên “~1.355” không được tìm thấy. Evidence
reproducible hiện có là:

- Git snapshot `0a7674d...`: manual-review list có 987 IDs.
- Git snapshot `ed6c6e0...`: manual-review list có 114 IDs.
- Current canonical V3: `data/processed_v3/metadata/manual_review_documents.json`
  có 1.359 IDs.

Do đó không được tuyên bố 1.359 chính là cùng population với narrative 1.355.

Phân loại primary, áp dụng cho current canonical list 1.359:

| Group | Count | Diễn giải | Trạng thái hiện tại |
|---|---:|---|---|
| GROUP_1_NO_ARTICLE_BY_NATURE | 491 | `fallback_chunkable_no_article=305` + `no_article_structure=186` | Có fallback chunks; vẫn cần review cấu trúc |
| GROUP_2_CLEANER_REMOVED_STRUCTURE | 0 | Audit không thấy dòng Điều/Khoản/Điểm bị cleaner xóa | Không có case được chứng minh |
| GROUP_3_OCR_OR_FORMATTING | 848 | `regex_recoverable=797` + `ocr_noise=51` | Có fallback/diagnostic coverage; cấu trúc cần review |
| GROUP_4_SOURCE_PASSAGE_MISSING | 20 | BTC passage rỗng từ nguồn | Chỉ giữ placeholder, chưa phục hồi substantive text |
| GROUP_5_OTHER | 0 | Không có bucket khác trong canonical breakdown | — |
| GROUP_6_UNRESOLVED | 0 trong current list | Không unresolved trong primary breakdown; membership lịch sử 1.355 vẫn unavailable | Historical population gap |

Full IDs, không rút gọn:

- All 1.359: `data/processed_v3/metadata/manual_review_documents.json`
- Group 1/3 breakdown: `data/processed_v3/metadata/manual_review_breakdown.json`
- Regex group: `data/processed_v3/metadata/manual_review_regex_candidates.json`
- OCR group: `data/processed_v3/metadata/manual_review_ocr_noise.json`
- Empty-source group: `data/processed_v3/metadata/manual_review_cleaner_cleared_text.json`

Fixes are deterministic and source-controlled; no manual edit evidence was found:

- Cleaner: `normalize_unicode_and_ocr`, `_remove_repeated_noise`,
  `merge_hard_wrapped_lines`; only deterministic page/TOC/header rules.
- Parser: `_logical_lines` and state-machine parser repair high-confidence split
  headings and preserve ordered legal structure.
- Chunker: `_chunk_manual_review_document` and `_chunk_unstructured_document`
  preserve searchable text as flagged fallback chunks.
- Empty source: `_chunk_empty_source_document` preserves ID and creates an
  explicit non-evidentiary placeholder; it does not claim to repair the passage.

## SURVIVAL

| Set comparison | Count |
|---|---:|
| RAW unique IDs | 8.532 |
| CLEANED IDs | 8.532 |
| CHUNKED IDs | 8.532 |
| RAW - CLEANED | 0 |
| CLEANED - CHUNKED | 0 |
| RAW - CHUNKED | 0 |

No document disappeared silently. `extract_errors.json` and `clean_errors.json`
are empty; `orphan_contexts.json` reports zero orphans. All 20 empty raw passages
survive as explicit `empty_source_placeholder` records, not as deleted documents.

## ARTICLE_PRESERVATION

`validation_report.json` reports 150.415 detected articles, 11.647 repaired split
articles and 594 suspected split articles. A line-anchored raw-vs-cleaned marker
check over the 8.532 LegalIR documents found:

- raw article-marker documents: 7.183;
- processed article-marker documents: 7.189;
- raw marker documents becoming processed marker-free: 0;
- raw marker occurrences: 151.359; processed marker occurrences: 151.803.

The marker check is diagnostic rather than a legal parser proof, but it finds no
case of the cleaner removing all visible article markers. Source assignment audit
also reports `assigned_source_line_count=source_nonempty_line_count=3.294.221`
and zero missing source tokens/bigrams.

## PASSAGE_COMPLETENESS

The 20 unique affected IDs are:

`10533, 131890, 149317, 177151, 181693, 187338, 191261, 196918, 208668,
210808, 232489, 255762, 263763, 288457, 34810, 55497, 56098, 57978, 67660,
71014`.

All 20 have a BTC `link`; all have empty raw `passage`; source fetch is not
present; supplementation is `NO`; final status is explicit placeholder. This is
`BTC_PASSAGE_INCOMPLETE`, not `OUR_CLEANER_LOST_PASSAGE`: removed-line count is
zero for these documents and the raw record is already empty.

## CHUNKS

Canonical processing metadata:

- documents: 8.532;
- chunks: 1.270.356;
- parents: 184.548;
- structure types: article 1.066.019, document_context 139.912,
  unstructured_fallback 64.405, empty_source_placeholder 20;
- empty chunks/parents: 0;
- duplicate chunk IDs: 0;
- orphan chunks: 0;
- cross-document parents: 0;
- non-null child `parent_text`: 0;
- missing metadata: 0;
- chunk contract: 192 tokens, overlap 32;
- source lines assigned: 100%; missing source tokens/bigrams: 0.

The implementation is not blind character chunking. It parses
document/chapter/section/article/clause/point order, chunks article units by
sentence boundaries, uses token boundaries only for overlong sentences, and
adds flagged same-source fallback/document-context chunks for uncovered source
regions. Tables/annexes or non-article regions are not silently deleted; they
are represented as `document_context` or fallback segments when outside parsed
article spans.

## PRIVATE_CORPUS

`PRIVATE_CORPUS_PROVENANCE = PASS_WITH_INDEX_HASH_CAVEAT`.

Evidence:

- private input: 2.080 query IDs, answer fields null;
- candidate union: 2.080 rows, 132.376 unique q-doc pairs, 8.180 unique docs;
- all candidate docs resolve to the 8.532 processed V3 documents;
- private retrieval index manifest reports the same 1.270.356 chunk count;
- the private asset registry points to `data/processed_v3` and its processing
  tree hash.

The FAISS manifest has its own index `corpus_hash/source_fingerprint`, not the
V3 `processed_corpus_tree_hash`, so exact byte-level equivalence of the old FAISS
build tree is not independently proven by the available manifests. This is a
lineage caveat, not evidence of an external or synthetic private document.

## CURRENT_ISSUES

1. Twenty BTC contexts are substantively empty. They have links, but no local
   source-fetch/supplement artifact is present; retrieval evidence for these IDs
   is therefore only a placeholder.
2. 1.359 documents remain manual-review flagged. The text is retained through
   fallback/structured recovery, but article-level semantics are not uniformly
   proven.
3. `semantic_completeness_gate_passed=false` and `quality_gate_passed=false`.
   `integrity_gate_passed=true`.
4. The exact historical ~1.355-ID list is not recoverable from the current repo;
   old snapshots are 987 and 114, while current V3 is 1.359.
5. Git history does not track the large raw/V3 physical trees, so a historical
   accidental physical-file deletion/addition audit cannot be proven beyond the
   current manifest/set equality. Current RAW/CLEANED/CHUNKED sets are exact.

## FILES_AT_RISK

- Empty-source IDs: see `manual_review_cleaner_cleared_text.json` and the list above.
- Regex/OCR/manual-review IDs: see the three canonical manual-review JSON reports.
- No unexplained processed document IDs were found.
- No processed chunk with an untraceable source path was found in the disk audit.

## CONCLUSION

`DATA_PIPELINE_HAS_ISSUES`.

The pipeline is provenance-safe with respect to the evidence available: raw
lineage is BTC-only, no external fetch or manual legal-text edit is evidenced,
all current documents survive into chunking, and the integrity audit is clean.
It is not safe to call fully clean/completely complete because 20 source
passages are empty, 1.359 documents remain review-flagged, and the exact old
~1.355 membership plus historical raw-file deletion history are unavailable.

## NEXT_SAFE_ACTION

Do not modify the current corpus or GPU validation. If recovery is authorized,
create a separate, provenance-recorded supplement audit for the 20 empty BTC
context IDs using only their BTC-provided links, then rerun read-only integrity
and semantic gates before any corpus promotion or production decision.

`GPU_RUNS_THIS_TASK: 0`

`MODAL_INFERENCE_THIS_TASK: 0`
