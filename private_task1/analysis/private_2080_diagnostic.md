# Private 2,080 Diagnostic Audit

## Scope and guardrails

- CPU/read-only audit only; no GPU, Modal, retrieval rerun, corpus edit, submission edit, weight tuning, or label overlay.
- Private leaderboard score `0.9168` is reported context only, not used as ground truth.
- F1?F4 calibration uses the existing validation artifacts and `train.json` answers only; Fold0 is excluded.

## Input and reproduction gate

- Private queries: **2080/2080**
- K20 q-docs: **41600/41600**
- Candidates/query: **20/20.0/20 (min/median/max)**
- Missing BGE scores: `0`; duplicate q-docs: `0`; non-finite scores: `0`.
- Final ranking reproduced exactly: **YES** (`2080/2080` queries, 5 docs/query).

## Diagnostic definitions

- `LOW_MARGIN`: top1?top2 margin ? F1?F4 P10 `0.01000000` or top5?top6 margin ? F1?F4 P10 `0.00119048`.
- `HIGH_DISAGREEMENT`: final Top5 mean Dense/BM25/KNN support ? F1?F4 P10 `1.6000`, or single-source Top5 count ? F1?F4 P90 `2.00`, or final Top1 has ?1 retrieval source.
- `STRONG_BGE_REORDER`: BGE Top5 differs from frozen retrieval Top5 and max absolute candidate-rank movement ? F1?F4 P90 `17.00`.
- Groups are mutually exclusive for counting: D (retrieval/corpus-risk) ? C (reranking-sensitive) ? B (ranking uncertainty) ? A (stable). They are diagnostic groups, not error labels.

## Private diagnostic summary

- GROUP_A_STABLE: **1357**
- GROUP_B_RANKING_UNCERTAINTY: **306**
- GROUP_C_RERANK_SENSITIVE: **375**
- GROUP_D_RETRIEVAL_CORPUS_RISK: **42**

- RRF top1?top2 margin: mean `0.07369769`.
- RRF top5?top6 margin: mean `0.01143700`.
- BGE changed retrieval Top5 composition: `2066/2080` queries.
- BGE moved final Top5 documents upward: `4697` document occurrences.
- Retrieval Top5 documents pushed down by BGE: `7414` occurrences.
- Median absolute rank movement: `4.50`; P90 query max movement: `17.00`.

### Source agreement

- Mean final Top5 coverage by source: Dense `2.86`, BM25 `3.16`, KNN `1.70`, BGE Top5 `3.39`.
- Mean pairwise Top5 overlaps: dense?bge `2.28`, dense?knn_word `0.94`, dense?bm25 `2.11`, bge?knn_word `1.11`, bge?bm25 `2.17`, knn_word?bm25 `1.06`.
- Final Top1 retrieval-source agreement distribution: 1 sources=11, 2 sources=722, 3 sources=1347.
- Across all 41,600 candidates, retrieval support counts: 1-source=890, 2-source=32213, 3-source=8497.
- Across final Top5 positions, retrieval support: 1-source=156, 2-source=6046, 3-source=4198.
- Source dominance by accumulated final-Top5 RRF contribution: bge=1292, bm25=788.

## Existing F1?F4 calibration

- Calibration status: **PASS**; validation queries `5600` and all are folds F1?F4; Fold0 used: `NO`.
| Diagnostic | Flagged queries | Flagged macro Recall@5 | Unflagged macro Recall@5 | Flagged zero-hit rate | Unflagged zero-hit rate |
|---|---:|---:|---:|---:|---:|
| LOW_MARGIN | 1085 | 0.878449 | 0.935585 | 0.100461 | 0.047398 |
| HIGH_DISAGREEMENT | 1188 | 0.842312 | 0.946649 | 0.135522 | 0.036718 |
| STRONG_BGE_REORDER | 587 | 0.893384 | 0.928160 | 0.081772 | 0.054857 |
| SINGLE_SOURCE_DOMINANCE | 514 | 0.710117 | 0.946182 | 0.266537 | 0.036571 |

These are associations on existing validation gold, not proof that any Private query is wrong. No calibration result is used to modify the submission.

## Corpus-quality audit of manual sample

- Manual sample: `30` queries, groups B/C/D where available; `140` unique Top5 documents.
- Documents with at least one heuristic quality signal: **115**.
- Queries containing at least one flagged document: **30**.
- Signals checked: missing/empty selected context, very short full document text, replacement/control characters, duplicate chunk text, recorded structure warnings, and explicit truncation markers.
- These are audit flags only; no corpus file was changed.

## Manual-review sample

### Query `34762` ? GROUP_B

Question: Trên trang trong của báo chí có các nội dung phải ghi nào?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=False; top1-top2 margin `0.00000000`, top5-top6 margin `0.00062729`, final Top5 support mean `2.60`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `19627` | 0.28333333 | 0.73586702 | 1 | 2 | 1 | 2 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Luat-Bao-chi-2016-280645; preview: 1. Trên trang nhất, bìa một đối với báo in, trang chủ, các trang đối với báo điện tử phải có các nội dung sau đây: |  |  |  |  |  |  |  |  |
| 2 | `177510` | 0.28333333 | 0.83963209 | 2 | 1 | 4 | 1 | 3 | duplicated_chunk_text |
|  | title: Nghi-dinh-51-2002-ND-CP-huong-dan-Luat-Bao-chi-Luat-sua-doi-bo-sung-mot-so-dieu-cua-Luat-Bao-chi-493; preview: Chương 5: Điều 15. Các nội dung phải ghi trên trang một, bìa một, trang trong của báo và tạp chí |  |  |  |  |  |  |  |  |
| 3 | `24232` | 0.11706385 | 0.12334464 | 15 | 3 | 7 | 11 | 3 | duplicated_chunk_text |
|  | title: Thong-tu-41-2020-TT-BTTTT-cap-giay-phep-hoat-dong-bao-in-tap-chi-in-va-bao-dien-tu-tap-chi-dien-tu-4; preview: 1. Cơ quan, tổ chức có nhu cầu thay đổi một trong các nội dung ghi trong giấy phép xuất bản đặc san phải gửi văn bản đề nghị Cục Báo chí cho phép thay đổi. Hồ sơ gồm có: |  |  |  |  |  |  |  |  |
| 4 | `225856` | 0.11000000 | 0.47141439 | 3 | - | 18 | 3 | 2 | none |
|  | title: Quyet-dinh-1418-QD-BTTTT-2022-tieu-chi-nhan-dien-bao-hoa-tap-chi-va-trang-dien-tu-tong-hop-523570; preview: 1. Về hình thức: - Mẫu trình bày tên gọi ấn phẩm, mẫu trình bày giao diện trang chủ không ghi hoặc ghi “tạp chí” rất nhỏ gây khó khăn trong việc nhận biết hoặc gây hiểu nhầm cho bạn đọc. - Chuyên trang nhưng mẫu trình bà |  |  |  |  |  |  |  |  |
| 5 | `201003` | 0.10083333 | 0.39373627 | 13 | 6 | - | 4 | 2 | duplicated_chunk_text |
|  | title: Luat-Quang-cao-2012-142541; preview: 4. Trên trang một của phụ trương quảng cáo phải ghi rõ các thông tin sau: |  |  |  |  |  |  |  |  |

### Query `103502` ? GROUP_B

Question: Thời hạn nộp lệ phí môn bài theo quy định của pháp luật là khi nào?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=False; top1-top2 margin `0.00023810`, top5-top6 margin `0.00068182`, final Top5 support mean `2.60`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `181088` | 0.21619048 | 0.75112838 | 4 | 5 | 3 | 1 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-126-2020-ND-CP-huong-dan-Luat-Quan-ly-thue-455733; preview: người nộp thuế được ghi nợ lệ phí trước bạ. 9. Lệ phí môn bài: a) Thời hạn nộp lệ phí môn bài chậm nhất là ngày 30 tháng 01 hàng năm. b) Đối với doanh nghiệp nhỏ và vừa chuyển đổi từ hộ kinh doanh (bao gồm cả đơn vị phụ  |  |  |  |  |  |  |  |  |
| 2 | `233287` | 0.21595238 | 0.74026495 | 1 | 3 | 12 | 2 | 3 | structure_warning |
|  | title: Nghi-dinh-139-2016-ND-CP-le-phi-mon-bai-315033; preview: 4. Thời hạn nộp lệ phí môn bài chậm nhất là ngày 30 tháng 01 hàng năm. Trường hợp người nộp lệ phí mới ra hoạt động sản xuất kinh doanh hoặc mới thành lập cơ sở sản xuất kinh doanh thì thời hạn nộp lệ phí môn bài chậm nh |  |  |  |  |  |  |  |  |
| 3 | `160848` | 0.18333333 | 0.27079135 | 2 | 1 | - | 7 | 2 | none |
|  | title: Thong-tu-302-2016-TT-BTC-huong-dan-le-phi-mon-bai-326995; preview: b) Nộp lệ phí môn bài Cá nhân, nhóm cá nhân, hộ gia đình thực hiện nộp lệ phí môn bài chậm nhất là ngày 30 tháng 01 hàng năm. Trường hợp người nộp lệ phí môn bài là cá nhân, nhóm cá nhân, hộ gia đình mới ra sản xuất, kin |  |  |  |  |  |  |  |  |
| 4 | `159430` | 0.17500000 | 0.52874953 | 3 | 2 | - | 3 | 2 | none |
|  | title: Nghi-dinh-22-2020-ND-CP-sua-doi-Nghi-dinh-139-2016-ND-CP-quy-dinh-le-phi-mon-bai-435348; preview: b) Hộ gia đình, cá nhân, nhóm cá nhân sản xuất, kinh doanh đã giải thể, ra hoạt động sản xuất, kinh doanh trở lại nộp lệ phí môn bài như sau: - Trường hợp ra hoạt động trong 6 tháng đầu năm thì thời hạn nộp lệ phí môn bà |  |  |  |  |  |  |  |  |
| 5 | `161949` | 0.11318182 | 0.22347252 | 9 | 13 | 2 | 10 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Luat-quan-ly-thue-2019-387595; preview: 3. Đối với các khoản thu khác thuộc ngân sách nhà nước từ đất, tiền cấp quyền khai thác tài nguyên nước, tài nguyên khoáng sản, lệ phí trước bạ, lệ phí môn bài thì thời hạn nộp theo quy định của Chính phủ. |  |  |  |  |  |  |  |  |

### Query `56650` ? GROUP_B

Question: Trong quá trình giải quyết công việc nếu có vấn đề liên quan đến lĩnh vực của Thứ trưởng khác thì Thứ trưởng Bộ Nội vụ giải quyết như thế nào?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=False; top1-top2 margin `0.00119048`, top5-top6 margin `0.00040404`, final Top5 support mean `2.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `107101` | 0.24285714 | 0.84082460 | 4 | 1 | 1 | 5 | 3 | structure_warning |
|  | title: Quyet-dinh-1079-QD-BNV-nam-2012-Quy-che-lam-viec-cua-Bo-Noi-vu-178937; preview: c) Chủ động giải quyết công việc được phân công, nếu có vấn đề liên quan đến lĩnh vực của Thứ trưởng khác thì trực tiếp phối hợp với Thứ trưởng đó để giải quyết. Trường hợp cần có ý kiến của Bộ trưởng hoặc giữa các Thứ t |  |  |  |  |  |  |  |  |
| 2 | `85275` | 0.24166667 | 0.93321842 | 1 | 2 | - | 1 | 2 | structure_warning |
|  | title: Quyet-dinh-355-QD-BGDDT-2022-Quy-che-lam-viec-cua-Bo-Giao-duc-va-Dao-tao-501866; preview: 3. Khi giải quyết công việc, nếu có vấn đề liên quan đến lĩnh vực của Thứ trưởng khác phụ trách thì trực tiếp phối hợp với Thứ trưởng đó để giải quyết. Trường hợp cần có ý kiến của Bộ trưởng hoặc giữa các Thứ trưởng còn  |  |  |  |  |  |  |  |  |
| 3 | `139692` | 0.15000000 | 0.84320658 | 3 | 3 | - | 4 | 2 | structure_warning |
|  | title: Quyet-dinh-244-QD-BCT-2017-Quy-che-lam-viec-cua-Bo-Cong-Thuong-394124; preview: c) Chủ động giải quyết công việc được phân công, nếu có vấn đề liên quan đến lĩnh vực của Thứ trưởng khác thì trực tiếp phối hợp với Thứ trưởng đó để giải quyết. Trường hợp cần có ý kiến của Bộ trưởng hoặc giữa các Thứ t |  |  |  |  |  |  |  |  |
| 4 | `240715` | 0.14007937 | 0.91585600 | 7 | 5 | - | 2 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Thong-tu-52-2017-TT-BQP-quy-che-lam-viec-Bo-Quoc-phong-348546; preview: g) Khi giải quyết công việc được phân công, nếu có vấn đề liên quan đến lĩnh vực do Thứ trưởng khác phụ trách thì chủ động phối hợp với Thứ trưởng đó để giải quyết; trường hợp còn có ý kiến khác nhau, thì Thứ trưởng đang |  |  |  |  |  |  |  |  |
| 5 | `39278` | 0.13790404 | 0.85760188 | 9 | 6 | 7 | 3 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Quyet-dinh-559-QD-UBDT-2017-Quy-che-lam-viec-cua-Uy-ban-Dan-toc-362716; preview: d) Chủ động giải quyết công việc được phân công, nếu có vấn đề liên quan đến lĩnh vực của Thứ trưởng, Phó Chủ nhiệm khác thì trực tiếp phối hợp với Thứ trưởng, Phó Chủ nhiệm đó để giải quyết. Trường hợp cần có ý kiến của |  |  |  |  |  |  |  |  |

### Query `93952` ? GROUP_B

Question: Chánh Thanh tra Bộ, cơ quan ngang Bộ phải đáp ứng tiêu chuẩn về trình độ và phẩm chất như thế nào?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=False; top1-top2 margin `0.00078725`, top5-top6 margin `0.00100900`, final Top5 support mean `2.20`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `6258` | 0.12192308 | 0.87441504 | 8 | 154 | - | 1 | 2 | none |
|  | title: Thong-tu-09-2014-TT-TTCP-tieu-chuan-chuc-danh-Pho-Chanh-Thanh-tra-Bo-ngang-Bo-tinh-thanh-pho-truc-th; preview: 3. Đối với chức danh Phó Chánh Thanh tra Bộ Công an, Bộ Quốc phòng thì ngoài các tiêu chuẩn Phó Chánh Thanh tra quy định tại Thông tư này còn phải đảm bảo các tiêu chuẩn bổ nhiệm lãnh đạo, chỉ huy theo quy định của Bộ Cô |  |  |  |  |  |  |  |  |
| 2 | `27125` | 0.12113583 | 0.50619531 | 59 | 5 | - | 2 | 2 | structure_warning |
|  | title: Quyet-dinh-1066-QD-BTP-2018-tieu-chuan-chuc-danh-lanh-dao-quan-ly-trong-cac-don-vi-Bo-Tu-phap-428426; preview: Điều 5. Chánh Văn phòng, Vụ trưởng Vụ Tổ chức cán bộ, Thủ trưởng các đơn vị thuộc Bộ chịu trách nhiệm thi hành Quyết định này./. Nơi nhận: - Như Điều 5; - Đảng ủy Bộ Tư pháp; - Các Thứ trưởng; - Cổng TTĐT Bộ Tư pháp; - L |  |  |  |  |  |  |  |  |
| 3 | `163187` | 0.10750000 | 0.41717631 | 2 | 38 | - | 4 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Quyet-dinh-1100-QD-BKHCN-2020-tieu-chuan-chuc-danh-lanh-dao-quan-ly-447684; preview: 3. Tiêu chuẩn chức danh Chánh Thanh tra, Phó Chánh Thanh tra Bộ ngoài các quy định tại khoản 1, 2 Điều này còn phải thực hiện theo quy định của pháp luật về thanh tra và các văn bản có liên quan. |  |  |  |  |  |  |  |  |
| 4 | `299093` | 0.07840007 | 0.32173836 | 9 | 46 | 16 | 5 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Quyet-dinh-3507-QD-BTNMT-2017-tieu-chuan-cong-chuc-lanh-dao-quan-ly-don-vi-to-chuc-thuoc-Bo-467261; preview: Căn cứ Thông tư số 03/2014/TT-BTTTT ngày 11 tháng 3 năm 2014 của Bộ trưởng Bộ Thông tin và Truyền thông quy định Chuẩn kỹ năng sử dụng công nghệ thông tin; Căn cứ Thông tư số 01/2014/TT-BGD&ĐT ngày 24 tháng 01 năm 2014 c |  |  |  |  |  |  |  |  |
| 5 | `75988` | 0.07362805 | 0.08484913 | 39 | - | 2 | 14 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Quyet-dinh-09-2010-QD-TTg-tieu-chuan-cac-ngach-Thanh-tra-vien-Cong-an-nhan-dan-101241; preview: Chương 3. Điều 7. Trách nhiệm thi hành Các Bộ trưởng, Thủ trưởng cơ quan ngang Bộ, Thủ trưởng cơ quan thuộc Chính phủ, Chủ tịch Ủy ban nhân dân tỉnh, thành phố trực thuộc Trung ương, Thủ trưởng các cơ quan, đơn vị và cá  |  |  |  |  |  |  |  |  |

### Query `40290` ? GROUP_B

Question: Thực hiện đề nghị cấp Giấy phép khảo nghiệm thuốc bảo vệ thực vật để đăng ký bổ sung khi có sự thay đổi về cách sử dụng bằng cách nào?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=False; top1-top2 margin `0.00000000`, top5-top6 margin `0.00181319`, final Top5 support mean `3.00`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `44086` | 0.27500000 | 0.46462360 | 2 | 1 | 2 | 2 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Quyet-dinh-238-QD-BNN-BVTV-2022-cong-bo-thu-tuc-hanh-chinh-linh-vuc-bao-ve-thuc-vat-500997; preview: quá 15 ngày làm việc. Trường hợp hồ sơ chưa đáp ứng yêu cầu, Cục Bảo vệ thực vật thông báo cho tổ chức, cá nhân bổ sung, hoàn thiện hồ sơ. - Bước 4: Cấp Giấy phép khảo nghiệm thuốc bảo vệ thực vật (theo mẫu quy định tại  |  |  |  |  |  |  |  |  |
| 2 | `245029` | 0.27500000 | 0.76681811 | 4 | 2 | 1 | 1 | 3 | structure_warning |
|  | title: Thong-tu-21-2015-TT-BNNPTNT-quan-ly-thuoc-bao-ve-thuc-vat-277987; preview: 2. Hồ sơ a) Đơn đề nghị cấp Giấy phép khảo nghiệm thuốc bảo vệ thực vật theo mẫu quy định tại Phụ lục I ban hành kèm theo Thông tư này; b) Bản sao chụp Giấy chứng nhận đăng ký thuốc bảo vệ thực vật đã được cấp (trường hợ |  |  |  |  |  |  |  |  |
| 3 | `205572` | 0.22666667 | 0.40554151 | 1 | 3 | 3 | 3 | 3 | duplicated_chunk_text |
|  | title: Luat-bao-ve-kiem-dich-thuc-vat-2013-215840; preview: b) Có sự thay đổi về phạm vi, quy mô hành nghề hoặc thông tin liên quan đến tổ chức đăng ký; |  |  |  |  |  |  |  |  |
| 4 | `165596` | 0.15818182 | 0.26707190 | 3 | 4 | 9 | 4 | 3 | duplicated_chunk_text |
|  | title: Nghi-dinh-31-2016-ND-CP-xu-phat-vi-pham-hanh-chinh-linh-vuc-giong-cay-trong-bao-ve-thuc-vat-2016-310; preview: Chương III HÀNH VI VI PHẠM HÀNH CHÍNH, HÌNH THỨC, MỨC XỬ PHẠT VÀ BIỆN PHÁP KHẮC PHỤC HẬU QUẢ TRONG LĨNH VỰC BẢO VỆ VÀ KIỂM DỊCH THỰC VẬT Điều 29. Vi phạm quy định về khảo nghiệm thuốc bảo vệ thực vật để đăng ký vào Danh  |  |  |  |  |  |  |  |  |
| 5 | `21119` | 0.10324176 | 0.07870373 | 11 | 5 | 6 | 13 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Quyet-dinh-3573-QD-BNN-BVTV-2022-cong-bo-thu-tuc-hanh-chinh-linh-vuc-bao-ve-thuc-vat-530301; preview: bằng biện pháp xông hơi khử trùng do Cục Bảo vệ thực vật cấp (chỉ nộp khi nhập khẩu lần đầu) đối với trường hợp nhập khẩu thuốc xông hơi khử trùng; - Báo cáo về tình hình nhập khẩu, sử dụng và mua bán methyl bromide theo |  |  |  |  |  |  |  |  |

### Query `80200` ? GROUP_B

Question: Tất cả các trường hợp tạm giữ phương tiện giao thông có phải niêm phong không?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=False; top1-top2 margin `0.00199134`, top5-top6 margin `0.00013952`, final Top5 support mean `2.00`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `89814` | 0.12532468 | 0.46194211 | 9 | 40 | - | 1 | 2 | duplicated_chunk_text, structure_warning |
|  | title: CHÍNH PHỦ; preview: g) Tự ý tháo gỡ niêm phong tài liệu, tang vật, phương tiện, nhà kho, trang thiết bị vi phạm đang bị niêm phong; tạm giữ hoặc tẩu tán tài liệu, tang vật vi phạm, tự ý làm thay đổi hiện trường vi phạm hành chính trong lĩnh |  |  |  |  |  |  |  |  |
| 2 | `155417` | 0.12333333 | 0.09224073 | 58 | 1 | - | 13 | 2 | structure_warning |
|  | title: Thong-tu-63-2020-TT-BCA-dieu-tra-giai-quyet-tai-nan-giao-thong-duong-bo-cua-Canh-sat-giao-thong-3038; preview: c) Thời hạn tạm giữ tang vật, phương tiện, giấy phép, chứng chỉ hành nghề liên quan đến vụ tai nạn giao thông không quá 07 ngày, kể từ ngày tạm giữ. Trường hợp có nhiều tình tiết phức tạp cần tiến hành xác minh thì cán b |  |  |  |  |  |  |  |  |
| 3 | `177170` | 0.12307692 | 0.12796418 | 2 | 4 | - | 11 | 2 | none |
|  | title: Nghi-dinh-138-2021-ND-CP-bao-quan-tang-vat-phuong-tien-vi-pham-hanh-chinh-bi-tam-giu-tich-thu-487715; preview: 7. Các trường hợp không giao phương tiện giao thông vi phạm hành chính cho tổ chức, cá nhân vi phạm giữ, bảo quản: |  |  |  |  |  |  |  |  |
| 4 | `99842` | 0.12142857 | 0.12619255 | 3 | 3 | - | 12 | 2 | structure_warning |
|  | title: Thong-tu-68-2020-TT-BCA-quy-trinh-tuan-tra-kiem-soat-va-xu-ly-vi-pham-hanh-chinh-Canh-sat-duong-thuy; preview: a) Khi phát hiện hành vi vi phạm theo quy định của pháp luật phải tạm giữ tang vật, phương tiện vi phạm hành chính, Tổ tuần tra, kiểm soát thông báo cho người vi phạm và những người có liên quan biết; tiến hành lập biên  |  |  |  |  |  |  |  |  |
| 5 | `237914` | 0.10288462 | 0.37930435 | 11 | 6 | - | 4 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Luat-67-2020-QH14-xu-ly-vi-pham-hanh-chinh-sua-doi-373520; preview: 5. Người lập biên bản tạm giữ, người ra quyết định tạm giữ có trách nhiệm bảo quản tang vật, phương tiện vi phạm hành chính, giấy phép, chứng chỉ hành nghề. Trong trường hợp tang vật, phương tiện vi phạm hành chính, giấy |  |  |  |  |  |  |  |  |

### Query `149310` ? GROUP_B

Question: Đối với kết luận giám định pháp y tâm thần bằng hình thức giám định tại chỗ mà có giám định viên không thống nhất thì xử lý như thế nào?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=False; top1-top2 margin `0.00047619`, top5-top6 margin `0.00219577`, final Top5 support mean `2.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `117272` | 0.25000000 | 0.73456866 | 4 | 2 | 1 | 2 | 3 | structure_warning |
|  | title: Thong-tu-23-2019-TT-BYT-ve-Quy-trinh-giam-dinh-phap-y-tam-than-423306; preview: giám định viên đó. Giám định viên có quyền độc lập đưa ra ý kiến bảo lưu kết luận của mình và chịu trách nhiệm trước pháp luật về kết luận đó. a) Kết luận theo tiêu chuẩn y học: - Căn cứ Tiêu chuẩn chẩn đoán của Tổ chức  |  |  |  |  |  |  |  |  |
| 2 | `232399` | 0.24952381 | 0.53408355 | 1 | 1 | 3 | 5 | 3 | none |
|  | title: Quyet-dinh-2999-QD-BYT-2022-quy-trinh-giam-dinh-phap-y-tam-than-30-benh-tam-than-thuong-gap-536565; preview: cầu hoặc người yêu cầu cung cấp. 3. Việc quản lý đối tượng giám định được thực hiện theo quy định tại khoản 4, Điều 27 Luật giám định tư pháp. B. QUY TRÌNH GIÁM ĐỊNH PHÁP Y TÂM THẦN ĐỐI VỚI TỪNG HÌNH THỨC GIÁM ĐỊNH Căn c |  |  |  |  |  |  |  |  |
| 3 | `68399` | 0.13500000 | 0.78040355 | 8 | 18 | - | 1 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Luat-Thi-hanh-an-hinh-su-2019-387991; preview: b) Kết luận của Hội đồng giám định pháp y tâm thần; |  |  |  |  |  |  |  |  |
| 4 | `261165` | 0.12666667 | 0.53736269 | 10 | 3 | - | 4 | 2 | structure_warning |
|  | title: Quyet-dinh-111-QD-VKSTC-2020-Quy-che-cong-tac-thuc-hanh-quyen-cong-to-440459; preview: 4. Nếu nội dung kết luận giám định pháp y tâm thần chưa rõ, chưa đầy đủ hoặc khi có nghi ngờ kết luận giám định pháp y tâm thần không chính xác thì việc giám định bổ sung, giám định lại được thực hiện theo quy định của B |  |  |  |  |  |  |  |  |
| 5 | `258607` | 0.10722222 | 0.60014331 | 7 | 10 | - | 3 | 2 | none |
|  | title: Thong-tu-22-2019-TT-BYT-quy-dinh-ty-le-phan-tram-ton-thuong-co-the-su-dung-trong-giam-dinh-phap-y-42; preview: b) Một người cần phải giám định tại hai tổ chức: (1) Giám định pháp y và (2) Giám định pháp y tâm thần: Ông Nguyễn Văn B (ông B) đã được tổ chức giám định pháp y giám định với kết luận tổng tỷ lệ % TTCT là 45% (T1). Sau  |  |  |  |  |  |  |  |  |

### Query `82538` ? GROUP_B

Question: Ai có trách nhiệm thực hiện giám sát hỗ trợ người bệnh thực hiện chế độ dinh dưỡng hàng ngày?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=False; top1-top2 margin `0.00166667`, top5-top6 margin `0.00101954`, final Top5 support mean `2.20`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `177463` | 0.22666667 | 0.22557361 | 1 | 1 | - | 3 | 2 | none |
|  | title: Thong-tu-18-2020-TT-BYT-quy-dinh-hoat-dong-dinh-duong-trong-benh-vien-457809; preview: b) Tổ chức tiếp nhận, hỗ trợ và giám sát người bệnh thực hiện chế độ dinh dưỡng tại khoa. |  |  |  |  |  |  |  |  |
| 2 | `22123` | 0.22500000 | 0.36344177 | 2 | 2 | - | 1 | 2 | duplicated_chunk_text |
|  | title: Quyet-dinh-1895-1997-QD-BYT-Quy-che-benh-vien-66676; preview: c. Bác sĩ điều trị ghi y lệnh dùng thuốc trong phiếu điều trị hàng ngày phải thực hiện các quy định trên; ngoài phần chỉ định thuốc còn có chỉ định chế độ chăm sóc, chế độ dinh dưỡng và phần nhận xét theo dõi người bệnh, |  |  |  |  |  |  |  |  |
| 3 | `215527` | 0.13285714 | 0.15539217 | 3 | 4 | - | 5 | 2 | none |
|  | title: BỘ Y TẾ; preview: b) Chăm sóc dinh dưỡng: thực hiện hoặc hỗ trợ người bệnh thực hiện chế độ dinh dưỡng phù hợp theo chỉ định của bác sỹ; theo dõi dung nạp, hài lòng về chế độ dinh dưỡng của người bệnh để báo cáo bác sỹ và người làm dinh d |  |  |  |  |  |  |  |  |
| 4 | `96286` | 0.10214286 | 0.23096146 | 26 | 13 | - | 2 | 2 | structure_warning |
|  | title: Quyet-dinh-2355-QD-BYT-2022-phong-lay-nhiem-SARSCOV2-trong-co-so-kham-chua-benh-527920; preview: động. - NVYT, nhân viên phục vụ và người thăm NB luôn mang khẩu trang. - Chỉ những người khỏe mạnh mới được vào chăm sóc/điều trị, thăm người thận nhân tạo. - Hướng dẫn NB luôn rửa tay đúng cách trước và sau khi ăn, sau  |  |  |  |  |  |  |  |  |
| 5 | `12781` | 0.09824176 | 0.05411449 | 11 | 5 | 8 | 13 | 3 | structure_warning |
|  | title: Quyet-dinh-1294-QD-BYT-2022-Ke-hoach-hanh-dong-Chien-luoc-Quoc-gia-ve-dinh-duong-513767; preview: đình có điểm FIES về thiếu an ninh thực phẩm bằng hoặc trên 5 (FAO) Giám sát dinh dưỡng Điều tra dinh dưỡng toàn quốc Hằng năm và cuối năm 2025 4. Tỷ lệ bệnh viện tổ chức thực hiện các hoạt động khám, tư vấn và điều trị  |  |  |  |  |  |  |  |  |

### Query `168416` ? GROUP_B

Question: Thế nào là Công chức Tài chính kế toán cấp xã theo quy định pháp luật hiện nay?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=False; top1-top2 margin `0.00166667`, top5-top6 margin `0.00111538`, final Top5 support mean `2.00`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `118485` | 0.21000000 | 0.61003518 | 2 | 3 | - | 1 | 2 | duplicated_chunk_text |
|  | title: Thong-tu-13-2019-TT-BNV-can-bo-cong-chuc-cap-xa-va-nguoi-hoat-dong-khong-chuyen-trach-o-cap-xa-42744; preview: b) Trực tiếp thực hiện các nhiệm vụ sau: Xây dựng dự toán thu, chi ngân sách cấp xã trình cấp có thẩm quyền phê duyệt; tổ chức thực hiện dự toán thu, chi ngân sách và các biện pháp khai thác nguồn thu trên địa bàn; Tổ ch |  |  |  |  |  |  |  |  |
| 2 | `145175` | 0.20833333 | 0.39860520 | 4 | 1 | - | 2 | 2 | duplicated_chunk_text |
|  | title: Nghi-dinh-33-2023-ND-CP-can-bo-cong-chuc-cap-xa-va-nguoi-hoat-dong-khong-chuyen-trach-o-cap-xa-56060; preview: 4. Công chức Tài chính - kế toán |  |  |  |  |  |  |  |  |
| 3 | `248942` | 0.17500000 | 0.39860520 | 3 | 2 | - | 3 | 2 | duplicated_chunk_text |
|  | title: CHÍNH PHỦ; preview: 4. Công chức Tài chính - kế toán |  |  |  |  |  |  |  |  |
| 4 | `226765` | 0.12089202 | 0.19442153 | 1 | 69 | - | 4 | 2 | none |
|  | title: Thong-tu-17-2019-TT-BTC-quy-dinh-xet-tang-Ky-niem-chuong-Vi-su-nghiep-Tai-chinh-Viet-Nam-410008; preview: d) Công chức làm công tác tài chính, kế toán tại UBND các xã, phường, thị trấn; |  |  |  |  |  |  |  |  |
| 5 | `95164` | 0.09250000 | 0.07702839 | 6 | 6 | - | 8 | 2 | none |
|  | title: Nghi-dinh-174-2016-ND-CP-huong-Luat-ke-toan-336391; preview: đ) Đơn vị kế toán ngân sách và tài chính xã, phường, thị trấn; |  |  |  |  |  |  |  |  |

### Query `22234` ? GROUP_B

Question: Kế toán trưởng có phải chịu trách nhiệm pháp lý trong trường hợp chủ doanh nghiệp gian lận về thuế hay không?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=False; top1-top2 margin `0.00008865`, top5-top6 margin `0.00298750`, final Top5 support mean `2.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `21398` | 0.17925532 | 0.18967804 | 45 | 1 | 6 | 4 | 3 | duplicated_chunk_text |
|  | title: Luat-Doanh-nghiep-so-59-2020-QH14-427301; preview: 2. Sau khi bán doanh nghiệp tư nhân, chủ doanh nghiệp tư nhân vẫn phải chịu trách nhiệm về các khoản nợ và nghĩa vụ tài sản khác của doanh nghiệp tư nhân phát sinh trong thời gian trước ngày chuyển giao doanh nghiệp, trừ |  |  |  |  |  |  |  |  |
| 2 | `234704` | 0.17916667 | 0.17044470 | 1 | 2 | - | 6 | 2 | none |
|  | title: Thong-tu-70-2015-TT-BTC-Chuan-muc-dao-duc-nghe-nghiep-ke-toan-kiem-toan-290066; preview: có thể phải chịu trách nhiệm đối với việc quản lý tài chính hiệu quả và các ý kiến tư vấn về các vấn đề liên quan đến hoạt động kinh doanh. 300.3 Kế toán viên, kiểm toán viên chuyên nghiệp trong doanh nghiệp có thể là ng |  |  |  |  |  |  |  |  |
| 3 | `130251` | 0.13583333 | 0.12498879 | 4 | 6 | 3 | 10 | 3 | duplicated_chunk_text |
|  | title: Luat-ke-toan-2015-298369; preview: 5. Doanh nghiệp, hộ kinh doanh dịch vụ kế toán và người được thuê làm kế toán, làm kế toán trưởng phải chịu trách nhiệm về thông tin, số liệu kế toán theo thỏa thuận trong hợp đồng. |  |  |  |  |  |  |  |  |
| 4 | `60562` | 0.13374384 | 0.50136870 | 5 | 56 | - | 1 | 2 | duplicated_chunk_text |
|  | title: Nghi-dinh-41-2018-ND-CP-quy-dinh-xu-phat-vi-pham-hanh-chinh-trong-linh-vuc-ke-toan-363484; preview: 3. Áp dụng các quy định của Nghị định này để xử lý đối với các hành vi vi phạm xảy ra trước ngày Nghị định này có hiệu lực như sau: Trong trường hợp Nghị định này không quy định trách nhiệm pháp lý hoặc quy định trách nh |  |  |  |  |  |  |  |  |
| 5 | `69835` | 0.10892157 | 0.31931680 | 49 | 8 | - | 2 | 2 | duplicated_chunk_text |
|  | title: Luat-Doanh-nghiep-2005-60-2005-QH11-7019; preview: Chương VI DOANH NGHIỆP TƯ NHÂN Điều 144. Cho thuê doanh nghiệp Chủ doanh nghiệp tư nhân có quyền cho thuê toàn bộ doanh nghiệp của mình nhưng phải báo cáo bằng văn bản kèm theo bản sao hợp đồng cho thuê có công chứng đến |  |  |  |  |  |  |  |  |

### Query `164868` ? GROUP_C

Question: Ai có quyền cấp chứng chỉ chuyên môn cho người điều khiển phương tiện giao thông?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=True; top1-top2 margin `0.00321372`, top5-top6 margin `0.00371849`, final Top5 support mean `2.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `50885` | 0.14730463 | 0.01151690 | 9 | 1 | 13 | 17 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-100-2019-ND-CP-xu-phat-vi-pham-hanh-chinh-linh-vuc-giao-thong-duong-bo-va-duong-sat-426369; preview: d) Điều khiển phương tiện, thiết bị thi công mà không có bằng, chứng chỉ chuyên môn theo quy định; |  |  |  |  |  |  |  |  |
| 2 | `99248` | 0.14409091 | 0.07558054 | 20 | 3 | - | 2 | 2 | none |
|  | title: Nghi-dinh-56-2022-ND-CP-chuc-nang-nhiem-vu-quyen-han-Bo-Giao-thong-van-tai-527429; preview: 7. Quy định việc đào tạo, huấn luyện, sát hạch, cấp, công nhận, thu hồi giấy phép, bằng, chứng chỉ chuyên môn cho người điều khiển phương tiện giao thông, người vận hành phương tiện, thiết bị chuyên dùng trong giao thông |  |  |  |  |  |  |  |  |
| 3 | `55002` | 0.12243590 | 0.11694060 | 10 | 50 | - | 1 | 2 | duplicated_chunk_text |
|  | title: Nghi-dinh-57-2013-ND-CP-to-chuc-va-hoat-dong-thanh-tra-nganh-Giao-thong-van-tai-191102; preview: d) Đào tạo, sát hạch, cấp bằng, giấy phép, chứng chỉ chuyên môn, nghiệp vụ cho người điều khiển phương tiện giao thông và người vận hành phương tiện, thiết bị chuyên dùng trong giao thông vận tải theo phân cấp; |  |  |  |  |  |  |  |  |
| 4 | `252814` | 0.11449106 | 0.00425849 | 16 | 11 | 1 | 20 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Thong-tu-33-2018-TT-BGTVT-tieu-chuan-chuc-danh-nhan-vien-duong-sat-truc-tiep-phuc-vu-chay-tau-383598; preview: a) Có bằng, chứng chỉ chuyên môn về lái phương tiện giao thông đường sắt phù hợp với loại phương tiện đảm nhiệm; |  |  |  |  |  |  |  |  |
| 5 | `91006` | 0.11264706 | 0.01778467 | 8 | 2 | - | 15 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-46-2016-ND-CP-xu-phat-vi-pham-hanh-chinh-giao-thong-duong-bo-duong-sat-288330; preview: d) Điều khiển phương tiện, thiết bị thi công mà không có bằng, chứng chỉ chuyên môn theo quy định; |  |  |  |  |  |  |  |  |

### Query `162058` ? GROUP_C

Question: Hội có quyền tham gia ý kiến vào các văn bản quy phạm pháp luật không?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=True; top1-top2 margin `0.00362155`, top5-top6 margin `0.00439394`, final Top5 support mean `2.00`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `134982` | 0.13195489 | 0.81596655 | 17 | 12 | - | 1 | 2 | structure_warning |
|  | title: Quyet-dinh-861-QD-BNV-Dieu-le-Hoi-My-thuat-Viet-Nam-181087; preview: 5. Tham gia ý kiến vào các văn bản quy phạm pháp luật có liên quan đến nội dung hoạt động của Hội theo quy định của pháp luật, kiến nghị với cơ quan nhà nước có thẩm quyền về các vấn đề liên quan tới sự phát triển Hội và |  |  |  |  |  |  |  |  |
| 2 | `89865` | 0.12833333 | 0.76697451 | 8 | 2 | - | 7 | 2 | structure_warning |
|  | title: Quyet-dinh-81-QD-BNV-2023-Dieu-le-sua-doi-Hoi-Giai-phau-benh-te-bao-benh-hoc-Viet-Nam-555026; preview: 5. Tham gia ý kiến vào các văn bản quy phạm pháp luật có liên quan đến nội dung hoạt động của Hội theo quy định của pháp luật. Kiến nghị với cơ quan nhà nước có thẩm quyền đối với các vấn đề liên quan đến sự phát triển c |  |  |  |  |  |  |  |  |
| 3 | `86787` | 0.11374916 | 0.79452294 | 1 | 69 | - | 5 | 2 | structure_warning |
|  | title: Quyet-dinh-537-QD-BNV-nam-2010-phe-duyet-Dieu-le-Hoi-Than-kinh-hoc-Viet-Nam-180728; preview: 3. Tham gia ý kiến vào các văn bản quy phạm pháp luật có liên quan đến nội dung hoạt động của Hội theo quy định của pháp luật. |  |  |  |  |  |  |  |  |
| 4 | `163273` | 0.10785714 | 0.80554003 | 5 | 68 | - | 2 | 2 | structure_warning |
|  | title: Quyet-dinh-77-2005-QD-BNV-Dieu-le-sua-doi-Hoi-Cuu-tro-tre-em-tan-tat-Viet-Nam-20960; preview: 8. Tham gia ý kiến vào các văn bản quy phạm pháp luật có liên quan đến nội dung hoạt động của Hội theo quy định của pháp luật. Đề đạt, kiến nghị với cơ quan nhà nước có thẩm quyền đối với các vấn đề liên quan tới sự phát |  |  |  |  |  |  |  |  |
| 5 | `53246` | 0.10666667 | 0.79824591 | 13 | 7 | - | 3 | 2 | structure_warning |
|  | title: Quyet-dinh-522-QD-BNV-2021-phe-duyet-Dieu-le-sua-doi-Hoi-Sinh-ly-hoc-Viet-Nam-472253; preview: 5. Tham gia ý kiến vào các văn bản quy phạm pháp luật có liên quan đến nội dung hoạt động của Hội theo quy định của pháp luật. Kiến nghị với cơ quan nhà nước có thẩm quyền đối với các vấn đề liên quan tới sự phát triển H |  |  |  |  |  |  |  |  |

### Query `9438` ? GROUP_C

Question: Trình tự thủ tục và hồ sơ đăng ký sau khi đền bù và yêu cầu mở lối đi ngang qua đất người khác

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=True; top1-top2 margin `0.00710105`, top5-top6 margin `0.00254486`, final Top5 support mean `2.00`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `59149` | 0.10917001 | 0.19029683 | 34 | 81 | - | 1 | 2 | duplicated_chunk_text |
|  | title: Nghi-dinh-181-2004-ND-CP-thi-hanh-Luat-Dat-dai-52514; preview: Chương XI TRÌNH TỰ THỦ TỤC HÀNH CHÍNH VỀ QUẢN LÝ VÀ SỬ DỤNG ĐẤT ĐAI Mục 1. THỦ TỤC HÀNH CHÍNH ÁP DỤNG CHUNG KHI NGƯỜI SỬ DỤNG ĐẤT THỰC HIỆN CÁC QUYỀN VÀ NGHĨA VỤ Điều 122. Việc nộp hồ sơ và trả lại kết quả giải quyết khi |  |  |  |  |  |  |  |  |
| 2 | `303096` | 0.10206897 | 0.12538910 | - | 143 | 2 | 4 | 2 | none |
|  | title: Nghi-dinh-08-2022-ND-CP-huong-dan-Luat-Bao-ve-moi-truong-479457; preview: quy định tại Phụ lục XIX ban hành kèm theo Nghị định này; c) Kết quả quan trắc và giám sát môi trường gần nhất theo quy định của pháp luật. 3. Trình tự, thủ tục đăng ký miễn trừ các chất POP: a) Tổ chức, cá nhân lập hồ s |  |  |  |  |  |  |  |  |
| 3 | `81598` | 0.10037577 | 0.18057042 | 85 | 11 | - | 2 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Bo-luat-dan-su-2015-296215; preview: 1. Chủ sở hữu có bất động sản bị vây bọc bởi các bất động sản của các chủ sở hữu khác mà không có hoặc không đủ lối đi ra đường công cộng, có quyền yêu cầu chủ sở hữu bất động sản vây bọc dành cho mình một lối đi hợp lý  |  |  |  |  |  |  |  |  |
| 4 | `50885` | 0.10030303 | 0.01455958 | 8 | - | 1 | 20 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-100-2019-ND-CP-xu-phat-vi-pham-hanh-chinh-linh-vuc-giao-thong-duong-bo-va-duong-sat-426369; preview: d) Điều khiển phương tiện, thiết bị thi công mà không có bằng, chứng chỉ chuyên môn theo quy định; |  |  |  |  |  |  |  |  |
| 5 | `113741` | 0.07226708 | 0.16447496 | 21 | 82 | - | 3 | 2 | duplicated_chunk_text |
|  | title: Nghi-dinh-43-2014-ND-CP-huong-dan-thi-hanh-Luat-Dat-dai-230680; preview: 3. Chủ đầu tư dự án nhà ở có trách nhiệm nộp 01 bộ hồ sơ đăng ký, cấp Giấy chứng nhận quyền sử dụng đất, quyền sở hữu nhà ở và tài sản khác gắn liền với đất thay cho người nhận chuyển nhượng quyền sử dụng đất, mua nhà ở, |  |  |  |  |  |  |  |  |

### Query `102358` ? GROUP_C

Question: Xử phạt hành chính đối với hành vi vi phạm chế độ hôn nhân một vợ một chồng như thế nào?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=True; top1-top2 margin `0.00746032`, top5-top6 margin `0.00931818`, final Top5 support mean `2.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `269739` | 0.15555556 | 0.44151887 | 7 | 1 | - | 7 | 2 | none |
|  | title: Thong-tu-lien-tich-01-2001-TTLT-BTP-BCA-TANDTC-VKSNDTC-quy-dinh-tai-Chuong-XV-toi-xam-pham-che-do-ho; preview: con vì thế mà tự sát, v.v... b) Người vi phạm chế độ một vợ, một chồng đã bị xử phạt hành chính về hành vi này mà còn vi phạm. 3.3. Trong trường hợp đã có quyết định của Tòa án tiêu huỷ việc kết hôn hoặc buộc phải chấm d |  |  |  |  |  |  |  |  |
| 2 | `208565` | 0.14809524 | 0.28248566 | 1 | 3 | - | 12 | 2 | duplicated_chunk_text |
|  | title: Quy-dinh-102-QD-TW-2017-xu-ly-ky-luat-dang-vien-vi-pham-368751; preview: a) Vi phạm quy định về cấm kết hôn, vi phạm chế độ hôn nhân một vợ, một chồng. |  |  |  |  |  |  |  |  |
| 3 | `245154` | 0.12875000 | 0.41087142 | 13 | 14 | 1 | 8 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Bo-luat-hinh-su-2015-296661; preview: Chương XVII CÁC TỘI XÂM PHẠM CHẾ ĐỘ HÔN NHÂN VÀ GIA ĐÌNH Điều 182. Tội vi phạm chế độ một vợ, một chồng |  |  |  |  |  |  |  |  |
| 4 | `51405` | 0.12506872 | 0.74388504 | 45 | 112 | 9 | 1 | 3 | duplicated_chunk_text |
|  | title: Nghi-dinh-162-2018-ND-CP-xu-phat-vi-pham-hanh-chinh-trong-linh-vuc-hang-khong-dan-dung-321856; preview: 1. Đối với mỗi hành vi vi phạm hành chính trong lĩnh vực hàng không dân dụng, cá nhân, tổ chức phải chịu một trong các hình thức xử phạt chính sau đây: |  |  |  |  |  |  |  |  |
| 5 | `76024` | 0.12500000 | 0.13937756 | 4 | 2 | - | 16 | 2 | duplicated_chunk_text |
|  | title: Nghi-dinh-110-2013-ND-CP-xu-phat-vi-pham-hanh-chinh-bo-tro-tu-phap-hanh-chinh-tu-phap-208274; preview: Chương 4. Điều 48. Hành vi vi phạm quy định về cấm kết hôn, vi phạm chế độ hôn nhân một vợ, một chồng; vi phạm quy định về ly hôn |  |  |  |  |  |  |  |  |

### Query `27372` ? GROUP_C

Question: Có xóa tên Đảng viên nếu Đảng viên không đóng Đảng phí ba tháng trong năm không?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=True; top1-top2 margin `0.00833333`, top5-top6 margin `0.00658730`, final Top5 support mean `2.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `65586` | 0.19166667 | 0.27793834 | 1 | 2 | 14 | 6 | 3 | structure_warning |
|  | title: Quy-dinh-24-QD-TW-2021-thi-hanh-Dieu-le-Dang-484359; preview: 8. Điều 8: Xoá tên đảng viên và giải quyết khiếu nại về xoá tên đảng viên 8.1. Xoá tên đảng viên. Chi bộ xem xét, đề nghị cấp uỷ có thẩm quyền quyết định xoá tên trong danh sách đảng viên đối với các trường hợp sau: Đảng |  |  |  |  |  |  |  |  |
| 2 | `78507` | 0.18333333 | 0.24274048 | 2 | 1 | - | 7 | 2 | structure_warning |
|  | title: Quy-dinh-29-QD-TW-thi-hanh-Dieu-le-Dang-2016-319797; preview: 8- Điều 8: Xóa tên đảng viên và giải quyết khiếu nại về xóa tên đảng viên 8.1- Xóa tên đảng viên Chi bộ xem xét, đề nghị cấp ủy có thẩm quyền quyết định xóa tên trong danh sách đảng viên đối với các trường hợp sau: đảng  |  |  |  |  |  |  |  |  |
| 3 | `102193` | 0.17500000 | 0.67576069 | 3 | 3 | - | 2 | 2 | structure_warning |
|  | title: Dieu-le-Dang-Cong-san-Viet-Nam-nam-2011-151840; preview: Chương I ĐẢNG VIÊN Điều 8. 1. Đảng viên bỏ sinh hoạt chi bộ hoặc không đóng đảng phí ba tháng trong năm mà không có lý do chính đáng; đảng viên giảm sút ý chí phấn đấu, không làm nhiệm vụ đảng viên, đã được chi bộ giáo d |  |  |  |  |  |  |  |  |
| 4 | `209697` | 0.15031328 | 0.05115276 | 6 | 5 | 1 | 17 | 3 | none |
|  | title: Huong-dan-01-HD-TW-2021-thi-hanh-Dieu-le-Dang-490529; preview: trở về tham gia sinh hoạt chi bộ theo quy định của Điều lệ Đảng thì đảng viên phải làm đơn báo cáo chi bộ xem xét cho tạm miễn sinh hoạt. Nếu đảng viên đi ra ngoài địa phương nơi cư trú (vì việc làm hoặc vì việc riêng) c |  |  |  |  |  |  |  |  |
| 5 | `154516` | 0.11944444 | 0.79486573 | 22 | 25 | - | 1 | 2 | structure_warning |
|  | title: Quyet-dinh-458-QD-BNV-nam-2010-phe-duyet-Dieu-le-Lien-doan-Dien-kinh-Viet-Nam-180701; preview: 4. Hội viên không đóng hội phí 6 tháng hoặc không sinh hoạt liên tục 6 tháng mà không có lý do chính đáng sẽ bị xóa tên trong danh sách của tổ chức cơ sở nơi sinh hoạt. |  |  |  |  |  |  |  |  |

### Query `159630` ? GROUP_C

Question: Cha mẹ giữ tiền của con có thể bị xử phạt như thế nào?

Reasons: low_margin=True, high_disagreement=False, strong_bge_reorder=True; top1-top2 margin `0.00958333`, top5-top6 margin `0.00153931`, final Top5 support mean `2.00`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `194863` | 0.13125000 | 0.31703994 | 14 | 14 | - | 1 | 2 | duplicated_chunk_text |
|  | title: Bo-luat-To-tung-Hinh-su-2003-19-2003-QH11-51701; preview: 4. Bị cáo bị xử phạt tù, nhưng được hưởng án treo; |  |  |  |  |  |  |  |  |
| 2 | `24778` | 0.12166667 | 0.10958439 | 1 | 8 | - | 10 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Bo-Luat-hinh-su-1999-15-1999-QH10-46056; preview: Chương XV CÁC TỘI XÂM PHẠM CHẾ ĐỘ HÔN NHÂN VÀ GIA ĐÌNH Điều 151. Tội ngược đãi hoặc hành hạ ông bà, cha mẹ, vợ chồng, con, cháu, người có công nuôi dưỡng mình Người nào ngược đãi hoặc hành hạ ông bà, cha mẹ, vợ chồng, co |  |  |  |  |  |  |  |  |
| 3 | `245154` | 0.11578947 | 0.03665941 | 2 | 4 | - | 17 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Bo-luat-hinh-su-2015-296661; preview: Chương XVII CÁC TỘI XÂM PHẠM CHẾ ĐỘ HÔN NHÂN VÀ GIA ĐÌNH Điều 182. Tội vi phạm chế độ một vợ, một chồng |  |  |  |  |  |  |  |  |
| 4 | `98892` | 0.10641026 | 0.05128743 | - | 11 | 1 | 16 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-144-2021-ND-CP-xu-phat-vi-pham-hanh-chinh-linh-vuc-an-ninh-an-toan-xa-hoi-425471; preview: Chương II HÀNH VI VI PHẠM HÀNH CHÍNH, HÌNH THỨC XỬ PHẠT VÀ BIỆN PHÁP KHẮC PHỤC HẬU QUẢ Mục 4. Điều 56. Hành vi ngăn cản việc thực hiện quyền, nghĩa vụ trong quan hệ gia đình giữa ông, bà và cháu; giữa cha, mẹ và con; giữ |  |  |  |  |  |  |  |  |
| 5 | `147066` | 0.10619048 | 0.12594631 | 4 | 5 | - | 8 | 2 | structure_warning |
|  | title: Bo-luat-Hinh-su-1985-17-LCT-HDNN7-37003; preview: quy định ở các Điều 88, 95, 96, 98 và Điều 99, thì có thể bị phạt tiền từ mười nghìn đồng (10. 000 đồng) đến một trăm nghìn đồng (100.000 đồng); ở Điều 97a, trong trường hợp bị xử phạt tiền thì có thể bị phạt tiền theo m |  |  |  |  |  |  |  |  |

### Query `103124` ? GROUP_C

Question: Mức xử phạt vi phạm hành chính đối với người thông thầu là bao nhiêu tiền?

Reasons: low_margin=False, high_disagreement=False, strong_bge_reorder=True; top1-top2 margin `0.01103896`, top5-top6 margin `0.01078373`, final Top5 support mean `2.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `17545` | 0.16785714 | 0.50898343 | - | 2 | 2 | 5 | 2 | duplicated_chunk_text |
|  | title: Luat-xu-ly-vi-pham-hanh-chinh-2012-142766; preview: 4. Mức tiền phạt cụ thể đối với một hành vi vi phạm hành chính là mức trung bình của khung tiền phạt được quy định đối với hành vi đó; nếu có tình tiết giảm nhẹ thì mức tiền phạt có thể giảm xuống nhưng không được giảm q |  |  |  |  |  |  |  |  |
| 2 | `186584` | 0.15681818 | 0.05159252 | 6 | 1 | 9 | 20 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-122-2021-ND-CP-xu-phat-vi-pham-hanh-chinh-linh-vuc-ke-hoach-285024; preview: 2. Thông thầu. |  |  |  |  |  |  |  |  |
| 3 | `60562` | 0.12517094 | 0.67400378 | 34 | 63 | 3 | 2 | 3 | duplicated_chunk_text |
|  | title: Nghi-dinh-41-2018-ND-CP-quy-dinh-xu-phat-vi-pham-hanh-chinh-trong-linh-vuc-ke-toan-363484; preview: 3. Áp dụng các quy định của Nghị định này để xử lý đối với các hành vi vi phạm xảy ra trước ngày Nghị định này có hiệu lực như sau: Trong trường hợp Nghị định này không quy định trách nhiệm pháp lý hoặc quy định trách nh |  |  |  |  |  |  |  |  |
| 4 | `292790` | 0.12512821 | 0.67917520 | 37 | 13 | - | 1 | 2 | duplicated_chunk_text |
|  | title: Nghi-dinh-18-2020-ND-CP-xu-phat-vi-pham-hanh-chinh-trong-linh-vuc-do-dac-va-ban-do-434367; preview: Chương II Nghị định này là mức phạt tiền đối với hành vi vi phạm hành chính của cá nhân, mức phạt tiền đối với hành vi vi phạm hành chính của tổ chức bằng 02 lần mức phạt tiền đối với cùng hành vi vi phạm hành chính của  |  |  |  |  |  |  |  |  |
| 5 | `26782` | 0.10238095 | 0.12558058 | - | 12 | 1 | 19 | 2 | duplicated_chunk_text |
|  | title: Nghi-dinh-12-2022-ND-CP-xu-phat-vi-pham-hanh-chinh-lao-dong-bao-hiem-nguoi-lam-viec-nuoc-ngoai-47931; preview: Chương V THẨM QUYỀN XỬ PHẠT VI PHẠM HÀNH CHÍNH VÀ LẬP BIÊN BẢN VI PHẠM HÀNH CHÍNH; THỦ TỤC XỬ PHẠT VI PHẠM HÀNH CHÍNH; THI HÀNH CÁC HÌNH THỨC XỬ PHẠT VI PHẠM HÀNH CHÍNH, CÁC BIỆN PHÁP KHẮC PHỤC HẬU QUẢ TRONG LĨNH VỰC LAO |  |  |  |  |  |  |  |  |

### Query `35162` ? GROUP_C

Question: Tình hình vận tải hàng hóa, hành khách dịp Tết 2023 được đảm bảo tăng cường như thế nào?

Reasons: low_margin=False, high_disagreement=False, strong_bge_reorder=True; top1-top2 margin `0.01541667`, top5-top6 margin `0.00180837`, final Top5 support mean `2.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `213014` | 0.17875000 | 0.29215464 | 8 | 1 | 3 | 14 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-10-2020-ND-CP-kinh-doanh-va-dieu-kien-kinh-doanh-van-tai-bang-xe-o-to-315260; preview: a) Tăng cường phương tiện vào các dịp Lễ, Tết và các kỳ thi trung học phổ thông Quốc gia, tuyển sinh đại học, cao đẳng: Doanh nghiệp, hợp tác xã khai thác tuyến cố định căn cứ vào nhu cầu đi lại, thống nhất với bến xe kh |  |  |  |  |  |  |  |  |
| 2 | `196603` | 0.16333333 | 0.35086977 | 13 | 2 | 2 | 10 | 3 | structure_warning |
|  | title: Thong-tu-12-2020-TT-BGTVT-quan-ly-hoat-dong-van-tai-bang-xe-o-to-va-dich-vu-ho-tro-van-tai-duong-bo-; preview: vị liên quan lập kế hoạch tăng cường phương tiện giải tỏa hành khách (trong đó có danh sách phương tiện, người lái xe được điều động) và tổ chức thực hiện. Thời gian ban hành kế hoạch tăng cường giải tỏa hành khách đảm b |  |  |  |  |  |  |  |  |
| 3 | `186125` | 0.15000000 | 0.60372925 | 1 | 34 | - | 2 | 2 | none |
|  | title: Chi-thi-03-CT-TTg-2023-don-doc-thuc-hien-nhiem-vu-trong-tam-sau-ky-nghi-Tet-nguyen-dan-551858; preview: cán bộ, người lao động trực, làm việc trong dịp Tết… Tuy nhiên, trong dịp nghỉ Tết, tình hình vi phạm quy định an toàn giao thông vẫn diễn ra phức tạp; còn xảy ra hiện tượng ùn tắc giao thông cục bộ, vận chuyển hành khác |  |  |  |  |  |  |  |  |
| 4 | `129555` | 0.11721925 | 0.69205987 | 15 | 53 | - | 1 | 2 | possible_truncation_marker |
|  | title: Nghi-quyet-10-NQ-CP-2023-phien-hop-Chinh-phu-thuong-ky-thang-01-voi-dia-phuong-553512; preview: Nhân dân trong không khí vui tươi, đầm ấm, sum vầy, lành mạnh, an toàn, tiết kiệm; mọi người, mọi nhà đều có Tết, bảo đảm không để ai không có Tết. Ngay từ ngày làm việc đầu tiên sau Tết, các bộ, ngành, địa phương, cơ qu |  |  |  |  |  |  |  |  |
| 5 | `288109` | 0.10052632 | 0.54366624 | - | 8 | 17 | 3 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Bo-luat-hang-hai-Viet-Nam-2015-298374; preview: 23. Vận tải biển nội địa là việc vận chuyển hàng hóa, hành khách, hành lý bằng tàu biển mà điểm nhận và điểm trả hàng hóa, hành khách, hành lý thuộc vùng biển Việt Nam. |  |  |  |  |  |  |  |  |

### Query `102332` ? GROUP_C

Question: Văn bản mật hiện nay được chia làm bao nhiêu cấp độ mật theo quy định của pháp luật?

Reasons: low_margin=False, high_disagreement=False, strong_bge_reorder=True; top1-top2 margin `0.01666667`, top5-top6 margin `0.00934524`, final Top5 support mean `2.00`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `193923` | 0.16666667 | 0.26512215 | 1 | 10 | - | 2 | 2 | structure_warning |
|  | title: Quyet-dinh-1119-QD-NHNN-Quy-che-su-dung-He-thong-thu-dien-tu-105475; preview: 5. Các văn bản có cấp độ “MẬT” trở lên thực hiện theo hướng dẫn riêng. |  |  |  |  |  |  |  |  |
| 2 | `134366` | 0.15000000 | 0.04675353 | 4 | 1 | - | 16 | 2 | structure_warning |
|  | title: Quyet-dinh-50-QD-BCT-2017-Quy-che-bao-ve-bi-mat-nha-nuoc-trong-nganh-Cong-Thuong-438464; preview: 2. Cán bộ, công chức, viên chức và người lao động làm công tác liên quan trực tiếp đến bí mật Nhà nước của ngành Công Thương phải cam kết bảo vệ bí mật Nhà nước bằng văn bản với thủ trưởng cơ quan (phụ lục VII). Văn bản  |  |  |  |  |  |  |  |  |
| 3 | `37130` | 0.11687805 | 0.36609510 | 39 | 23 | - | 1 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Quyet-dinh-01-QD-TANDTC-2022-Quy-che-van-thu-trong-he-thong-Toa-an-nhan-dan-499476; preview: c) Văn bản mật đi được đăng ký theo quy định của pháp luật về bảo vệ bí mật nhà nước thuộc Tòa án nhân dân và lập sổ theo dõi riêng. |  |  |  |  |  |  |  |  |
| 4 | `218546` | 0.10666667 | 0.16602387 | 28 | 4 | - | 4 | 2 | duplicated_chunk_text, structure_warning |
|  | title: CHÍNH PHỦ; preview: 2. Mẫu vật, tài liệu, thông tin, dữ liệu được phân loại theo cấp độ mật (nếu có yêu cầu) và có chính sách bảo vệ an toàn thông tin cho từng loại tương ứng phù hợp với các quy định của pháp luật. |  |  |  |  |  |  |  |  |
| 5 | `169478` | 0.09017857 | 0.05632469 | 2 | 12 | - | 14 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Quyet-dinh-2175-QD-KTNN-nam-2014-Quy-che-cong-tac-van-thu-luu-tru-Kiem-toan-Nha-nuoc-259832; preview: a) Được giải mật theo quy định của pháp luật về bảo vệ bí mật nhà nước; |  |  |  |  |  |  |  |  |

### Query `98314` ? GROUP_C

Question: Đối tượng nào phải nộp thuế thu nhập cá nhân?

Reasons: low_margin=False, high_disagreement=False, strong_bge_reorder=True; top1-top2 margin `0.01835470`, top5-top6 margin `0.00202020`, final Top5 support mean `2.60`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `240117` | 0.19391026 | 0.17387997 | 1 | 11 | 1 | 6 | 3 | none |
|  | title: Luat-thue-thu-nhap-ca-nhan-2007-04-2007-QH12-59652; preview: b) Cá nhân đã nộp thuế nhưng có thu nhập tính thuế chưa đến mức phải nộp thuế; |  |  |  |  |  |  |  |  |
| 2 | `171570` | 0.17555556 | 0.21489441 | 4 | 3 | 7 | 3 | 3 | duplicated_chunk_text |
|  | title: Thong-tu-111-2013-TT-BTC-Huong-dan-Luat-thue-thu-nhap-ca-nhan-va-Nghi-dinh-65-2013-ND-CP-205356; preview: 1. Đối tượng phải đăng ký thuế Theo quy định tại Điều 27 Nghị định số 65/2013/NĐ-CP thì đối tượng phải đăng ký thuế thu nhập cá nhân bao gồm: |  |  |  |  |  |  |  |  |
| 3 | `247150` | 0.17142857 | 0.19500111 | 5 | 1 | - | 5 | 2 | structure_warning |
|  | title: Thong-tu-205-2013-TT-BTC-huong-dan-Hiep-dinh-tranh-danh-thue-hai-lan-Viet-Nam-voi-cac-nuoc-217929; preview: Chương II THUẾ ĐỐI VỚI CÁC LOẠI THU NHẬP Mục 9. THU NHẬP TỪ HOẠT ĐỘNG DỊCH VỤ CÁ NHÂN ĐỘC LẬP Điều 29. Xác định nghĩa vụ thuế đối với thu nhập từ hoạt động dịch vụ cá nhân độc lập Theo quy định tại Hiệp định, một đối tượ |  |  |  |  |  |  |  |  |
| 4 | `174178` | 0.12231707 | 0.26330745 | - | 39 | 3 | 2 | 2 | none |
|  | title: TỔNG CỤC THUẾ CỤC THUẾ TP HÀ NỘI; preview: tổ chức, cá nhân trả thu nhập nào thì nộp hồ sơ khai quyết toán thuế tại cơ quan thuế trực tiếp quản lý tổ chức, cá nhân trả thu nhập đó. Trường hợp cá nhân có thay đổi nơi làm việc và tại tổ chức, cá nhân trả thu nhập c |  |  |  |  |  |  |  |  |
| 5 | `288898` | 0.11944444 | 0.10368624 | 16 | 2 | 13 | 13 | 3 | duplicated_chunk_text, structure_warning |
|  | title: Thong-tu-92-2015-TT-BTC-huong-dan-thue-gia-tri-gia-tang-thue-thu-nhap-ca-nhan-282089; preview: tạm nộp trong năm, số thuế đã nộp ở nước ngoài (nếu có). Cá nhân cam kết chịu trách nhiệm về tính chính xác của các thông tin trên bản chụp đó. Trường hợp tổ chức trả thu nhập không cấp chứng từ khấu trừ thuế cho cá nhân |  |  |  |  |  |  |  |  |

### Query `115756` ? GROUP_D

Question: Chi phí làm sổ đỏ bao gồm những gì?

Reasons: low_margin=True, high_disagreement=True, strong_bge_reorder=False; top1-top2 margin `0.00833333`, top5-top6 margin `0.00333333`, final Top5 support mean `1.00`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `297690` | 0.13333333 | 0.42707899 | 4 | - | - | 1 | 1 | none |
|  | title: Nghi-quyet-27-2016-NQ-HDND-muc-thu-mien-giam-thu-nop-quan-ly-su-dung-cac-khoan-phi-le-phi-Thanh-Hoa-; preview: b) Mức thu (không bao gồm chi phí in ấn, sao chụp hồ sơ, tài liệu): Đơn vị tính: Đồng/hồ sơ, tài liệu STT Nội dung Mức thu 1 Phí khai thác, sử dụng tài liệu đất đai 300.000 2 Các loại bản đồ chuyên đề khác (trừ bản đồ hà |  |  |  |  |  |  |  |  |
| 2 | `35496` | 0.12500000 | 0.20205158 | 2 | - | - | 2 | 1 | structure_warning |
|  | title: Nghi-quyet-06-2020-NQ-HDND-muc-thu-mot-so-loai-phi-va-le-phi-tinh-Phu-Tho-448090; preview: liên hiệp hợp tác xã (bao gồm cả giấy chứng nhận đăng ký chi nhánh, văn phòng đại diện, địa điểm kinh doanh của hợp tác xã, liên hiệp hợp tác xã). b) Đối tượng miễn lệ phí: - Bổ sung, thay đổi những thông tin về số điện  |  |  |  |  |  |  |  |  |
| 3 | `119286` | 0.11428571 | 0.02286812 | - | 1 | - | 19 | 1 | structure_warning |
|  | title: Nghi-dinh-03-2011-ND-CP-huong-dan-bien-phap-thi-hanh-Luat-Hoat-dong-chu-thap-do-117180; preview: thập đỏ dũng cảm cứu người, cứu tài sản của Nhà nước và nhân dân, nếu bị thiệt hại về tính mạng thì được xem xét để công nhận là liệt sỹ; nếu bị thương làm suy giảm khả năng lao động từ 21% trở lên thì được xem xét để đư |  |  |  |  |  |  |  |  |
| 4 | `113741` | 0.10000000 | 0.10532485 | - | - | 1 | 7 | 1 | duplicated_chunk_text |
|  | title: Nghi-dinh-43-2014-ND-CP-huong-dan-thi-hanh-Luat-Dat-dai-230680; preview: 3. Chủ đầu tư dự án nhà ở có trách nhiệm nộp 01 bộ hồ sơ đăng ký, cấp Giấy chứng nhận quyền sử dụng đất, quyền sở hữu nhà ở và tài sản khác gắn liền với đất thay cho người nhận chuyển nhượng quyền sử dụng đất, mua nhà ở, |  |  |  |  |  |  |  |  |
| 5 | `163254` | 0.09333333 | 0.18298931 | - | - | 4 | 3 | 1 | none |
|  | title: Thong-tu-120-2021-TT-BTC-muc-thu-mot-so-khoan-phi-ho-tro-doi-tuong-anh-huong-Covid19-498794; preview: nhiệm vụ, giải pháp chủ yếu thúc đẩy tăng trưởng kinh tế, giải ngân vốn đầu tư công và xuất khẩu bền vững những tháng cuối năm 2021 và đầu năm 2022; chỉ đạo của Thủ tướng Chính phủ tại công văn số 8374/VPCP-KTTH ngày 15  |  |  |  |  |  |  |  |  |

### Query `76050` ? GROUP_D

Question: Nộp đơn khởi kiện dân sự có cần công chứng giấy tờ không?

Reasons: low_margin=True, high_disagreement=True, strong_bge_reorder=False; top1-top2 margin `0.00813268`, top5-top6 margin `0.00380952`, final Top5 support mean `1.20`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `102434` | 0.13540541 | 0.08868424 | 35 | 8 | - | 1 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Bo-luat-to-tung-hinh-su-2015-296884; preview: 2. Trường hợp có đơn hoặc giấy tờ gửi qua dịch vụ bưu chính thì thời hạn được tính theo dấu bưu chính nơi gửi. Nếu có đơn hoặc giấy tờ gửi qua cơ sở giam giữ thì thời hạn được tính từ ngày Trưởng Nhà tạm giữ, Trưởng Buồn |  |  |  |  |  |  |  |  |
| 2 | `229347` | 0.12727273 | 0.01443556 | - | 1 | - | 9 | 1 | structure_warning |
|  | title: BỘ CÔNG AN; preview: 5. Các cơ sở y tế hoặc cơ quan, đơn vị không được quy định thêm các thủ tục hành chính trong khám bệnh, chữa bệnh bảo hiểm y tế ngoài các quy định tại Thông tư này. Trường hợp cơ sở y tế hoặc cơ quan, đơn vị cần sao chụp |  |  |  |  |  |  |  |  |
| 3 | `305455` | 0.10833333 | 0.06140118 | 4 | - | - | 2 | 1 | duplicated_chunk_text |
|  | title: Luat-dat-dai-2013-215836; preview: b) Khởi kiện tại Tòa án nhân dân có thẩm quyền theo quy định của pháp luật về tố tụng dân sự; |  |  |  |  |  |  |  |  |
| 4 | `187506` | 0.10000000 | 0.04300366 | 3 | - | - | 3 | 1 | duplicated_chunk_text |
|  | title: Bo-luat-To-tung-dan-su-2004-24-2004-QH11-52189; preview: 2. Trong trường hợp do tình thế khẩn cấp, cần phải bảo vệ ngay bằng chứng, ngăn chặn hậu quả nghiêm trọng có thể xảy ra thì cá nhân, cơ quan, tổ chức có quyền nộp đơn yêu cầu Tòa án có thẩm quyền ra quyết định áp dụng bi |  |  |  |  |  |  |  |  |
| 5 | `46918` | 0.09666667 | 0.01860141 | 1 | - | - | 8 | 1 | duplicated_chunk_text, structure_warning |
|  | title: Bo-luat-to-tung-dan-su-2015-296861; preview: e) Danh mục tài liệu, chứng cứ người khởi kiện nộp kèm theo đơn khởi kiện; |  |  |  |  |  |  |  |  |

### Query `142774` ? GROUP_D

Question: BOG là gì?

Reasons: low_margin=False, high_disagreement=True, strong_bge_reorder=True; top1-top2 margin `0.01022727`, top5-top6 margin `0.00170592`, final Top5 support mean `1.20`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `242302` | 0.13750000 | 0.88109076 | - | 6 | - | 1 | 1 | none |
|  | title: Thong-tu-40-2022-TT-BCT-Quy-chuan-ky-thuat-an-toan-kho-chua-khi-thien-nhien-hoa-long-549089; preview: và bay hơi nhanh khi nhập LNG vào bồn chứa hoặc xuất LNG cho các phương tiện chuyên chở. BOG phải được tái hóa lỏng và sử dụng làm nhiên liệu khí. BOG từ phương tiện chuyên chở LNG (chỉ áp dụng tại các kho cảng) phải đượ |  |  |  |  |  |  |  |  |
| 2 | `184968` | 0.12727273 | 0.01085815 | - | 1 | - | 9 | 1 | duplicated_chunk_text, structure_warning |
|  | title: Thong-tu-25-2020-TT-BCT-lap-bao-cao-thuc-hien-ke-hoach-su-dung-nang-luong-tiet-kiem-va-hieu-qua-4544; preview: tiết kiệm NL (%)(3) Tiết kiệm chi phí (Tr. đồng) Lợi ích khác (là gì?) Mức tiết kiệm NL (Đơn vị đo) Mức tiết kiệm NL (%)(1) Tiết kiệm chi phí (Tr. đồng) Lợi ích khác (là gì?) (3) So với mục đích sử dụng (ví dụ chiếu sáng |  |  |  |  |  |  |  |  |
| 3 | `165017` | 0.10500000 | 0.01239336 | - | 2 | - | 8 | 1 | duplicated_chunk_text, structure_warning, possible_truncation_marker |
|  | title: Thong-tu-03-2022-TT-BTP-danh-gia-tac-dong-thu-tuc-hanh-chinh-lap-de-nghi-xay-dung-van-ban-504451; preview: hợp với điều kiện phát triển kinh tế - xã hội của địa phương và bảo đảm quyền, nghĩa vụ và lợi ích hợp pháp của cá nhân, tổ chức nêu trên? Cơ quan chủ trì soạn thảo cần trình bày rõ lý do lựa chọn đối với từng thủ tục hà |  |  |  |  |  |  |  |  |
| 4 | `232471` | 0.09750000 | 0.01333362 | - | 3 | - | 6 | 1 | duplicated_chunk_text, possible_truncation_marker |
|  | title: Quyet-dinh-1390-QD-BCT-2020-cau-hoi-kiem-tra-de-xac-nhan-da-tap-huan-ve-an-toan-thuc-pham-448394; preview: c) An toàn thực phẩm là việc bảo đảm để thực phẩm được sản xuất, kinh doanh trong điều kiện an toàn 󠄗 Câu 3 Sản xuất thực phẩm là gì? |  |  |  |  |  |  |  |  |
| 5 | `53007` | 0.09456306 | 0.17788967 | 51 | 17 | - | 2 | 2 | none |
|  | title: Cong-uoc-quoc-te-buon-ban-cac-loai-dong-thuc-vat-hoang-da-nguy-cap-CITES-107575; preview: chỉ tái xuất hoặc bất kỳ một giấy phép xuất khẩu nào qua đường bưu điện liên quan đến việc nhập khẩu những mẫu vật đó sau khi đã dùng xong. 7. Ở đâu có điều kiện và thuận lợi thì cơ quan thẩm quyền quản lý nên đánh dấu l |  |  |  |  |  |  |  |  |

### Query `54374` ? GROUP_D

Question: Công chức nhận tiền của người dân sẽ bị xử lý như thế nào?

Reasons: low_margin=False, high_disagreement=True, strong_bge_reorder=True; top1-top2 margin `0.03166667`, top5-top6 margin `0.01292151`, final Top5 support mean `1.20`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `245154` | 0.18166667 | 0.49551013 | 3 | - | 1 | 2 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Bo-luat-hinh-su-2015-296661; preview: Chương XVII CÁC TỘI XÂM PHẠM CHẾ ĐỘ HÔN NHÂN VÀ GIA ĐÌNH Điều 182. Tội vi phạm chế độ một vợ, một chồng |  |  |  |  |  |  |  |  |
| 2 | `24778` | 0.15000000 | 0.67981684 | 2 | - | - | 1 | 1 | duplicated_chunk_text, structure_warning |
|  | title: Bo-Luat-hinh-su-1999-15-1999-QH10-46056; preview: Chương XV CÁC TỘI XÂM PHẠM CHẾ ĐỘ HÔN NHÂN VÀ GIA ĐÌNH Điều 151. Tội ngược đãi hoặc hành hạ ông bà, cha mẹ, vợ chồng, con, cháu, người có công nuôi dưỡng mình Người nào ngược đãi hoặc hành hạ ông bà, cha mẹ, vợ chồng, co |  |  |  |  |  |  |  |  |
| 3 | `126466` | 0.11875000 | 0.05651094 | - | 1 | - | 14 | 1 | structure_warning, possible_truncation_marker |
|  | title: VIỆN KIỂM SÁT NHÂN DÂN TỐI CAO; preview: viên trong Ban Dồn điền đổi thửa thôn D đã thu tiền xử lý đất các hộ dân lấn chiếm với tổng diện tích là 521,1m2 và số tiền là 133.000.000 đồng. Ngoài ra, do thiếu kinh phí xây dựng Nhà văn hóa thôn nên năm 2015, các thà |  |  |  |  |  |  |  |  |
| 4 | `184002` | 0.10000000 | 0.12851457 | 1 | - | - | 7 | 1 | duplicated_chunk_text, structure_warning |
|  | title: Luat-Thuc-hien-dan-chu-o-co-so-nam-2022-546085; preview: 3. Cán bộ, công chức, viên chức lợi dụng chức vụ, quyền hạn vi phạm quy định của Luật này, xâm phạm lợi ích của Nhà nước, quyền và lợi ích hợp pháp của tổ chức, cá nhân thì tùy theo tính chất, mức độ vi phạm mà bị xử lý  |  |  |  |  |  |  |  |  |
| 5 | `173521` | 0.09078947 | 0.04167954 | - | 2 | - | 17 | 1 | possible_truncation_marker |
|  | title: TÒA ÁN NHÂN DÂN TỐI CAO; preview: 10.000.000 đồng và 20 chỉ vàng. Khi Tòa án tiến hành hòa giải, A chỉ yêu cầu B trả 10.000.000 đồng, đối với 20 chỉ vàng A không yêu cầu B trả. Như vậy, Thẩm phán ra quyết định công nhận sự thỏa thuận của đương sự và xử l |  |  |  |  |  |  |  |  |

### Query `66174` ? GROUP_D

Question: Hồ sơ, trình tự thực hiện như thế nào?

Reasons: low_margin=False, high_disagreement=True, strong_bge_reorder=True; top1-top2 margin `0.03571429`, top5-top6 margin `0.00333333`, final Top5 support mean `1.20`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `189268` | 0.15238095 | 0.94235575 | 2 | 124 | - | 1 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-105-2020-ND-CP-quy-dinh-chinh-sach-phat-trien-giao-duc-mam-non-452005; preview: 3. Hồ sơ, trình tự thực hiện |  |  |  |  |  |  |  |  |
| 2 | `263594` | 0.11666667 | 0.40879691 | 1 | - | - | 4 | 1 | structure_warning |
|  | title: Thong-tu-lien-tich-35-2014-TTLT-BGDDT-BTC-huong-dan-66-2013-QD-TTg-ho-tro-chi-phi-hoc-tap-dan-toc-th; preview: Điều 4. Trình tự, thủ tục và hồ sơ |  |  |  |  |  |  |  |  |
| 3 | `163500` | 0.11666667 | 0.07033022 | - | 1 | - | 16 | 1 | structure_warning, possible_truncation_marker |
|  | title: Quyet-dinh-1944-QD-TCHQ-2014-Quy-che-Kiem-soat-thu-tuc-hanh-chinh-hai-quan-242416; preview: căn cứ cho việc đưa ra các phương án xử lý tối ưu. III. RÀ SOÁT, ĐÁNH GIÁ TÍNH HỢP LÝ, HỢP PHÁP CỦA CÁC BỘ PHẬN CẤU THÀNH TTHC Câu 1. Tên TTHC Tên TTHC được coi là rõ ràng, thống nhất nếu tên của một TTHC được quy định c |  |  |  |  |  |  |  |  |
| 4 | `55778` | 0.10000000 | 0.43098971 | 3 | - | - | 3 | 1 | structure_warning |
|  | title: Thong-tu-lien-tich-11-2015-TTLT-BVHTTDL-BTC-BGDDT-uu-dai-sinh-vien-nghe-thuat-truyen-thong-dac-thu-2; preview: 1. Trình tự, thủ tục và hồ sơ |  |  |  |  |  |  |  |  |
| 5 | `118434` | 0.10000000 | 0.11588966 | - | 2 | - | 10 | 1 | structure_warning, possible_truncation_marker |
|  | title: TỔ CÔNG TÁC ĐẶC BIỆT CỦA THỦ TƯỚNG CHÍNH PHỦ VỀ RÀ SOÁT, THÁO GỠ KHÓ KHĂN, VƯỚNG MẮC VÀ THÚC ĐẨY THỰC HIỆN DỰ ÁN ĐẦU TƯ; preview: nước giao đất, cho thuê đất, cho phép chuyển mục đích sử dụng đất để thực hiện dự án đầu tư được quy định tại Điều 14 Nghị định số 43/2014/NĐ-CP. 28. Việc xem xét các điều kiện đối với chủ đầu tư theo quy định Luật Nhà ở |  |  |  |  |  |  |  |  |

### Query `114634` ? GROUP_D

Question: Hồ sơ xin phép thành lập Hội đấu tranh vì nữ quyền gồm những giấy tờ tài liệu gì?

Reasons: low_margin=True, high_disagreement=True, strong_bge_reorder=False; top1-top2 margin `0.00309524`, top5-top6 margin `0.00395115`, final Top5 support mean `1.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `23975` | 0.14166667 | 0.53512603 | 1 | - | - | 2 | 1 | structure_warning |
|  | title: Quyet-dinh-616-QD-BNV-nam-2013-phe-duyet-Dieu-le-Hoi-Xa-hoi-hoc-Viet-Nam-190909; preview: 1. Hồ sơ xin gia nhập Hội gồm: |  |  |  |  |  |  |  |  |
| 2 | `304262` | 0.13857143 | 0.25750750 | 5 | 3 | - | 4 | 2 | none |
|  | title: Nghi-dinh-45-2010-ND-CP-to-chuc-hoat-dong-quan-ly-hoi-104561; preview: Chương 2. Điều 7. Hồ sơ xin phép thành lập hội |  |  |  |  |  |  |  |  |
| 3 | `76024` | 0.12727273 | 0.08511664 | - | 1 | - | 9 | 1 | duplicated_chunk_text |
|  | title: Nghi-dinh-110-2013-ND-CP-xu-phat-vi-pham-hanh-chinh-bo-tro-tu-phap-hanh-chinh-tu-phap-208274; preview: Chương 4. Điều 48. Hành vi vi phạm quy định về cấm kết hôn, vi phạm chế độ hôn nhân một vợ, một chồng; vi phạm quy định về ly hôn |  |  |  |  |  |  |  |  |
| 4 | `51256` | 0.11576923 | 0.74895430 | - | 50 | 18 | 1 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-02-2023-ND-CP-huong-dan-Luat-Tai-nguyen-nuoc-513343; preview: 4. Giấy tờ, tài liệu nộp kèm theo Đơn này gồm có: - Báo cáo hiện trạng khai thác, sử dụng nước và tình hình thực hiện giấy phép. - Phiếu kết quả phân tích chất lượng nguồn nước dưới đất không quá sáu (06) tháng tính đến  |  |  |  |  |  |  |  |  |
| 5 | `293727` | 0.09375000 | 0.03946662 | - | 2 | - | 14 | 1 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-82-2020-ND-CP-xu-phat-hanh-chinh-linh-vuc-hon-nhan-thi-hanh-an-pha-san-doanh-nghiep-392611; preview: c) Sử dụng tài liệu giả của cơ quan, tổ chức trong hồ sơ xin phép thành lập, hồ sơ đăng ký hoạt động văn phòng giám định tư pháp; |  |  |  |  |  |  |  |  |

### Query `132332` ? GROUP_D

Question: Nơi tiếp nhận phạt nguội

Reasons: low_margin=True, high_disagreement=True, strong_bge_reorder=True; top1-top2 margin `0.00357143`, top5-top6 margin `0.00708333`, final Top5 support mean `1.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `253103` | 0.12500000 | 0.12925307 | 6 | - | - | 1 | 1 | structure_warning |
|  | title: Thong-tu-37-2017-TT-BGTVT-mau-bien-ban-quyet-dinh-trong-xu-phat-vi-pham-hanh-chinh-giao-thong-364646; preview: vi phạm/người đại diện tổ chức vi phạm. Trường hợp không xác định được đối tượng vi phạm hành chính, thì ghi «Không xác định được đối tượng vi phạm hành chính». Trường hợp cá nhân chết, mất tích hoặc tổ chức giải thể, ph |  |  |  |  |  |  |  |  |
| 2 | `90853` | 0.12142857 | 0.05427800 | - | 1 | - | 12 | 1 | none |
|  | title: Nghi-quyet-326-2016-UBTVQH14-muc-thu-mien-giam-thu-nop-quan-ly-su-dung-an-phi-le-phi-Toa-an-337085; preview: 4. Trường hợp tranh chấp hợp đồng mua bán tài sản, chuyển nhượng quyền sử dụng đất, một bên yêu cầu trả lại tiền, đặt cọc và phạt cọc, một bên chấp nhận trả số tiền cọc đã nhận và không chấp nhận phạt cọc, mà Tòa án chấp |  |  |  |  |  |  |  |  |
| 3 | `193708` | 0.11538462 | 0.07597501 | 11 | - | 1 | 7 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Thong-tu-65-2020-TT-BCA-quy-trinh-tuan-tra-xu-ly-vi-pham-hanh-chinh-ve-giao-thong-duong-bo-427300; preview: bị kỹ thuật nghiệp vụ theo quy định; Giao quyết định xử phạt vi phạm hành chính cho người bị xử phạt hoặc người đại diện hợp pháp, người được ủy quyền; Tiếp nhận, kiểm tra, đối chiếu biên lai thu tiền phạt (hoặc chứng từ |  |  |  |  |  |  |  |  |
| 4 | `10663` | 0.10416667 | 0.08518657 | 1 | - | - | 6 | 1 | duplicated_chunk_text |
|  | title: Thong-tu-06-2017-TT-BGTVT-quy-trinh-su-dung-phuong-tien-thiet-bi-ky-thuat-nghiep-vu-giao-thong-duong; preview: 4. Trường hợp lái xe hoặc chủ xe không có mặt tại nơi cân kiểm tra tải trọng xe, người sử dụng thiết bị cân in và ký vào phiếu cân kiểm tra tải trọng xe theo quy định. Trong thời hạn 03 ngày kể từ ngày cân, bộ phận cân p |  |  |  |  |  |  |  |  |
| 5 | `50885` | 0.10208333 | 0.04671412 | 4 | - | 2 | 14 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Nghi-dinh-100-2019-ND-CP-xu-phat-vi-pham-hanh-chinh-linh-vuc-giao-thong-duong-bo-va-duong-sat-426369; preview: d) Điều khiển phương tiện, thiết bị thi công mà không có bằng, chứng chỉ chuyên môn theo quy định; |  |  |  |  |  |  |  |  |

### Query `18778` ? GROUP_D

Question: Việc doanh nghiệp tạm ngưng hoạt động được quy định như thế nào?

Reasons: low_margin=True, high_disagreement=True, strong_bge_reorder=True; top1-top2 margin `0.01166667`, top5-top6 margin `0.00110390`, final Top5 support mean `1.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `72609` | 0.12666667 | 0.22839886 | 1 | - | - | 3 | 1 | duplicated_chunk_text, structure_warning |
|  | title: Thong-tu-120-2016-TT-BTC-Dieu-le-hoat-dong-Cong-ty-trach-nhiem-huu-han-Xo-so-dien-toan-Viet-Nam-3196; preview: 1. Chủ sở hữu Công ty quyết định việc tạm ngừng kinh doanh của Công ty theo đề nghị của Chủ tịch Công ty. Quyết định tạm ngừng kinh doanh của chủ sở hữu Công ty phải được lập thành văn bản. |  |  |  |  |  |  |  |  |
| 2 | `281238` | 0.11500000 | 0.05123386 | - | 1 | - | 18 | 1 | duplicated_chunk_text, structure_warning, possible_truncation_marker |
|  | title: Thong-tu-133-2016-TT-BTC-huong-dan-che-do-ke-toan-doanh-nghiep-nho-va-vua-284997; preview: vốn hóa. - Tỷ lệ vốn hóa được sử dụng để xác định chi phí đi vay được vốn hóa trong kỳ: Nêu rõ tỷ lệ vốn hóa này là bao nhiêu? (9) Nguyên tắc ghi nhận vốn chủ sở hữu: - Vốn góp của chủ sở hữu có được ghi nhận theo số vốn |  |  |  |  |  |  |  |  |
| 3 | `72196` | 0.10885734 | 0.24980810 | 27 | 151 | - | 1 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Thong-tu-05-2018-TT-NHNN-chap-thuan-thay-doi-danh-sach-bau-nhan-su-to-chuc-tin-dung-la-hop-tac-xa-37; preview: Chương II QUY ĐỊNH CỤ THỂ Mục 1. NHỮNG THAY ĐỔI PHẢI ĐƯỢC CHẤP THUẬN Điều 10. Tạm ngừng hoạt động kinh doanh từ 05 ngày làm việc trở lên, trừ trường hợp tạm ngừng hoạt động do sự kiện bất khả kháng |  |  |  |  |  |  |  |  |
| 4 | `69835` | 0.10220588 | 0.24169974 | 6 | 134 | - | 2 | 2 | duplicated_chunk_text |
|  | title: Luat-Doanh-nghiep-2005-60-2005-QH11-7019; preview: Chương VI DOANH NGHIỆP TƯ NHÂN Điều 144. Cho thuê doanh nghiệp Chủ doanh nghiệp tư nhân có quyền cho thuê toàn bộ doanh nghiệp của mình nhưng phải báo cáo bằng văn bản kèm theo bản sao hợp đồng cho thuê có công chứng đến |  |  |  |  |  |  |  |  |
| 5 | `42223` | 0.09500000 | 0.08256922 | - | 2 | - | 13 | 1 | duplicated_chunk_text, structure_warning, possible_truncation_marker |
|  | title: Thong-tu-200-2014-TT-BTC-huong-dan-Che-do-ke-toan-Doanh-nghiep-263599; preview: chưa sử dụng đang ghi trên sổ kế toán. (17) Nguyên tắc ghi nhận doanh thu chưa thực hiện - Doanh thu chưa thực hiện được ghi nhận trên cơ sở nào? - Phương pháp phân bổ doanh thu chưa thực hiện. (18) Nguyên tắc ghi nhận t |  |  |  |  |  |  |  |  |

### Query `73712` ? GROUP_D

Question: Khái niệm về sổ đỏ là gì?

Reasons: low_margin=False, high_disagreement=True, strong_bge_reorder=False; top1-top2 margin `0.02166667`, top5-top6 margin `0.00296791`, final Top5 support mean `1.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `67945` | 0.15000000 | 0.08149764 | 2 | - | - | 1 | 1 | duplicated_chunk_text |
|  | title: Luat-Dat-dai-2003-13-2003-QH11-51685; preview: 1. Bản đồ địa chính là thành phần của hồ sơ địa chính phục vụ thống nhất quản lý nhà nước về đất đai. |  |  |  |  |  |  |  |  |
| 2 | `107674` | 0.12833333 | 0.04000708 | 3 | - | 13 | 2 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Thong-tu-24-2014-TT-BTNMT-ho-so-dia-chinh-236560; preview: 3. Đăng ký biến động đất đai, tài sản gắn liền với đất (sau đây gọi là đăng ký biến động) là việc thực hiện thủ tục để ghi nhận sự thay đổi về một hoặc một số thông tin đã đăng ký vào hồ sơ địa chính theo quy định của ph |  |  |  |  |  |  |  |  |
| 3 | `99335` | 0.11875000 | 0.00083572 | - | 1 | - | 14 | 1 | structure_warning, possible_truncation_marker |
|  | title: Quyet-dinh-2885-QD-BYT-2022-tai-lieu-truyen-thong-truc-tiep-cham-soc-suc-khoe-sinh-san-vi-thanh-nien; preview: Bút bi Các bản chiếu 5 Xu hướng tính dục Nêu khái niệm xu hướng tính dục. Hỏi: theo các em, con người có những xu hướng tính dục như thế nào? Phát cho mỗi HS 1 thẻ màu, mời các em viết 1 xu hướng tính dục lên thẻ. GV yêu |  |  |  |  |  |  |  |  |
| 4 | `116389` | 0.09666667 | 0.00876249 | 1 | - | - | 8 | 1 | structure_warning |
|  | title: Thong-tu-lien-tich-06-2018-TTLT-VKSNDTC-TANDTC-BCA-BTP-BLDTBXH-ve-thu-tuc-to-tung-hinh-su-403552; preview: b) Đã được đào tạo, tập huấn, bồi dưỡng về kỹ năng giải quyết các vụ án hình sự có người tham gia tố tụng là người dưới 18 tuổi; |  |  |  |  |  |  |  |  |
| 5 | `81598` | 0.09160428 | 0.02669817 | - | 185 | 3 | 4 | 2 | duplicated_chunk_text, structure_warning |
|  | title: Bo-luat-dan-su-2015-296215; preview: 1. Chủ sở hữu có bất động sản bị vây bọc bởi các bất động sản của các chủ sở hữu khác mà không có hoặc không đủ lối đi ra đường công cộng, có quyền yêu cầu chủ sở hữu bất động sản vây bọc dành cho mình một lối đi hợp lý  |  |  |  |  |  |  |  |  |

### Query `5190` ? GROUP_D

Question: Chủ tịch Hội doanh nghiệp vừa và nhỏ Việt Đức có quyền hạn và nhiệm vụ như thế nào?

Reasons: low_margin=False, high_disagreement=True, strong_bge_reorder=False; top1-top2 margin `0.02508503`, top5-top6 margin `0.00375000`, final Top5 support mean `1.40`.

| Rank | Document | RRF | BGE score (FT) | Dense | BM25 | KNN | BGE rank | Support | Corpus audit |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `92156` | 0.16666667 | 0.80327201 | 1 | - | - | 1 | 1 | structure_warning |
|  | title: Quyet-dinh-763-QD-BNV-nam-2013-phe-duyet-Dieu-le-Hoi-Tam-ly-hoc-xa-hoi-Viet-Nam-199805; preview: 2. Chủ tịch Hội có nhiệm vụ và quyền hạn: |  |  |  |  |  |  |  |  |
| 2 | `115130` | 0.14158163 | 0.70696545 | 47 | 1 | - | 6 | 2 | structure_warning |
|  | title: Quyet-dinh-53-2004-QD-BNV-phe-duyet-dieu-le-hoi-doanh-nghiep-nho-va-vua-Viet-Duc-18590; preview: Điều 3. Chủ tịch Hội doanh nghiệp nhỏ và vừa Việt - Đức, Vụ trưởng Vụ Tổ chức phi chính phủ chịu trách nhiệm thi hành Quyết định này./. KT. BỘ TRƯỞNG BỘ NỘI VỤ THỨ TRƯỞNG Đặng Quốc Tiến ĐIỀU LỆ HỘI DOANH NGHIỆP NHỎ VÀ VỪ |  |  |  |  |  |  |  |  |
| 3 | `248571` | 0.11077236 | 0.78334993 | 4 | 121 | - | 2 | 2 | structure_warning |
|  | title: Quyet-dinh-762-QD-BNV-nam-2013-phe-duyet-Dieu-le-Hoi-Huu-nghi-Viet-Nam-Duc-199804; preview: 2. Chủ tịch Hội có nhiệm vụ, quyền hạn: |  |  |  |  |  |  |  |  |
| 4 | `60889` | 0.11000000 | 0.78334993 | 2 | - | - | 3 | 1 | structure_warning |
|  | title: Quyet-dinh-388-QD-BNV-nam-2014-Dieu-le-Hoi-Chien-si-Thanh-co-Quang-Tri-1972-227563; preview: 2. Chủ tịch Hội có nhiệm vụ, quyền hạn: |  |  |  |  |  |  |  |  |
| 5 | `90799` | 0.09375000 | 0.13693199 | - | 2 | - | 14 | 1 | duplicated_chunk_text, structure_warning |
|  | title: Thong-tu-15-2022-TT-BKHCN-quan-ly-Chuong-trinh-ho-tro-doanh-nghiep-nang-cao-nang-suat-532445; preview: Chương IV TỔ CHỨC THỰC HIỆN Điều 45. Liên đoàn Thương mại và Công nghiệp Việt Nam, Hiệp hội doanh nghiệp nhỏ và vừa Việt Nam, Liên minh Hợp tác xã Việt Nam và các hiệp hội doanh nghiệp trung ương và địa phương |  |  |  |  |  |  |  |  |

## Conclusion

- PRIMARY_SIGNAL: **MIXED**.
- Evidence shows meaningful reranking movement and RRF uncertainty, while retrieval support disagreement is also present. The current audit does not establish a strong corpus-quality signal: corpus flags in the bounded manual sample are heuristic and are not sufficient to authorize a processed-pv1 rebuild.
- NEXT_ACTION: **INVESTIGATE_RANKING_BEFORE_CORPUS_REBUILD**.
- No submission was changed or created by this diagnostic audit.

