# Ranking Failure Attribution ? F1?F4 K20

## Scope

- CPU/read-only failure attribution only. No GPU, Modal, retrieval/BGE rerun, corpus/data edit, weight search, submission creation, or submission modification.
- Gold is used only for existing F1?F4 validation (`train.json`); Fold0 is excluded. Private answers are not used.

## Reproduction gate

- Validation queries: **5600/5600**; F1?F4 only; Fold0: `NO`.
- K20 q-docs: **112000/112000**; candidates/query: `20`; identity/finite checks: `PASS`.
- Current `RETRIEVAL_RRF_NO_LABEL` Recall@5: **0.924514880952381**; exact reproduction: **YES**.
- Candidate oracle macro Recall@5 ceiling: **0.966494047619048**.

## Miss decomposition

- Retrieval miss: **136** (2.43%) ? no relevant document in K20.
- Ranking miss: **187** (3.34%) ? relevant document exists in K20 but current Top5 misses it.
- Successful queries: **5277** (94.23%).
- Oracle gap: **0.041979166666667**.
- Query-count decomposition: candidate absence `136` vs ranking/fusion failure `187`.

## Relevant-document locations for ranking misses

- Rank distributions count relevant-document occurrences. Missing source rank is represented as `21` (not present in that source?s K20 list).
- Current RRF rank 6?10: `141`; 11?15: `40`; 16?20: `14`.
| Source | Median relevant-doc rank | Absent occurrences (rank 21) |
|---|---:|---:|
| dense | 11 | 58 |
| bm25 | 15 | 76 |
| knn_word | 17 | 94 |

## Fixed source ablations (no weight sweep)

| Policy | Recall@5 | F1 | F2 | F3 | F4 |
|---|---:|---:|---:|---:|---:|
| CURRENT | 0.924514881 | 0.929821429 | 0.928452381 | 0.921785714 | 0.918000000 |
| NO_BGE | 0.913235119 | 0.915357143 | 0.916726190 | 0.907976190 | 0.912880952 |
| NO_BM25 | 0.904812500 | 0.912559524 | 0.903690476 | 0.896964286 | 0.906035714 |
| NO_DENSE | 0.913726190 | 0.914107143 | 0.916547619 | 0.911964286 | 0.912285714 |
| NO_KNN | 0.904500000 | 0.910892857 | 0.910654762 | 0.898690476 | 0.897761905 |
| BGE_ONLY | 0.842357143 | 0.856071429 | 0.839761905 | 0.836309524 | 0.837285714 |
| BM25_ONLY | 0.845258929 | 0.854821429 | 0.847559524 | 0.831011905 | 0.847642857 |
| DENSE_ONLY | 0.816166667 | 0.820714286 | 0.828750000 | 0.807738095 | 0.807464286 |
| KNN_ONLY | 0.489568452 | 0.493630952 | 0.487797619 | 0.479583333 | 0.497261905 |

## Source rescue/harm versus CURRENT

| Removed source | Rescued | Harmed | Net |
|---|---:|---:|---:|
| bge | 52 | 122 | -70 |
| bm25 | 55 | 157 | -102 |
| dense | 28 | 87 | -59 |
| knn_word | 21 | 125 | -104 |

- Rescued = CURRENT miss ? removal hits; harmed = CURRENT hit ? removal misses. These are validation transitions, not Private labels.

## Ranking-miss cause signals

| Signal | Queries | Share of ranking misses |
|---|---:|---:|
| SINGLE_SOURCE_DOMINANCE | 68 | 36.36% |
| HIGH_DISAGREEMENT | 68 | 36.36% |
| STRONG_BGE_REORDER | 32 | 17.11% |
| LOW_MARGIN | 63 | 33.69% |

Relevant-doc best source counts: dense=61, bm25=54, knn_word=72.
Relevant-doc lowest source counts: dense=53, bm25=31, knn_word=103.

## BGE attribution

- BGE_DEMOTION: **744** relevant q-doc occurrences (best retrieval rank ?5, BGE rank >5).
- BGE_RESCUE: **77** relevant q-doc occurrences (best retrieval rank >5, BGE rank ?5).
- BGE-only Recall@5: `0.842357143`.
- Dense-only Recall@5: `0.816166667`.
- BGE-only minus Dense-only: `+0.026190476`.

## Source contribution summary

| Source | Remove-source Recall delta | Rescues | Harms | Net |
|---|---:|---:|---:|---:|
| bge | -0.011279762 | 52 | 122 | -70 |
| bm25 | -0.019702381 | 55 | 157 | -102 |
| dense | -0.010788690 | 28 | 87 | -59 |
| knn_word | -0.020014881 | 21 | 125 | -104 |

## Private pattern transfer (diagnostic only)

- Queries with at least one validation-proven pattern: **2080/2080**.
- HIGH_DISAGREEMENT-like: `42`.
- SINGLE_SOURCE_DOMINANCE-like: `42`.
- STRONG_BGE_REORDER-like: `375`.
- BGE_DEMOTION-like: `2080`.
- LOW_MARGIN-like: `395`.
- Private `BGE_DEMOTION-like` is only a candidate-level proxy: a frozen retrieval Top5 candidate moves below BGE rank 5. Without Private gold, it is not a claim about a relevant document.
- These are not Private error claims and do not change Private Top5.

## Private manual-review shortlist (20 max)

| Query | Risk flags | Current Top5 | Reason |
|---|---:|---|---|
| `88908` | 5 | 65968 273469 31877 28679 68024 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, STRONG_BGE_REORDER, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00190476 |
|  |  | question: Thời báo Ngân hàng là đơn vị trực thuộc cơ quan nào? | Top10 ranks: 1:65968(d=2,b=5,k=-,bg=3); 2:273469(d=1,b=4,k=-,bg=7); 3:31877(d=72,b=1,k=9,bg=13); 4:28679(d=4,b=-,k=-,bg=1); 5:68024(d=-,b=-,k=3,bg=2); 6:204342(d=-,b=12,k=1,bg=10); 7:6656(d=32,b=2,k=-,bg=8); 8:32812(d=86,b=6,k=-,bg=5); 9:243709(d=-,b=3,k=-,bg=17); 10:25061(d=94,b=7,k=-,bg=6) |
| `132332` | 5 | 253103 90853 193708 10663 50885 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, STRONG_BGE_REORDER, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00708333 |
|  |  | question: Nơi tiếp nhận phạt nguội | Top10 ranks: 1:253103(d=6,b=-,k=-,bg=1); 2:90853(d=-,b=1,k=-,bg=12); 3:193708(d=11,b=-,k=1,bg=7); 4:10663(d=1,b=-,k=-,bg=6); 5:50885(d=4,b=-,k=2,bg=14); 6:3551(d=-,b=2,k=-,bg=13); 7:283257(d=30,b=37,k=-,bg=2); 8:219419(d=19,b=-,k=4,bg=5); 9:53565(d=-,b=-,k=6,bg=3); 10:53190(d=-,b=3,k=-,bg=11) |
| `18778` | 5 | 72609 281238 72196 69835 42223 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, STRONG_BGE_REORDER, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00110390 |
|  |  | question: Việc doanh nghiệp tạm ngưng hoạt động được quy định như thế nào? | Top10 ranks: 1:72609(d=1,b=-,k=-,bg=3); 2:281238(d=-,b=1,k=-,bg=18); 3:72196(d=27,b=151,k=-,bg=1); 4:69835(d=6,b=134,k=-,bg=2); 5:42223(d=-,b=2,k=-,bg=13); 6:200355(d=9,b=26,k=3,bg=10); 7:129823(d=-,b=132,k=1,bg=16); 8:249551(d=8,b=13,k=-,bg=9); 9:115832(d=2,b=-,k=-,bg=17); 10:265035(d=14,b=177,k=-,bg=4) |
| `40498` | 4 | 206598 69 270233 149091 144472 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00359848 |
|  |  | question: Khái niệm về kiểm soát tài liệu? | Top10 ranks: 1:206598(d=-,b=1,k=-,bg=3); 2:69(d=8,b=6,k=-,bg=1); 3:270233(d=10,b=3,k=-,bg=4); 4:149091(d=26,b=2,k=-,bg=6); 5:144472(d=1,b=-,k=-,bg=9); 6:25241(d=2,b=22,k=20,bg=14); 7:215923(d=69,b=28,k=-,bg=2); 8:23402(d=-,b=-,k=1,bg=16); 9:292871(d=3,b=58,k=-,bg=7); 10:191651(d=6,b=16,k=-,bg=10) |
| `114634` | 4 | 23975 304262 76024 51256 293727 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00395115 |
|  |  | question: Hồ sơ xin phép thành lập Hội đấu tranh vì nữ quyền gồm những giấy tờ tài liệu gì? | Top10 ranks: 1:23975(d=1,b=-,k=-,bg=2); 2:304262(d=5,b=3,k=-,bg=4); 3:76024(d=-,b=1,k=-,bg=9); 4:51256(d=-,b=50,k=18,bg=1); 5:293727(d=-,b=2,k=-,bg=14); 6:268656(d=85,b=4,k=-,bg=6); 7:100761(d=51,b=5,k=-,bg=5); 8:2113(d=-,b=-,k=1,bg=15); 9:114415(d=-,b=148,k=11,bg=3); 10:34866(d=2,b=-,k=-,bg=16) |
| `53432` | 4 | 254991 53504 243916 235502 200355 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00666667 |
|  |  | question: Cơ quan kiểm tra các hoạt động kinh doanh thuốc lá có trách nhiệm và quyền hạn thế nào? | Top10 ranks: 1:254991(d=1,b=3,k=2,bg=2); 2:53504(d=2,b=1,k=-,bg=1); 3:243916(d=10,b=4,k=-,bg=3); 4:235502(d=-,b=2,k=-,bg=7); 5:200355(d=-,b=-,k=1,bg=13); 6:51210(d=26,b=8,k=-,bg=5); 7:234609(d=3,b=195,k=-,bg=8); 8:247318(d=-,b=68,k=12,bg=4); 9:261451(d=-,b=10,k=16,bg=10); 10:108598(d=15,b=28,k=-,bg=6) |
| `115756` | 4 | 297690 35496 119286 113741 163254 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00333333 |
|  |  | question: Chi phí làm sổ đỏ bao gồm những gì? | Top10 ranks: 1:297690(d=4,b=-,k=-,bg=1); 2:35496(d=2,b=-,k=-,bg=2); 3:119286(d=-,b=1,k=-,bg=19); 4:113741(d=-,b=-,k=1,bg=7); 5:163254(d=-,b=-,k=4,bg=3); 6:20457(d=-,b=2,k=-,bg=18); 7:101752(d=1,b=-,k=-,bg=12); 8:290221(d=-,b=3,k=-,bg=11); 9:153999(d=3,b=-,k=-,bg=6); 10:283304(d=7,b=90,k=-,bg=4) |
| `142774` | 4 | 242302 184968 165017 232471 53007 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, STRONG_BGE_REORDER, BGE_DEMOTION; top5-top6 margin=0.00170592 |
|  |  | question: BOG là gì? | Top10 ranks: 1:242302(d=-,b=6,k=-,bg=1); 2:184968(d=-,b=1,k=-,bg=9); 3:165017(d=-,b=2,k=-,bg=8); 4:232471(d=-,b=3,k=-,bg=6); 5:53007(d=51,b=17,k=-,bg=2); 6:99335(d=-,b=4,k=-,bg=5); 7:81598(d=-,b=-,k=3,bg=4); 8:6411(d=-,b=-,k=1,bg=15); 9:184384(d=1,b=-,k=-,bg=19); 10:166280(d=33,b=25,k=-,bg=3) |
| `76050` | 4 | 102434 229347 305455 187506 46918 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00380952 |
|  |  | question: Nộp đơn khởi kiện dân sự có cần công chứng giấy tờ không? | Top10 ranks: 1:102434(d=35,b=8,k=-,bg=1); 2:229347(d=-,b=1,k=-,bg=9); 3:305455(d=4,b=-,k=-,bg=2); 4:187506(d=3,b=-,k=-,bg=3); 5:46918(d=1,b=-,k=-,bg=8); 6:42262(d=-,b=5,k=-,bg=4); 7:298992(d=-,b=2,k=-,bg=20); 8:160120(d=2,b=-,k=-,bg=7); 9:134404(d=-,b=-,k=1,bg=17); 10:46291(d=-,b=3,k=-,bg=16) |
| `122546` | 4 | 177170 17545 279475 42262 205344 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00000000 |
|  |  | question: Đặt tiền bảo lãnh phương tiện giao thông trong những trường hợp nào? | Top10 ranks: 1:177170(d=1,b=-,k=-,bg=1); 2:17545(d=3,b=-,k=7,bg=2); 3:279475(d=-,b=1,k=-,bg=14); 4:42262(d=-,b=2,k=14,bg=10); 5:205344(d=2,b=-,k=-,bg=7); 6:129823(d=-,b=-,k=1,bg=16); 7:196603(d=-,b=3,k=-,bg=11); 8:21010(d=-,b=4,k=-,bg=8); 9:193708(d=-,b=81,k=4,bg=5); 10:288109(d=31,b=30,k=-,bg=3) |
| `93374` | 4 | 130251 15491 281238 99335 92115 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00112546 |
|  |  | question: Nhiệm vụ và yêu cầu đối với một người kế toán như thế nào? | Top10 ranks: 1:130251(d=2,b=-,k=6,bg=1); 2:15491(d=1,b=-,k=-,bg=2); 3:281238(d=-,b=1,k=-,bg=6); 4:99335(d=-,b=2,k=-,bg=17); 5:92115(d=27,b=-,k=7,bg=3); 6:95164(d=11,b=113,k=8,bg=4); 7:122329(d=-,b=-,k=1,bg=20); 8:262949(d=6,b=13,k=-,bg=7); 9:177925(d=-,b=3,k=-,bg=18); 10:265921(d=7,b=50,k=-,bg=5) |
| `71110` | 4 | 129823 102434 19236 208105 46291 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, STRONG_BGE_REORDER, BGE_DEMOTION; top5-top6 margin=0.00142857 |
|  |  | question: Điều kiện để xử lý kỷ luật người lao động bằng hình thức sa thải | Top10 ranks: 1:129823(d=1,b=-,k=1,bg=4); 2:102434(d=64,b=11,k=7,bg=1); 3:19236(d=-,b=1,k=-,bg=15); 4:208105(d=2,b=-,k=-,bg=3); 5:46291(d=-,b=2,k=-,bg=8); 6:143446(d=5,b=-,k=-,bg=2); 7:298992(d=-,b=3,k=-,bg=7); 8:95879(d=-,b=6,k=11,bg=6); 9:245154(d=6,b=95,k=-,bg=5); 10:187506(d=3,b=-,k=-,bg=9) |
| `66174` | 4 | 189268 263594 163500 55778 118434 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, STRONG_BGE_REORDER, BGE_DEMOTION; top5-top6 margin=0.00333333 |
|  |  | question: Hồ sơ, trình tự thực hiện như thế nào? | Top10 ranks: 1:189268(d=2,b=124,k=-,bg=1); 2:263594(d=1,b=-,k=-,bg=4); 3:163500(d=-,b=1,k=-,bg=16); 4:55778(d=3,b=-,k=-,bg=3); 5:118434(d=-,b=2,k=-,bg=10); 6:189230(d=-,b=-,k=1,bg=8); 7:70872(d=49,b=79,k=-,bg=2); 8:875(d=-,b=3,k=-,bg=12); 9:280282(d=4,b=-,k=5,bg=19); 10:246534(d=-,b=101,k=3,bg=9) |
| `54374` | 4 | 245154 24778 126466 184002 173521 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, STRONG_BGE_REORDER, BGE_DEMOTION; top5-top6 margin=0.01292151 |
|  |  | question: Công chức nhận tiền của người dân sẽ bị xử lý như thế nào? | Top10 ranks: 1:245154(d=3,b=-,k=1,bg=2); 2:24778(d=2,b=-,k=-,bg=1); 3:126466(d=-,b=1,k=-,bg=14); 4:184002(d=1,b=-,k=-,bg=7); 5:173521(d=-,b=2,k=-,bg=17); 6:17545(d=10,b=54,k=5,bg=9); 7:294399(d=30,b=37,k=-,bg=3); 8:257055(d=59,b=44,k=-,bg=4); 9:115640(d=25,b=34,k=-,bg=6); 10:241299(d=43,b=53,k=-,bg=5) |
| `91268` | 4 | 221604 198723 299694 111409 229585 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00000000 |
|  |  | question: Quỹ Nước sạch và Bảo vệ môi trường Việt Nam có chức năng là gì? | Top10 ranks: 1:221604(d=1,b=1,k=-,bg=1); 2:198723(d=26,b=5,k=3,bg=2); 3:299694(d=35,b=8,k=1,bg=5); 4:111409(d=-,b=3,k=-,bg=3); 5:229585(d=2,b=-,k=-,bg=4); 6:87797(d=-,b=2,k=-,bg=10); 7:188274(d=4,b=16,k=-,bg=7); 8:75382(d=-,b=4,k=-,bg=11); 9:303096(d=-,b=6,k=16,bg=12); 10:142223(d=5,b=-,k=-,bg=6) |
| `32794` | 4 | 282279 119471 300748 27511 160381 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, STRONG_BGE_REORDER, BGE_DEMOTION; top5-top6 margin=0.00254813 |
|  |  | question: Thẩm tra hồ sơ pháp lý thực hiện như thế nào? | Top10 ranks: 1:282279(d=1,b=1,k=-,bg=2); 2:119471(d=3,b=-,k=-,bg=1); 3:300748(d=-,b=87,k=4,bg=3); 4:27511(d=-,b=2,k=-,bg=14); 5:160381(d=-,b=186,k=1,bg=19); 6:118434(d=-,b=3,k=-,bg=13); 7:191413(d=2,b=117,k=-,bg=9); 8:103779(d=7,b=105,k=-,bg=4); 9:112039(d=4,b=100,k=-,bg=6); 10:280282(d=-,b=-,k=2,bg=18) |
| `75296` | 4 | 105724 240806 139192 15810 191413 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, STRONG_BGE_REORDER, BGE_DEMOTION; top5-top6 margin=0.01578089 |
|  |  | question: Để thăng hạng kỹ thuận lên phi công giảng viên bay quân sự cấp 3 thì cần phải đáp ứng được những tiêu chuẩn nào? | Top10 ranks: 1:105724(d=1,b=1,k=1,bg=1); 2:240806(d=-,b=2,k=-,bg=3); 3:139192(d=2,b=14,k=2,bg=18); 4:15810(d=-,b=-,k=3,bg=2); 5:191413(d=4,b=24,k=-,bg=4); 6:15628(d=20,b=4,k=-,bg=13); 7:227296(d=-,b=3,k=-,bg=14); 8:112316(d=3,b=-,k=-,bg=6); 9:199066(d=-,b=11,k=16,bg=5); 10:128503(d=5,b=162,k=-,bg=10) |
| `66112` | 4 | 192853 122601 19035 113145 118099 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00102814 |
|  |  | question: Vùng hoạt động vui chơi, giải trí dưới nước được quy định như thế nào? | Top10 ranks: 1:192853(d=1,b=1,k=2,bg=1); 2:122601(d=88,b=122,k=1,bg=9); 3:19035(d=-,b=2,k=-,bg=11); 4:113145(d=-,b=3,k=-,bg=8); 5:118099(d=-,b=4,k=-,bg=6); 6:303096(d=82,b=31,k=-,bg=2); 7:290394(d=32,b=38,k=-,bg=3); 8:170808(d=2,b=-,k=-,bg=13); 9:139585(d=-,b=183,k=9,bg=5); 10:226472(d=4,b=22,k=-,bg=18) |
| `42158` | 4 | 234480 218843 228085 9980 63721 | HIGH_DISAGREEMENT, SINGLE_SOURCE_DOMINANCE, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00021978 |
|  |  | question: Giám đốc Nhà văn hóa Lao động phải trình độ như thế nào? | Top10 ranks: 1:234480(d=1,b=1,k=1,bg=1); 2:218843(d=4,b=-,k=-,bg=2); 3:228085(d=-,b=2,k=-,bg=12); 4:9980(d=80,b=33,k=7,bg=3); 5:63721(d=-,b=3,k=-,bg=11); 6:272553(d=3,b=-,k=-,bg=5); 7:38738(d=2,b=-,k=-,bg=8); 8:121997(d=90,b=-,k=3,bg=6); 9:208105(d=6,b=76,k=-,bg=4); 10:62543(d=-,b=4,k=-,bg=18) |
| `81068` | 3 | 72265 64367 236791 98892 282223 | STRONG_BGE_REORDER, BGE_DEMOTION, LOW_MARGIN; top5-top6 margin=0.00142340 |
|  |  | question: Đăng ký thường trú tại chỗ ở mới thì cần phải có những giấy tờ gì? | Top10 ranks: 1:72265(d=1,b=1,k=2,bg=2); 2:64367(d=2,b=2,k=1,bg=1); 3:236791(d=3,b=5,k=-,bg=3); 4:98892(d=16,b=11,k=3,bg=20); 5:282223(d=40,b=6,k=-,bg=5); 6:6411(d=21,b=4,k=-,bg=10); 7:208990(d=6,b=63,k=-,bg=4); 8:188894(d=29,b=8,k=-,bg=6); 9:77650(d=4,b=149,k=8,bg=16); 10:144551(d=20,b=7,k=-,bg=9) |

## Conclusion

- PRIMARY_BOTTLENECK: **RANK_FUSION_BOTTLENECK**.
- The main gap is ranking/fusion: `187` ranking-miss queries vs `136` retrieval-miss queries; current-to-oracle macro Recall gap is `0.041979`.
- Fixed ablations show mixed source effects; this task does not choose weights or implement a new policy.
- NEXT_ACTION: **DESIGN_TARGETED_RANKING_V2**.
- No Private submission was changed or created.

