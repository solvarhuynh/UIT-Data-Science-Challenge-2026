# Báo cáo tổng hợp cải thiện điểm UDSC2026 Task1 LegalIR

**Mốc so sánh**
- Baseline public trước đây: **~0.930**
- Submission mới: **~0.939**
- Mức tăng tuyệt đối: **+0.009**
- Tương đương: **+0.9 điểm phần trăm**
- Tăng tương đối so với baseline: **~0.97%**

> Lưu ý: leaderboard chỉ cho biết điểm cuối cùng của cả pipeline. Vì vậy không thể khẳng định chính xác mỗi thay đổi riêng lẻ đóng góp bao nhiêu vào +0.009 nếu không có ablation public riêng. Phần dưới tổng hợp các tác động có bằng chứng từ OOF/fold validation và từ thay đổi pipeline thực tế.

## 1. Thay baseline ~0.930 bằng pipeline V3A có kiểm soát

Baseline cũ chủ yếu lấy top-5 từ hệ thống retrieval/reranking trước đó. Pipeline mới không thay toàn bộ ranking một cách tự do mà dùng **V3A residual policy** để quyết định có nên thay đúng **một document** trong top-5 hay không.

Thiết kế chính:
- bảo vệ **top-1, top-2, top-3**;
- chỉ cho phép drop ở **rank 4 hoặc rank 5**;
- tối đa **một swap/query**;
- dùng feature của candidate hiện tại và candidate mới để quyết định swap.

Tác động:
- giảm nguy cơ làm hỏng những query baseline vốn đã đúng;
- tập trung sửa các lỗi nằm ở phần cuối top-5;
- biến một hệ thống retrieval rộng thành một policy sửa lỗi có kiểm soát.

Validation lịch sử:
- strict OOF baseline: **0.924445**
- V3A folds 1–4: **0.929649**
- baseline folds 1–4: **0.925929**
- cải thiện folds 1–4: **+0.003720**
- Fold 0: từ **0.918512** lên **0.921726** (**+0.003214**)

Điểm quan trọng là V3A cho cải thiện dương trên các fold đã kiểm tra, nên được chọn cho production thay vì các policy có mức tăng nhỏ hơn hoặc kém ổn định hơn.

## 2. Fresh Public Dense K500 mở rộng recall

Pipeline mới tạo lại dense retrieval cho public với:
- **1000 public queries**
- **500 candidates/query**

Artifact:
`artifacts/task1/recovery_096/final_public_v3/public_raw_k500.jsonl`

Tác động:
- tăng khả năng đưa document đúng vào candidate pool;
- cung cấp vùng tìm kiếm rộng hơn so với chỉ dựa vào top-200/top-5;
- là nguồn đầu vào cho adaptive expansion và candidate union.

Đây là thay đổi quan trọng về **recall ceiling**: policy phía sau chỉ có thể sửa ranking nếu document tốt đã xuất hiện trong candidate set.

## 3. BGE K200 được chạy lại tương thích với fresh K500

Cache public BGE cũ không còn tương thích với fresh K500 mới, nên không được tái sử dụng.

Pipeline mới chạy lại BGE trên **first 200 fresh K500 chunks/query** bằng canonical reranker:
`models/reranker`

Output:
`public_bge_k200_compatible.jsonl`

Tác động:
- loại bỏ mismatch giữa candidate ordering mới và reranker score cũ;
- đảm bảo các feature dựa trên reranker thực sự tương ứng với candidate hiện tại;
- tránh silent feature corruption, một lỗi có thể làm policy chọn sai swap dù model policy đúng.

Đây là một tác động vừa về chất lượng ranking, vừa về **feature parity/correctness**.

## 4. Adaptive K500 chỉ mở rộng khi query có nguy cơ miss ở K200

Thay vì dùng K500 cho mọi query một cách mù quáng, pipeline dùng classifier đã final-fit để xác định query nào có khả năng bị miss nếu chỉ dùng K200.

Model:
`adaptive_k500_final_fit.joblib`

Thông số production:
- selected budget: **0.25**
- threshold: **0.038521777982483386**
- training queries: **7000**
- không dùng public labels

Tác động:
- giữ candidate pool gọn ở các query dễ;
- dùng phần K201–K500 cho các query có khả năng cần recall bổ sung;
- giảm noise so với việc mở rộng sâu cho tất cả query;
- tận dụng thêm candidate tốt mà K200 có thể bỏ lỡ.

Adaptive K500 vì vậy tác động chủ yếu lên **candidate recall có chọn lọc**.

## 5. Candidate Union kết hợp nhiều nguồn retrieval độc lập

Candidate set production không phụ thuộc vào một retriever duy nhất. Nó hợp nhất các nguồn:
- `adaptive_k500`
- `bm25`
- `knn_word`
- `knn_char`

Sau đó áp dụng bounded union / RRF với cap 200.

Public union đã PASS:
- query count: **1000**
- max candidates: **200**
- RRF k: **60**
- source keys: 4 nguồn trên

Tác động:
- dense retrieval bắt semantic similarity;
- BM25 bổ sung exact lexical matching;
- word KNN và char KNN bổ sung các trường hợp tên luật, cụm từ, spelling, tokenization hoặc pattern bề mặt;
- giảm nguy cơ tất cả lỗi cùng xảy ra ở một retrieval family.

Đây là một trong những yếu tố chính giúp tăng **candidate diversity** mà vẫn giữ pool bounded.

## 6. Public shortlist giới hạn phạm vi scoring nhưng giữ candidate có giá trị

Pipeline prepare tạo shortlist từ:
- baseline top-5;
- compatible union top-20;
- dedupe;
- tối đa khoảng 25 docs/query;
- evidence theo true-S2, tối đa 3 chunks/doc.

Tác động:
- giữ nguyên baseline strong candidates;
- bổ sung một số candidate tốt nhất từ union;
- không đưa hàng trăm document vào expensive frozen reranker;
- tập trung reranker vào những document có xác suất thực sự cạnh tranh top-5.

Điều này vừa tiết kiệm inference vừa làm feature phía sau ít nhiễu hơn.

## 7. True-S2 evidence giúp reranker nhìn đúng phần nội dung quan trọng

Mỗi document shortlist không được đại diện bằng toàn bộ text một cách thô. Pipeline chọn evidence chunks tốt nhất, tối đa **3 chunks/doc**, rồi mới chạy frozen reranker.

Tác động:
- tăng signal-to-noise;
- giúp cross-encoder tập trung vào đoạn pháp lý liên quan trực tiếp đến câu hỏi;
- giảm khả năng một document dài bị đánh giá thấp chỉ vì phần lớn nội dung không liên quan.

Đây là bước quan trọng giữa retrieval cấp document và reranking cấp semantic evidence.

## 8. Frozen reranker tạo feature semantic mạnh cho policy

PUBLIC Frozen scorer dùng canonical reranker để score shortlist/evidence và tạo feature phục vụ policy.

Output mới:
- `public_frozen_features.jsonl`
- `public_frozen_features_report.json`

Tác động:
- đưa semantic cross-encoder signal vào quyết định swap;
- giúp policy phân biệt candidate chỉ có retrieval score cao với candidate thật sự phù hợp về nội dung;
- giữ inference frozen, không fine-tune bằng public data, tránh leakage.

Việc Frozen chạy đúng trên public shortlist là mắt xích bắt buộc để V3A có đủ feature như lúc validation.

## 9. 58 feature của V3A kết hợp retrieval, rank và reranker signal

V3A final-fit dùng **58 action features**.

Trong đó có các nhóm signal như:
- baseline rank;
- candidate rank;
- source-specific rank;
- retrieval/source overlap;
- frozen reranker score;
- quan hệ giữa outgoing và incoming candidate;
- các feature mô tả chất lượng và độ tin cậy của swap.

Các source rank production được giữ đúng semantic:
- `bm25`
- `adaptive_k500`
- `knn_char`
- `knn_word`

Tác động:
- policy không quyết định từ một score đơn lẻ;
- có thể nhận diện candidate được nhiều retrieval family đồng thuận;
- tận dụng cả lexical và semantic evidence.

## 10. Chính sách “max one swap” là cơ chế bảo toàn baseline

Một observation quan trọng từ audit train:
- có rất nhiều action candidates;
- nhưng số query thật sự có opportunity để sửa là nhỏ hơn nhiều;
- rank-5 thường là vị trí yếu nhất trong các query có opportunity.

Vì vậy V3A không rerank toàn bộ top-5 mà chỉ cho phép:
- giữ nguyên, hoặc
- thay tối đa một document.

Tác động:
- hạn chế regression;
- chuyển bài toán từ “xếp hạng lại toàn bộ” thành “sửa đúng lỗi có xác suất cao”;
- phù hợp với thực tế baseline ~0.93 đã khá mạnh.

Đây là lý do kiến trúc residual có thể tăng điểm dù mức thay đổi ranking thực tế rất nhỏ.

# Những thứ KHÔNG phải nguyên nhân trực tiếp làm tăng điểm

### Beam account / GPU / A10G / RTX4090
Chỉ ảnh hưởng khả năng chạy inference. Không thay đổi thuật toán nếu model, batch semantics và output contract giữ nguyên.

### Preflight / fail-closed / manifest / hash checks
Không trực tiếp tăng ranking score, nhưng ngăn:
- dùng nhầm artifact;
- cache không tương thích;
- sai feature schema;
- public/train path lẫn nhau.

Chúng có tác động gián tiếp rất lớn vì bảo đảm submission thực sự là pipeline đã validation.

### Download/upload lại artifacts
Chỉ là vận hành, không làm tăng quality.

# Những thử nghiệm đã không được đưa vào production

Các hướng từng thử nhưng bị loại vì không ổn định hoặc không đủ tốt:
- handcrafted Pairwise BENEFIT
- LambdaMART / LTR
- Delta K10
- Opportunity-Gated K20 V1/V2
- hard negatives
- delta-recall residual
- rescue variants
- V3B dù aggregate hơi cao hơn nhưng kém ổn định hơn V3A

Việc **không đưa các thử nghiệm này vào submission** cũng quan trọng: production giữ V3A vì consistency trên folds thay vì chọn một cấu hình chỉ có peak score tốt hơn.

# Chuỗi tác động tổng thể

```text
Fresh Dense K500
        ↓
Adaptive K500
        ↓
BM25 + word KNN + char KNN + dense union
        ↓
Candidate recall/diversity tốt hơn
        ↓
Bounded public shortlist
        ↓
True-S2 evidence selection
        ↓
Frozen cross-encoder semantic features
        ↓
58-feature V3A residual policy
        ↓
Protect top-1..3 + max one swap ở rank 4/5
        ↓
Ít regression hơn + sửa được một phần lỗi baseline
        ↓
Public leaderboard ~0.930 → ~0.939
```

# Kết luận

Mức tăng từ **~0.930 lên ~0.939** không đến từ một thay đổi đơn lẻ. Nó đến từ việc kết hợp ba lớp cải thiện:

1. **Recall tốt hơn** — Fresh K500 + Adaptive K500 + multi-source candidate union.
2. **Evidence và semantic scoring tốt hơn** — Compatible BGE, true-S2 evidence, frozen reranker.
3. **Decision policy an toàn hơn** — V3A 58 features, bảo vệ top-3, chỉ sửa rank 4/5, tối đa một swap.

Nếu phải chọn yếu tố có ảnh hưởng kiến trúc lớn nhất, đó là **candidate diversity/recall tốt hơn kết hợp với V3A residual one-swap policy**. Candidate generation tạo cơ hội sửa; frozen reranker cung cấp semantic evidence; V3A quyết định khi nào cơ hội đó đủ đáng tin để thay baseline.

**Kết quả cuối: +0.009 absolute / +0.9 percentage points so với submission ~0.930 trước đó.**
