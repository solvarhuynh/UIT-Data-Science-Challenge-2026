# TV4 nên làm việc như thế nào trong Workflow B?

TV4 không phải trợ lý support cho TV2. TV4 sở hữu một nhánh nghiên cứu độc lập nhằm cải thiện score và thứ hạng cuối của Task1. Experiment chính hiện tại là B1 Direct Learning-to-Rank. Tuy vậy, sở hữu một nhánh không có nghĩa là chạy mọi ý tưởng trong roadmap. Công việc của TV4 là tìm đúng nút thắt đang quan trọng nhất, chọn experiment nhỏ nhất có thể trả lời câu hỏi kế tiếp, rồi để evidence – bằng chứng từ dữ liệu và kết quả chạy – quyết định hướng sau đó.

## Khi mở repo lên, TV4 nên đọc gì trước?

TV4 nên đọc theo một trật tự cố định. Trước hết là [workflow_B_tv2_tv4_overview.md](workflow_B_tv2_tv4_overview.md) để hiểu câu chuyện chung: TV2 đang kiểm tra semantic understanding, còn TV4 đang kiểm tra ranking. Sau đó đọc [workflow_B_execution_rules.md](workflow_B_execution_rules.md) để nắm quy tắc dùng dữ liệu, GPU, artifact, freeze và báo cáo áp dụng cho cả dự án.

Tiếp theo đọc [workflow_B_tv4.md](workflow_B_tv4.md) để biết nhánh TV4 hiện được định nghĩa ra sao, rồi đọc [experiments/workflow_B_tv4_experiments.md](experiments/workflow_B_tv4_experiments.md) để phân biệt experiment đang hoạt động với các hướng chỉ mới là điều kiện tương lai. Cuối cùng đọc [reports/task1/progress_log.md](../../../reports/task1/progress_log.md) và [reports/task1/workflow_b/tv4/task_results.md](../../../reports/task1/workflow_b/tv4/task_results.md). Hai file này cho biết điều gì thực sự đã xảy ra, thay vì điều gì từng được dự định.

Không nên mở một report cũ độc lập rồi coi đó là kế hoạch hiện tại. Một report có thể đúng trong bối cảnh lúc nó được viết nhưng đã bị thay thế bởi contract, gate hoặc kết quả mới. Đọc theo thứ tự trên giúp TV4 bắt đầu từ bức tranh chung, đi qua quy tắc, rồi mới đến trạng thái thực tế của nhánh mình.

## TV4 nên chọn task theo tiêu chí nào?

TV4 không chọn task chỉ vì task đó thú vị hoặc dễ chạy. Câu hỏi đúng là: “Trong nhánh TV4, điều gì đang là uncertainty quan trọng nhất và experiment nào cho ta nhiều thông tin hữu ích nhất với chi phí hợp lý?”. Đây là cách tập trung vào score leverage – khả năng tạo ra cải thiện đáng kể cho top-5 – thay vì đếm số experiment đã chạy.

Hiện B1 vẫn ở trạng thái NOT_YET_EVALUATED. Vì vậy câu hỏi chính chưa đổi: “Direct LTR có giúp ranking tốt hơn không?”. **Learning-to-Rank (LTR)** là cách huấn luyện mô hình để đưa ứng viên tốt lên trước ứng viên kém trong cùng một query. Khi chưa biết B1 có tín hiệu hay không, TV4 không nên nhảy đồng thời sang feature expansion, hard-negative curriculum, retrieval mới, neural model mới hoặc routing model. Các hướng đó có thể hợp lý về sau, nhưng hiện chưa có evidence khiến chúng trở thành ưu tiên.

## Trước khi chạy một experiment, TV4 phải tự trả lời những gì?

Trước khi Codex thực hiện bất kỳ thao tác chạy nào, TV4 cần nói bằng tiếng Việt đơn giản:

“Tôi đang gặp vấn đề gì? Bằng chứng nào cho thấy đây là vấn đề thật? Experiment này thay đổi đúng MỘT thứ gì? Những thứ nào giữ nguyên? Nếu kết quả tốt, tôi học được gì? Nếu kết quả xấu, tôi học được gì? Kết quả nào khiến tôi STOP?”

Các câu trả lời này biến một ý tưởng thành một **experiment contract** – bản cam kết trước về câu hỏi, dữ liệu, input, phần frozen, metric và điều kiện dừng. Nếu không biết kết quả tốt hay xấu sẽ giúp phân biệt những hướng nào, thì chưa nên mở run. Việc ghi rõ thứ giữ nguyên cũng quan trọng như ghi thứ được thay đổi: nếu đổi nhiều thứ cùng lúc, TV4 sẽ không biết nguyên nhân của cải thiện hay suy giảm.

## Tại sao phải test nhỏ trước khi chạy lớn?

Trước khi lái xe 1.000 km, ta kiểm tra xe có nổ máy không, phanh có hoạt động không và hệ thống nhiên liệu có ổn không. Với ML cũng vậy: trước hết kiểm tra environment smoke, sau đó chạy một deterministic run nhỏ, kiểm tra artifact, rồi mới chạy toàn bộ experiment.

Technical smoke chỉ trả lời câu hỏi kỹ thuật: model có load được không, GPU có hoạt động không, input có đúng schema không, output có được ghi đúng nơi không. Nó không trả lời câu hỏi khoa học. Một model có thể load thành công trên GPU nhưng vẫn tạo Recall thấp hơn baseline. Vì vậy smoke PASS chỉ mở cổng cho run được ủy quyền; nó không phải scientific success.

Nếu job dài, output nên được persist incrementally – ghi dần trong quá trình chạy – và hỗ trợ resume/checkpoint khi an toàn về khoa học. Những thay đổi như dtype, quantization, attention backend, checkpoint hoặc max_length có thể làm kết quả mang ý nghĩa khác. Không đổi chúng chỉ để smoke hoặc runtime chạy qua; nếu bắt buộc phải đổi, phải STOP, review và cập nhật contract.

## TV4 nên dùng Codex như thế nào?

Quy tắc là **ONE Codex prompt = ONE technical task**. Mỗi prompt cần nêu rõ file hoặc data được đọc, chính xác phải làm gì, không được làm gì, output nào phải tạo, lưu ở đâu, thế nào là PASS và khi nào STOP.

Không giao yêu cầu mơ hồ như “hãy tối ưu model”. Một yêu cầu tốt hơn là: “Execute frozen B1 configuration and produce the specified OOF evaluation artifacts”, trong đó contract đã nói rõ cấu hình frozen, dữ liệu, artifact và cổng dừng. **OOF** là out-of-fold, tức điểm hoặc dự đoán được tạo khi mỗi phần dữ liệu được đánh giá bởi mô hình không học trực tiếp trên phần đó; nếu B1 contract dùng khái niệm này thì artifact phải theo đúng định nghĩa đã chốt.

Codex thực hiện thao tác kỹ thuật trong phạm vi đó. Codex không tự quyết định hướng khoa học bằng cách thấy một score chưa đẹp rồi tự tune thêm, đổi threshold hoặc chạy model khác. Hướng đi đến từ experiment contract và evidence sau run.

## Nếu task BLOCKED thì TV4 nên làm gì?

BLOCKED không có nghĩa là phải ép task chạy cho bằng được. Trước hết phân loại nguyên nhân: infrastructure block, data block, scientific contract block hoặc model/runtime block. Sau đó điều tra root cause, lưu bằng chứng, và xác định hành động kế tiếp. Không đổi scientific configuration chỉ để thoát khỏi một vấn đề kỹ thuật.

B2a là ví dụ dễ hiểu. Context audit đã thực thi thành công nhưng phát hiện context không khả thi với các tài liệu cực dài. Đó là evidence hữu ích: nó nói rằng cần hiểu nguyên nhân độ dài hoặc thiết kế long-document handling trước khi chạy Qwen hợp lệ. Nó không chứng minh Qwen thất bại khoa học. TV4 nên áp dụng cùng kỷ luật cho B1: nếu GPU unavailable, artifact sai, model drift hoặc contract mơ hồ, ghi nhận BLOCKED và dừng ở gate tương ứng.

## TV4 nên làm gì sau khi B1 có kết quả?

Nếu B1 cải thiện Recall một cách đáng kể, TV4 trước tiên phải hiểu B1 thắng ở đâu: query nào, loại ứng viên nào, hoặc kiểu lỗi nào được sửa. Chỉ khi vùng thắng đã rõ, một targeted extension – mở rộng có mục tiêu – mới có cơ sở. Nếu B1 có tín hiệu nhưng mức tăng khiêm tốn, một giả thuyết nhỏ về feature expansion có thể đáng thử; không nên lập tức thêm mọi feature có sẵn.

Nếu sau này TV2 tạo được semantic score Qwen hợp lệ và đã **prediction freeze** – niêm phong dự đoán trước khi xem truth – TV4 có thể cân nhắc **Semantic-Augmented LTR**, tức đưa tín hiệu semantic đã khóa vào mô hình LTR để kiểm tra xem hai loại thông tin có bổ sung cho nhau không. Nhánh này phải chờ artifact TV2 hợp lệ; không được tự tạo điểm semantic tạm thời rồi coi đó là cầu nối khoa học.

Nếu cả hai hướng nhiều lần thất bại trên các tài liệu dễ gây nhầm, **hard negative** – ứng viên trông rất giống tài liệu đúng nhưng thực ra sai – trở thành manh mối để phân tích hoặc xây dựng dữ liệu huấn luyện khó hơn. Nếu ranking đã tốt lên nhưng Recall bão hòa, nghĩa là thứ tự trong K77 được cải thiện nhưng các tài liệu đúng vẫn không xuất hiện đủ, TV4 nên điều tra K77 ceiling và khả năng truy hồi ứng viên.

Đây là cây quyết định bằng prose, không phải danh sách lệnh chạy tự động. Một kết quả chỉ mở ra giả thuyết kế tiếp khi nó cung cấp đúng evidence; TV4 không cần thực hiện tất cả targeted features, semantic LTR, hard negatives và K77 analysis.

## TV4 được dùng GPU khi nào?

TV4 không bị cấm dùng GPU. Tuy nhiên, allocation dùng chung dựa trên leverage và rủi ro kiến trúc. Nếu chỉ có một nhánh GPU-heavy quan trọng và được biện minh, nhánh đó thuộc TV2 theo quy tắc hiện tại. Nếu có ít nhất hai nhánh GPU độc lập cùng có contract hợp lệ, TV2 giữ nhánh có expected score leverage hoặc architecture/integration risk cao hơn; TV4 có thể nhận nhánh còn lại.

TV4 không nên tạo một GPU experiment chỉ vì compute đang rảnh. GPU là tài nguyên để trả lời một câu hỏi đã đủ quan trọng, không phải lý do để mở thêm task. Khi được cấp GPU, TV4 dùng Modal hiện có nếu phù hợp, ghi provider/GPU/runtime, chạy smoke không label trước, rồi mới chạy inference hoặc training tốn kém. Fine-tuning Qwen, nếu được mở sau review, vẫn thuộc TV2 chứ không trở thành task TV4 vì TV4 đang có GPU.

## TV4 có được tự tạo submission không?

Có. Khi TV4 có kết quả experiment hợp lệ và prediction lineage rõ ràng, TV4 có thể tạo submission candidate, validation report, hash và manifest dưới khu vực report của experiment TV4, chẳng hạn reports/task1/tv4/b1/submission/. **Submission candidate** là bản dự thi nội bộ đã chuẩn bị để kiểm tra; nó không đồng nghĩa với public leaderboard upload.

Public upload chỉ được thực hiện sau shared deployment/regression gate: active scorer contract đã được xác minh, cấu trúc submission đã validate, diff với deployment incumbent đã ghi nhận, hash artifact đã lưu, quy tắc không tune theo public score được tuân thủ, và regression risk đã được review. Public score không được dùng làm feedback để chỉnh hyperparameter, threshold, feature hoặc model. Việc tạo candidate và kiểm tra cục bộ được phép; upload công khai là quyết định có cổng riêng.

## TV4 phải ghi lại kết quả như thế nào?

Sau mỗi task meaningful, trước hết lưu detailed artifact/report theo experiment contract. Sau đó append một bản tóm tắt ngắn vào [reports/task1/workflow_b/tv4/task_results.md](../../../reports/task1/workflow_b/tv4/task_results.md). Bản tóm tắt phải trả lời: mục tiêu là gì, đã làm gì, kết quả quan trọng nhất là gì, ta thực sự kết luận được gì, điều gì chưa được phép kết luận, artifact nằm ở đâu, và đúng một next action là gì. Không dán raw log vào ledger.

Đồng thời phải append entry canonical vào [reports/task1/progress_log.md](../../../reports/task1/progress_log.md). Ledger TV4 là bản đọc nhanh cho người; progress log là lịch sử khoa học/vận hành chung. Hai nơi có thể cùng nhắc B1 nhưng không nên sao chép toàn bộ chi tiết hoặc tạo hai “sự thật hiện tại” khác nhau.

## Khi nào TV4 phải dừng và hỏi lại?

TV4 phải stop/escalate nếu metric semantics thay đổi, K77 hoặc candidate population thay đổi, nhánh training mới mở, xuất hiện rủi ro Fold0/public data, một kiến trúc GPU mới trở nên high leverage, tham số frozen của B1 cần sửa, scientific comparator chưa rõ, hoặc quyết định public deployment còn mơ hồ. Những tình huống này thay đổi ý nghĩa của experiment nên cần review trước khi tiếp tục.

Routine engineering implementation – như sửa lỗi đường dẫn hoặc tạo folder đúng contract – không cần liên tục hỏi professor. Điểm phân biệt là việc đó có thay đổi câu hỏi, dữ liệu, model, metric, comparator hoặc ranh giới deployment hay không.

## TV4 nên hiểu vai trò của TV2 như thế nào?

TV2 hiện sở hữu semantic/neural architecture branch; TV4 sở hữu Direct LTR/ranking branch. Hai nhóm giải hai uncertainty quan trọng khác nhau và không nhóm nào chỉ là support cho nhóm kia. Họ có thể thực thi command của nhau khi thuận tiện, nhưng experiment ownership và artifact path vẫn giữ nguyên: B2a thuộc TV2, B1 thuộc TV4.

TV4 nên dùng kết quả TV2 như evidence khi một experiment tương lai thực sự phụ thuộc vào nó. Semantic-Augmented LTR là ví dụ: nó cần semantic score TV2 đã được freeze và có lineage hợp lệ. Chờ đúng artifact giúp TV4 kết hợp hai tín hiệu mà không làm lẫn ownership hoặc nhìn truth quá sớm.

## Một ngày làm việc tốt của TV4 trông như thế nào?

Buổi sáng, TV4 đọc trạng thái hiện tại và chọn đúng một câu hỏi đang active: hiện nay đó là B1 Direct LTR có cải thiện ranking hay không. Sau đó TV4 kiểm tra contract: input nào được dùng, cái gì frozen, output nằm đâu, success rule và STOP rule là gì.

TV4 chạy technical gate nhỏ nhất cần thiết. Nếu gate PASS, TV4 tiếp tục đúng experiment đã được ủy quyền. Nếu gate BLOCKED, TV4 điều tra block và không tự ý đổi cấu hình khoa học. Khi có kết quả, TV4 lưu artifact, viết entry ngắn vào task_results, cập nhật progress log, rồi nêu đúng một next action. Một ngày làm việc tốt không kết thúc bằng năm experiment mới mở nhưng không experiment nào đủ lineage để diễn giải.

## Nguyên tắc quan trọng nhất là gì?

Mục tiêu không phải tối đa hóa số experiment. Mục tiêu là tối đa hóa lượng thông tin hữu ích và expected Task1 score improvement trên mỗi experiment. Một experiment tốt có thể tăng score, loại bỏ một hướng sai, chỉ ra nút thắt thật, hoặc cung cấp bằng chứng để mở một nhánh có leverage cao hơn.

Đó là cách TV4 làm việc độc lập mà vẫn không trôi khỏi kế hoạch chung: bắt đầu từ câu hỏi hiện tại, thay đổi ít nhất có thể, kiểm tra kỹ thuật trước, tôn trọng STOP, niêm phong và ghi lại kết quả, rồi để evidence chọn đúng một bước kế tiếp.

## Các tài liệu TV4 cần dùng cùng guide này

- [Tổng quan TV2 và TV4](workflow_B_tv2_tv4_overview.md)
- [Quy tắc thực thi Workflow B](workflow_B_execution_rules.md)
- [Nhánh hiện tại của TV4](workflow_B_tv4.md)
- [Danh mục experiment TV4](experiments/workflow_B_tv4_experiments.md)
- [Ledger kết quả TV4](../../../reports/task1/tv4/task_results.md)
- [Lịch sử tiến độ chung](../../../reports/task1/progress_log.md)
