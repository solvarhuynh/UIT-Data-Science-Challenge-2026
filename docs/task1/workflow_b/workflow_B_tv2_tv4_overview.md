# Workflow B — TV2 và TV4 đang làm gì?

Task1 cuối cùng cần trả về cho mỗi câu hỏi một tập không quá 5 tài liệu pháp lý phù hợp nhất. **Workflow A** đã khép lại, nên **Workflow B** hiện có hai người nghiên cứu cùng nhìn vào 77 ứng viên đã đi tới vòng cuối, nhưng đặt hai câu hỏi khác nhau. TV2 hỏi: “Hệ thống có thật sự hiểu nội dung câu hỏi và văn bản của từng ứng viên không?”. TV4 hỏi: “Nếu thông tin hiện có đã hữu ích, ta có đang xếp hạng các ứng viên sai thứ tự không?”. Vì vậy, chưa có nhánh nào được mặc nhiên xem là đúng; bằng chứng từ từng thử nghiệm mới quyết định bước kế tiếp.

# Phần 1 — TV2

## Vì sao TV2 lại mở nhánh Neural Semantic Reranker?

Hệ thống hiện tại đã có nhiều tín hiệu để tìm và xếp hạng tài liệu. Tuy nhiên, một tín hiệu được tính gián tiếp chưa chắc đã nắm được mối quan hệ nghĩa giữa một câu hỏi pháp lý tiếng Việt và chính phần nội dung pháp lý trả lời câu hỏi đó. Đây mới là giả thuyết về nút thắt **semantic** – tức là mức độ hệ thống hiểu đúng ý nghĩa – chứ chưa phải kết luận rằng biểu diễn hiện tại đã hỏng.

TV2 vì thế thử một **semantic reranking** – bước xếp hạng lại dựa trên nghĩa bằng cách cho mô hình đọc trực tiếp cả Query và Document. Query là câu hỏi người dùng đặt ra; Document là tài liệu pháp lý đang được cân nhắc. Mô hình sẽ ước lượng xem cặp câu hỏi–tài liệu đó có thật sự liên quan không. Hãy hình dung một người tuyển sinh đọc nguyên bài luận của từng thí sinh thay vì chỉ nhìn vài điểm số tóm tắt. Nếu cách đọc trực tiếp này giúp chọn đúng hơn, ta có bằng chứng rằng hiểu nghĩa sâu hơn có thể cải thiện top-5 cuối cùng; nếu không, ta không nên tiếp tục đổ thời gian vào mô hình lớn chỉ vì nó nghe có vẻ hiện đại.

## K77 là gì và tại sao TV2 chưa thay đổi retrieval ngay?

Mỗi query hiện đã có 77 tài liệu ứng viên chuẩn hóa. Tập 77 tài liệu này gọi là **K77** – có thể hiểu như danh sách 77 người đã vượt qua vòng sơ tuyển. **Retrieval** là bước lấy các ứng viên ban đầu từ toàn bộ kho tài liệu; nhưng trong B2a-0, TV2 không mở lại bước đó mà giữ nguyên danh sách K77 cho từng query.

Lý do là TV2 muốn cô lập câu hỏi đang kiểm tra: liệu mô hình có xếp hạng tốt hơn khi đã được đưa đúng 77 ứng viên hay không. Nếu đồng thời đổi retrieval và reranking, kết quả tăng hoặc giảm có thể đến từ việc lấy được ứng viên mới, từ cách đọc nghĩa mới, hoặc từ cả hai. Giống như đổi cùng lúc loại gạo, nước sốt, thịt và nhiệt độ nấu rồi thấy món ăn ngon hơn: ta không biết thành phần nào thực sự có tác dụng. Giữ K77 nguyên vẹn giúp kết quả B2a-0 nói rõ hơn về tín hiệu semantic của TV2, từ đó quyết định có cần mở nhánh kiến trúc khác hay không.

## B2a-0 **Zero-shot** thực sự đang kiểm tra điều gì?

**Zero-shot** nghĩa là dùng năng lực đã có sẵn của mô hình trước khi huấn luyện thêm trên ví dụ riêng của Task1. Với B2a-0, mô hình **Qwen/Qwen3-Reranker-0.6B** đọc từng cặp **Query** + **Document** trong K77, cho ra 77 điểm liên quan semantic, rồi giữ lại 5 điểm cao nhất. Nói đơn giản, 77 người được đọc hồ sơ lần lượt, sau đó chỉ 5 người có vẻ phù hợp nhất được đưa vào danh sách cuối. Năm tài liệu đó chính là **top-5** dự đoán của query.

Cách thử này khớp trực tiếp với mục tiêu set-based của Task1: ta không chỉ hỏi mô hình có chấm một tài liệu cao hay thấp, mà hỏi 5 tài liệu nó chọn có trùng với các tài liệu đúng hay không. Vì vậy B2a-0 là một phép kiểm tra rõ ràng về giá trị của tín hiệu semantic khi retrieval đầu vào được giữ cố định. Nó chưa phải bằng chứng rằng Qwen sẽ giải quyết toàn bộ bài toán, và cũng chưa phải bước **Fine-tuning**.

## Tại sao chúng ta chưa chạy Qwen full ngay?

Trước hết, nguồn gốc mô hình đã được kiểm tra và đạt. Checkpoint được đóng băng là **Qwen/Qwen3-Reranker-0.6B**, tại revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`. “Đóng băng” ở đây có nghĩa là ta xác định chính xác phiên bản sẽ được đánh giá, để kết quả không âm thầm thay đổi vì dùng một bản khác.

Sau đó, context audit – cuộc kiểm tra xem mô hình có đủ “chỗ đọc” hay không – đã xem toàn bộ 431.200 cặp query–document. Các con số độ dài không phải là vài trường hợp lẻ: p50 = 18.455 nghĩa là một nửa số cặp có độ dài không quá khoảng 18 nghìn token; p90 = 64.539 nghĩa là 10% cặp còn dài hơn khoảng 64 nghìn token; p95 = 95.289 nghĩa là 5% cặp vượt khoảng 95 nghìn token; p99 = 158.503 nghĩa là 1% cặp vượt khoảng 158 nghìn token. Token là những mảnh văn bản mà mô hình dùng để đọc và tính toán, không hoàn toàn trùng với số từ.

Điểm quan trọng nằm ở p99. Nếu cửa sổ context của checkpoint hiện tại khoảng 32.768 token, thì phần đuôi 1% dài hơn sức chứa của mô hình rất nhiều. Nói đời thường, ta đang cố đưa một quyển sách dày vào chiếc hộp chỉ chứa được một phần nhỏ quyển sách; chạy thẳng có thể bị cắt mất nội dung, lỗi hoặc tạo ra điểm số không còn so sánh công bằng. Giá trị max = 2.145.502 token còn đáng ngờ hơn: có thể kho pháp lý thật sự chứa những hồ sơ cực dài, nhưng cũng có thể cách ghép input đã lặp hoặc nối nhầm dữ liệu. Trước khi thiết kế cách xử lý phức tạp, phải phân biệt hai khả năng đó.

Vì vậy trạng thái hiện tại rất cụ thể: model audit PASS; context audit execution PASS; nhưng context feasibility là `BLOCKED_LONG_DOCUMENT`. “Blocked” ở đây là chưa đủ điều kiện chạy thí nghiệm hợp lệ, không phải Qwen đã thất bại về mặt khoa học. Chưa có kết quả **Recall** nào từ B2a-0, nên không được nói rằng mô hình làm tốt hay làm kém.

## TV2 phải làm gì tiếp theo?

Việc trước mắt là long-document root-cause audit – truy nguyên vì sao xuất hiện các đầu vào quá dài. Nếu độ dài cực đoan do lỗi dữ liệu hoặc lỗi dựng input, dữ liệu và cách dựng input phải được sửa dưới một contract đã được rà soát. Nếu văn bản thực sự dài, ta cần thiết kế riêng một thí nghiệm reranking cho tài liệu dài. Có thể về sau cân nhắc **chunk-aware reranking** – chia tài liệu thành các đoạn rồi xếp hạng dựa trên các đoạn; **section-aware reranking** – chú ý các phần như điều, khoản, mục; hoặc **multi-chunk aggregation** – gộp bằng chứng từ nhiều đoạn. Chưa ý tưởng nào trong số này được chọn ở hiện tại.

Chỉ sau khi cách xử lý context hợp lệ về mặt khoa học, TV2 mới làm **Modal GPU smoke** – một lần chạy nhỏ để kiểm tra mô hình có sống được trên môi trường GPU và pipeline có vận hành đúng không – rồi mới scoring zero-shot đầy đủ, tạo top-5, thực hiện **Prediction freeze**, và tính Recall/Precision trên F1–F4. Trình tự này quan trọng vì một lệnh chạy thành công về kỹ thuật vẫn có thể dựa trên input bị cắt sai và không tạo ra kết quả khoa học đáng tin.

## Prediction freeze là gì và tại sao phải làm trước khi xem label?

**Prediction freeze** là niêm phong danh sách top-5 mà mô hình đã chọn trước khi nhìn nhãn đúng. Ta lưu các dự đoán đã chốt cùng một hash – dấu vân tay dùng để phát hiện danh sách có bị sửa hay không. Hình ảnh dễ hiểu là nộp và niêm phong bài làm trước rồi mới mở đáp án.

Sau khi top-5 đã được freeze, mới được nối chúng với truth của F1–F4 để đánh giá. Mục đích là ngăn việc nhìn label rồi vô tình chỉnh ngưỡng, đổi cách chọn hoặc chọn checkpoint có lợi cho nhãn. Nhờ vậy Recall/Precision phản ánh một dự đoán đã tạo độc lập, chứ không phải câu trả lời được sửa sau khi biết đáp án. F1–F4 vẫn là các fold phát triển mang tính exploratory, nên kết quả của chúng giúp định hướng và sàng lọc giả thuyết; không nên mô tả chúng như bằng chứng xác nhận cuối cùng.

## Recall và Precision sẽ nói cho TV2 điều gì?

**Recall** – nôm na là trong số những tài liệu đúng mà đáng lẽ phải tìm được, hệ thống lấy lại được bao nhiêu. Recall thấp cho thấy có thể tài liệu đúng bị bỏ lỡ, hoặc mô hình không đẩy chúng vào top-5. **Precision** – trong 5 tài liệu hệ thống đã chọn, có bao nhiêu tài liệu thật sự đúng. Precision giúp nhìn xem việc tăng Recall có phải trả giá bằng một danh sách đầy tài liệu không liên quan hay không.

Trong quyết định của B2a, Recall là tín hiệu chính vì Task1 cần giữ được các tài liệu pháp lý liên quan trong tập cuối. Precision là hàng rào an toàn: không thể gọi là cải thiện nếu chỉ nhặt thêm tài liệu đúng bằng cách làm chất lượng 5 lựa chọn sụp đổ. Nếu Recall tăng mà Precision vẫn chấp nhận được, giả thuyết semantic đáng được theo đuổi; nếu Recall không tăng, ta cần xem lại long-document handling, checkpoint hoặc tầng ứng viên trước khi kết luận về giá trị của semantic reranking.

## Fine-tuning nằm ở đâu trong câu chuyện này?

**Fine-tuning** là huấn luyện tiếp mô hình pretrained bằng các ví dụ đặc thù của Task1. Nếu Zero-shot là mời một luật sư giàu kinh nghiệm vào làm và yêu cầu họ bắt tay xử lý ngay, Fine-tuning là cho luật sư đó học thêm phong cách hồ sơ và cách ưu tiên riêng của tổ chức. **LoRA** là một cách điều chỉnh gọn hơn, chỉ học thêm một phần nhỏ tham số thay vì thay đổi toàn bộ mô hình.

Fine-tuning thuộc TV2, nhưng không thuộc B2a-0 và hiện chưa được cấp phép. Trình tự phải là B2a-0 hợp lệ → top-5 → freeze → Recall/Precision → giáo sư review → chỉ khi đó mới cân nhắc B2a-1 Fine-tuning hoặc LoRA. Nếu huấn luyện ngay, ta sẽ tiêu GPU và thời gian để biến đổi mô hình trước khi biết tín hiệu semantic nguyên bản đã giúp được bao nhiêu; khi ấy rất khó nói cải thiện đến từ năng lực pretrained hay từ dữ liệu huấn luyện thêm.

## Khi nào TV2 mới thay Embedding hoặc Retrieval?

**Embedding** là cách biến câu hỏi hoặc tài liệu thành một biểu diễn số để những nội dung gần nghĩa có thể được nhận ra; giống như sắp các cuốn sách gần chủ đề vào cùng khu vực thư viện. **Retrieval** là dùng biểu diễn hoặc tín hiệu đó để lấy một nhóm ứng viên ban đầu. Hai thứ này khác với long-document handling: xử lý tài liệu quá dài là tìm cách cho mô hình đọc một ứng viên dài đúng cách, còn thay Embedding/Retrieval là thay cách tìm ứng viên.

TV2 chỉ nên mở lại tầng này nếu bằng chứng cho thấy tài liệu đúng thường xuyên vắng khỏi K77, hoặc có mặt nhưng được biểu diễn quá kém để được tìm thấy. Khi đó câu chuyện tương lai có thể là retrieval tốt hơn → candidate pool mạnh hơn → semantic reranker → top-5 cuối. Đây là nhánh kiến trúc về sau, không phải việc của B2a-0 hiện tại.

## Vậy toàn bộ hướng đi của TV2 là gì?

Hiện tại, TV2 chỉ cần làm cho B2a-0 chạy được một cách khoa học và lấy được Recall/Precision zero-shot. Sau bằng chứng đó, nhóm mới chọn hướng phù hợp: Fine-tuning nếu tín hiệu gốc có ích nhưng cần học phong cách Task1; kiến trúc long-document nếu độ dài là nút thắt; checkpoint mạnh hơn nếu năng lực mô hình chưa đủ; Embedding/Retrieval nếu K77 bỏ sót tài liệu đúng; hoặc tích hợp nhiều tầng nếu các tín hiệu bổ sung thực sự đem lại giá trị. Đây là các nhánh rẽ theo bằng chứng, không phải danh sách việc bắt buộc phải chạy hết.

# Phần 2 — TV4

## TV4 đang giải một vấn đề khác TV2 như thế nào?

TV2 đặt câu hỏi liệu hệ thống có thiếu hiểu biết semantic khi đọc Query và Document hay không. TV4 bắt đầu từ một giả định khác: các thông tin và điểm số hiện có có thể đã hữu ích, nhưng phương pháp sắp xếp chưa khai thác chúng đúng cách. Hãy tưởng tượng hồ sơ của các thí sinh đã có nhiều điểm đánh giá đáng tin; TV2 muốn đọc lại bài luận để hiểu người thật, còn TV4 hỏi liệu hội đồng tuyển sinh có đang xếp những hồ sơ đã chấm sai thứ tự không.

Vì câu hỏi khác nhau, TV4 không phải bản phụ trợ cho TV2. TV4 sở hữu một nhánh cải thiện score riêng: kiểm tra xem một bộ xếp hạng được huấn luyện đúng mục tiêu có thể đưa ứng viên tốt lên trên hay không. Kết quả của TV2 có thể hữu ích về sau, nhưng không phải điều kiện để TV4 được xem là một thử nghiệm độc lập.

## Direct Learning-to-Rank là gì và tại sao đáng thử?

**Learning-to-Rank (LTR)** là cách huấn luyện một mô hình chuyên học thứ tự: ứng viên nào nên đứng trước ứng viên nào trong cùng một query. Thay vì chỉ dự đoán một xác suất hay một hành động trung gian rồi hy vọng thứ tự cuối cùng sẽ tốt, Direct LTR học trực tiếp mục tiêu xếp hạng. Giống như huấn luyện hội đồng tuyển sinh tập trung vào câu hỏi “ai nên đứng trước ai trong danh sách trúng tuyển”, chứ không chỉ giao cho họ một loạt điểm rời rạc.

Thử nghiệm hiện tại của TV4 là B1 Direct LTR. Đây là một thử nghiệm cải thiện score thực sự, không phải công việc hỗ trợ hay chuẩn bị hộ cho nhánh khác. Trạng thái khoa học của B1 hiện là `NOT_YET_EVALUATED`, nghĩa là chưa có Recall/Precision hợp lệ để tuyên bố LTR tốt hơn cách cũ.

## TV4 đang làm gì ngay bây giờ?

Trước mắt TV4 hoàn tất runtime/execution đóng băng của B1: kiểm tra pipeline và khả năng chạy đúng theo thiết kế đã chốt. Nếu hạ tầng vượt qua kiểm tra survivability – tức là chạy ổn định đủ để hoàn thành thử nghiệm – TV4 sẽ chạy toàn bộ B1, tạo predictions, chọn top-5, rồi tính Recall/Precision. Sau khi đã xem kết quả, TV4 không được “rescue-tune”, tức là chỉnh lại thử nghiệm chỉ để cứu một kết quả không đẹp; làm vậy sẽ phá ý nghĩa của phép so sánh.

Historical reranker autopsy – mổ xẻ các reranker lịch sử để hiểu chúng đã thất bại hoặc mất tín hiệu ở đâu – có thể chạy khi còn năng lực. Nhưng đó là việc thứ yếu sau B1. Nó chỉ đáng làm nếu giúp giải thích hoặc chọn bước tiếp theo, chứ không thay thế kết quả của Direct LTR.

## Nếu B1 tốt nhưng chưa đủ thì TV4 làm gì?

Nếu B1 cải thiện nhưng chưa đưa top-5 đến mức cần thiết, bằng chứng sẽ quyết định một nhánh hẹp hơn. **Targeted LTR feature expansion** là bổ sung có chọn lọc những đặc trưng mà kết quả B1 cho thấy còn thiếu, thay vì nhồi mọi tín hiệu mới vào mô hình. **Semantic-Augmented LTR** là thêm điểm semantic đã được đóng băng vào LTR; nếu sau này TV2 có điểm Qwen hợp lệ và TV4 có kết quả LTR hợp lệ, hai nhóm có thể kiểm tra xem kết hợp chúng có tạo thêm giá trị hay không. Đây là chiếc cầu tự nhiên giữa hai câu hỏi nghiên cứu, chưa phải workload hiện tại.

**Hard negative** là một ứng viên trông rất giống tài liệu đúng nhưng thực ra sai; huấn luyện với những ca khó như vậy giúp LTR học cách phân biệt tinh hơn, giống như cho hội đồng xem các hồ sơ gần như ngang nhau để luyện khả năng chọn. **Secondary ranker** là một bộ xếp hạng thứ hai ở tầng sau, dùng để soi kỹ nhóm ứng viên đã được lọc. Mỗi hướng chỉ nên được mở nếu B1 chỉ ra đúng nhu cầu đó; không có nghĩa TV4 phải chạy tất cả.

## Khi nào TV4 mới đụng đến K77 hoặc Retrieval?

Nếu ranking cải thiện nhưng Recall bắt đầu bão hòa, nghĩa là thứ tự trong 77 ứng viên đã tốt hơn nhưng vẫn không thể lấy được nhiều tài liệu đúng hơn, TV4 mới phân tích **K77 recoverability** – khả năng tài liệu đúng có thể được lấy lại từ chính danh sách 77 hay không – và candidate ceiling – trần kết quả do danh sách ứng viên áp đặt. Nếu tài liệu liên quan thường xuyên không xuất hiện trong K77, đó là bằng chứng tầng tạo ứng viên có thể cần mở lại; xếp hạng giỏi đến đâu cũng không thể chọn một tài liệu chưa từng được đưa vào danh sách.

Một kiến trúc Retrieval mới có đòn bẩy lớn vẫn phải tuân theo quy tắc phân bổ GPU và ưu tiên của dự án. Nói cách khác, kết quả B1 có thể làm lộ ra nhu cầu mở K77, nhưng không tự động cấp phép cho một cuộc đại tu retrieval ngay lập tức.


## Vậy hướng đi của TV4 trong thời gian tới là gì?

Hiện tại TV4 tập trung vào B1 Direct LTR. Khi có capacity, historical reranker autopsy có thể giúp đọc lại các tín hiệu cũ. Chỉ sau khi có bằng chứng hợp lệ, TV4 mới cân nhắc mở rộng feature có mục tiêu, Semantic-LTR, hard negatives, secondary ranker, phân tích trần K77, hoặc một nhánh GPU thứ hai nếu dự án ưu tiên. TV4 có roadmap dài hạn, nhưng roadmap là bản đồ các lựa chọn có điều kiện chứ không phải lịch chạy toàn bộ.

TV2 và TV4 vì vậy đang tấn công hai điều chưa chắc chắn khác nhau. TV2 hỏi: “Ta có cần hiểu semantic sâu hơn không?”. TV4 hỏi: “Ta có thể xếp hạng tốt hơn thông tin đã có không?”. Một kết quả tốt, xấu, hoặc bị chặn đúng cách đều làm thay đổi lựa chọn tiếp theo: có thể cần Fine-tuning, xử lý tài liệu dài, mở lại candidate pool, hoặc chỉ cần giữ hướng hiện tại. Nhóm không tiến lên bằng cách chạy hết mọi ý tưởng trong roadmap; nhóm tiến lên bằng cách niêm phong từng thử nghiệm, đọc đúng bằng chứng, rồi để bằng chứng đó chọn bước kế tiếp.
