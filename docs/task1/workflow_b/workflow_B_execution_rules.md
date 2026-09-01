# Workflow B — Quy tắc chạy experiment và quản trị kết quả

Tài liệu này là rulebook chuẩn cho TV2 và TV4 khi mở hoặc thực hiện một task/experiment. Mục tiêu là để mỗi kết quả có thể truy nguyên, không bị lẫn giữa các nhánh, không bị **leakage** – việc thông tin đáp án len vào quá trình thiết kế – và không tiêu GPU trước khi biết thí nghiệm có đáng chạy hay không. Quy tắc áp dụng cho cả experiment thành công lẫn bị chặn.

## Khi nào một ý tưởng được phép trở thành experiment?

Một ý tưởng chỉ được gọi là experiment sau khi có mô tả trước khi chạy. Bản mô tả tối thiểu phải ghi: **Experiment ID**, owner (TV2 hoặc TV4), câu hỏi khoa học, lý do mở ngay lúc này, evidence/trigger, quần thể dữ liệu chính xác, input/feature/model, phần nào giữ nguyên, metric, success rule, kill/stop rule, đường dẫn artifact, nhu cầu GPU và việc có cần professor review hay không.

Phần “vì sao bây giờ” rất quan trọng: nó nối experiment với bằng chứng trước đó, thay vì biến repository thành nơi thử ngẫu nhiên. Các câu như “thử xem sao”, “tune thêm chút”, hoặc “chạy model khác xem có cao hơn không” không phải scientific contract. Không được chạy một sweep mù về tham số hoặc model. Nếu chưa nói rõ kết quả sẽ trả lời câu hỏi nào và điều kiện nào khiến ta dừng, ý tưởng vẫn là giả thuyết chứ chưa phải run được phép thực hiện.

## Một prompt Codex được phép làm bao nhiêu việc?

Quy tắc chuẩn là **ONE prompt = ONE concrete technical task**: một prompt chỉ giao một nhiệm vụ kỹ thuật cụ thể. Prompt phải nêu exact inputs, exact outputs, allowed operations, forbidden operations, STOP conditions và final console output. Nhờ vậy người đọc biết lệnh đang làm gì và biết lúc nào nó phải dừng.

Không gộp audit + training + evaluation + deployment trong một prompt, trừ khi scientific contract đã cho phép đúng pipeline tuần tự đó. Một prompt sửa đường dẫn không tự động được hiểu là được chạy model; một prompt audit không được âm thầm huấn luyện; một prompt đánh giá không được tự ý tạo submission. Tách việc cũng giúp artifact và trách nhiệm thuộc đúng experiment.

## CPU và GPU được dùng như thế nào?

Các việc nhẹ và có thể tái lập nên chạy CPU khi phù hợp: data audit, tokenization/context audit, hashing, preprocessing nhẹ, evaluation, tạo report và các CPU model nếu contract yêu cầu. **GPU-heavy run** là run cần nhiều tính toán song song; đối với Task1, nó phải dùng hạ tầng **Modal** hiện có, trừ khi một môi trường khác đã được phê duyệt rõ ràng.

Khi experiment cần CUDA/GPU, thứ tự chuẩn là: (1) dùng Modal nếu có; (2) ghi provider, loại GPU và runtime; (3) chạy một technical smoke nhỏ, không dùng label; (4) chỉ khi smoke PASS mới chạy inference/training tốn kém; (5) ghi output tăng dần trong job dài; (6) hỗ trợ resume/checkpoint khi điều đó an toàn về khoa học. **Smoke** là một lần chạy nhỏ để kiểm tra hạ tầng và pipeline, không phải bằng chứng về score.

Không được âm thầm đổi GPU type-dependent semantics, dtype, quantization, model checkpoint, attention backend hoặc quy tắc max_length nếu thay đổi đó có thể làm experiment khác đi. Nếu cần đổi, phải STOP → review → cập nhật contract trước khi chạy. Chạy được trên máy khác không có nghĩa là kết quả còn cùng ý nghĩa khoa học.

## GPU được chia cho TV2 và TV4 thế nào?

Phân bổ GPU dựa trên leverage và rủi ro, không dựa trên việc làm cho số task của hai nhóm bằng nhau. Nếu tại một thời điểm chỉ có một experiment GPU-heavy quan trọng và đã được biện minh, GPU thuộc về TV2. Nếu có từ hai experiment độc lập trở lên cùng đủ điều kiện, TV2 giữ experiment có expected score leverage cao nhất hoặc architecture/integration risk cao nhất; TV4 có thể nhận nhánh GPU còn lại.

Quy tắc này không tự động cấp GPU cho một ý tưởng chưa có contract. Fine-tuning Qwen, nếu sau này được cho phép, thuộc TV2. Việc một người của nhóm kia có thời gian hoặc gõ lệnh không làm thay đổi allocation hay ownership.

## Model phải được khóa như thế nào?

Với external pretrained model, phải ghi model ID, immutable revision, license, model card, tokenizer revision, dependency versions, dtype/backend, context contract và provenance assessment. **Provenance** là nguồn gốc và danh tính có thể kiểm chứng của model; nếu revision trôi dạt, hai lần chạy có thể không còn là cùng experiment.

Không được dùng một revision mới chỉ vì nó đang có sẵn trong cache. Ví dụ model B2a hiện được khóa là **Qwen/Qwen3-Reranker-0.6B**, revision e61197ed45024b0ed8a2d74b80b4d909f1255473. Đây chỉ là ví dụ về cách ghi model provenance, không phải authorization cho experiment khác.

## Dữ liệu nào tuyệt đối không được dùng sai?

F1–F4 có thể được dùng theo active experiment contract. **Fold0** không được dùng như một holdout khoa học mới hoặc nguồn để chọn training/model/feature. **Public labels** bị cấm. **Leaderboard feedback** không được dùng để tune model, threshold, feature hoặc hyperparameter.

Lý do rất dễ hiểu: nếu cứ nhìn đáp án trong lúc thiết kế, điểm cuối cùng sẽ giống bài làm đã được sửa theo đáp án và mất khả năng cho biết hệ thống có tổng quát hóa hay không. Một kết quả tốt chỉ có ý nghĩa khi dữ liệu dùng để ra quyết định đã được contract cho phép và lineage được ghi lại.

## Prediction freeze là luật gì?

Với experiment tạo dự đoán trước khi đánh giá, pipeline bắt buộc là: scores → top-5 → prediction file → **SHA256 freeze** → chỉ sau đó truth join → Recall/Precision. **Prediction freeze** là niêm phong danh sách dự đoán trước khi nhìn nhãn; SHA256 là dấu vân tay để phát hiện file bị đổi.

Trước freeze không được xem truth để sửa top-5, threshold hay tie-break. Record phải có prediction hash, pair count, query count, missing/duplicate count và tie-break rule. Cách này tách “mô hình đã chọn gì” khỏi “đáp án nói gì”, qua đó ngăn label-guided selection.

## STOP có nghĩa là gì?

BLOCKED không đồng nghĩa với experiment thất bại. Nó nói rằng một cổng đã chặn việc diễn giải hoặc chạy tiếp. Ví dụ gồm GPU unavailable, context infeasible, provenance uncertain, data contract ambiguous, model revision drift, missing candidate pairs hoặc unexpected truncation.

Khi preregistered gate nói STOP thì phải STOP, không tự phát minh workaround chỉ để có log “chạy thành công”. Trước tiên phân loại technical block, scientific block, data block hay contract block; sau đó review hành động kế tiếp. B2a là ví dụ: context audit đã chạy thành công nhưng feasibility bị chặn bởi tài liệu dài, nên Qwen chưa hề thất bại khoa học và chưa có Recall result.

## Tên experiment và artifact đặt như thế nào?

Dùng ID ổn định, chẳng hạn TV2 có B2a-0, B2a-1, TV2-E2; TV4 có B1, TV4-E1, TV4-E2. Folder nên dùng tên lowercase ổn định khi thực tế cho phép. Tên artifact theo mẫu <experiment_id>_<stage>_<artifact>.<ext>, ví dụ b2a0_context_length_audit.json, b2a0_scores.jsonl, b2a0_top5_predictions.json, b2a0_prediction_freeze.json, b1_oof_scores.parquet, b1_evaluation.json.

Không dùng tên mơ hồ như final2.json, new_best.json, test_ok.json hoặc last_version.zip; các tên đó không nói được file thuộc experiment nào, tạo ở stage nào, và có còn là bản đang dùng hay không.

## Output của TV2 và TV4 để đâu?

reports/task1/ là root dùng chung của toàn Task1. Riêng toàn bộ artifact khoa học của Workflow-B phải nằm dưới reports/task1/workflow_b/. Trong đó reports/task1/workflow_b/tv2/ là root cho artifact Workflow-B thuộc TV2, reports/task1/workflow_b/tv4/ là root cho artifact thuộc TV4, còn reports/task1/workflow_b/shared/ là nơi chứa các định nghĩa Workflow-B dùng chung. Đây là các root canonical; không tạo các bản hiện hành song song ở reports/task1/tv2/, reports/task1/tv4/ hoặc reports/task1/shared/.

reports/task1/progress_log.md vẫn là một file append-only duy nhất cho lịch sử toàn Task1 và không được di chuyển. Các contract toàn Task1 đã có sẵn dưới reports/task1/contracts/ cũng giữ nguyên, trừ khi được xác định rõ là contract riêng của Workflow-B. Artifact của experiment TV2 đặt dưới root TV2; artifact của experiment TV4 đặt dưới root TV4. Các định nghĩa dùng chung và immutable của Workflow-B đặt dưới shared/.

Ownership đi theo experiment, không đi theo người thực thi. B2a thuộc TV2 và có canonical path reports/task1/workflow_b/tv2/b2a/; B1 thuộc TV4 và có canonical path reports/task1/workflow_b/tv4/b1/. Nếu TV4 chạy B2a trên Modal, output vẫn thuộc TV2. Nếu TV2 giúp chạy B1, output vẫn thuộc TV4. Khi liên kết artifact giữa hai nhánh, ghi rõ nguồn và hash thay vì sao chép thành các bản “current” mâu thuẫn nhau.

## Một experiment nên có những loại artifact nào?

Khi có liên quan, có thể tổ chức dưới contracts/, audit/, runtime/, models/, inference/, evaluation/, reports/ và submission/. Không bắt buộc experiment nào cũng tạo đủ mọi folder; chỉ tạo phần cần thiết. Model weights lớn nên ở Hugging Face cache hoặc Modal Volume/cache. Repository thường chỉ giữ manifest, revision và hash để tránh nhân bản dữ liệu lớn.

## Sau mỗi task phải ghi kết quả thế nào?

Sau mỗi task meaningful đã hoàn tất hoặc bị block, owner phải append đúng một entry ngắn vào ledger của mình: reports/task1/workflow_b/tv2/task_results.md hoặc reports/task1/workflow_b/tv4/task_results.md. Ledger là bản tóm tắt cho người đọc, không thay thế scientific report chi tiết.

Dùng mẫu: ## YYYY-MM-DD — <Experiment ID> — <Task name>. Ghi **Mục tiêu** (1–2 câu), **Đã làm** (1–3 câu), **Kết quả chính**, **Trạng thái** (PASS / BLOCKED / FAIL / NOT_YET_EVALUATED), **Điều rút ra**, **Chưa được kết luận**, **Artifact chính** và **Bước tiếp theo**. Mỗi entry khoảng 80–180 từ và bước tiếp theo phải là đúng một next action. Không dán full log.

## progress_log.md khác task_results.md thế nào?

Chỉ giữ một reports/task1/progress_log.md dùng chung và append-only. Đây là canonical scientific/operational history, tức lịch sử đầy đủ về tiến triển và vận hành. Hai ledger TV2/TV4 là bản tóm tắt human-readable theo owner. Chúng có thể cùng trỏ tới một experiment, nhưng không lặp lại các chi tiết kỹ thuật lớn; chi tiết nằm ở report và progress log.

## Khi nào được tạo submission?

TV2 hoặc TV4 được tạo **submission candidate** riêng sau khi có prediction lineage hợp lệ về mặt khoa học. Ví dụ có thể dùng reports/task1/tv4/b1/submission/ hoặc reports/task1/tv2/b2a/submission/. Tạo candidate và local validation được phép; hashing là bắt buộc.

Public upload không tự động được phép. Trước upload phải vượt shared deployment gate: active scorer contract đã xác minh, submission structure đã validate, diff với deployment incumbent đã ghi nhận, artifact hash đã lưu, public tuning rule được tuân thủ và regression risk đã review. Deployment incumbent không được nhầm với scientific comparator; một bản đang triển khai không mặc nhiên là baseline khoa học.

## Không được tự động chạy experiment kế tiếp

Sau khi một experiment kết thúc, không tự động fine-tune, thử model lớn hơn, đổi embedding, đổi retrieval, thêm feature, đổi threshold hay chạy model sweep. Trước tiên phải diễn giải kết quả rồi mới chọn nhánh do evidence dẫn dắt. Ví dụ B2a zero-shot hữu ích có thể mở Fine-tuning; B2a bị block bởi tài liệu thật sự dài có thể mở long-document branch; K77 thiếu tài liệu đúng có thể mở retrieval branch. Không ví dụ nào là lệnh chạy tự động.

## Khi nào cần hỏi professor?

Professor review cần thiết khi semantics khoa học thay đổi, mở nhánh training model mới, đổi candidate population, đổi retrieval/embedding, mở Fine-tuning, đổi kiến trúc truncation/chunking, xuất hiện mơ hồ về metric/comparator, preregistered STOP condition kích hoạt, hoặc quyết định public deployment còn mơ hồ về khoa học. Không cần hỏi professor cho việc tạo file thông thường hay chi tiết implementation không làm thay đổi contract.

## Repo/file hygiene cần giữ ra sao?

Dùng git mv cho tracked-file rename. Không xóa historical scientific evidence chỉ vì workflow đã đóng; hãy đánh dấu rõ đó là tài liệu lịch sử. Không giữ nhiều bản sao hiện hành mâu thuẫn của cùng một contract. Khi di chuyển artifact quan trọng, ghi old path, new path và hash trước/sau khi phù hợp.

## Checklist tối thiểu trước mỗi run là gì?

Trước khi chạy, trả lời ngắn các câu sau: experiment nào, ai owner, vì sao chạy bây giờ, phần nào frozen, phần nào được đổi, dùng data nào, có Fold0/public labels không (phải là NO nếu chưa có phép rõ ràng), CPU hay Modal GPU, output path là gì, success rule và STOP rule ra sao, có prediction freeze không, và professor authorization đã đủ chưa. Nếu một câu chưa trả lời được, chưa bắt đầu run.
