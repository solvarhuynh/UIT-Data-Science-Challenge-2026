# Technical debt — Legal Structure Parser

## Quyết định hiện tại

Parser ưu tiên hoàn thành luồng `Extract → Clean → Parse → Chunk` trên tập mẫu
để bàn giao dữ liệu cho Retrieval. Không mở rộng state machine hay thay đổi schema
cây phân cấp trước khi Parent-Child Chunking chạy end-to-end.

Ngoại lệ an toàn đã thực hiện: `Khoản1` và `Điểma` được nhận diện vì pattern
vẫn neo ở đầu dòng và chỉ được gọi khi parser đang ở trong một `Điều`. Test hồi
quy đảm bảo câu `Tài khoản 1 ...` không bị coi là cấu trúc pháp luật.

## Hạng mục defer

- Khối sửa đổi/trích dẫn: `Điều X` trong phần trích có thể đóng Điều cha sai.
  Defer vì suspend/resume cần tiêu chí kết thúc khối rõ ràng. Trước khi làm, thu
  thập corpus sửa đổi và viết test fail cho mở/đóng khối và nested quote.
- Tiêu đề nhiều dòng: title của Điều có thể ở dòng sau `Điều 10.`. Defer vì
  look-ahead có thể nhầm câu nội dung ngắn thành title. Cần fixture PDF thật và
  tín hiệu tin cậy (style reader hoặc rule ngắn, không phải marker cấp thấp).
- Cấp `Phần`: mất ngữ cảnh vĩ mô ở Bộ luật lớn. Đây là thay đổi schema xuyên
  parser, chunker, metadata và citation; chỉ làm sau khi hợp đồng chunk/TV2 ổn
  định, kèm `Part` và migration-compatibility test.
- Format dị thường khác: OCR/justify có thể làm lệch pattern. Mỗi biến thể
  phải có test fail riêng; chỉ nới rule nếu toàn bộ regression suite vẫn xanh.

## Quy trình xử lý sau MVP

1. Ghi nhận văn bản lỗi và rút fixture tối thiểu, đã ẩn dữ liệu nhạy cảm nếu có.
2. Thêm test tái hiện lỗi trước; xác nhận test đang fail.
3. Sửa rule nhỏ nhất có thể, không thay đổi public contract nếu không cần.
4. Chạy toàn bộ test parser và end-to-end chunking; kiểm tra không thay đổi bất
   ngờ số lượng article/chunk trên corpus baseline.
