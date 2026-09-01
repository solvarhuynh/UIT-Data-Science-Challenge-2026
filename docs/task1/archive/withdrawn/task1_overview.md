**UIT DATA SCIENCE CHALLENGE 2026**

# **Task 1 – Legal Information Retrieval (LegalIR)**

*Truy vấn thông tin pháp luật tiếng Việt*

| Mục tiêu | Với một câu hỏi pháp luật, hệ thống xác định văn bản hành chính/pháp lý chứa thông tin cần thiết và trả về ID của văn bản đó. |
| :---- | :---- |
| **Quy mô dữ liệu** | Khoảng 8.500 văn bản hành chính \+ 10.000 câu hỏi. |

# **1\. Mô tả bài toán**

* Input: một câu hỏi tiếng Việt cần truy vấn thông tin pháp luật.  
* Output: ID của văn bản có thông tin cần thiết để trả lời câu hỏi.

| "id": {         "question": "Việc tổ chức vận động, tiếp nhận, sử dụng nguồn đóng góp tự nguyện được thực hiện dựa trên nguyên tắc nào?",         "answer": \[             "177504"         \]     } |
| :---- |

# **2\. Cấu trúc dữ liệu cung cấp**

| Tệp  | Nội dung |
| :---- | :---- |
| **train.json** | Tập dữ liệu huấn luyện cho các đội phát triển phương pháp. |
| **warmup.json** | Tập dữ liệu mẫu phục vụ vòng Warm-up, giúp làm quen bài toán và quy trình submission. |
| **public-official.json** | Tập dữ liệu dùng trong giai đoạn Public Test theo cấu hình chính thức. |
| **private-official.json** | Tập dữ liệu chính thức của Private Test. |
| **selected-contexts.zip** | Kho văn bản được chọn; gồm nhiều tệp context\_\*.json.  |

# **3\. Ví dụ cấu trúc một văn bản**

**context\_\*.json**

* id: mã định danh duy nhất của văn bản  
* name: tiêu đề văn bản  
* link: đường dẫn nguồn  
* passage: nội dung văn bản được sử dụng để truy vấn

| {     "link": "https://thuvienphapluat.vn/van-ban/Bo-may-hanh-chinh/Quyet-dinh-5868-QD-BYT-2018-co-cau-to-chuc-cua-Vu-Trang-thiet-bi-va-Cong-trinh-y-te-396608.aspx",     "name": "Quyet-dinh-5868-QD-BYT-2018-co-cau-to-chuc-cua-Vu-Trang-thiet-bi-va-Cong-trinh-y-te-396608",     "passage": "BỘ Y TẾ\\r\\n\\n  -------\\n\\nCỘNG HÒA XÃ HỘI\\r\\n\\n  CHỦ NGHĨA VIỆT NAM\\r\\n\\n  Độc lập \- Tự do \- Hạnh phúc \\r\\n\\n  ---------------\\n\\nSố: 5868/QĐ-BYT\\n\\nHà Nội, ngày 28\\r\\n\\n  tháng 9 năm 2018\\n\\n\\n\\nQUYẾT ĐỊNH\\n\\nQUY\\r\\n\\nĐỊNH CHỨC NĂNG, NHIỆM VỤ, QUYỀN HẠN VÀ CƠ CẤU TỔ CHỨC CỦA VỤ TRANG THIẾT BỊ VÀ CÔNG\\r\\n\\nTRÌNH Y TẾ THUỘC BỘ Y TẾ\\n\\nBỘ TRƯỞNG BỘ Y TẾ\\n\\nCăn cứ Nghị định số 75/2017/NĐ-CP … \- Lưu: VT, TCCB, TTB, PC.\\n\\nBỘ TRƯỞNG\\n\\n\\r\\n\\n  Nguyễn Thị Kim Tiến\\n\\n",     "id": 740 }  |
| :---- |

# **4\. Đánh giá tác vụ**

Hệ thống cần trả về danh sách các document\_id theo thứ tự giảm dần về mức độ liên quan. Điểm xếp hạng sử dụng MRR làm độ đo chính và Recall@3 làm độ đo phụ.

| Độ đo | Vai trò | Mô tả |
| :---- | :---- | :---- |
| **MRR** | Độ đo chính | Đánh giá vị trí của văn bản đúng trong danh sách kết quả. Văn bản đúng càng ở vị trí cao thì điểm càng lớn. |
| **Recall@3** | Độ đo phụ | Đo tỷ lệ câu hỏi mà văn bản đúng xuất hiện trong 03 kết quả đầu tiên. |

