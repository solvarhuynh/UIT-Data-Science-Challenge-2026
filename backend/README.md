# UDSC Local Grader

Đây là backend độc lập cho giao diện `../frontend/`. Nó không import package,
model, corpus hay prediction nào từ repo UDSC hiện tại. Có thể xóa toàn bộ thư
mục `backend/` và `frontend/` mà không ảnh hưởng pipeline chính.

## Chạy local

Từ thư mục gốc:

```powershell
python backend/server.py
```

Mở `http://127.0.0.1:8765`. Có thể đổi cổng bằng `--port 9000` hoặc biến môi
trường `UDSC_GRADER_PORT`.

Server dùng SQLite trong `backend/data/grader.sqlite3` và giữ file upload trong
`backend/data/uploads/`. Đây là dữ liệu runtime riêng của grader.

Muốn QA dùng đúng nhánh scorer có WordNet/OMW và ROUGE-L của BTC, cài thêm:

```powershell
python -m pip install -r backend/requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

Nếu không cài các gói này, server vẫn chạy nhưng sẽ đánh dấu kết quả QA là
`approximate-fallback` để không nhầm với điểm scorer chính thức.

## File submission được nhận

- `.json` trực tiếp;
- `.zip` chứa đúng một file `submission.json`.

LegalIR có dạng object:

```json
{
  "question-id": {"answer": ["document-id-1", "document-id-2"]}
}
```

LegalQA có dạng object:

```json
{
  "question-id": {"answer": "câu trả lời"}
}
```

IR chấm macro Recall và Precision theo quy tắc BTC, tối đa năm document ID cho
mỗi câu. QA chấm macro METEOR và ROUGE-L. Nếu máy thiếu WordNet/OMW hoặc
`rouge-score`, giao diện sẽ ghi rõ rằng kết quả QA là fallback xấp xỉ.

## Reference độc lập

Backend không tự đọc `data/raw/btc/` và không dùng train làm reference. Đặt các
file riêng vào:

```text
backend/data/references/
├── ir/
│   ├── warmup.json
│   ├── public.json
│   ├── private.json
│   └── holdout.json
└── qa/
    ├── warmup.json
    ├── public.json
    ├── private.json
    └── holdout.json
```

Mỗi file reference dùng cùng schema với submission nhưng chứa `answer` gold.
Stage chưa có file reference vẫn nhận và lưu submission, nhưng trạng thái là
`Chờ reference` và không tạo điểm.

`public-official.json` trong repo hiện chỉ chứa câu hỏi với `answer: null`, vì
vậy nó đủ để tạo submission nhưng không đủ để chấm điểm. Muốn có điểm local có
ý nghĩa, cần một holdout/reference được giữ ngoài quá trình fit và tune. Nếu
chỉ tách một phần từ train sau khi model đã học toàn bộ train, hãy gọi đó là
diagnostic nội bộ, không coi là test độc lập.

Có thể đặt reference ở thư mục khác bằng:

```powershell
$env:UDSC_GRADER_REFERENCE_DIR = 'D:\du-lieu-bi-mat\udsc-references'
python backend/server.py
```
