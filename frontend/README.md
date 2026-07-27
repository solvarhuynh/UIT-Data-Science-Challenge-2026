# HCMUTE-SHIPCODE Frontend

Giao diện tra cứu và hỏi đáp pháp luật Việt Nam của đội HCMUTE-SHIPCODE nằm
trong thư mục `giao dien`.

## Chạy local

Yêu cầu Node.js 20 trở lên.

```powershell
cd "frontend/giao dien"
npm install
npm run dev
```

## Kiểm tra trước khi phát hành

```powershell
cd "frontend/giao dien"
npm run lint
npm run build
```

Frontend gọi `POST /api/v1/query`. Khi backend chưa sẵn sàng, ứng dụng sử dụng
dữ liệu mẫu để phục vụ trình diễn. Chi tiết về công nghệ, cấu trúc và cách kết
nối backend được mô tả trong `giao dien/README.md`.
