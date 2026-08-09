# HCMUTE-SHIPCODE — Trợ lý pháp luật Việt Nam

Giao diện tra cứu và hỏi đáp pháp luật của đội HCMUTE-SHIPCODE. Ứng dụng tập trung
vào ba trải nghiệm chính:

- hỏi đáp dựa trên căn cứ pháp lý được truy hồi;
- tra cứu văn bản theo luật, số hiệu, điều, khoản và điểm;
- kiểm tra nguồn trích dẫn và điểm truy hồi ngay bên cạnh câu trả lời.

## Chạy dự án

Yêu cầu Node.js `^20.19.0` hoặc `>=22.12.0`.

```powershell
npm install
npm run dev
```

Build bản phát hành:

```powershell
npm run build
```

## Kết nối backend

Frontend gọi `POST /api/v1/query`. Khi backend chưa sẵn sàng, ứng dụng tự dùng
dữ liệu mẫu tiếng Việt để phục vụ trình diễn. Phần kết nối được tách riêng trong
`src/lib/legal/api.ts`.

## Công nghệ

- TanStack Start và React
- TypeScript
- Tailwind CSS
- shadcn/ui
- Motion

Logo chính thức nằm tại `LOGO SHIPCODE.png`.

Họa tiết nền trống đồng Đông Sơn sử dụng bản minh họa CC0 từ Wikimedia Commons:
`public/dong-son-drum.png`.
