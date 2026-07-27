# UDSC2026 Frontend

Frontend React cho hệ thống RAG pháp luật UDSC2026. Ứng dụng chỉ chịu trách nhiệm hiển thị giao diện chat, gửi request tới FastAPI backend và render citation/streaming response.

## Tech Stack

- React
- Vite
- TailwindCSS
- Backend API: `http://localhost:8000`

## Chạy local

```powershell
cd frontend
npm install
npm run dev
```

Frontend mặc định chạy ở:

```text
http://localhost:5173
```

## Biến môi trường

```env
VITE_API_BASE_URL=http://localhost:8000
```

## Phạm vi frontend

- Ô nhập câu hỏi pháp luật.
- Khung chat gửi/nhận API.
- Streaming response từ backend nếu endpoint SSE được bật.
- Render Markdown câu trả lời.
- Hiển thị citation theo `law_name`, `article`, `clause`, `quote`.

Frontend không chứa logic retrieval, prompt, embedding hoặc LLM.
