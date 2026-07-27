# Test Cases Reference cho LegalIR & LegalQA

## Mục tiêu

Bộ test này dùng làm benchmark tham chiếu cho TV5 khi đánh giá routing, retrieval, QA, citation, tool calling và web search. Đây không phải test tự động hoàn chỉnh, mà là nguồn câu hỏi chuẩn để chuyển thành unit/integration test.

## Follow-up Questions và Query Rewriting

```json
[
  {
    "context": "Người dùng vừa hỏi về thủ tục thành lập công ty",
    "follow_up": "Còn điều kiện đăng ký thì sao?",
    "expected_rewrite": "Điều kiện đăng ký thành lập doanh nghiệp theo pháp luật Việt Nam"
  },
  {
    "context": "Thảo luận về hợp đồng lao động",
    "follow_up": "Nó có hiệu lực bao lâu?",
    "expected_rewrite": "Thời hạn hiệu lực của hợp đồng lao động theo Bộ luật Lao động"
  },
  {
    "context": "Câu hỏi về thuế thu nhập cá nhân",
    "follow_up": "Làm sao để khai báo đây?",
    "expected_rewrite": "Cách thức khai báo thuế thu nhập cá nhân theo quy định"
  },
  {
    "context": "Hỏi về quyền thừa kế",
    "follow_up": "Có bao nhiêu hàng thừa kế vậy?",
    "expected_rewrite": "Số hàng thừa kế theo Bộ luật Dân sự Việt Nam"
  }
]
```

## Route Detection: Legal RAG

```json
[
  {
    "query": "Quyền và nghĩa vụ của người lao động theo Bộ luật Lao động 2019",
    "expected_route": "legal_rag",
    "description": "Tra cứu văn bản pháp luật cụ thể"
  },
  {
    "query": "Thủ tục đăng ký kết hôn tại UBND",
    "expected_route": "legal_rag",
    "description": "Thủ tục hành chính theo quy định"
  },
  {
    "query": "Điều kiện để được ly hôn đơn phương",
    "expected_route": "legal_rag",
    "description": "Điều kiện pháp lý cụ thể"
  },
  {
    "query": "Trách nhiệm hình sự của người chưa thành niên",
    "expected_route": "legal_rag",
    "description": "Quy định về trách nhiệm hình sự"
  }
]
```

## Route Detection: Agent Tools

```json
[
  {
    "query": "Tính tiền phạt hợp đồng 500 triệu chậm 45 ngày với lãi suất 0.15% mỗi ngày",
    "expected_route": "agent_tools",
    "description": "Tính toán phạt hợp đồng"
  },
  {
    "query": "Kiểm tra người sinh năm 2006 có đủ tuổi ký hợp đồng lao động không?",
    "expected_route": "agent_tools",
    "description": "Kiểm tra tuổi pháp lý"
  },
  {
    "query": "Chia thừa kế cho 3 con với tài sản 2 tỷ đồng theo luật",
    "expected_route": "agent_tools",
    "description": "Tính toán chia thừa kế"
  },
  {
    "query": "Công ty ABC có hợp lệ theo quy định đặt tên doanh nghiệp không?",
    "expected_route": "agent_tools",
    "description": "Kiểm tra quy tắc đặt tên"
  }
]
```

## Route Detection: Web Search

```json
[
  {
    "query": "Luật Đất đai 2024 có những thay đổi gì mới nhất?",
    "expected_route": "web_search",
    "description": "Thông tin pháp luật mới"
  },
  {
    "query": "Mức lương tối thiểu vùng năm 2024 hiện tại",
    "expected_route": "web_search",
    "description": "Thông tin cập nhật gần đây"
  },
  {
    "query": "Vụ án tham nhũng ở Quảng Ninh vừa xét xử gần đây",
    "expected_route": "web_search",
    "description": "Tin tức pháp lý hiện tại"
  }
]
```

## Route Detection: General Chat

```json
[
  {
    "query": "Xin chào, bạn có thể giúp tôi được không?",
    "expected_route": "general_chat",
    "description": "Chào hỏi"
  },
  {
    "query": "Cảm ơn bạn đã hỗ trợ",
    "expected_route": "general_chat",
    "description": "Cảm ơn"
  },
  {
    "query": "Hôm nay thời tiết Hà Nội thế nào?",
    "expected_route": "general_chat",
    "description": "Chủ đề ngoài pháp luật"
  }
]
```

## Complex Legal Questions

```json
[
  {
    "query": "Người nước ngoài có thể sở hữu nhà ở Việt Nam không?",
    "complexity": "high",
    "expected_docs": ["Luật Nhà ở", "Luật Đầu tư", "Nghị định 99/2015"]
  },
  {
    "query": "Điều kiện để được miễn thuế thu nhập doanh nghiệp",
    "complexity": "medium",
    "expected_docs": ["Luật Thuế TNDN", "Nghị định 218/2013"]
  }
]
```

## Web Search Integration

```json
[
  {
    "query": "Nghị định mới về giao thông 2024",
    "search_type": "tavily_search_legal",
    "expected_sources": ["thuvienphapluat.vn", "baochinhphu.vn"],
    "expected_content": "Thông tin về văn bản pháp luật mới"
  },
  {
    "query": "Lương tối thiểu vùng 1 năm 2024",
    "search_type": "tavily_qna",
    "expected_answer": "Mức lương tối thiểu cụ thể"
  }
]
```

## Expected Flow

```json
{
  "query": "Luật Đất đai 2024 thay đổi gì",
  "expected_flow": [
    "Phát hiện từ khóa mới nhất hoặc năm hiện tại",
    "Route sang web_search",
    "Gọi tavily_search_tool",
    "Tổng hợp kết quả kèm nguồn",
    "Trả lời qua QA layer với warning nếu nguồn chưa được xác minh trong corpus local"
  ]
}
```
