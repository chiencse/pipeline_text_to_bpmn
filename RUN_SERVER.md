# Hướng dẫn chạy Server

## Cách 1: Chạy FastAPI Server (Khuyến nghị)

### Bước 1: Kích hoạt virtual environment (nếu có)
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Hoặc Command Prompt
venv\Scripts\activate.bat
```

### Bước 2: Chạy server với uvicorn
```powershell
# Từ thư mục gốc của project
uvicorn app.main:app --reload --port 8000
```

**Giải thích các tham số:**
- `--reload`: Tự động reload khi code thay đổi (chỉ dùng khi development)
- `--port 8000`: Chạy trên cổng 8000
- `--host 0.0.0.0`: Cho phép truy cập từ các máy khác (tùy chọn)

### Bước 3: Kiểm tra server đã chạy
Mở trình duyệt hoặc dùng curl:
```
http://localhost:8000/docs
```

Bạn sẽ thấy Swagger UI với tất cả các endpoints.

## Cách 2: Chạy với Python trực tiếp

```powershell
python -m app.main
```

**Lưu ý:** Cách này sẽ chạy code trong `__main__` block, không phải FastAPI server.

## Cách 3: Chạy với LangGraph Dev (cho LangGraph Studio)

```powershell
langgraph dev
```

Server sẽ chạy trên cổng mặc định của LangGraph (thường là 8123).

## Các Endpoints chính

### 1. Start Pipeline B (với user feedback)
```bash
POST http://localhost:8000/pipeline/b/start
Content-Type: application/json

{
  "text": "Send an email to finance, attach the quotation, wait for reply, then update the invoice in the SAP system.",
  "options": {}
}
```

### 2. Get Pending Feedback
```bash
GET http://localhost:8000/pipeline/b/feedback/{thread_id}
```

### 3. Submit Feedback
```bash
POST http://localhost:8000/pipeline/b/feedback/{thread_id}
Content-Type: application/json

{
  "user_decision": "approve"
}
```

### 4. Get Pipeline Status
```bash
GET http://localhost:8000/pipeline/b/status/{thread_id}
```

### 5. Legacy Endpoints (chạy đồng bộ, không có interrupt)
```bash
POST http://localhost:8000/pipeline/a
POST http://localhost:8000/pipeline/b
```

## Troubleshooting

### Lỗi: Port đã được sử dụng
```powershell
# Thay đổi port
uvicorn app.main:app --reload --port 8001
```

### Lỗi: Module không tìm thấy
```powershell
# Đảm bảo bạn đang ở thư mục gốc của project
cd D:\BK\DACN\langGraph
uvicorn app.main:app --reload --port 8000
```

### Lỗi: Checkpointer conflict với LangGraph API
Nếu chạy `langgraph dev`, server sẽ tự động phát hiện và không dùng custom checkpointer.

Nếu chạy `uvicorn` và muốn tắt checkpointer:
```powershell
# Windows PowerShell
$env:LANGGRAPH_USE_CHECKPOINTER="false"
uvicorn app.main:app --reload --port 8000
```

## Kiểm tra Server đang chạy

### Xem tất cả endpoints
Mở trình duyệt: `http://localhost:8000/docs`

### Health check
```bash
GET http://localhost:8000/
```

### Xem OpenAPI schema
```bash
GET http://localhost:8000/openapi.json
```

## Ví dụ sử dụng với curl

```bash
# Start pipeline
curl -X POST "http://localhost:8000/pipeline/b/start" \
  -H "Content-Type: application/json" \
  -d '{"text": "Send email and create spreadsheet"}'

# Get feedback (thay {thread_id} bằng ID từ response trên)
curl "http://localhost:8000/pipeline/b/feedback/{thread_id}"

# Submit feedback
curl -X POST "http://localhost:8000/pipeline/b/feedback/{thread_id}" \
  -H "Content-Type: application/json" \
  -d '{"user_decision": "approve"}'
```

## Ví dụ sử dụng với Python requests

```python
import requests

# Start pipeline
response = requests.post(
    "http://localhost:8000/pipeline/b/start",
    json={"text": "Send email and create spreadsheet"}
)
data = response.json()
thread_id = data["thread_id"]

# Get feedback
feedback = requests.get(f"http://localhost:8000/pipeline/b/feedback/{thread_id}")
print(feedback.json())

# Submit feedback
result = requests.post(
    f"http://localhost:8000/pipeline/b/feedback/{thread_id}",
    json={"user_decision": "approve"}
)
print(result.json())
```



