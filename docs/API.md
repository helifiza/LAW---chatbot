# API

Base URL: `http://127.0.0.1:8000/api/v1`; Swagger: `http://127.0.0.1:8000/docs`.

| Method | Path | Mục đích |
|---|---|---|
| GET | `/health` | trạng thái API, Ollama và hai model local |
| POST | `/sessions` | tạo phiên |
| GET | `/sessions/{session_id}` | khôi phục phiên |
| DELETE | `/sessions/{session_id}` | đóng/xóa phiên |
| POST | `/sessions/{session_id}/documents` | upload multipart nhiều file |
| DELETE | `/sessions/{session_id}/documents` | xóa toàn bộ file |
| DELETE | `/sessions/{session_id}/documents/{document_id}` | xóa một file |
| POST | `/sessions/{session_id}/questions` | retrieve + generation |

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions
curl -X POST -F "files=@tai_lieu.pdf" http://127.0.0.1:8000/api/v1/sessions/SESSION_ID/documents
curl -X POST -H "Content-Type: application/json" -d '{"question":"Điều kiện là gì?","top_k":5}' http://127.0.0.1:8000/api/v1/sessions/SESSION_ID/questions
```

`GET /health` trả thêm `ollama_available`, `embedding_provider`, `embedding_model`, `embedding_model_available`, `generation_provider`, `generation_model` và `generation_model_available`.

Lỗi có dạng `{"error":{"code":"...","message":"..."}}`. Các code chính: `SESSION_NOT_FOUND`, `SESSION_EXPIRED`, `DOCUMENT_LIMIT_REACHED`, `FILE_TOO_LARGE`, `UNSUPPORTED_FILE_TYPE`, `DOCUMENT_PARSE_ERROR` và `OLLAMA_REQUEST_FAILED`.
