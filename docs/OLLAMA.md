# Ollama trong SLaw

## Hai model, một tiến trình local

```text
Tài liệu/câu hỏi → bge-m3 → vector → Chroma
Ngữ cảnh retrieve → qwen3:4b → câu trả lời có nguồn
```

Backend gọi API native tại `http://127.0.0.1:11434`:

- `GET /api/tags`: health check và danh sách model.
- `POST /api/embed`: embedding tài liệu và câu hỏi.
- `POST /api/chat`: generation, `stream=false`, `think=false`.

Mọi HTTP/retry/validate phản hồi nằm trong `app/clients/ollama_client.py`. `EmbeddingService` và `GenerationService` chỉ xử lý nghiệp vụ của từng tầng.

## Cài trên Windows

```powershell
ollama --version
ollama pull bge-m3
ollama pull qwen3:4b
ollama list
.\scripts\check-ollama.ps1
```

Nếu `ollama` chưa vào PATH nhưng đã cài:

```powershell
$OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
& $OllamaExe pull bge-m3
& $OllamaExe pull qwen3:4b
```

## Health check

`GET /api/v1/health` phân biệt rõ:

- Ollama server có kết nối được không.
- `bge-m3` có sẵn không.
- `qwen3:4b` có sẵn không.
- Chroma đang có bao nhiêu vector.

## Lỗi thường gặp

- `ollama_available=false`: mở Ollama hoặc chạy `ollama serve`.
- `embedding_model_available=false`: chạy `ollama pull bge-m3`.
- `generation_model_available=false`: chạy `ollama pull qwen3:4b`.
- `OLLAMA_REQUEST_FAILED` có `model ... not found`: tên model trong `.env` chưa được pull.
- Generation chậm ở câu đầu: Ollama đang nạp model vào RAM/GPU.
- Hết RAM/VRAM: đóng model/app khác hoặc chọn generation model nhỏ hơn.
- Collection báo khác embedding model: đổi `CHROMA_COLLECTION_NAME` rồi upload lại.

## Đổi model

Ví dụ đổi generation:

```powershell
ollama pull qwen3:8b
```

```env
OLLAMA_GENERATION_MODEL=qwen3:8b
```

Sau đó khởi động lại backend; không cần lập chỉ mục lại vì embedding vẫn là `bge-m3`.

Nếu đổi embedding model, phải tạo collection/database mới và upload lại vì vector từ hai model không cùng không gian biểu diễn.
