# Debug trong VS Code

1. Chạy `scripts/setup.ps1`.
2. Chọn interpreter `backend/.venv` bằng **Python: Select Interpreter**.
3. Chạy `ollama pull bge-m3`, `ollama pull qwen3:4b` và `scripts/check-ollama.ps1`.
4. Điền `backend/.env`.
5. Chọn **SLaw: Backend + Frontend** trong Run and Debug.

| Hiện tượng | Kiểm tra đầu tiên | Tầng tiếp theo |
|---|---|---|
| Không kết nối | `/api/v1/health`, URL frontend | CORS/backend terminal |
| `ollama_available=false` | Ollama đang chạy, cổng 11434 | `OLLAMA_BASE_URL` |
| `embedding_model_available=false` | `ollama list` | `ollama pull bge-m3` |
| `generation_model_available=false` | `ollama list` | `ollama pull qwen3:4b` |
| Upload lỗi | `_save_temporary_upload` | `DocumentParser.parse` |
| Chunk sai Điều/trang | `chunk_pages` | test chunking |
| Không có vector | `_embed_batch` | `VectorRepository.upsert` |
| Retrieve sai phiên | metadata `session_id` | `VectorRepository.query` |
| Không tìm thấy | score, `MIN_SIMILARITY` | chunk size/top_k |
| Generation lỗi | `/api/chat`, tên model | `GenerationService.generate` |
| Sai nguồn | context trong `RagService` | system prompt generation |
| Refresh mất chat | localStorage | SQLite messages |

Đặt `LOG_LEVEL=DEBUG` để xem thêm. Chạy test:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -vv
```

Dữ liệu runtime: `backend/data/slaw_ollama.db`, `backend/data/chroma_ollama/`, `backend/data/tmp/`. Muốn reset dev, dừng server rồi xóa `backend/data` (thao tác này xóa toàn bộ phiên/vector cục bộ).
