# Debug trong VS Code

1. Chạy `scripts/setup.ps1`.
2. Chọn interpreter `backend/.venv` bằng **Python: Select Interpreter**.
3. Điền `GEMINI_API_KEY` trong `backend/.env`.
4. Chạy `scripts/check-gemini.ps1`.
5. Chọn **SLaw: Backend + Frontend** trong Run and Debug.

| Hiện tượng | Kiểm tra đầu tiên | Tầng tiếp theo |
|---|---|---|
| Không kết nối | `/api/v1/health`, URL frontend | CORS/backend terminal |
| `gemini_configured=false` | `GEMINI_API_KEY` trong `backend/.env` | tạo key Google AI Studio |
| Gemini 401/403 | key và project | quota/quyền API |
| Gemini 404 | `GEMINI_GENERATION_MODEL` | dùng model stable trong `.env.example` |
| Gemini 429 | rate limit | retry/quota/tier |
| Upload lỗi | `_save_temporary_upload` | `DocumentParser.parse` |
| Chunk sai Điều/trang | `chunk_pages` | test chunking |
| Không có vector | `EmbeddingService.embed_texts` | `VectorRepository.upsert` |
| Retrieve sai phiên | metadata `session_id` | `query`/`list_chunks` |
| Không tìm thấy | response `trace` | chunk size/candidate_k |
| Rerank chậm lần đầu | tải BGE từ Hugging Face | cache/mạng/dung lượng |
| Generation lỗi | model/key/quota | `GenerationService.generate` |
| Sai nguồn | context trong `RagService` | system prompt generation |
| Refresh mất chat | localStorage | SQLite messages |

Đặt `LOG_LEVEL=DEBUG` để xem thêm. Chạy test:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -vv
```

Dữ liệu runtime: `backend/data/slaw_gemini.db`,
`backend/data/chroma_gemini/`, `backend/data/tmp/`. Muốn reset dev, dừng
server rồi xóa `backend/data` (thao tác này xóa toàn bộ phiên/vector cục bộ).
