# Các thay đổi từ notebook RAG sang ứng dụng

## File bắt buộc sửa

| File | Thay đổi |
|---|---|
| `backend/app/core/config.py` | Gemini key/model, dimension 768, HyDE, candidate/RRF/rerank và collection mới |
| `backend/app/core/errors.py` | lỗi Gemini và reranker |
| `backend/app/clients/gemini_client.py` | Google Gen AI SDK, retry, validate, chuẩn hóa L2 |
| `backend/app/services/embedding_service.py` | `RETRIEVAL_DOCUMENT` khi index, `RETRIEVAL_QUERY` khi tìm |
| `backend/app/services/query_rewrite_service.py` | HyDE và fallback câu hỏi gốc |
| `backend/app/services/hybrid_retrieval_service.py` | Gemini dense + BM25 + RRF |
| `backend/app/services/reranking_service.py` | lazy-load `BAAI/bge-reranker-v2-m3` |
| `backend/app/services/generation_service.py` | sinh câu trả lời bằng Gemini |
| `backend/app/services/rag_service.py` | ghép 5 stage và tạo trace |
| `backend/app/repositories/vector_repository.py` | validate provider/model/dimension và đọc chunks cho BM25 |
| `backend/app/container.py` | nối dependency mới |
| `backend/app/main.py` | đóng Gemini client khi shutdown |
| `backend/app/api/schemas.py` | `debug`, `trace`, health mới |
| `backend/app/api/routers/chat.py` | chuyển cờ debug vào pipeline |
| `backend/app/api/routers/health.py` | báo cấu hình Gemini/reranker |
| `backend/requirements.txt` | `google-genai`, `rank-bm25`, `sentence-transformers`, `numpy` |
| `backend/.env.example` | biến môi trường mới, không chứa key thật |

## File được thay thế/xóa

- Xóa `backend/app/clients/ollama_client.py`.
- Xóa `backend/tests/test_ollama_client.py`, thay bằng
  `backend/tests/test_gemini_client.py`.
- Xóa `docs/OLLAMA.md`, thay bằng `docs/GEMINI.md`.
- Xóa `scripts/check-ollama.ps1`, thay bằng `scripts/check-gemini.ps1`.

## Test và tài liệu cũng được cập nhật

| File | Mục đích |
|---|---|
| `backend/tests/test_embedding_service.py` | kiểm tra batch và task type Gemini |
| `backend/tests/test_generation_service.py` | kiểm tra prompt/call Gemini |
| `backend/tests/test_gemini_client.py` | kiểm tra config SDK, L2, thiếu key |
| `backend/tests/test_query_rewrite_service.py` | kiểm tra HyDE và fallback |
| `backend/tests/test_hybrid_retrieval_service.py` | kiểm tra dense/BM25/RRF/rerank |
| `backend/tests/test_rag_pipeline.py` | kiểm tra pipeline end-to-end và trace |
| `backend/tests/test_vector_repository.py` | kiểm tra `list_chunks` theo session |
| `README.md` | hướng dẫn cài/chạy và bật debug |
| `docs/API.md` | schema health/debug mới |
| `docs/ARCHITECTURE.md` | kiến trúc pipeline mới |
| `docs/DEBUGGING.md` | xử lý lỗi Gemini/reranker |
| `docs/TEST_REPORT.md` | kết quả kiểm thử bàn giao |
| `scripts/setup.ps1` | bỏ bước cài/pull Ollama |

## Khác biệt có chủ đích so với cell Colab

Notebook giữ chunks trong biến RAM `result`; backend đọc chunks theo
`session_id` từ Chroma nên vẫn chạy đúng sau khi restart.

Notebook dùng HyDE cho cả dense và BM25. Backend dùng:

- HyDE cho dense semantic retrieval;
- câu hỏi gốc cho BM25 và CrossEncoder.

Như vậy các từ khóa như tên luật, số Điều, loại thuế không bị đoạn HyDE làm
loãng. Hai danh sách vẫn được hợp nhất bằng RRF giống notebook.

Nguồn giữ đúng thứ tự rerank, không chuyển qua `set`. Reranker được lazy-load
và có fallback RRF để API không chết nếu máy thiếu RAM/mạng.

## Việc phải làm sau khi nhận ZIP

1. Chạy `scripts/setup.ps1`.
2. Điền `GEMINI_API_KEY` vào `backend/.env`.
3. Tạo phiên chat mới.
4. Upload lại toàn bộ tài liệu để embedding bằng Gemini.
5. Hỏi với `"debug": true` để kiểm tra trace trước khi tắt debug.
