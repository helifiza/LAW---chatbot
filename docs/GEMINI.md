# Gemini trong SLaw

## Cấu hình key

1. Tạo API key trong Google AI Studio.
2. Chạy `scripts/setup.ps1` để tạo `backend/.env`.
3. Điền:

```dotenv
GEMINI_API_KEY=your_real_key
```

Không đặt key trong notebook, source Python, frontend hoặc Git. Backend đọc
key từ biến môi trường; file `.env` thật đã nằm trong `.gitignore`.

## Model

- Embedding: `gemini-embedding-001`, 768 chiều.
- Generation/HyDE: `gemini-3.5-flash-lite`.
- Reranker local: `BAAI/bge-reranker-v2-m3`.

Tài liệu được embed với `RETRIEVAL_DOCUMENT`; truy vấn dense dùng
`RETRIEVAL_QUERY`. Vector 768 chiều của `gemini-embedding-001` được chuẩn hóa
L2 trước khi lưu/truy vấn cosine.

Reranker được lazy-load ở câu hỏi đầu tiên và có thể cần tải khoảng vài GB.
Nếu tải hoặc inference lỗi, backend ghi cảnh báo và dùng thứ tự RRF thay thế.

## Bắt buộc embedding lại

Vector Ollama `bge-m3` cũ không cùng không gian vector với Gemini. Cấu hình
mặc định vì vậy dùng đường dẫn và collection mới:

```dotenv
CHROMA_PERSIST_DIR=data/chroma_gemini
CHROMA_COLLECTION_NAME=slaw_documents_gemini_embedding_001_768_v1
```

Sau khi nâng cấp, tạo phiên mới và upload lại file. Không đổi collection về
tên cũ. Nếu đổi model hoặc số chiều sau này, tiếp tục dùng collection mới.

## Kiểm tra

```powershell
.\scripts\check-gemini.ps1
```

Sau đó gọi câu hỏi với `"debug": true` để xem trace từng tầng.
