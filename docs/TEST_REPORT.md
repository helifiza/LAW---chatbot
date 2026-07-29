# Báo cáo kiểm thử

Ngày kiểm tra: 2026-07-28.

## Kết quả

```text
Ruff: All checks passed
Pytest: 18 passed
FastAPI GET /api/v1/health: HTTP 200
```

Các test bao phủ:

- Gemini client, config task type và chuẩn hóa L2;
- embedding document/query;
- generation prompt grounded;
- HyDE và fallback về câu hỏi gốc;
- dense + BM25 + RRF + CrossEncoder giả lập;
- Chroma query/list/delete theo `session_id`;
- pipeline index → retrieve → rerank → generate → lưu lịch sử.

Test không dùng API key thật, không gọi Gemini thật và không tải trọng số
`BAAI/bge-reranker-v2-m3`. Sau khi điền key, chạy
`scripts/check-gemini.ps1`, rồi thực hiện một câu hỏi thật với `debug=true`
để kiểm tra quota, model availability và khả năng tải reranker trên máy đích.
