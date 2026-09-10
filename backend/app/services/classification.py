from __future__ import annotations

import logging
import re
from typing import Sequence

from app.clients.gemini_client import GeminiClient
from app.services.query_router_service import LINH_VUC_TAXONOMY

logger = logging.getLogger(__name__)

_LINH_VUC_LIST_BLOCK = "\n".join(f"  {code}. {name}" for code, name in LINH_VUC_TAXONOMY.items())

_CLASSIFY_SYSTEM_PROMPT = f"""Bạn phân loại văn bản pháp luật Việt Nam vào ĐÚNG 1 mã lĩnh vực
duy nhất trong danh mục sau, dựa trên tên file và phần nội dung trích ở đầu văn bản.

Danh mục lĩnh vực:
{_LINH_VUC_LIST_BLOCK}

Chỉ trả về DUY NHẤT mã số 2 chữ số (ví dụ: 05), không thêm bất kỳ ký tự, giải
thích, hay markdown nào khác. Nếu văn bản thực sự bao trùm nhiều lĩnh vực
ngang nhau và không có lĩnh vực nào nổi trội hơn hẳn, trả về 19.
"""

_CODE_PATTERN = re.compile(r"\b(0[1-9]|1[0-9])\b")

DEFAULT_MAX_EXCERPT_CHARS = 6000


class LinhVucClassificationService:
    """Phân loại 1 lĩnh vực (linh_vuc) cho toàn bộ document, chạy 1 lần khi
    indexing xong — dùng chung taxonomy với QueryRouterService để đảm bảo
    mã trả về ở 2 nơi luôn khớp nhau (LINH_VUC_TAXONOMY là nguồn duy nhất).

    Đây là tính năng BỔ TRỢ cho retrieval (lọc phạm vi tìm kiếm), không phải
    core path — mọi lỗi (Gemini lỗi, parse sai định dạng, model trả mã lạ)
    đều được nuốt và trả về None thay vì raise, để không làm hỏng indexing.
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        model: str,
        max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
    ) -> None:
        self.gemini_client = gemini_client
        self.model = model
        self.max_excerpt_chars = max_excerpt_chars

    def classify(self, file_name: str, chunk_texts: Sequence[str]) -> str | None:
        excerpt = "\n\n".join(chunk_texts)[: self.max_excerpt_chars].strip()
        if not excerpt:
            return None
        prompt = f"TÊN FILE: {file_name}\n\nNỘI DUNG TRÍCH (đầu văn bản):\n{excerpt}"
        try:
            raw = self.gemini_client.generate(
                model=self.model,
                prompt=prompt,
                system_instruction=_CLASSIFY_SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=20,
            )
        except Exception:
            logger.exception("Phân loại lĩnh vực thất bại cho file %s, bỏ qua", file_name)
            return None

        match = _CODE_PATTERN.search(raw.strip())
        if match is None:
            logger.warning(
                "Model trả về mã lĩnh vực không đọc được (%r) cho file %s", raw, file_name
            )
            return None
        code = match.group(1)
        if code not in LINH_VUC_TAXONOMY:
            logger.warning("Model trả về mã lĩnh vực không hợp lệ (%s) cho file %s", code, file_name)
            return None
        return code