from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from app.clients.gemini_client import GeminiClient
from app.core.errors import GeminiServiceError


HYDE_SYSTEM_PROMPT = """Bạn tạo một đoạn văn bản giả định chỉ để hỗ trợ truy hồi tài liệu pháp luật.
Không trình bày suy luận nội bộ. Không khẳng định đây là tư vấn pháp lý.
Chỉ trả về đoạn văn bản giả định, không thêm tiêu đề hay lời dẫn."""


@dataclass(frozen=True)
class RewriteResult:
    retrieval_query: str
    used_hyde: bool
    warning: str | None = None


class QueryRewriteService:
    def __init__(
        self,
        client: GeminiClient,
        model: str,
        enabled: bool,
        temperature: float,
        max_tokens: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.enabled = enabled
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logger or logging.getLogger(__name__)

    def rewrite(
        self,
        question: str,
        history: Sequence[tuple[str, str]],
    ) -> RewriteResult:
        if not self.enabled:
            return RewriteResult(question, False)

        history_text = "\n".join(
            f"{role.upper()}: {content}" for role, content in history
        ) or "(không có)"
        prompt = (
            "Dựa vào câu hỏi và lịch sử hội thoại, hãy viết một câu trả lời "
            "giả định dài khoảng 100-200 từ. Đoạn này nên chứa các thuật ngữ "
            "pháp luật có khả năng xuất hiện trong tài liệu nguồn để dùng làm "
            "truy vấn semantic search.\n\n"
            f"LỊCH SỬ:\n{history_text}\n\n"
            f"CÂU HỎI: {question}"
        )
        try:
            rewritten = self.client.generate(
                model=self.model,
                prompt=prompt,
                system_instruction=HYDE_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            return RewriteResult(rewritten, True)
        except GeminiServiceError as exc:
            warning = (
                "HyDE không khả dụng; đã dùng nguyên câu hỏi cho dense retrieval: "
                f"{exc.message}"
            )
            self.logger.warning(warning)
            return RewriteResult(question, False, warning)
