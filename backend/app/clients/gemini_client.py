from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from google import genai
from google.genai import errors, types

from app.core.errors import GeminiServiceError


TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
T = TypeVar("T")


class GeminiClient:
    """Bao Google Gen AI SDK, retry và kiểm tra định dạng phản hồi."""

    def __init__(
        self,
        api_key: str,
        retry_count: int,
        retry_delay_seconds: float,
        sdk_client: Any | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.retry_count = max(1, retry_count)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._client = sdk_client
        if self._client is None and self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def close(self) -> None:
        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _require_client(self) -> Any:
        if self._client is None:
            raise GeminiServiceError(
                "Chưa cấu hình GEMINI_API_KEY. Hãy sao chép backend/.env.example "
                "thành backend/.env rồi điền khóa Gemini."
            )
        return self._client

    @staticmethod
    def _error_message(exc: errors.APIError) -> str:
        message = getattr(exc, "message", None)
        return str(message or exc).strip()[:500]

    def _with_retry(self, operation: Callable[[], T], action: str) -> T:
        delay = self.retry_delay_seconds
        for attempt in range(1, self.retry_count + 1):
            try:
                return operation()
            except errors.APIError as exc:
                code = int(getattr(exc, "code", 0) or 0)
                transient = code in TRANSIENT_STATUS_CODES
                if not transient or attempt == self.retry_count:
                    raise GeminiServiceError(
                        f"Gemini lỗi khi {action} (HTTP {code or 'không rõ'}): "
                        f"{self._error_message(exc)}"
                    ) from exc
                time.sleep(delay)
                delay *= 2
            except GeminiServiceError:
                raise
            except Exception as exc:
                raise GeminiServiceError(
                    f"Gemini trả về lỗi không hợp lệ khi {action}: {exc}"
                ) from exc
        raise RuntimeError("Không thể gọi Gemini")

    @staticmethod
    def _normalize(vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]

    def embed(
        self,
        model: str,
        texts: Sequence[str],
        *,
        task_type: str,
        output_dimension: int,
    ) -> list[list[float]]:
        if not texts:
            return []
        client = self._require_client()

        def request() -> Any:
            return client.models.embed_content(
                model=model,
                contents=list(texts),
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=output_dimension,
                ),
            )

        response = self._with_retry(request, "tạo embedding")
        raw_embeddings = getattr(response, "embeddings", None)
        if not raw_embeddings or len(raw_embeddings) != len(texts):
            actual = len(raw_embeddings or [])
            raise GeminiServiceError(
                f"Gemini trả về {actual} embedding cho {len(texts)} văn bản"
            )

        vectors: list[list[float]] = []
        for item in raw_embeddings:
            values = getattr(item, "values", None)
            if values is None or len(values) != output_dimension:
                raise GeminiServiceError(
                    "Gemini trả về embedding thiếu hoặc sai số chiều"
                )
            vector = [float(value) for value in values]
            # gemini-embedding-001 cần chuẩn hóa thủ công khi dùng < 3072 chiều.
            if model == "gemini-embedding-001" and output_dimension != 3072:
                vector = self._normalize(vector)
            vectors.append(vector)
        return vectors

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system_instruction: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        client = self._require_client()

        def request() -> Any:
            return client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )

        response = self._with_retry(request, "sinh nội dung")
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise GeminiServiceError(
                f"Gemini model {model} không trả về nội dung văn bản"
            )
        return text.strip()
