from __future__ import annotations

import time
from typing import Sequence

import httpx

from app.core.errors import OllamaServiceError


TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class OllamaClient:
    """Bao API native của Ollama để service không phụ thuộc HTTP chi tiết."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        retry_count: int,
        retry_delay_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        # Tự tương thích với file .env của bản cũ dùng OpenAI-compatible `/v1/`.
        if normalized_url.endswith("/v1"):
            normalized_url = normalized_url[:-3]
        self.base_url = normalized_url
        self.retry_count = max(1, retry_count)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            transport=transport,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def model_is_available(configured_model: str, available_models: set[str]) -> bool:
        configured = configured_model.strip().lower()
        normalized = {model.strip().lower() for model in available_models}
        if configured in normalized:
            return True
        return ":" not in configured and f"{configured}:latest" in normalized

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                return str(payload["error"])
        except ValueError:
            pass
        return response.text.strip()[:300] or f"HTTP {response.status_code}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        retry_count: int | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        attempts = max(1, retry_count or self.retry_count)
        delay = self.retry_delay_seconds
        for attempt in range(1, attempts + 1):
            try:
                request_options: dict[str, object] = {"json": json}
                if timeout_seconds is not None:
                    request_options["timeout"] = timeout_seconds
                response = self._client.request(
                    method,
                    path,
                    **request_options,
                )
                if (
                    response.status_code in TRANSIENT_STATUS_CODES
                    and attempt < attempts
                ):
                    time.sleep(delay)
                    delay *= 2
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Phản hồi JSON không phải object")
                return payload
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == attempts:
                    raise OllamaServiceError(
                        "Không kết nối hoặc không nhận được phản hồi từ Ollama tại "
                        f"{self.base_url} sau {attempts} lần thử. "
                        "Hãy kiểm tra ứng dụng Ollama đang chạy."
                    ) from exc
                time.sleep(delay)
                delay *= 2
            except httpx.HTTPStatusError as exc:
                raise OllamaServiceError(
                    f"Ollama từ chối {method} {path}: "
                    f"{self._error_detail(exc.response)}"
                ) from exc
            except ValueError as exc:
                raise OllamaServiceError(
                    f"Ollama trả về dữ liệu không hợp lệ cho {method} {path}: {exc}"
                ) from exc
        raise RuntimeError("Không thể gọi Ollama")

    def list_models(self, timeout_seconds: float = 2.0) -> set[str]:
        payload = self._request(
            "GET",
            "/api/tags",
            retry_count=1,
            timeout_seconds=timeout_seconds,
        )
        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaServiceError("Ollama không trả về danh sách model hợp lệ")
        names: set[str] = set()
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if name:
                names.add(str(name))
        return names

    def status(self, required_models: Sequence[str]) -> tuple[bool, dict[str, bool]]:
        try:
            available_models = self.list_models()
        except OllamaServiceError:
            return False, {model: False for model in required_models}
        availability = {
            model: self.model_is_available(model, available_models)
            for model in required_models
        }
        return True, availability

    def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        payload = self._request(
            "POST",
            "/api/embed",
            json={"model": model, "input": list(texts), "truncate": True},
        )
        raw_embeddings = payload.get("embeddings")
        if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(texts):
            raise OllamaServiceError(
                f"Ollama model {model} trả về thiếu hoặc sai định dạng embedding"
            )
        if any(not isinstance(vector, list) for vector in raw_embeddings):
            raise OllamaServiceError(
                f"Ollama model {model} trả về vector không hợp lệ"
            )
        try:
            return [
                [float(value) for value in vector]
                for vector in raw_embeddings
            ]
        except (TypeError, ValueError) as exc:
            raise OllamaServiceError(
                f"Ollama model {model} trả về vector không hợp lệ"
            ) from exc

    def chat(
        self,
        model: str,
        messages: Sequence[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = self._request(
            "POST",
            "/api/chat",
            json={
                "model": model,
                "messages": list(messages),
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise OllamaServiceError(
                f"Ollama model {model} không trả về nội dung câu trả lời"
            )
        return content.strip()
