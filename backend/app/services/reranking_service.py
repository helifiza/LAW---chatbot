from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

from app.core.errors import RerankerServiceError
from app.domain.models import SearchResult


class CrossEncoderReranker:
    """Lazy-load BGE CrossEncoder để backend không tải model lúc khởi động."""

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        max_length: int,
        model_loader: Callable[[], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self._model_loader = model_loader
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                if self._model_loader is not None:
                    self._model = self._model_loader()
                else:
                    from sentence_transformers import CrossEncoder

                    kwargs: dict[str, object] = {
                        "max_length": self.max_length,
                    }
                    if self.device.lower() != "auto":
                        kwargs["device"] = self.device
                    self._model = CrossEncoder(self.model_name, **kwargs)
            except Exception as exc:
                raise RerankerServiceError(
                    f"Không tải được reranker {self.model_name}: {exc}"
                ) from exc
        return self._model

    @staticmethod
    def _to_float_list(raw_scores: Any) -> list[float]:
        values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
        if not isinstance(values, (list, tuple)):
            values = [values]
        flattened: list[float] = []
        for value in values:
            if isinstance(value, (list, tuple)):
                if len(value) != 1:
                    raise ValueError("Reranker trả về score nhiều chiều")
                value = value[0]
            flattened.append(float(value))
        return flattened

    def rerank(
        self,
        question: str,
        candidates: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        model = self._load_model()
        pairs = [[question, item.chunk.text] for item in candidates]
        try:
            with self._predict_lock:
                raw_scores = model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
            scores = self._to_float_list(raw_scores)
        except RerankerServiceError:
            raise
        except Exception as exc:
            raise RerankerServiceError(
                f"Reranker {self.model_name} không chấm điểm được: {exc}"
            ) from exc

        if len(scores) != len(candidates):
            raise RerankerServiceError(
                "Số điểm rerank không khớp số candidate"
            )
        ranked = [
            SearchResult(chunk=item.chunk, score=score)
            for item, score in zip(candidates, scores)
        ]
        return sorted(
            ranked, key=lambda item: item.score, reverse=True
        )[:top_k]
