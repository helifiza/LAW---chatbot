from __future__ import annotations

from typing import Sequence

from app.clients.gemini_client import GeminiClient


class EmbeddingService:
    def __init__(
        self,
        client: GeminiClient,
        model: str,
        batch_size: int,
        output_dimension: int,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = max(1, batch_size)
        self.output_dimension = output_dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            embeddings.extend(
                self.client.embed(
                    self.model,
                    texts[start : start + self.batch_size],
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimension=self.output_dimension,
                )
            )
        if len(embeddings) != len(texts):
            raise RuntimeError("Gemini trả về thiếu embedding")
        return embeddings

    def embed_query(self, question: str) -> list[float]:
        return self.client.embed(
            self.model,
            [question],
            task_type="RETRIEVAL_QUERY",
            output_dimension=self.output_dimension,
        )[0]
