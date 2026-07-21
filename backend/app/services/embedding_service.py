from __future__ import annotations

from typing import Sequence

from app.clients.ollama_client import OllamaClient


class EmbeddingService:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        batch_size: int,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = max(1, batch_size)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            embeddings.extend(
                self.client.embed(
                    self.model,
                    texts[start : start + self.batch_size],
                )
            )
        if len(embeddings) != len(texts):
            raise RuntimeError("Ollama trả về thiếu embedding")
        return embeddings

    def embed_query(self, question: str) -> list[float]:
        return self.client.embed(self.model, [question])[0]
