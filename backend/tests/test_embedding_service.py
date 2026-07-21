import unittest

from app.services.embedding_service import EmbeddingService


class FakeOllamaClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def embed(self, model: str, texts) -> list[list[float]]:
        batch = list(texts)
        self.calls.append((model, batch))
        return [[float(len(text)), float(index)] for index, text in enumerate(batch)]


class EmbeddingServiceTests(unittest.TestCase):
    def test_batches_and_preserves_input_order(self) -> None:
        client = FakeOllamaClient()
        service = EmbeddingService(client, "bge-m3", batch_size=2)

        result = service.embed_texts(["a", "bbbb", "cc"])

        self.assertEqual(
            client.calls,
            [("bge-m3", ["a", "bbbb"]), ("bge-m3", ["cc"])],
        )
        self.assertEqual(result, [[1.0, 0.0], [4.0, 1.0], [2.0, 0.0]])

    def test_query_uses_the_same_embedding_model(self) -> None:
        client = FakeOllamaClient()
        service = EmbeddingService(client, "bge-m3", batch_size=16)

        self.assertEqual(service.embed_query("Điều kiện là gì?"), [16.0, 0.0])
        self.assertEqual(client.calls[0][0], "bge-m3")


if __name__ == "__main__":
    unittest.main()
