import json
import unittest

import httpx

from app.clients.ollama_client import OllamaClient
from app.core.errors import OllamaServiceError


class OllamaClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests: list[tuple[str, dict[str, object] | None]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content) if request.content else None
            self.requests.append((request.url.path, payload))
            if request.url.path == "/api/tags":
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {"name": "bge-m3:latest"},
                            {"name": "qwen3:4b"},
                        ]
                    },
                )
            if request.url.path == "/api/embed":
                inputs = payload["input"]
                return httpx.Response(
                    200,
                    json={"embeddings": [[float(len(text))] for text in inputs]},
                )
            if request.url.path == "/api/chat":
                return httpx.Response(
                    200,
                    json={"message": {"role": "assistant", "content": "  OK  "}},
                )
            return httpx.Response(404, json={"error": "not found"})

        self.client = OllamaClient(
            "http://127.0.0.1:11434/v1/",
            timeout_seconds=30,
            retry_count=1,
            retry_delay_seconds=0,
            transport=httpx.MockTransport(handler),
        )

    def tearDown(self) -> None:
        self.client.close()

    def test_normalizes_old_v1_url_and_checks_both_models(self) -> None:
        available, models = self.client.status(["bge-m3", "qwen3:4b"])

        self.assertEqual(self.client.base_url, "http://127.0.0.1:11434")
        self.assertTrue(available)
        self.assertEqual(models, {"bge-m3": True, "qwen3:4b": True})

    def test_uses_native_embed_and_chat_endpoints(self) -> None:
        vectors = self.client.embed("bge-m3", ["a", "abcd"])
        answer = self.client.chat(
            "qwen3:4b",
            [{"role": "user", "content": "test"}],
            temperature=0.2,
            max_tokens=100,
        )

        self.assertEqual(vectors, [[1.0], [4.0]])
        self.assertEqual(answer, "OK")
        self.assertEqual(
            [path for path, _ in self.requests],
            ["/api/embed", "/api/chat"],
        )
        chat_payload = self.requests[1][1]
        self.assertEqual(chat_payload["model"], "qwen3:4b")
        self.assertFalse(chat_payload["stream"])
        self.assertFalse(chat_payload["think"])
        self.assertEqual(chat_payload["options"]["num_predict"], 100)

    def test_surfaces_ollama_error_detail(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                request=request,
                json={"error": "model 'missing' not found"},
            )

        client = OllamaClient(
            "http://127.0.0.1:11434",
            30,
            1,
            0,
            transport=httpx.MockTransport(handler),
        )
        try:
            with self.assertRaisesRegex(OllamaServiceError, "model 'missing' not found"):
                client.chat("missing", [], 0.2, 10)
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
