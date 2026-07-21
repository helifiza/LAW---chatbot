import unittest

from app.services.generation_service import GenerationService


class FakeOllamaClient:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def chat(self, model, messages, temperature, max_tokens) -> str:
        self.request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        return "Câu trả lời local. [luat.pdf, trang 2]"


class GenerationServiceTests(unittest.TestCase):
    def test_builds_grounded_prompt_and_calls_ollama(self) -> None:
        client = FakeOllamaClient()
        service = GenerationService(client, "qwen3:4b", 0.2, 1000)

        result = service.generate(
            "Điều kiện là gì?",
            "[Đoạn 1 | luat.pdf | trang 2]\nNội dung điều kiện.",
            [("user", "Câu hỏi trước"), ("assistant", "Câu trả lời trước")],
        )

        self.assertEqual(result, "Câu trả lời local. [luat.pdf, trang 2]")
        assert client.request is not None
        self.assertEqual(client.request["model"], "qwen3:4b")
        messages = client.request["messages"]
        self.assertIn("Chỉ trả lời", messages[0]["content"])
        self.assertIn("NGỮ CẢNH", messages[1]["content"])
        self.assertIn("Nội dung điều kiện", messages[1]["content"])
        self.assertIn("Câu hỏi trước", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
