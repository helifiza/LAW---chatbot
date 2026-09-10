import unittest
import json

from app.services.generation_service import GenerationService


class FakeGeminiClient:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def generate(self, model, prompt, system_instruction, temperature, max_output_tokens) -> str:
        self.request = {
            "model": model,
            "prompt": prompt,
            "system_instruction": system_instruction,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        return "Câu trả lời local. [luat.pdf, trang 2]"


class GenerationServiceTests(unittest.TestCase):
    def test_builds_grounded_prompt_and_calls_ollama(self) -> None:
        client = FakeGeminiClient()
        service = GenerationService(client, "qwen3:4b", 0.2, 1000)

        result = service.generate(
            "Điều kiện là gì?",
            "[Đoạn 1 | luat.pdf | trang 2]\nNội dung điều kiện.",
            [("user", "Câu hỏi trước"), ("assistant", "Câu trả lời trước")],
        )

        self.assertEqual(result, "Câu trả lời local. [luat.pdf, trang 2]")
        assert client.request is not None
        self.assertEqual(client.request["model"], "qwen3:4b")
        messages = [
            {"content": client.request["system_instruction"]},
            {"content": client.request["prompt"]},
        ]
        self.assertIn("Chỉ trả lời", messages[0]["content"])
        self.assertIn("NGỮ CẢNH", messages[1]["content"])
        self.assertIn("Nội dung điều kiện", messages[1]["content"])
        self.assertIn("Câu hỏi trước", messages[1]["content"])

    def test_parses_structured_claim_contract(self) -> None:
        raw = json.dumps({
            "answer": "Doanh nghiệp phải đăng ký. [luat.pdf, trang 2]",
            "claims": [{"claim_id":"C1","text":"Doanh nghiệp phải đăng ký.",
                "claim_type":"legal_conclusion","importance":2,
                "citations":["[luat.pdf, trang 2]"]}],
        }, ensure_ascii=False)

        parsed = GenerationService.parse_grounded_answer(raw)

        self.assertTrue(parsed.structured)
        self.assertEqual(parsed.claims[0].claim_type, "legal_conclusion")
        self.assertEqual(parsed.claims[0].citations, ("[luat.pdf, trang 2]",))

    def test_fallback_attaches_standalone_citation_to_previous_claim(self) -> None:
        parsed = GenerationService.parse_grounded_answer(
            "Doanh nghiệp phải đăng ký. [luat.pdf, trang 2]"
        )

        self.assertFalse(parsed.structured)
        self.assertEqual(len(parsed.claims), 1)
        self.assertEqual(parsed.claims[0].citations, ("[luat.pdf, trang 2]",))


if __name__ == "__main__":
    unittest.main()
