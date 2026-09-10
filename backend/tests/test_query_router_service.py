import json
import unittest

from app.services.query_router_service import QueryIntent, QueryRouterService


class FakeGeminiClient:
    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.request = None

    def generate(self, **kwargs):
        self.request = kwargs
        if self.error is not None:
            raise self.error
        return json.dumps(self.response)


class QueryRouterServiceTests(unittest.TestCase):
    def test_uses_gemini_contract_and_extracts_document_slot(self) -> None:
        client = FakeGeminiClient({
            "intent": "document_summary",
            "scope": "single_document",
            "coverage": "multi_aspect",
            "target_documents": ["Nghị định 123"],
            "loai_van_ban_filter": ["nghị định"],
            "linh_vuc": [],
            "comparison_aspects": ["đối tượng áp dụng", "mức xử phạt"],
            "as_of_date": "2026-09-08",
        })
        result = QueryRouterService(client, "gemini-test").route(
            "Tóm tắt Nghị định 123"
        )

        self.assertEqual(result.intent, QueryIntent.DOCUMENT_SUMMARY)
        self.assertEqual(result.target_documents, ["Nghị định 123"])
        self.assertEqual(result.comparison_aspects, ["đối tượng áp dụng", "mức xử phạt"])
        self.assertEqual(result.as_of_date, "2026-09-08")
        self.assertEqual(client.request["prompt"], "Tóm tắt Nghị định 123")
        self.assertIn("system_instruction", client.request)
        self.assertNotIn("system_prompt", client.request)

    def test_keyword_route_is_used_only_when_gemini_fails(self) -> None:
        client = FakeGeminiClient(error=RuntimeError("offline"))
        result = QueryRouterService(client, "gemini-test").route("Tóm tắt tài liệu")

        self.assertEqual(result.intent, QueryIntent.DOCUMENT_SUMMARY)
        self.assertEqual(result.source, "keyword")


if __name__ == "__main__":
    unittest.main()
