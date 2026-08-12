import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from google.genai import errors as genai_errors

from app.clients.gemini_client import GeminiClient
from app.core.errors import GeminiServiceError


def make_api_error(code: int, message: str) -> genai_errors.APIError:
    """Tạo APIError giả mà không cần gọi __init__ thật của SDK (constructor
    của google-genai yêu cầu response_json/httpx.Response nội bộ khá phức
    tạp). GeminiClient chỉ đọc `code` và `message` qua getattr nên chỉ cần
    gán 2 thuộc tính này là đủ để mô phỏng chính xác hành vi thật.
    """
    exc = genai_errors.APIError.__new__(genai_errors.APIError)
    exc.code = code
    exc.message = message
    return exc


class GeminiClientConfigurationTests(unittest.TestCase):
    def test_is_configured_false_without_api_key_and_sdk_client(self) -> None:
        client = GeminiClient(api_key="", retry_count=3, retry_delay_seconds=0)

        self.assertFalse(client.is_configured)

    def test_is_configured_true_with_injected_sdk_client(self) -> None:
        sdk_client = MagicMock()
        client = GeminiClient(
            api_key="",
            retry_count=3,
            retry_delay_seconds=0,
            sdk_client=sdk_client,
        )

        self.assertTrue(client.is_configured)

    def test_creates_default_sdk_client_when_api_key_provided(self) -> None:
        with patch("app.clients.gemini_client.genai.Client") as client_ctor:
            client_ctor.return_value = MagicMock()
            client = GeminiClient(
                api_key=" my-key ",
                retry_count=1,
                retry_delay_seconds=0,
            )

        self.assertEqual(client.api_key, "my-key")
        client_ctor.assert_called_once_with(api_key="my-key")
        self.assertTrue(client.is_configured)

    def test_retry_count_and_delay_are_clamped_to_non_negative(self) -> None:
        client = GeminiClient(
            api_key="",
            retry_count=0,
            retry_delay_seconds=-5,
            sdk_client=MagicMock(),
        )

        self.assertEqual(client.retry_count, 1)
        self.assertEqual(client.retry_delay_seconds, 0.0)

    def test_close_calls_underlying_client_close(self) -> None:
        sdk_client = MagicMock()
        client = GeminiClient(
            api_key="",
            retry_count=1,
            retry_delay_seconds=0,
            sdk_client=sdk_client,
        )

        client.close()

        sdk_client.close.assert_called_once()

    def test_close_is_noop_when_not_configured(self) -> None:
        client = GeminiClient(api_key="", retry_count=1, retry_delay_seconds=0)

        # Không được raise dù chưa cấu hình client.
        client.close()


class GeminiClientEmbedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sdk_client = MagicMock()
        self.client = GeminiClient(
            api_key="",
            retry_count=1,
            retry_delay_seconds=0,
            sdk_client=self.sdk_client,
        )

    def test_embed_empty_texts_skips_api_call(self) -> None:
        vectors = self.client.embed(
            "gemini-embedding-001",
            [],
            task_type="RETRIEVAL_DOCUMENT",
            output_dimension=768,
        )

        self.assertEqual(vectors, [])
        self.sdk_client.models.embed_content.assert_not_called()

    def test_embed_returns_normalized_vectors_for_reduced_dimension(self) -> None:
        raw = [3.0, 4.0]  # norm = 5 -> normalized: [0.6, 0.8]
        self.sdk_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=raw)]
        )

        vectors = self.client.embed(
            "gemini-embedding-001",
            ["xin chào"],
            task_type="RETRIEVAL_DOCUMENT",
            output_dimension=2,
        )

        self.assertEqual(len(vectors), 1)
        self.assertAlmostEqual(vectors[0][0], 0.6)
        self.assertAlmostEqual(vectors[0][1], 0.8)

        _, kwargs = self.sdk_client.models.embed_content.call_args
        self.assertEqual(kwargs["model"], "gemini-embedding-001")
        self.assertEqual(kwargs["contents"], ["xin chào"])
        self.assertEqual(kwargs["config"].task_type, "RETRIEVAL_DOCUMENT")
        self.assertEqual(kwargs["config"].output_dimensionality, 2)

    def test_embed_no_normalization_when_dimension_is_3072(self) -> None:
        raw = [3.0, 4.0, 0.0] + [0.0] * 3069  # đủ 3072 phần tử
        self.sdk_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=raw)]
        )

        vectors = self.client.embed(
            "gemini-embedding-001",
            ["a"],
            task_type="RETRIEVAL_QUERY",
            output_dimension=3072,
        )

        # Không chuẩn hóa -> giữ nguyên giá trị gốc.
        self.assertEqual(vectors[0][0], 3.0)
        self.assertEqual(vectors[0][1], 4.0)

    def test_embed_raises_when_embedding_count_mismatch(self) -> None:
        self.sdk_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2])]
        )

        with self.assertRaisesRegex(GeminiServiceError, "2 văn bản"):
            self.client.embed(
                "gemini-embedding-001",
                ["a", "b"],
                task_type="RETRIEVAL_DOCUMENT",
                output_dimension=2,
            )

    def test_embed_raises_when_vector_dimension_mismatch(self) -> None:
        self.sdk_client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
        )

        with self.assertRaisesRegex(GeminiServiceError, "sai số chiều"):
            self.client.embed(
                "gemini-embedding-001",
                ["a"],
                task_type="RETRIEVAL_DOCUMENT",
                output_dimension=2,
            )

    def test_embed_without_configured_client_raises(self) -> None:
        client = GeminiClient(api_key="", retry_count=1, retry_delay_seconds=0)

        with self.assertRaisesRegex(GeminiServiceError, "GEMINI_API_KEY"):
            client.embed(
                "gemini-embedding-001",
                ["a"],
                task_type="RETRIEVAL_DOCUMENT",
                output_dimension=768,
            )


class GeminiClientGenerateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sdk_client = MagicMock()
        self.client = GeminiClient(
            api_key="",
            retry_count=1,
            retry_delay_seconds=0,
            sdk_client=self.sdk_client,
        )

    def test_generate_returns_stripped_text(self) -> None:
        self.sdk_client.models.generate_content.return_value = SimpleNamespace(
            text="  Xin chào, đây là câu trả lời.  "
        )

        answer = self.client.generate(
            "gemini-3.5-flash-lite",
            "Hỏi gì đó",
            system_instruction="Bạn là trợ lý pháp lý.",
            temperature=0.2,
            max_output_tokens=1000,
        )

        self.assertEqual(answer, "Xin chào, đây là câu trả lời.")

        _, kwargs = self.sdk_client.models.generate_content.call_args
        self.assertEqual(kwargs["model"], "gemini-3.5-flash-lite")
        self.assertEqual(kwargs["contents"], "Hỏi gì đó")
        self.assertEqual(kwargs["config"].system_instruction, "Bạn là trợ lý pháp lý.")
        self.assertEqual(kwargs["config"].temperature, 0.2)
        self.assertEqual(kwargs["config"].max_output_tokens, 1000)

    def test_generate_raises_when_response_text_missing(self) -> None:
        self.sdk_client.models.generate_content.return_value = SimpleNamespace(text=None)

        with self.assertRaisesRegex(GeminiServiceError, "không trả về nội dung"):
            self.client.generate(
                "gemini-3.5-flash-lite",
                "Hỏi gì đó",
                system_instruction="",
                temperature=0.2,
                max_output_tokens=1000,
            )

    def test_generate_raises_when_response_text_blank(self) -> None:
        self.sdk_client.models.generate_content.return_value = SimpleNamespace(text="   ")

        with self.assertRaisesRegex(GeminiServiceError, "không trả về nội dung"):
            self.client.generate(
                "gemini-3.5-flash-lite",
                "Hỏi gì đó",
                system_instruction="",
                temperature=0.2,
                max_output_tokens=1000,
            )


class GeminiClientRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sdk_client = MagicMock()

    def test_retries_on_transient_status_then_succeeds(self) -> None:
        client = GeminiClient(
            api_key="",
            retry_count=3,
            retry_delay_seconds=1,
            sdk_client=self.sdk_client,
        )
        self.sdk_client.models.generate_content.side_effect = [
            make_api_error(503, "service unavailable"),
            SimpleNamespace(text="OK sau khi retry"),
        ]

        with patch("app.clients.gemini_client.time.sleep") as sleep_mock:
            answer = client.generate(
                "gemini-3.5-flash-lite",
                "prompt",
                system_instruction="",
                temperature=0.2,
                max_output_tokens=100,
            )

        self.assertEqual(answer, "OK sau khi retry")
        self.assertEqual(self.sdk_client.models.generate_content.call_count, 2)
        sleep_mock.assert_called_once_with(1)

    def test_uses_exponential_backoff_between_retries(self) -> None:
        client = GeminiClient(
            api_key="",
            retry_count=3,
            retry_delay_seconds=1,
            sdk_client=self.sdk_client,
        )
        self.sdk_client.models.generate_content.side_effect = [
            make_api_error(429, "rate limited"),
            make_api_error(500, "server error"),
            SimpleNamespace(text="OK cuối cùng"),
        ]

        with patch("app.clients.gemini_client.time.sleep") as sleep_mock:
            answer = client.generate(
                "gemini-3.5-flash-lite",
                "prompt",
                system_instruction="",
                temperature=0.2,
                max_output_tokens=100,
            )

        self.assertEqual(answer, "OK cuối cùng")
        sleep_mock.assert_has_calls([call(1), call(2)])

    def test_non_transient_status_raises_immediately_without_retry(self) -> None:
        client = GeminiClient(
            api_key="",
            retry_count=5,
            retry_delay_seconds=0,
            sdk_client=self.sdk_client,
        )
        self.sdk_client.models.generate_content.side_effect = make_api_error(
            400, "invalid argument"
        )

        with patch("app.clients.gemini_client.time.sleep") as sleep_mock:
            with self.assertRaisesRegex(GeminiServiceError, "invalid argument"):
                client.generate(
                    "gemini-3.5-flash-lite",
                    "prompt",
                    system_instruction="",
                    temperature=0.2,
                    max_output_tokens=100,
                )

        self.assertEqual(self.sdk_client.models.generate_content.call_count, 1)
        sleep_mock.assert_not_called()

    def test_raises_after_exhausting_all_retries(self) -> None:
        client = GeminiClient(
            api_key="",
            retry_count=2,
            retry_delay_seconds=0,
            sdk_client=self.sdk_client,
        )
        self.sdk_client.models.generate_content.side_effect = make_api_error(
            503, "vẫn quá tải"
        )

        with patch("app.clients.gemini_client.time.sleep"):
            with self.assertRaisesRegex(GeminiServiceError, "vẫn quá tải"):
                client.generate(
                    "gemini-3.5-flash-lite",
                    "prompt",
                    system_instruction="",
                    temperature=0.2,
                    max_output_tokens=100,
                )

        self.assertEqual(self.sdk_client.models.generate_content.call_count, 2)

    def test_unexpected_exception_is_wrapped_in_gemini_service_error(self) -> None:
        client = GeminiClient(
            api_key="",
            retry_count=1,
            retry_delay_seconds=0,
            sdk_client=self.sdk_client,
        )
        self.sdk_client.models.generate_content.side_effect = ValueError("bug lạ")

        with self.assertRaisesRegex(GeminiServiceError, "lỗi không hợp lệ"):
            client.generate(
                "gemini-3.5-flash-lite",
                "prompt",
                system_instruction="",
                temperature=0.2,
                max_output_tokens=100,
            )

    def test_gemini_service_error_propagates_without_rewrapping(self) -> None:
        client = GeminiClient(
            api_key="",
            retry_count=3,
            retry_delay_seconds=0,
            sdk_client=self.sdk_client,
        )
        original = GeminiServiceError("lỗi gốc")
        self.sdk_client.models.generate_content.side_effect = original

        with self.assertRaises(GeminiServiceError) as ctx:
            client.generate(
                "gemini-3.5-flash-lite",
                "prompt",
                system_instruction="",
                temperature=0.2,
                max_output_tokens=100,
            )

        self.assertIs(ctx.exception, original)
        self.assertEqual(self.sdk_client.models.generate_content.call_count, 1)


if __name__ == "__main__":
    unittest.main()