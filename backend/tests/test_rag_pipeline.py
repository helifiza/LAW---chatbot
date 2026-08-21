import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.repositories.history_repository import HistoryRepository
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import LegalChunkingService
from app.services.document_parser import DocumentParser
from app.services.indexing_service import IndexingService
from app.services.history_service import HistoryService
from app.services.query_router_service import QueryRouterService
from app.services.rag_service import (
    RagService,
    ANTI_HALLUCINATION_ANSWER,
    NOT_FOUND_ANSWER,
)



class FakeEmbeddingService:
    def embed_texts(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, _question):
        return [1.0, 0.0, 0.0]


class FakeGenerationService:
    """Trả lời cố định, có citation hợp lệ trỏ đúng vào nguồn đã index."""

    model = "fake-generation-model"

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def generate(self, question, context, history):
        return self._answer


class FakeQueryRewriteService:
    """Không rewrite, không dùng HyDE — trả nguyên câu hỏi làm retrieval query."""

    model = "fake-rewrite-model"

    def rewrite(self, question, history):
        return SimpleNamespace(retrieval_query=question, used_hyde=False, warning=None)


class FakeRetrievalService:
    """Bọc VectorRepository thật + FakeEmbeddingService để retrieval đi qua
    đúng dữ liệu đã index thật, không cần HybridRetrievalService/reranker
    (mô hình nặng, không phù hợp cho unit/integration test nhanh)."""

    def __init__(self, vector_repository: VectorRepository, embedding_service: FakeEmbeddingService) -> None:
        self._vector_repository = vector_repository
        self._embedding_service = embedding_service

    def retrieve(self, history_id, original_query, dense_query, top_k):
        query_vec = self._embedding_service.embed_query(dense_query)
        results = self._vector_repository.query(history_id, query_vec, top_k)
        return SimpleNamespace(results=results, trace={"warnings": []})


class FakeSummarizeService:
    """Cho phép ép kết quả run() để test riêng nhánh SUMMARY của RagService."""

    def __init__(self, answer: str, sources=(), scope_document_ids=()) -> None:
        self._answer = answer
        self._sources = tuple(sources)
        self._scope_document_ids = tuple(scope_document_ids)

    def run(self, history_id, question):
        return SimpleNamespace(
            answer=self._answer,
            sources=self._sources,
            scope_document_ids=self._scope_document_ids,
        )


def _make_highlight(file_name: str, text: str, page: int = 1):
    return SimpleNamespace(
        document_id="doc-1",
        file_name=file_name,
        page_number=page,
        page_end_number=page,
        dieu=None,
        chuong=None,
        text=text,
    )


class RagServiceTestBase(unittest.TestCase):
    """Setup chung: index 1 document thật vào 1 history thật."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        logger = logging.getLogger("test.rag_service")

        self.histories = HistoryRepository(root / "state.db")
        self.vectors = VectorRepository(root / "chroma", "rag_service_test", "fake-model")
        self.history_service = HistoryService(self.histories, self.vectors, logger)
        self.embeddings = FakeEmbeddingService()
        self.indexing = IndexingService(
            self.history_service,
            self.histories,
            self.vectors,
            DocumentParser(),
            LegalChunkingService(500, 50, logger),
            self.embeddings,
            logger,
        )

        self.history = self.history_service.create()
        upload = root / "upload.txt"
        upload.write_text(
            "Điều 1: Nghĩa vụ thanh toán\nBên A phải thanh toán đúng hạn.",
            encoding="utf-8",
        )
        self.document = self.indexing.index_file(
            self.history.id,
            "test-user",           # user_id
            upload,                 # temp_path
            "hop_dong.txt",          # original_file_name
            "text/plain",            # mime_type
            upload.stat().st_size,   # size_bytes
        )
        self.assertEqual(self.document.status, "ready")

    def _build_rag_service(
        self,
        generation_answer: str = "",
        summarize_service: FakeSummarizeService | None = None,
    ) -> RagService:
        return RagService(
            history_service=self.history_service,
            history_repository=self.histories,
            query_rewrite_service=FakeQueryRewriteService(),
            retrieval_service=FakeRetrievalService(self.vectors, self.embeddings),
            generation_service=FakeGenerationService(generation_answer),
            embedding_client=self.embeddings,
            history_limit=8,
            query_router_service=QueryRouterService(gemini_client=None, model="fake-router-model"),
            vector_repository=self.vectors,
            summarize_service=summarize_service or FakeSummarizeService(answer=""),
        )


class SpecificRouteTests(RagServiceTestBase):
    def test_index_retrieve_generate_and_persist_history(self) -> None:
        rag = self._build_rag_service(
            generation_answer="Bên A phải thanh toán đúng hạn. [hop_dong.txt, trang 1]"
        )

        answer = rag.ask(self.history.id, "Nghĩa vụ là gì?", 5)

        self.assertEqual(answer.answer, "Bên A phải thanh toán đúng hạn. [hop_dong.txt, trang 1]")
        self.assertEqual(answer.sources[0].document_id, self.document.id)
        self.assertEqual(
            [item.role for item in self.histories.list_messages(self.history.id)],
            ["user", "assistant"],
        )

    def test_generation_without_valid_citation_falls_back_to_anti_hallucination(self) -> None:
        rag = self._build_rag_service(
            generation_answer="Bên A phải thanh toán đúng hạn."  # không có citation nào
        )

        answer = rag.ask(self.history.id, "Nghĩa vụ là gì?", 5)

        self.assertEqual(answer.answer, ANTI_HALLUCINATION_ANSWER)

    def test_no_retrieval_results_returns_not_found_without_calling_generation(self) -> None:
        rag = self._build_rag_service(generation_answer="không nên xuất hiện")

        # Câu hỏi không liên quan gì tới nội dung đã index -> tuỳ retrieval
        # thật có thể vẫn trả kết quả (vì FakeEmbeddingService luôn trả cùng
        # 1 vector). Ở đây ta test trực tiếp nhánh rỗng bằng history khác
        # chưa index tài liệu nào -> ask() raise NoReadyDocumentError, không
        # phải NOT_FOUND_ANSWER; nên kiểm tra NOT_FOUND_ANSWER qua unit nhỏ
        # hơn ở _ask_specific với retrieval_service giả trả rỗng.
        class EmptyRetrievalService:
            def retrieve(self, **kwargs):
                return SimpleNamespace(results=[], trace={"warnings": []})

        rag._retrieval_service_override_for_test = None  # no-op, giữ rõ ý định
        rag.retrieval_service = EmptyRetrievalService()

        answer = rag.ask(self.history.id, "Câu hỏi bất kỳ", 5)

        self.assertEqual(answer.answer, NOT_FOUND_ANSWER)


class SummaryRouteBugFixTests(RagServiceTestBase):
    """Test trực tiếp bug đã fix: sources rỗng ở nhánh SUMMARY không được
    ghi đè answer hợp lệ ("không tìm thấy nội dung liên quan") thành
    ANTI_HALLUCINATION_ANSWER."""

    NOT_FOUND_SUMMARY_ANSWER = (
        "Không tìm thấy nội dung liên quan trong các tài liệu của phiên "
        "để tổng hợp câu trả lời này."
    )

    def test_empty_sources_keeps_original_not_found_answer(self) -> None:
        rag = self._build_rag_service(
            summarize_service=FakeSummarizeService(
                answer=self.NOT_FOUND_SUMMARY_ANSWER,
                sources=(),
                scope_document_ids=(self.document.id,),
            )
        )

        answer = rag.ask(self.history.id, "tóm tắt tài liệu này", 5)

        # Trước khi fix: answer này sẽ bị ghi đè nhầm thành ANTI_HALLUCINATION_ANSWER
        # vì _validate_citations trả total citation = 0 -> has_valid_citation = False.
        self.assertEqual(answer.answer, self.NOT_FOUND_SUMMARY_ANSWER)
        self.assertNotEqual(answer.answer, ANTI_HALLUCINATION_ANSWER)
        self.assertEqual(answer.sources, ())

    def test_empty_sources_trace_reports_zero_citations_without_override(self) -> None:
        rag = self._build_rag_service(
            summarize_service=FakeSummarizeService(
                answer=self.NOT_FOUND_SUMMARY_ANSWER,
                sources=(),
            )
        )

        answer = rag.ask(self.history.id, "tổng hợp nội dung", 5, include_trace=True)

        citation_validation = answer.trace["citation_validation"]
        self.assertEqual(citation_validation["citation_count"], 0)
        self.assertFalse(citation_validation["has_valid_citation"])
        self.assertEqual(answer.answer, self.NOT_FOUND_SUMMARY_ANSWER)

    def test_non_empty_sources_with_valid_citation_keeps_answer(self) -> None:
        rag = self._build_rag_service(
            summarize_service=FakeSummarizeService(
                answer="Bên A phải thanh toán đúng hạn. [hop_dong.txt, trang 1]",
                sources=[_make_highlight("hop_dong.txt", "Bên A phải thanh toán đúng hạn.")],
            )
        )

        answer = rag.ask(self.history.id, "khái quát nội dung hợp đồng", 5)

        self.assertEqual(
            answer.answer,
            "Bên A phải thanh toán đúng hạn. [hop_dong.txt, trang 1]",
        )

    def test_non_empty_sources_with_invalid_citation_falls_back(self) -> None:
        # Regression guard: fix không được làm lỏng validation khi sources
        # có thật nhưng citation trong answer trỏ sai file.
        rag = self._build_rag_service(
            summarize_service=FakeSummarizeService(
                answer="Nội dung không rõ nguồn. [file_khong_ton_tai.txt, trang 99]",
                sources=[_make_highlight("hop_dong.txt", "Bên A phải thanh toán đúng hạn.")],
            )
        )

        answer = rag.ask(self.history.id, "tóm tắt hợp đồng", 5)

        self.assertEqual(answer.answer, ANTI_HALLUCINATION_ANSWER)


if __name__ == "__main__":
    unittest.main()