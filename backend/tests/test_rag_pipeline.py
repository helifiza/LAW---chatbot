import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.domain.legal_document import matches_legal_reference
from app.domain.models import MessageRole
from app.repositories.history_repository import HistoryRepository
from app.repositories.user_repository import UserRepository
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import LegalChunkingService
from app.services.document_parser import DocumentParser
from app.services.indexing_service import IndexingService
from app.services.history_service import HistoryService
from app.services.query_router_service import QueryRouterService, QueryScope
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


class FakeDocumentParser:
    def parse(self, file_path, original_file_name):
        return [(1, file_path.read_text(encoding="utf-8"))]


class FakeGenerationService:
    """Trả lời cố định, có citation hợp lệ trỏ đúng vào nguồn đã index."""

    model = "fake-generation-model"

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def generate(self, question, context, history):
        return self._answer

    def generate_comparison(self, question, context_by_document, history):
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

    def retrieve(
        self, history_id, original_query, dense_query, top_k,
        document_ids=None, linh_vuc_filter=None, document_type_filter=None,
    ):
        query_vec = self._embedding_service.embed_query(dense_query)
        results = self._vector_repository.query(
            history_id,
            query_vec,
            top_k,
            document_ids=document_ids,
            linh_vuc_filter=linh_vuc_filter,
            document_type_filter=document_type_filter,
        )
        return SimpleNamespace(results=results, trace={"warnings": []})


class FakeSummarizeService:
    """Cho phép ép kết quả run() để test riêng nhánh SUMMARY của RagService."""

    def __init__(self, answer: str, sources=(), scope_document_ids=()) -> None:
        self._answer = answer
        self._sources = tuple(sources)
        self._scope_document_ids = tuple(scope_document_ids)

    def run(self, history_id, question, target_documents=None):
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
        user = UserRepository(root / "state.db").create_user(
            "Test", "test@example.com", "hash"
        )
        self.vectors = VectorRepository(root / "chroma", "rag_service_test", "fake-model")
        self.addCleanup(self.vectors.close)
        self.history_service = HistoryService(self.histories, self.vectors, logger)
        self.embeddings = FakeEmbeddingService()
        self.indexing = IndexingService(
            self.history_service,
            self.histories,
            self.vectors,
            FakeDocumentParser(),
            LegalChunkingService(500, 50, logger),
            self.embeddings,
            logger,
        )

        self.history = self.history_service.create(user["id"], "Test")
        upload = root / "upload.pdf"
        upload.write_text(
            "Điều 1: Nghĩa vụ thanh toán\nBên A phải thanh toán đúng hạn.",
            encoding="utf-8",
        )
        self.document = self.indexing.index_file(
            self.history.id,
            "test-user",           # user_id
            upload,                 # temp_path
            "hop_dong.txt",          # logical fixture name; parser is replaced above
            "application/pdf",       # mime_type
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


class ScopeResolutionTests(RagServiceTestBase):
    """scope single/selected phải thu hẹp retrieval về văn bản được chỉ định."""

    def test_multi_document_scope_is_unrestricted(self) -> None:
        rag = self._build_rag_service()

        resolution = rag._resolve_scope(
            self.history.id, QueryScope.MULTI_DOCUMENT, ["hop_dong"]
        )

        self.assertIsNone(resolution.document_ids)
        self.assertEqual(resolution.source, "unrestricted")

    def test_single_document_scope_resolves_target_by_file_name(self) -> None:
        rag = self._build_rag_service()

        resolution = rag._resolve_scope(
            self.history.id, QueryScope.SINGLE_DOCUMENT, ["hop_dong"]
        )

        self.assertEqual(resolution.document_ids, (self.document.id,))
        self.assertEqual(resolution.source, "targets")

    def test_unresolved_target_returns_not_found_without_retrieval(self) -> None:
        rag = self._build_rag_service(generation_answer="không nên xuất hiện")
        calls: list[dict] = []

        class RecordingRetrievalService:
            def retrieve(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(results=[], trace={"warnings": []})

        rag.retrieval_service = RecordingRetrievalService()
        resolution = rag._resolve_scope(
            self.history.id, QueryScope.SINGLE_DOCUMENT, ["Nghị định 99/2099"]
        )

        answer, sources, count, trace = rag._ask_specific(
            self.history.id, "Điều 1 quy định gì?", 5, scope_resolution=resolution
        )

        self.assertTrue(resolution.is_unresolved)
        self.assertIn("Nghị định 99/2099", answer)
        self.assertEqual((sources, count), ((), 0))
        self.assertEqual(calls, [])
        self.assertEqual(trace["scope_resolution"]["unresolved_targets"], ["Nghị định 99/2099"])

    def test_history_reference_uses_documents_from_last_answer(self) -> None:
        rag = self._build_rag_service()
        self.histories.add_message(
            self.history.id,
            MessageRole.ASSISTANT,
            "câu trả lời trước",
            sources=json.dumps([{"document_id": self.document.id}]),
        )

        resolution = rag._resolve_scope(
            self.history.id, QueryScope.SINGLE_DOCUMENT, [], requires_history_context=True
        )

        self.assertEqual(resolution.document_ids, (self.document.id,))
        self.assertEqual(resolution.source, "history")

    def test_specific_passes_document_ids_to_retrieval(self) -> None:
        rag = self._build_rag_service(
            generation_answer="Bên A phải thanh toán đúng hạn. [hop_dong.txt, trang 1]"
        )
        calls: list[dict] = []
        real_retrieval = rag.retrieval_service

        class RecordingRetrievalService:
            def retrieve(self, **kwargs):
                calls.append(kwargs)
                return real_retrieval.retrieve(**kwargs)

        rag.retrieval_service = RecordingRetrievalService()
        resolution = rag._resolve_scope(
            self.history.id, QueryScope.SINGLE_DOCUMENT, ["hop_dong"]
        )

        rag._ask_specific(self.history.id, "Nghĩa vụ là gì?", 5, scope_resolution=resolution)

        self.assertEqual(calls[0]["document_ids"], (self.document.id,))


class TwoDocumentTestBase(RagServiceTestBase):
    def setUp(self) -> None:
        super().setUp()
        upload = Path(self._tmp.name) / "quy_che.txt"
        upload.write_text(
            "Điều 1: Thời hạn thanh toán\nThanh toán trong 30 ngày.",
            encoding="utf-8",
        )
        self.second_document = self.indexing.index_file(
            self.history.id, "test-user", upload, "quy_che.txt",
            "application/pdf", upload.stat().st_size,
        )
        self.assertEqual(self.second_document.status, "ready")


class CompareScopeTests(TwoDocumentTestBase):
    """Compare không bao giờ tự so sánh toàn phiên khi thiếu văn bản chỉ định."""

    def _recording_rag(self, generation_answer: str = ""):
        rag = self._build_rag_service(generation_answer=generation_answer)
        calls: list[dict] = []
        real_retrieval = rag.retrieval_service

        class RecordingRetrievalService:
            def retrieve(self, **kwargs):
                calls.append(kwargs)
                return real_retrieval.retrieve(**kwargs)

        rag.retrieval_service = RecordingRetrievalService()
        return rag, calls

    def test_without_targets_requires_explicit_documents(self) -> None:
        rag, calls = self._recording_rag()

        answer, sources, _, trace = rag._ask_compare(
            self.history.id, "So sánh các văn bản", None, 5
        )

        self.assertTrue(answer.startswith("Cần xác định ít nhất hai văn bản"))
        self.assertEqual(sources, ())
        self.assertEqual(calls, [])
        self.assertEqual(trace["scope_resolution"]["document_ids"], [])

    def test_reports_unresolved_target(self) -> None:
        rag, calls = self._recording_rag()

        answer, _, _, trace = rag._ask_compare(
            self.history.id, "So sánh", ["hop_dong", "Luật 99/2099"], 5
        )

        self.assertIn("Luật 99/2099", answer)
        self.assertEqual(calls, [])
        self.assertEqual(trace["scope_resolution"]["unresolved_targets"], ["Luật 99/2099"])

    def test_targets_limit_retrieval_to_each_document(self) -> None:
        rag, calls = self._recording_rag(
            "Hợp đồng [hop_dong.txt, trang 1] khác quy chế [quy_che.txt, trang 1]"
        )

        _, _, _, trace = rag._ask_compare(
            self.history.id, "So sánh thanh toán", ["hop_dong", "quy_che"], 5
        )

        self.assertEqual(
            [call["document_ids"] for call in calls],
            [[self.document.id], [self.second_document.id]],
        )
        self.assertEqual(
            trace["scope_resolution"]["document_ids"],
            [self.document.id, self.second_document.id],
        )

    def test_history_reference_uses_documents_from_last_answer(self) -> None:
        rag, calls = self._recording_rag()
        self.histories.add_message(
            self.history.id,
            MessageRole.ASSISTANT,
            "câu trả lời trước",
            sources=json.dumps([
                {"document_id": self.document.id},
                {"document_id": self.second_document.id},
            ]),
        )

        _, _, _, trace = rag._ask_compare(
            self.history.id, "So sánh 2 văn bản trên", None, 5,
            requires_history_context=True,
        )

        self.assertEqual(trace["scope_resolution"]["source"], "history")
        self.assertEqual(len(calls), 2)


class RecordingSummarizeService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, history_id, question, target_documents=None,
            full_enumeration=False, document_type_filter=None, document_ids=None):
        self.calls.append({"target_documents": target_documents, "document_ids": document_ids})
        return SimpleNamespace(
            answer="Không tìm thấy nội dung liên quan.",
            sources=(),
            scope_document_ids=tuple(document_ids or ()),
        )


class SummaryScopeTests(TwoDocumentTestBase):
    """Summary phải được chỉ rõ văn bản, không mặc định tóm tắt toàn phiên."""

    def setUp(self) -> None:
        super().setUp()
        self.summarizer = RecordingSummarizeService()
        self.rag = self._build_rag_service(summarize_service=self.summarizer)

    def test_without_target_requires_explicit_document(self) -> None:
        answer, _, _, trace = self.rag._ask_summary(self.history.id, "Tóm tắt nội dung")

        self.assertIn("nêu rõ", answer)
        self.assertEqual(self.summarizer.calls, [])
        self.assertEqual(trace["scope_resolution"]["source"], "unspecified")

    def test_unresolved_target_returns_not_found(self) -> None:
        answer, _, _, _ = self.rag._ask_summary(
            self.history.id, "Tóm tắt Luật 99/2099", ["Luật 99/2099"]
        )

        self.assertIn("Luật 99/2099", answer)
        self.assertEqual(self.summarizer.calls, [])

    def test_target_passes_resolved_document_ids(self) -> None:
        self.rag._ask_summary(self.history.id, "Tóm tắt quy chế", ["quy_che"])

        self.assertEqual(self.summarizer.calls[0]["document_ids"], (self.second_document.id,))

    def test_file_name_in_question_is_explicit(self) -> None:
        self.rag._ask_summary(self.history.id, "Tóm tắt hop_dong giúp tôi")

        self.assertEqual(self.summarizer.calls[0]["document_ids"], (self.document.id,))

    def test_history_reference_uses_documents_from_last_answer(self) -> None:
        self.histories.add_message(
            self.history.id, MessageRole.ASSISTANT, "trả lời trước",
            sources=json.dumps([{"document_id": self.second_document.id}]),
        )

        self.rag._ask_summary(
            self.history.id, "Tóm tắt văn bản trên", requires_history_context=True
        )

        self.assertEqual(self.summarizer.calls[0]["document_ids"], (self.second_document.id,))


class LegalReferenceMatchTests(unittest.TestCase):
    def test_reference_with_document_type_matches_number(self) -> None:
        self.assertTrue(matches_legal_reference("Nghị định 15/2020/NĐ-CP", "15/2020/NĐ-CP"))

    def test_partial_number_and_unaccented_reference_match(self) -> None:
        self.assertTrue(matches_legal_reference("15/2020", "15/2020/NĐ-CP"))
        self.assertTrue(matches_legal_reference("nghi dinh 15/2020/ND-CP", "15/2020/NĐ-CP"))

    def test_token_boundary_prevents_false_match(self) -> None:
        self.assertFalse(matches_legal_reference("Nghị định 115/2020/NĐ-CP", "15/2020/NĐ-CP"))

    def test_short_number_is_not_matched_in_reverse(self) -> None:
        self.assertFalse(matches_legal_reference("Nghị định 15/2020/NĐ-CP", "15"))


class FakeLegalGraphRepository:
    def __init__(self, documents, relations) -> None:
        self._documents = documents
        self._relations = relations

    def list_documents(self, history_id):
        return self._documents

    def list_relation_details(self, history_id, document_type_filter=None):
        return list(self._relations)


class GraphTestBase(TwoDocumentTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.rag = self._build_rag_service()
        self.rag.legal_graph_repository = FakeLegalGraphRepository(
            documents=[
                {"id": "lg-1", "upload_document_id": self.document.id,
                 "document_number": "15/2020/NĐ-CP", "title": "", "document_type": "Nghị định"},
                {"id": "lg-2", "upload_document_id": self.second_document.id,
                 "document_number": "01/2021/TT-BTC", "title": "", "document_type": "Thông tư"},
            ],
            relations=[{
                "upload_document_id": self.second_document.id,
                "source_file_name": "quy_che.txt",
                "source_document_id": "lg-2", "target_document_id": "lg-1",
                "source_number": "01/2021/TT-BTC", "target_number": "15/2020/NĐ-CP",
                "raw_target_reference": "Nghị định 15/2020/NĐ-CP",
                "relation_type": "HUONG_DAN_THI_HANH",
                "page_number": 1, "page_end_number": 1,
                "confidence": 0.9, "quote": "hướng dẫn thi hành Nghị định 15/2020/NĐ-CP",
            }],
        )


class RelationshipScopeTests(GraphTestBase):
    def test_target_with_document_type_finds_relation(self) -> None:
        answer, sources, _, trace = self.rag._ask_relationship(
            self.history.id, "quan hệ", ["Nghị định 15/2020/NĐ-CP"]
        )

        self.assertEqual(len(sources), 1)
        self.assertIn("hướng dẫn thi hành", answer)
        self.assertEqual(trace["scope_resolution"]["unresolved_targets"], [])

    def test_single_document_scope_without_target_requires_explicit_document(self) -> None:
        answer, sources, _, _ = self.rag._ask_relationship(
            self.history.id, "quan hệ", None, scope=QueryScope.SINGLE_DOCUMENT
        )

        self.assertIn("nêu rõ", answer)
        self.assertEqual(sources, ())

    def test_multi_document_scope_without_target_lists_all(self) -> None:
        _, sources, _, _ = self.rag._ask_relationship(
            self.history.id, "quan hệ", None, scope=QueryScope.MULTI_DOCUMENT
        )

        self.assertEqual(len(sources), 1)

    def test_history_reference_filters_by_either_side(self) -> None:
        self.histories.add_message(
            self.history.id, MessageRole.ASSISTANT, "trả lời trước",
            sources=json.dumps([{"document_id": self.document.id}]),
        )

        _, sources, _, trace = self.rag._ask_relationship(
            self.history.id, "văn bản trên được hướng dẫn bởi gì?", None,
            scope=QueryScope.SINGLE_DOCUMENT, requires_history_context=True,
        )

        self.assertEqual(len(sources), 1)  # văn bản trước là đầu đích của quan hệ
        self.assertEqual(trace["scope_resolution"]["source"], "history")


class RecordingEffectService:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def evaluate(self, history_id, targets, as_of, document_type_filter=None):
        self.calls.append(list(targets))
        return {"answer": "Văn bản còn hiệu lực.", "sources": [],
                "status": "EFFECTIVE", "warnings": []}


class CurrentEffectScopeTests(GraphTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.effect = RecordingEffectService()
        self.rag.legal_effect_service = self.effect

    def test_targets_are_passed_through(self) -> None:
        self.rag._ask_current_effect(self.history.id, ["Nghị định 15/2020/NĐ-CP"], None)

        self.assertEqual(self.effect.calls, [["Nghị định 15/2020/NĐ-CP"]])

    def test_history_reference_resolves_to_document_number(self) -> None:
        self.histories.add_message(
            self.history.id, MessageRole.ASSISTANT, "trả lời trước",
            sources=json.dumps([{"document_id": self.document.id}]),
        )

        _, _, _, trace = self.rag._ask_current_effect(
            self.history.id, [], None, requires_history_context=True
        )

        self.assertEqual(self.effect.calls, [["15/2020/NĐ-CP"]])
        self.assertEqual(trace["scope_resolution"]["source"], "history")

    def test_history_reference_with_many_documents_requires_explicit(self) -> None:
        self.histories.add_message(
            self.history.id, MessageRole.ASSISTANT, "trả lời trước",
            sources=json.dumps([
                {"document_id": self.document.id},
                {"document_id": self.second_document.id},
            ]),
        )

        answer, _, _, _ = self.rag._ask_current_effect(
            self.history.id, [], None, requires_history_context=True
        )

        self.assertIn("nêu rõ", answer)
        self.assertEqual(self.effect.calls, [])

    def test_multi_document_scope_without_target_requires_explicit(self) -> None:
        answer, _, _, _ = self.rag._ask_current_effect(
            self.history.id, [], None, scope=QueryScope.ALL_MATCHING
        )

        self.assertIn("nêu rõ", answer)
        self.assertEqual(self.effect.calls, [])


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
