import unittest
from types import SimpleNamespace

from app.domain.models import ChunkDetail
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.services.summary_service import SummarizeService


class _HistoryRepository:
    def list_documents(self, _history_id):
        return [
            SimpleNamespace(id="decree", file_name="nghi_dinh.pdf", status="ready"),
            SimpleNamespace(id="law", file_name="luat.pdf", status="ready"),
        ]


class _LegalGraphRepository:
    def list_upload_document_ids_by_types(self, _history_id, document_types):
        return {"decree"} if "nghị định" in document_types else set()


class SummaryDocumentTypeFilterTests(unittest.TestCase):
    def test_filters_ready_documents_before_resolving_explicit_target(self):
        service = SummarizeService(
            history_repo=_HistoryRepository(),
            vector_repo=SimpleNamespace(),
            gemini_client=SimpleNamespace(),
            model="test",
            legal_graph_repo=_LegalGraphRepository(),
        )

        scope = service._resolve_scope(
            "history",
            "Tóm tắt luật",
            target_documents=["luat"],
            document_type_filter=["nghị định"],
        )

        self.assertEqual(scope, [])


class _VectorRepository:
    def __init__(self, chunks):
        self.chunks = chunks
        self.received_filter = None

    def query(self, *_args, document_type_filter=None, **_kwargs):
        self.received_filter = document_type_filter
        return []

    def list_chunks(self, _history_id):
        return self.chunks


class _EmbeddingService:
    def embed_query(self, _query):
        return [1.0, 0.0]


class _Reranker:
    def rerank(self, _query, results, top_k):
        return list(results[:top_k])


class HybridDocumentTypeFilterTests(unittest.TestCase):
    @staticmethod
    def _chunk(element_id, document_type, text="nghĩa vụ thanh toán"):
        return ChunkDetail(
            history_id="history",
            document_id=element_id,
            user_id="user",
            element_id=element_id,
            file_name=f"{element_id}.pdf",
            page_number=1,
            page_end_number=1,
            content_type="text",
            chuong=None,
            muc=None,
            dieu=None,
            text=text,
            chunk_index=0,
            char_count=20,
            token_count=4,
            created_at="2026-01-01T00:00:00+00:00",
            document_type=document_type,
        )

    def test_applies_same_filter_to_dense_and_bm25_candidates(self):
        vector_repository = _VectorRepository([
            self._chunk("decree", "nghị định"),
            self._chunk("decree-2", "nghị định", "quy định khác"),
            self._chunk("decree-3", "nghị định", "điều khoản riêng"),
            self._chunk("law", "luật"),
        ])
        service = HybridRetrievalService(
            vector_repository=vector_repository,
            embedding_service=_EmbeddingService(),
            reranker=_Reranker(),
            candidate_k=10,
            rerank_candidate_k=10,
            rrf_k=60,
            min_dense_similarity=0.0,
        )

        outcome = service.retrieve(
            "history",
            "thanh toán",
            "thanh toán",
            5,
            document_type_filter=["NGHI DINH"],
        )

        self.assertEqual(vector_repository.received_filter, ["NGHI DINH"])
        self.assertEqual(
            [result.chunk.element_id for result in outcome.results],
            ["decree"],
        )
        self.assertEqual(outcome.trace["document_type_filter"], ["nghị định"])


if __name__ == "__main__":
    unittest.main()
