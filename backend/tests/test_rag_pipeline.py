import tempfile
import unittest
from pathlib import Path

from app.repositories.session_repository import SessionRepository
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import LegalChunkingService
from app.services.document_parser import DocumentParser
from app.services.indexing_service import IndexingService
from app.services.rag_service import RagService
from app.services.session_service import SessionService


class FakeEmbeddingService:
    def embed_texts(self, texts): return [[1.0, 0.0, 0.0] for _ in texts]
    def embed_query(self, _question): return [1.0, 0.0, 0.0]


class FakeGenerationService:
    def generate(self, question, context, history):
        assert question == "Nghĩa vụ là gì?"
        assert "Điều 1" in context
        assert history == []
        return "Bên A phải thanh toán đúng hạn. [hop_dong.txt, trang 1]"


class RagPipelineTests(unittest.TestCase):
    def test_index_retrieve_generate_and_persist_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = SessionRepository(root / "state.db")
            vectors = VectorRepository(root / "chroma", "pipeline_test", "fake-model")
            session_service = SessionService(sessions, vectors, 30, 5)
            embeddings = FakeEmbeddingService()
            indexing = IndexingService(
                session_service, sessions, vectors, DocumentParser(),
                LegalChunkingService(500, 50), embeddings,
            )
            rag = RagService(
                session_service, sessions, vectors, embeddings,
                FakeGenerationService(), 0.2, 8,
            )
            session = session_service.create()
            upload = root / "upload.txt"
            upload.write_text(
                "Điều 1: Nghĩa vụ thanh toán\nBên A phải thanh toán đúng hạn.",
                encoding="utf-8",
            )
            document = indexing.index_file(
                session.id, upload, "hop_dong.txt", "text/plain", upload.stat().st_size
            )
            answer = rag.ask(session.id, "Nghĩa vụ là gì?", 5)
            self.assertEqual(document.status, "ready")
            self.assertEqual(answer.sources[0].document_id, document.id)
            self.assertEqual(
                [item.role for item in sessions.list_messages(session.id)],
                ["user", "assistant"],
            )


if __name__ == "__main__":
    unittest.main()
