import tempfile
import unittest
from pathlib import Path

from app.domain.models import DocumentStatus, MessageRole
from app.repositories.session_repository import SessionRepository


class SessionRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.repository = SessionRepository(Path(self.temp_directory.name) / "test.db")

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_session_keeps_documents_and_history(self) -> None:
        session = self.repository.create_session(30)
        document = self.repository.create_document(
            session.id, "luat.pdf", "application/pdf", 1024
        )
        self.repository.update_document_status(
            document.id, DocumentStatus.READY.value, chunk_count=4
        )
        self.repository.add_message(session.id, MessageRole.USER, "Điều 1 là gì?")
        self.repository.add_message(session.id, MessageRole.ASSISTANT, "Trả lời")
        self.assertEqual(self.repository.list_documents(session.id)[0].chunk_count, 4)
        self.assertEqual(
            [item.role for item in self.repository.list_messages(session.id)],
            ["user", "assistant"],
        )

    def test_delete_session_cascades_state(self) -> None:
        session = self.repository.create_session(30)
        self.repository.create_document(session.id, "test.txt", "text/plain", 12)
        self.repository.add_message(session.id, MessageRole.USER, "test")
        self.assertTrue(self.repository.delete_session(session.id))
        self.assertEqual(self.repository.list_documents(session.id), [])
        self.assertEqual(self.repository.list_messages(session.id), [])


if __name__ == "__main__":
    unittest.main()
