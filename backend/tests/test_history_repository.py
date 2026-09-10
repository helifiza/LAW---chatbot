import tempfile
import unittest
from pathlib import Path

from app.domain.models import DocumentStatus, MessageRole
from app.repositories.history_repository import HistoryRepository
from app.repositories.user_repository import UserRepository


class HistoryRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database = Path(self.temp_directory.name) / "test.db"
        user = UserRepository(database).create_user("Test", "test@example.com", "hash")
        self.user_id = user["id"]
        self.repository = HistoryRepository(database)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_history_keeps_documents_and_history(self) -> None:
        history = self.repository.create_history(self.user_id, "Test")
        document = self.repository.create_document(
            history.id, "luat.pdf", "application/pdf", 1024
        )
        self.repository.update_document_status(
            document.id, DocumentStatus.READY.value, chunk_count=4
        )
        self.repository.add_message(history.id, MessageRole.USER, "Điều 1 là gì?")
        self.repository.add_message(history.id, MessageRole.ASSISTANT, "Trả lời")
        self.assertEqual(self.repository.list_documents(history.id)[0].chunk_count, 4)
        self.assertEqual(
            [item.role for item in self.repository.list_messages(history.id)],
            ["user", "assistant"],
        )

    def test_delete_history_cascades_state(self) -> None:
        history = self.repository.create_history(self.user_id, "Test")
        self.repository.create_document(history.id, "test.txt", "text/plain", 12)
        self.repository.add_message(history.id, MessageRole.USER, "test")
        self.assertTrue(self.repository.delete_history(history.id))
        self.assertEqual(self.repository.list_documents(history.id), [])
        self.assertEqual(self.repository.list_messages(history.id), [])


if __name__ == "__main__":
    unittest.main()
