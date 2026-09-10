import tempfile
import unittest
from pathlib import Path

from app.core.errors import DocumentLimitError, HistoryNotFoundError
from app.repositories.history_repository import HistoryRepository
from app.repositories.user_repository import UserRepository
from app.services.history_service import HistoryService


class FakeVectorRepository:
    def __init__(self) -> None:
        self.deleted_histories: list[str] = []

    def delete_history(self, history_id: str) -> None:
        self.deleted_histories.append(history_id)

    def delete_document(self, _history_id: str, _document_id: str) -> None:
        return None


class HistoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        database = Path(self.temp_directory.name) / "state.db"
        self.user = UserRepository(database).create_user("Test", "test@example.com", "hash")
        self.repository = HistoryRepository(database)
        self.vectors = FakeVectorRepository()

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_missing_history_raises_domain_error(self) -> None:
        service = HistoryService(self.repository, self.vectors)
        with self.assertRaises(HistoryNotFoundError):
            service.require_active("missing")

    def test_document_limit_is_enforced_in_backend(self) -> None:
        service = HistoryService(self.repository, self.vectors, max_documents=1)
        history = service.create(self.user["id"], "Test")
        service.start_document(history.id, "one.pdf", "application/pdf", 10)
        with self.assertRaises(DocumentLimitError):
            service.start_document(history.id, "two.pdf", "application/pdf", 10)


if __name__ == "__main__":
    unittest.main()
