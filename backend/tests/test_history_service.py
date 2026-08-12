import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from app.repositories.history_repository import HistoryRepository
from app.services.history_service import HistoryService


class FakeVectorRepository:
    def __init__(self) -> None:
        self.deleted_histories: list[str] = []

    def delete_history(self, history_id: str) -> None:
        self.deleted_histories.append(history_id)

    def delete_document(self, _history_id: str, _document_id: str) -> None:
        return None


class HistoryServiceTests(unittest.TestCase):
    def test_cleanup_removes_expired_sqlite_and_vector_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = HistoryRepository(Path(directory) / "state.db")
            vectors = FakeVectorRepository()
            service = HistoryService(repository, vectors, 1, 5)
            history = service.create()
            removed = service.cleanup_expired(history.expires_at + timedelta(seconds=1))
            self.assertEqual(removed, 1)
            self.assertIsNone(repository.get_history(history.id))
            self.assertEqual(vectors.deleted_histories, [history.id])

    def test_create_uses_default_values_when_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = HistoryRepository(Path(directory) / "state.db")
            vectors = FakeVectorRepository()
            service = HistoryService(repository, vectors, 1, 5)

            history = service.create()

            self.assertEqual(history.user_id, "local")
            self.assertEqual(history.title, "Cuộc trò chuyện mới")

    def test_list_by_user_returns_latest_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = HistoryRepository(Path(directory) / "state.db")
            vectors = FakeVectorRepository()
            service = HistoryService(repository, vectors, 1, 5)

            first = service.create("user-1", "Đầu tiên")
            second = service.create("user-1", "Thứ hai")

            histories = service.list_by_user("user-1")

            self.assertEqual([item.id for item in histories], [second.id, first.id])


if __name__ == "__main__":
    unittest.main()
