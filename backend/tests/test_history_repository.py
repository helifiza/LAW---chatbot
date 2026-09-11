import sqlite3
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

    def test_soft_delete_history_hides_but_preserves_state(self) -> None:
        history = self.repository.create_history(self.user_id, "Test")
        self.repository.create_document(history.id, "test.txt", "text/plain", 12)
        self.repository.add_message(history.id, MessageRole.USER, "test")
        self.assertTrue(self.repository.soft_delete_history(history.id))

        self.assertIsNone(self.repository.get_history(history.id))
        self.assertEqual(self.repository.list_histories_by_user(self.user_id), [])
        deleted = self.repository.get_history(history.id, include_deleted=True)
        self.assertIsNotNone(deleted)
        self.assertIsNotNone(deleted.deleted_at)
        self.assertEqual(len(self.repository.list_documents(history.id)), 1)
        self.assertEqual(len(self.repository.list_messages(history.id)), 1)

        restored = self.repository.restore_history(history.id)
        self.assertIsNotNone(restored)
        self.assertIsNone(restored.deleted_at)
        self.assertEqual(
            [item.id for item in self.repository.list_histories_by_user(self.user_id)],
            [history.id],
        )

    def test_existing_database_is_migrated_with_deleted_at(self) -> None:
        legacy_database = Path(self.temp_directory.name) / "legacy.db"
        connection = sqlite3.connect(legacy_database)
        try:
            connection.execute(
                """CREATE TABLE history (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.commit()
        finally:
            connection.close()

        migrated_repository = HistoryRepository(legacy_database)
        with migrated_repository._connection() as connection:
            columns = [
                row["name"] for row in connection.execute("PRAGMA table_info(history)")
            ]
        self.assertIn("deleted_at", columns)


if __name__ == "__main__":
    unittest.main()
