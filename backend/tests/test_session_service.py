import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from app.repositories.session_repository import SessionRepository
from app.services.session_service import SessionService


class FakeVectorRepository:
    def __init__(self) -> None:
        self.deleted_sessions: list[str] = []

    def delete_session(self, session_id: str) -> None:
        self.deleted_sessions.append(session_id)

    def delete_document(self, _session_id: str, _document_id: str) -> None:
        return None


class SessionServiceTests(unittest.TestCase):
    def test_cleanup_removes_expired_sqlite_and_vector_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SessionRepository(Path(directory) / "state.db")
            vectors = FakeVectorRepository()
            service = SessionService(repository, vectors, 1, 5)
            session = service.create()
            removed = service.cleanup_expired(session.expires_at + timedelta(seconds=1))
            self.assertEqual(removed, 1)
            self.assertIsNone(repository.get_session(session.id))
            self.assertEqual(vectors.deleted_sessions, [session.id])


if __name__ == "__main__":
    unittest.main()
