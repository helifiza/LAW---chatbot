import tempfile
import unittest
from pathlib import Path

from app.domain.models import ChunkDetail
from app.repositories.vector_repository import VectorRepository


def make_chunk(session_id: str, document_id: str, element_id: str) -> ChunkDetail:
    text = f"Nội dung {session_id}"
    return ChunkDetail(
        session_id, document_id, element_id, f"{document_id}.txt", 1, 1,
        "text", None, None, None, text, 0, len(text), 4,
        "2026-07-20T00:00:00+00:00",
    )


class VectorRepositoryTests(unittest.TestCase):
    def test_query_and_delete_are_scoped_to_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = VectorRepository(
                Path(directory) / "chroma", "test_vectors", "test-model"
            )
            repository.upsert(
                [make_chunk("session-a", "doc-a", "chunk-a"), make_chunk("session-b", "doc-b", "chunk-b")],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            )
            results = repository.query("session-a", [1.0, 0.0, 0.0], 5)
            self.assertEqual([item.chunk.element_id for item in results], ["chunk-a"])
            repository.delete_document("session-a", "doc-a")
            self.assertEqual(repository.query("session-a", [1.0, 0.0, 0.0], 5), [])
            self.assertEqual(repository.count(), 1)


if __name__ == "__main__":
    unittest.main()
