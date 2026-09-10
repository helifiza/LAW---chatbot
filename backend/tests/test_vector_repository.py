import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.domain.models import ChunkDetail
from app.repositories.vector_repository import VectorRepository


def make_chunk(history_id: str, document_id: str, element_id: str) -> ChunkDetail:
    text = f"Nội dung {history_id}"
    return ChunkDetail(
        history_id, document_id, "user", element_id, f"{document_id}.txt", 1, 1,
        "text", None, None, None, text, 0, len(text), 4,
        "2026-07-20T00:00:00+00:00",
    )


class VectorRepositoryTests(unittest.TestCase):
    def test_query_and_delete_are_scoped_to_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = VectorRepository(
                Path(directory) / "chroma", "test_vectors", "test-model"
            )
            repository.upsert(
                [make_chunk("history-a", "doc-a", "chunk-a"), make_chunk("history-b", "doc-b", "chunk-b")],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            )
            results = repository.query("history-a", [1.0, 0.0, 0.0], 5)
            self.assertEqual([item.chunk.element_id for item in results], ["chunk-a"])
            repository.delete_document("history-a", "doc-a")
            self.assertEqual(repository.query("history-a", [1.0, 0.0, 0.0], 5), [])
            self.assertEqual(repository.count(), 1)
            repository.close()

    def test_document_type_is_stored_updated_and_used_as_query_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = VectorRepository(
                Path(directory) / "chroma", "document_type_vectors", "test-model"
            )
            decree = replace(
                make_chunk("history-a", "decree", "chunk-decree"),
                document_type="nghị định",
            )
            law = replace(
                make_chunk("history-a", "law", "chunk-law"),
                document_type="luật",
            )
            repository.upsert([decree, law], [[1.0, 0.0], [1.0, 0.0]])

            decree_results = repository.query(
                "history-a", [1.0, 0.0], 5,
                document_type_filter=["NGHI DINH"],
            )
            self.assertEqual(
                [item.chunk.element_id for item in decree_results],
                ["chunk-decree"],
            )
            self.assertEqual(decree_results[0].chunk.document_type, "nghị định")

            updated_count = repository.update_document_metadata(
                "history-a", "law", document_type="nghị định"
            )
            self.assertEqual(updated_count, 1)
            updated_results = repository.query(
                "history-a", [1.0, 0.0], 5,
                document_type_filter=["nghị định"],
            )
            self.assertEqual(
                {item.chunk.element_id for item in updated_results},
                {"chunk-decree", "chunk-law"},
            )
            repository.close()


if __name__ == "__main__":
    unittest.main()
