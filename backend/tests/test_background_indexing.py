import tempfile
import unittest
from pathlib import Path

from app.api.routers.documents import _index_in_background


class FakeIndexingService:
    def __init__(self) -> None:
        self.calls = []

    def index_file(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))


class FakeContainer:
    def __init__(self) -> None:
        self.indexing_service = FakeIndexingService()


class BackgroundIndexingTests(unittest.TestCase):
    def test_task_indexes_existing_record_and_removes_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "upload.pdf"
            path.write_bytes(b"pdf")
            container = FakeContainer()
            document = object()

            _index_in_background(
                container, "history", "user", path, "upload.pdf",
                "application/pdf", 3, document,
            )

            self.assertFalse(path.exists())
            self.assertEqual(len(container.indexing_service.calls), 1)
            self.assertIs(container.indexing_service.calls[0][1]["document"], document)


if __name__ == "__main__":
    unittest.main()
