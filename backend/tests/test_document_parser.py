import unittest
from pathlib import Path

from app.core.errors import UnsupportedFileTypeError
from app.services.document_parser import DocumentParser


class DocumentParserTests(unittest.TestCase):
    def test_rejects_text(self) -> None:
        with self.assertRaises(UnsupportedFileTypeError):
            DocumentParser().parse(Path("test.txt"), "test.txt")

    def test_rejects_json(self) -> None:
        with self.assertRaises(UnsupportedFileTypeError):
            DocumentParser().parse(Path("test.json"), "test.json")


if __name__ == "__main__":
    unittest.main()
