import tempfile
import unittest
from pathlib import Path

from app.services.document_parser import DocumentParser


class DocumentParserTests(unittest.TestCase):
    def test_reads_utf8_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.txt"
            path.write_text("Nội dung tiếng Việt", encoding="utf-8")
            self.assertEqual(DocumentParser().parse(path, path.name)[0][1], "Nội dung tiếng Việt")

    def test_pretty_prints_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            path.write_text('{"ten":"Luật"}', encoding="utf-8")
            text = DocumentParser().parse(path, path.name)[0][1]
            self.assertIn('"ten": "Luật"', text)


if __name__ == "__main__":
    unittest.main()
