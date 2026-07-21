from __future__ import annotations

import json
from pathlib import Path

from app.core.errors import DocumentParseError, UnsupportedFileTypeError


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}


class DocumentParser:
    """Tầng chỉ chịu trách nhiệm đổi file thành danh sách (trang, text)."""

    def parse(self, file_path: Path, original_file_name: str) -> list[tuple[int, str]]:
        extension = Path(original_file_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Chưa hỗ trợ định dạng {extension or 'không xác định'}"
            )
        try:
            if extension == ".pdf":
                pages = self._parse_pdf(file_path)
            elif extension == ".docx":
                pages = self._parse_docx(file_path)
            else:
                pages = self._parse_text(file_path, extension)
        except (UnsupportedFileTypeError, DocumentParseError):
            raise
        except Exception as exc:
            raise DocumentParseError(
                f"Không đọc được {original_file_name}: {exc}"
            ) from exc
        if not pages or not any(text.strip() for _, text in pages):
            raise DocumentParseError(
                "Không trích xuất được nội dung. PDF scan cần OCR trước khi upload."
            )
        return pages

    @staticmethod
    def _parse_pdf(file_path: Path) -> list[tuple[int, str]]:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        return [
            (page_number, page.extract_text() or "")
            for page_number, page in enumerate(reader.pages, start=1)
        ]

    @staticmethod
    def _parse_docx(file_path: Path) -> list[tuple[int, str]]:
        from docx import Document

        document = Document(str(file_path))
        return [(1, "\n".join(paragraph.text for paragraph in document.paragraphs))]

    @staticmethod
    def _parse_text(file_path: Path, extension: str) -> list[tuple[int, str]]:
        raw = file_path.read_text(encoding="utf-8-sig")
        if extension == ".json":
            try:
                raw = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
            except json.JSONDecodeError as exc:
                raise DocumentParseError(f"JSON không hợp lệ: {exc}") from exc
        return [(1, raw)]
