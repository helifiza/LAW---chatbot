from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Sequence

from app.domain.models import ChunkDetail


PATTERN_CHUONG = re.compile(
    r"^[ \t]*CHƯƠNG[ \t]+[IVXLCDM\d]+", re.IGNORECASE | re.MULTILINE
)
PATTERN_MUC = re.compile(
    r"^[ \t]*Mục[ \t]+\d+[A-Za-z]?(?:[.:-])?", re.IGNORECASE | re.MULTILINE
)
PATTERN_DIEU = re.compile(
    r"^[ \t]*Điều[ \t]+\d+[A-Za-z]?[ \t]*[.:-]",
    re.IGNORECASE | re.MULTILINE,
)
HEADING_NEXT = (
    r"[ \t]*(?:"
    r"CHƯƠNG[ \t]+[IVXLCDM\d]+"
    r"|Mục[ \t]+\d+[A-Za-z]?(?:[ \t]*[.:-]|(?=[ \t]*(?:\r?\n|\f|$)))"
    r"|Điều[ \t]+\d+[A-Za-z]?[ \t]*[.:-]"
    r")"
)
FINE_SPLIT_SEPARATORS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n[ \t]*\n"),
    re.compile(r"\n"),
    re.compile(r"(?<=[.?!])[ \t]+"),
    re.compile(r"(?<=[;:])[ \t]+"),
    re.compile(r"[ \t]+"),
)


@lru_cache(maxsize=1)
def _token_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    encoder = _token_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class SourcePart:
    page_number: int
    text: str


@dataclass(frozen=True)
class RawBlock:
    chuong: str | None
    muc: str | None
    dieu: str | None
    text: str
    spans: tuple[tuple[int, int, int], ...]


class LegalChunkingService:
    """Hybrid chunking: cấu trúc pháp luật trước, ký tự sau."""

    def __init__(
        self,
        chunk_size_chars: int = 2500,
        overlap_chars: int = 250,
        logger: logging.Logger | None = None,
    ) -> None:
        if chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars phải lớn hơn 0")
        if overlap_chars < 0 or overlap_chars >= chunk_size_chars:
            raise ValueError("overlap_chars phải >= 0 và nhỏ hơn chunk_size_chars")
        self.chunk_size_chars = chunk_size_chars
        self.overlap_chars = overlap_chars
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def clean_text(text: str, bare_page_number: int | None = None) -> str:
        if not text:
            return ""
        value = unicodedata.normalize("NFC", text)
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        if bare_page_number is not None:
            lines = value.split("\n")
            nonempty = [index for index, line in enumerate(lines) if line.strip()]
            for index in set(nonempty[:1] + nonempty[-1:]):
                if lines[index].strip() == str(bare_page_number):
                    lines[index] = ""
            value = "\n".join(lines)
        value = re.sub(
            r"(?im)^[ \t]*(?:trang[ \t]*\d+(?:/\d+)?|-[ \t]*\d+[ \t]*-)[ \t]*$",
            "",
            value,
        )
        value = re.sub(r"[ \t]+(?=\n)", "", value)
        value = re.sub(
            rf"\n(?!\n|{HEADING_NEXT})", " ", value, flags=re.IGNORECASE
        )
        value = re.sub(r"\n{3,}", "\n\n", value)
        value = re.sub(r"[ \t]{2,}", " ", value)
        return value.strip()

    @staticmethod
    def _join_parts(
        parts: Sequence[SourcePart],
    ) -> tuple[str, tuple[tuple[int, int, int], ...]]:
        text_parts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        cursor = 0
        for part in parts:
            value = part.text.strip()
            if not value:
                continue
            if text_parts:
                text_parts.append("\n")
                cursor += 1
            start = cursor
            text_parts.append(value)
            cursor += len(value)
            spans.append((start, cursor, part.page_number))
        return "".join(text_parts), tuple(spans)

    def _parse_structure(
        self,
        pages: Sequence[tuple[int, str]],
        remove_bare_page_numbers: bool,
    ) -> list[RawBlock]:
        blocks: list[RawBlock] = []
        current_chuong: str | None = None
        current_muc: str | None = None
        current_dieu: str | None = None
        buffer: list[SourcePart] = []

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            full_text, spans = self._join_parts(buffer)
            buffer = []
            if full_text and spans:
                blocks.append(
                    RawBlock(
                        current_chuong,
                        current_muc,
                        current_dieu,
                        full_text,
                        spans,
                    )
                )

        for page_number, raw_text in pages:
            cleaned = self.clean_text(
                raw_text,
                page_number if remove_bare_page_numbers else None,
            )
            if not cleaned:
                continue
            for line in cleaned.splitlines():
                line = line.strip()
                if not line:
                    continue
                if PATTERN_CHUONG.match(line):
                    flush()
                    current_chuong, current_muc, current_dieu = line, None, None
                    continue
                if PATTERN_MUC.match(line):
                    flush()
                    current_muc, current_dieu = line, None
                    continue
                dieu_match = PATTERN_DIEU.match(line)
                if dieu_match:
                    flush()
                    current_dieu = dieu_match.group().rstrip(".:- ").strip()
                    buffer.append(SourcePart(page_number, line))
                    continue
                buffer.append(SourcePart(page_number, line))
        flush()
        self.logger.debug("Tách được %s khối cấu trúc", len(blocks))
        return blocks

    @staticmethod
    def _split_one_tier(
        text: str,
        start: int,
        end: int,
        pattern: re.Pattern[str],
    ) -> list[tuple[int, int]]:
        pieces: list[tuple[int, int]] = []
        cursor = start
        for match in pattern.finditer(text, start, end):
            piece_end = match.end()
            if piece_end > cursor:
                pieces.append((cursor, piece_end))
                cursor = piece_end
        if cursor < end:
            pieces.append((cursor, end))
        return pieces

    def _recursive_split(
        self,
        text: str,
        start: int,
        end: int,
        tier_index: int = 0,
    ) -> list[tuple[int, int]]:
        if end - start <= self.chunk_size_chars:
            return [(start, end)]
        if tier_index >= len(FINE_SPLIT_SEPARATORS):
            return [
                (cursor, min(cursor + self.chunk_size_chars, end))
                for cursor in range(start, end, self.chunk_size_chars)
            ]
        pieces = self._split_one_tier(
            text, start, end, FINE_SPLIT_SEPARATORS[tier_index]
        )
        if len(pieces) <= 1:
            return self._recursive_split(text, start, end, tier_index + 1)
        result: list[tuple[int, int]] = []
        for piece_start, piece_end in pieces:
            if piece_end - piece_start <= self.chunk_size_chars:
                result.append((piece_start, piece_end))
            else:
                result.extend(
                    self._recursive_split(
                        text, piece_start, piece_end, tier_index + 1
                    )
                )
        return result

    @staticmethod
    def _avoid_cutting_word(text: str, index: int, lower: int, upper: int) -> int:
        while lower < index < upper and not text[index - 1].isspace():
            index += 1
        return index

    def _merge_with_overlap(
        self, text: str, pieces: Sequence[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        if not pieces:
            return []
        chunks: list[tuple[int, int]] = []
        chunk_start, chunk_end = pieces[0]
        for piece_start, piece_end in pieces[1:]:
            if piece_end - chunk_start <= self.chunk_size_chars:
                chunk_end = piece_end
                continue
            chunks.append((chunk_start, chunk_end))
            new_start = max(
                chunk_start,
                piece_end - self.chunk_size_chars,
                chunk_end - self.overlap_chars,
            )
            new_start = self._avoid_cutting_word(
                text, new_start, chunk_start, piece_start
            )
            chunk_start, chunk_end = new_start, piece_end
        chunks.append((chunk_start, chunk_end))
        return chunks

    @staticmethod
    def _pages_for_range(
        spans: Sequence[tuple[int, int, int]], start: int, end: int
    ) -> tuple[int, int]:
        pages = [
            page
            for span_start, span_end, page in spans
            if span_end > start and span_start < end
        ]
        if pages:
            return pages[0], pages[-1]
        nearest = min(spans, key=lambda span: abs(span[0] - start))
        return nearest[2], nearest[2]

    def _split_block(self, block: RawBlock) -> list[tuple[str, int, int]]:
        if len(block.text) <= self.chunk_size_chars:
            return [(block.text, block.spans[0][2], block.spans[-1][2])]
        atomic = self._recursive_split(block.text, 0, len(block.text))
        result: list[tuple[str, int, int]] = []
        for start, end in self._merge_with_overlap(block.text, atomic):
            piece = block.text[start:end].strip()
            if piece:
                start_page, end_page = self._pages_for_range(block.spans, start, end)
                result.append((piece, start_page, end_page))
        return result

    def chunk_pages(
        self,
        session_id: str,
        document_id: str,
        file_name: str,
        pages: Sequence[tuple[int, str]],
        remove_bare_page_numbers: bool = True,
    ) -> list[ChunkDetail]:
        blocks = self._parse_structure(pages, remove_bare_page_numbers)
        chunks: list[ChunkDetail] = []
        created_at = datetime.now(timezone.utc).isoformat()
        for block in blocks:
            pieces = self._split_block(block)
            is_partial = len(pieces) > 1
            if block.dieu is not None:
                content_type = "dieu_partial" if is_partial else "dieu"
            elif block.muc is not None:
                content_type = "muc"
            elif block.chuong is not None:
                content_type = "chuong"
            else:
                content_type = "paragraph"
            for text, start_page, end_page in pieces:
                chunk_index = len(chunks)
                chunks.append(
                    ChunkDetail(
                        session_id=session_id,
                        document_id=document_id,
                        element_id=f"{document_id}:chunk:{chunk_index}",
                        file_name=file_name,
                        page_number=start_page,
                        page_end_number=end_page,
                        content_type=content_type,
                        chuong=block.chuong,
                        muc=block.muc,
                        dieu=block.dieu,
                        text=text,
                        chunk_index=chunk_index,
                        char_count=len(text),
                        token_count=count_tokens(text),
                        created_at=created_at,
                    )
                )
        self.logger.info(
            "Chunking hoàn tất | session=%s document=%s chunks=%s",
            session_id,
            document_id,
            len(chunks),
        )
        return chunks
