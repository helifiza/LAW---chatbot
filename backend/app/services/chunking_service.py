from __future__ import annotations

import logging
import re #thư viện Regex để tìm kiếm hoặc xử lý chuỗi theo mẫu
import unicodedata #thư viện xử lý unicode
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Sequence

from app.domain.models import ChunkDetail#định nghĩa các phần tử ở file model


PATTERN_CHUONG = re.compile(
    #nhận diện CHƯƠNG bằng: đầu dòng ^; khoảng trắng đầu dòng [ \t]*; định dạng CHƯƠNG; có ít nhất 1 khoảng trắng với CHƯƠNG [ \t]+; số la mã hoặc số thường [IVXLCDM\d]+
    #dùng cờ re.IGNORECASE để không phân biệt chữ hoa/thường của chuỗi CHƯƠNG
    #cờ re.MULTILINE để nhận dạng chương trên dòng, trước nó không có kí tự chữ hay số nào cả, đằng sau thì có thể; Ví dụ: CHƯƠNG I QUY ĐỊNH CHUNG(đúng)/CHƯƠNG I
    # không có kí tự $ ở cuối để có thể văn bản pháp lý sẽ viết tiêu đề chương ở cùng dòng ngay sau chuỗi CHƯƠNG 
    # ^, CHƯƠNG, [ \t]+, [IVXLCDM\d]+ bắt buộc phải có, còn các kí tự khác thì không cần thiết
    r"^[ \t]*CHƯƠNG[ \t]+[IVXLCDM\d]+", re.IGNORECASE | re.MULTILINE
)
PATTERN_MUC = re.compile(
    #ngoại trừ ^, MỤC, [ \t]+, \d+ bắt buộc có thì các kí tự kia không cần thiết
    #trước mục được nhận dạng giống chương
    #\d+ để nhận diện phần số sau chữ mục, dấu + nằm sau \d là có thể có nhiều hơn 1 số;[A-Za-z]?có thể có 1 chữ cái trong danh sách đưa ra; (?:[.:-])? có thể có 1 trong các ký tự .:- thì vẫn chấp nhận đó là mục
    r"^[ \t]*Mục[ \t]+\d+[A-Za-z]?(?:[.:-])?", re.IGNORECASE | re.MULTILINE
)
PATTERN_DIEU = re.compile(
    #tương tự chương và mục
    #theo văn bản pháp lý thì bắt buộc có ^, Điều,[ \t]+, \d+, [.:-], các kí tự khác thì không cần thiết
    r"^[ \t]*Điều[ \t]+\d+[A-Za-z]?[ \t]*[.:-]",
    re.IGNORECASE | re.MULTILINE,
)

#hỗ trợ re.search() một lần là tìm được bất kì chương/mục/điều
HEADING_NEXT = (
    r"[ \t]*(?:"
    r"CHƯƠNG[ \t]+[IVXLCDM\d]+"
    r"|Mục[ \t]+\d+[A-Za-z]?(?:[ \t]*[.:-]|(?=[ \t]*(?:\r?\n|\f|$)))"
    r"|Điều[ \t]+\d+[A-Za-z]?[ \t]*[.:-]"
    r")"
)

#chia nhỏ chunk
FINE_SPLIT_SEPARATORS: tuple[re.Pattern[str], ...] = (
    #được ưu tiên cắt theo thứ tự từ trên xuống nếu chunk cắt xong vẫn dài
    re.compile(r"\n[ \t]*\n"),#cắt theo đoạn
    re.compile(r"\n"),#cắt theo từng dòng
    re.compile(r"(?<=[.?!])[ \t]+"),#cắt theo dấu chấm, chấm hỏi, chấm than
    re.compile(r"(?<=[;:])[ \t]+"),#cắt theo dấu chấm phẩy, dấu hai chấm
    re.compile(r"[ \t]+"),#cắt theo khoảng trắng
)


@lru_cache(maxsize=1)#lưu nhớ kết quả của hàm _token_encoder() để tránh việc import tiktoken nhiều lần, chỉ giữ lại 1 kết quả gần nhất
def _token_encoder():
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")#chỉ dùng để đếm số lượng token có trong văn bản, không dùng để mã hóa hay giải mã
    except Exception:
        return None

#tiktoken sẽ chia văn bản thành các token rồi ánh xạ mỗi token thành 1 id số nguyên
def count_tokens(text: str) -> int:
    if not text:
        return 0
    encoder = _token_encoder()
    if encoder is not None:
        return len(encoder.encode(text))#đếm token dựa vào cl100k_base
    return max(1, (len(text) + 3) // 4)#nếu không có thư viện tiktoken thì ước lượng số token dựa vào số ký tự, trung bình 1 token ~4 ký tự, nên chia cho 4 và làm tròn lên, tối thiểu là 1 token


@dataclass(frozen=True)
class SourcePart:#đối tượng này có thuộc tính dùng để lưu trữ thông tin về một phần của văn bản nguồn, bao gồm số trang và nội dung văn bản
    page_number: int
    text: str


@dataclass(frozen=True)
class RawBlock:#đối tượng này có thuộc tính dùng để lưu trữ thông tin về một khối văn bản đã được phân tích cấu trúc
    chuong: str | None
    muc: str | None
    dieu: str | None
    text: str
    spans: tuple[tuple[int, int, int], ...]

#XỬ LÝ VĂN BẢN TRƯỚC KHI PHÂN TÍCH CẤU TRÚC
class LegalChunkingService:
    """Hybrid chunking: cấu trúc pháp luật trước, ký tự sau."""
    #hàm khởi tạo dịch vụ
    def __init__(
        self,
        chunk_size_chars: int = 2500,#kích thước chunk mặc định là 2500 ký tự, có thể thay đổi khi khởi tạo dịch vụ
        overlap_chars: int = 250,#số ký tự chồng lấp giữa các chunk để giữ được ngữ cảnh
        logger: logging.Logger | None = None,
    ) -> None:
        if chunk_size_chars <= 0:
            raise ValueError("chunk_size_chars phải lớn hơn 0")
        if overlap_chars < 0 or overlap_chars >= chunk_size_chars:
            raise ValueError("overlap_chars phải >= 0 và nhỏ hơn chunk_size_chars")
        self.chunk_size_chars = chunk_size_chars
        self.overlap_chars = overlap_chars
        self.logger = logger or logging.getLogger(__name__)

    #hàm làm sạch văn bản
    @staticmethod
    def clean_text(text: str, bare_page_number: int | None = None) -> str:
        if not text:
            return ""
        value = unicodedata.normalize("NFC", text)
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        #nếu số trang(chỉ số) được thấy ở đầu trang or cuối trang thì xóa đi, tránh nhầm lẫn với nội dung văn bản
        if bare_page_number is not None:
            lines = value.split("\n")
            nonempty = [index for index, line in enumerate(lines) if line.strip()]
            for index in set(nonempty[:1] + nonempty[-1:]):
                if lines[index].strip() == str(bare_page_number):
                    lines[index] = ""
            value = "\n".join(lines)
        #tìm những dòng ví dụ Trang 15; trang15;TRANG   20;Trang 15/100; -15-;- 15 - để xóa đi
        value = re.sub(
            r"(?im)^[ \t]*(?:trang[ \t]*\d+(?:/\d+)?|-[ \t]*\d+[ \t]*-)[ \t]*$",
            "",
            value,
        )
        #xóa khoảng trắng hoặc tab ở cuối mỗi dòng, nhưng không xóa khoảng trắng giữa các từ => để tránh văn bản bị bẩn và khi ghép các dòng or so sánh không bị lỗi, tránh embedding và chunking thừa  
        value = re.sub(r"[ \t]+(?=\n)", "", value)
        #Ghép các dòng bị ngắt giữa câu và giữ nguyên ranh giới cấu trúc của chương/mục/điều, giữ đúng cấu trúc văn bản theo các đoạn để tránh mất ngữ nghĩa
        value = re.sub(
            rf"\n(?!\n|{HEADING_NEXT})", " ", value, flags=re.IGNORECASE
        )
        value = re.sub(r"\n{3,}", "\n\n", value)#nếu có >= 3 dòng trống thì thay bằng 2 dòng trống để giữ ranh giới đoạn
        value = re.sub(r"[ \t]{2,}", " ", value)# nếu có >= 2 khoảng trắng hoặc tab thì thay bằng 1 khoảng trắng để tránh văn bản bị bẩn và khi ghép các dòng or so sánh không bị lỗi, tránh embedding và chunking thừa
        return value.strip()# xóa khoảng trắng đầu và cuối văn bản
#PHÂN TÍCH CẤU TRÚC VĂN BẢN
    #Giai đoạn 1: phân tích cấu trúc văn bản từ các trang đã được làm sạch và trả về danh sách các khối văn bản thô (RawBlock) chứa thông tin về chương, mục, điều, nội dung văn bản và vị trí của từng phần trong chuỗi văn bản
    @staticmethod
    def _join_parts(# gộp các phần văn bản đã được phân tích cấu trúc thành một chuỗi văn bản duy nhất và trả về cùng với các thông tin về vị trí của từng phần trong chuỗi văn bản đó
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

    #hàm phân tích này hoạt động như sau:
    #buffer: vùng chứa tạm, flush: "đóng gói dữ liệu trong buffer vào RawBlock"
    #đọc từng trang, nhận diện CHƯƠNG->Mục tại chương đó -> Khi này mục vào chương chưa được thêm vào buffer được đưa vào current -> chưa có block
    #khi đọc đến Điều -> đọc hết thông tin trong điều đó -> buffer sẽ thêm thông tin của điều
    #khi đọc hết điều đó thì buffer sẽ được flush ra block
    def _parse_structure(#phân tích cấu trúc văn bản từ các trang đã được làm sạch và trả về danh sách các khối văn bản thô (RawBlock) chứa thông tin về chương, mục, điều, nội dung văn bản và vị trí của từng phần trong chuỗi văn bản
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

    #Giai đoạn 2: chia nhỏ các khối thô nếu chúng vượt quá kích thước chunk_size_chars
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

    #hàm điều chỉnh vị trí của overlap bằng isspace() để tránh cắt từ giữa câu, giữa từ
    @staticmethod
    def _avoid_cutting_word(text: str, index: int, lower: int, upper: int) -> int:
        while lower < index < upper and not text[index - 1].isspace():
            index += 1
        return index

    #hàm gộp các chunk có chồng lấp với nhau để giữ ngữ cảnh, tránh mất ý nghĩa khi chia nhỏ văn bản
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

    #hàm này để xác định một chunk(phần nhỏ) bắt đầu và kết thúc ở trang nào
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

    #hàm hỗ trợ chia nhỏ thêm các RawBlock nếu chúng vượt quá kích thước chunk_size_chars, đồng thời giữ nguyên thông tin về chương, mục, điều và vị trí của từng phần trong chuỗi văn bản
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

    #hàm chính gộp cả giai đoạn 1 và 2
    def chunk_pages(
        self,
        history_id: str,
        document_id: str,
        user_id: str,
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
                        history_id=history_id,
                        document_id=document_id,
                        user_id=user_id,
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
            "Chunking hoàn tất | history=%s document=%s chunks=%s",
            history_id,
            document_id,
            len(chunks),
        )
        return chunks
