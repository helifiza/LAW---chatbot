"""Indexing tài liệu pháp luật theo Chương/Mục/Điều.

Điểm quan trọng của pipeline này:
- Parse toàn bộ tài liệu liên tục, không coi mỗi trang là một tài liệu riêng.
- Giữ trạng thái Chương/Mục/Điều khi chuyển trang.
- Một Điều chỉ kết thúc khi gặp Điều/Mục/Chương mới hoặc hết tài liệu.
- chunk_index tăng liên tục và duy nhất trong toàn bộ tài liệu.
- page_number là trang bắt đầu; page_end_number là trang kết thúc chunk.
"""

import os
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional, Sequence, Tuple
from functools import lru_cache


_token_counter: Optional[Callable[[str], int]] = None

@lru_cache(maxsize=1)
def _tiktoken_encoder():
  try:
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")
  except Exception:
    return None

def _xu_ly_token(text: str) ->int:
  if not text:
    return 0
  encoder = _tiktoken_encoder()
  if encoder is None:
    raise RuntimeError(
        "Không tìm thấy tokenizer"
    )
  return len(encoder.encode(text))

def dem_token(text: str) -> int:
  counter = _token_counter or _xu_ly_token
  return counter(text)

@dataclass
class in4_detail:
    """Thông tin một chunk và vị trí nguồn của nó trong tài liệu."""

    document_id: str
    element_id: str
    file_name: str
    page_number: int  # Trang bắt đầu của chunk
    content_type: str
    chuong: Optional[str]
    muc: Optional[str]
    dieu: Optional[str]
    text: str
    chunk_index: int = 0
    page_end_number: Optional[int] = None  # Trang kết thúc của chunk
    char_count: int = field(init=False)
    token_count: int = field(init=False)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.page_end_number is None:
            self.page_end_number = self.page_number
        self.char_count = len(self.text)
        self.token_count = dem_token(self.text)


# Chỉ dùng space/tab trong regex heading. Không dùng \s vì \s ăn cả xuống dòng.
PATTERN_CHUONG = re.compile(
    r"^[ \t]*CHƯƠNG[ \t]+[IVXLCDM\d]+",
    re.IGNORECASE | re.MULTILINE,
)
PATTERN_MUC = re.compile(
    r"^[ \t]*Mục[ \t]+\d+[A-Za-z]?(?:[.:-])?",
    re.IGNORECASE | re.MULTILINE,
)
PATTERN_DIEU = re.compile(
    # Dấu phân cách là bắt buộc để không nhầm câu "Điều 3 của Luật..."
    # ở đầu một dòng bị wrap thành heading mới.
    r"^[ \t]*Điều[ \t]+\d+[A-Za-z]?[ \t]*[.:-]",
    re.IGNORECASE | re.MULTILINE,
)

HEADING_NEXT = (
    r"[ \t]*(?:"
    r"CHƯƠNG[ \t]+[IVXLCDM\d]+"
    r"|Mục[ \t]+\d+[A-Za-z]?(?:[.:-])?"
    r"|Điều[ \t]+\d+[A-Za-z]?[ \t]*[.:-]"
    # Không đặt \b sau toàn bộ nhóm: dấu chấm là ký tự non-word nên
    # chuỗi "Điều 1." theo sau bởi space không tạo word boundary.
    r")"
)


def lam_sach_text(
    text: str,
    bare_page_number: Optional[int] = None,
) -> str:
    """Làm sạch text nhưng luôn giữ ranh giới trước heading pháp luật."""
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Một số PDF đặt số trang trần (ví dụ dòng chỉ có "8") ở đầu/cuối trang.
    # Chỉ xóa khi caller cung cấp đúng số trang và nó nằm ở biên trang, nhờ đó
    # không xóa nhầm một ô bảng hoặc dữ liệu số ở giữa nội dung.
    if bare_page_number is not None:
        lines = text.split("\n")
        nonempty_indexes = [i for i, line in enumerate(lines) if line.strip()]
        for index in nonempty_indexes[:1] + nonempty_indexes[-1:]:
            if lines[index].strip() == str(bare_page_number):
                lines[index] = ""
        text = "\n".join(lines)

    # Xóa dòng chỉ chứa số trang. [ \t] được dùng để không ăn sang dòng khác.
    text = re.sub(
        r"(?im)^[ \t]*(?:trang[ \t]*\d+(?:/\d+)?|-[ \t]*\d+[ \t]*-)[ \t]*$",
        "",
        text,
    )

    # pypdf thường để space ở cuối dòng; cần xóa trước khi xử lý xuống dòng.
    text = re.sub(r"[ \t]+(?=\n)", "", text)

    # Nối các dòng bị wrap, nhưng không nối mất dòng bắt đầu bằng heading.
    text = re.sub(
        rf"\n(?!\n|{HEADING_NEXT})",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


@dataclass(frozen=True)
class _NguonText:
    """Một phần text cùng trang nguồn, dùng nội bộ khi gom Điều xuyên trang."""

    page_number: int
    text: str


def _kiem_tra_cau_hinh_chunk(max_chunk_chars: int, overlap_chars: int) -> None:
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars phải lớn hơn 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars không được âm")
    if overlap_chars >= max_chunk_chars:
        raise ValueError("overlap_chars phải nhỏ hơn max_chunk_chars")


def _ghep_text_va_vi_tri_trang(
    parts: Sequence[_NguonText],
) -> Tuple[str, List[Tuple[int, int, int]]]:
    """Ghép buffer và tạo map (char_start, char_end, page_number)."""
    text_parts: List[str] = []
    spans: List[Tuple[int, int, int]] = []
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

    return "".join(text_parts), spans

#Hybrid chunking
#gđ1: tách theo cấu trúc chương/mục/điều
@dataclass(frozen=True)
class _khoiTho:
  chuong: Optional[str]
  muc: Optional[str]
  dieu: Optional[str]
  text: str
  spans: List[Tuple[int,int,int]]

def _chia_khoi_theo_cau_truc(
    pages: Sequence[Tuple[int,str]],
    remove_bare_page_numbers: bool,
)->List[_khoiTho]:
    khoi_list: List[_khoiTho] = []
    current_chuong: Optional[str] = None
    current_muc: Optional[str] = None
    current_dieu: Optional[str] = None
    buffer: List[_NguonText] = []

    def flush_buffer() -> None:
      nonlocal buffer
      if not buffer:
        return
      full_text, spans = _ghep_text_va_vi_tri_trang(buffer)
      buffer = []
      if not full_text or not spans:
        return
      khoi_list.append(
          _khoiTho(
              chuong=current_chuong,
              muc=current_muc,
              dieu=current_dieu,
              text=full_text,
              spans=spans,
          )
      )
    for page_number, raw_text in pages:
      clean_text = lam_sach_text(
          raw_text,
          bare_page_number=page_number if remove_bare_page_numbers else None,
      )
      if not clean_text:
        continue
      for line in clean_text.splitlines():
          line = line.strip()
          if not line:
            continue
          chuong_match = PATTERN_CHUONG.match(line)
          if chuong_match:
            flush_buffer()
            current_chuong = line
            current_muc = None
            current_dieu = None
            continue
          muc_match = PATTERN_MUC.match(line)
          if muc_match:
            flush_buffer()
            current_muc = line
            current_dieu = None
            continue
          dieu_match = PATTERN_DIEU.match(line)
          if dieu_match:
            flush_buffer()
            current_dieu = dieu_match.group().rstrip(".:- ").strip()
            buffer.append(_NguonText(page_number, line))
            continue
          buffer.append(_NguonText(page_number, line))
    flush_buffer()
    return khoi_list

# gđ2: nếu chunk ở gđ1 dài quá thì tiếp tục chunk nhỏ hơn(recursive)
FINE_SPLIT_SEPARATORS: Tuple["re.Pattern[str]", ...] = (
    re.compile(r"\n[ \t]*\n"),             # ranh giới đoạn văn (dòng trống)
    re.compile(r"\n"),                     # xuống dòng đơn (mỗi dòng nguồn)
    re.compile(r"(?<=[\.\?\!])[ \t]+"),    # ranh giới câu, sau . ? !
    re.compile(r"(?<=[;:])[ \t]+"),        # ranh giới mệnh đề, sau ; :
    re.compile(r"[ \t]+"),                 # khoảng trắng bất kỳ (mịn nhất)
)

def _trang_cua_khoang(
    spans: Sequence[Tuple[int, int, int]],
    start: int,
    end: int,
) -> Tuple[int,int]:
    pages = [page for span_start, span_end, page in spans if span_end > start and span_start <end]
    if not pages:
        nearest = min(spans, key = lambda span: abs(span[0] - start))
        return nearest[2], nearest[2]
    return pages[0], pages[-1]
def _tach_theo_1_tier(
    text: str,
    start: int,
    end: int,
    pattern: "re.Pattern[str]",
)->List[Tuple[int, int]]:
    pieces: List[Tuple[int, int]] = []
    cursor = start
    for m in pattern.finditer(text, start, end):
      piece_end = m.end()
      if piece_end <= cursor:
        continue
      pieces.append((cursor, piece_end))
      cursor = piece_end
    if cursor < end:
      pieces.append((cursor, end))
    return pieces

def recursive_tach_thanh_manh(
    text: str,
    start: int,
    end: int,
    max_chunk_chars: int,
    tiers:Sequence["re.Pattern[str]"] = FINE_SPLIT_SEPARATORS,
)->List[Tuple[int, int]]:
  if end - start <= max_chunk_chars:
    return [(start, end)]
  if tier_index >= len(tiers):
    ket_qua: List[Tuple[int,int]] = []
    cursor = start
    while cursor < end:
      piece_end = min(cursor + max_chunk_chars, end)
      ket_qua.append((cursor, piece_end))
      cursor = piece_end
    return ket_qua
  pieces = _tach_theo_1_tier(text, start, end, tiers[tier_index])
  if len(pieces) <= 1:
    return recursive_tach_thanh_manh(text, start, end, max_chunk_chars, tiers, tier_index+1)
  ket_qua = []
  for piece_start, piece_end in pieces:
    if piece_end - piece_start <= max_chunk_chars:
      ket_qua.append((piece_start, piece_end))
    else:
      ket_qua.extend(
          recursive_tach_thanh_manh(
              text,
              piece_start,
              piece_end,
              max_chunk_chars,
              tiers,
              tier_index + 1,
          )
      )
  return ket_qua

def _tranh_cat_giua_tu(text: str, index: int, gioi_han_duoi: int, gioi_han_tren: int) -> int:
  while gioi_han_duoi < index < gioi_han_tren and not text[index-1].isspace():
    index += 1
  return index

def overlap(
    text: str,
    manh_nho: Sequence[Tuple[int,int]],
    max_chunk_chars: int,
    overla_chars: int,
)->List[Tuple[int,int]]:
    if not manh_nho:
        return []
    chunks: List[Tuple[int,int]] = []
    chunk_start, chunk_end = manh_nho[0]
    for piece_start, piece_end in manh_nho[1:]:
        if piece_end - chunk_start <= max_chunk_chars:
            chunk_end = piece_end
            continue
        chunks.append((chunk_start, chunk_end))
        new_start = max(
            chunk_start,
            piece_end - max_chunk_chars,
            chunk_end - overlap_chars,
        )
        new_start = _tranh_cat_giua_tu(text, new_start, chunk_start, piece_start)
        chunk_start, chunk_end = new_start, piece_end
    chunks.append((chunk_start, chunk_end))
    return chunks

def _chia_nhoi_khoi_tho(
    khoi: _khoiTho,
    max_chunk_chars: int,
    overlap_chars: int,
    fine_split_separators: Sequence["re.Pattern[str]"] = FINE_SPLIT_SEPARATORS,
)-> List[Tuple[str, int, int]]:
    text = khoi.text
    if len(text) <= max_chunk_chars:
        start_page, end_page = khoi.spans[0][2], khoi.spans[-1][-2]
        return [(text, start_page, end_page)]

    manh_nho = recursive_tach_thanh_manh(
        text,
        0,
        len(text),
        max_chunk_chars,
        fine_split_separators,
    )
    khoang_chunk = overlap(
        text,
        manh_nho,
        max_chunk_chars,
        overlap_chars
    )
    ket_qua: List[Tuple[str, int, int]] = []
    for start, end in khoang_chunk:
      piece = text[start:end].strip()
      if not piece:
        continue
      start_page, end_page = _trang_cua_khoang(khoi.spans, start, end)
      ket_qua.append((piece, start_page, end_page))
    return ket_qua

def trich_xuat_theo_trang(
    pages: Sequence[Tuple[int, str]],
    file_name: str,
    document_id: Optional[str] = None,
    max_chunk_chars: int = 5000,
    overlap_chars: int = 150,
    remove_bare_page_numbers: bool = False,
    fine_split_separators: Sequence["re.Pattern[str]"] = FINE_SPLIT_SEPARATORS,
) -> List[in4_detail]:
    _kiem_tra_cau_hinh_chunk(max_chunk_chars, overlap_chars)
    doc_id = document_id or str(uuid.uuid4())
    khoi_tho_list = _chia_khoi_theo_cau_truc(
        pages,
        remove_bare_page_numbers
    )
    chunks: List[in4_detail] = []
    for khoi in khoi_tho_list:
        pieces = _chia_nhoi_khoi_tho(
            khoi,
            max_chunk_chars,
            overlap_chars,
            fine_split_separators
        )
        is_partial = len(pieces) > 1
        for text_piece, start_page, end_page in pieces:
            chunks.append(
                in4_detail(
                    document_id=doc_id,
                    element_id=str(uuid.uuid4()),
                    file_name=file_name,
                    page_number=start_page,
                    page_end_number = end_page,
                    content_type= (
                        "dieu_partial"
                        if khoi.dieu is not None and is_partial
                        else "paragraph"
                    ),
                    chuong = khoi.chuong,
                    muc = khoi.muc,
                    dieu = khoi.dieu,
                    text = text_piece,
                    chunk_index = len(chunks),
                )
            )
    return chunks

def trich_xuat_smart(
    raw_text: str,
    file_name: str,
    page_number: int,
    document_id: Optional[str] = None,
    max_chunk_chars: int = 5000,
    overlap_chars: int = 150,
) -> List[in4_detail]:
    """API tương thích cho text đơn hoặc một trang.

    Với PDF nhiều trang, phải dùng ``trich_xuat_theo_trang`` thông qua
    ``xu_ly_pdf`` để trạng thái cấu trúc được giữ xuyên trang.
    """
    return trich_xuat_theo_trang(
        pages=[(page_number, raw_text)],
        file_name=file_name,
        document_id=document_id,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=overlap_chars,
        remove_bare_page_numbers=False,
    )


@dataclass
class so_sanh_doc:
    chunk: in4_detail
    score: float
    rerank_score: Optional[float] = None

    def diem_cuoi(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score

    @staticmethod
    def loc_trung_lap(
        ket_qua: List["so_sanh_doc"],
        nguong_giong_nhau: float = 0.9,
    ) -> List["so_sanh_doc"]:
        def jaccard(a: str, b: str) -> float:
            set_a, set_b = set(a.lower().split()), set(b.lower().split())
            if not set_a or not set_b:
                return 0.0
            return len(set_a & set_b) / len(set_a | set_b)

        ket_qua_sorted = sorted(
            ket_qua,
            key=lambda item: item.diem_cuoi(),
            reverse=True,
        )
        giu_lai: List[so_sanh_doc] = []
        for kq in ket_qua_sorted:
            trung = any(
                jaccard(kq.chunk.text, kept.chunk.text) >= nguong_giong_nhau
                for kept in giu_lai
            )
            if not trung:
                giu_lai.append(kq)
        return giu_lai

    @staticmethod
    def du_context(
        ket_qua: List["so_sanh_doc"],
        nguong_score: float = 0.5,
    ) -> bool:
        return bool(ket_qua) and ket_qua[0].diem_cuoi() >= nguong_score


def xu_ly_pdf(
    file_path: str,
    document_id: Optional[str] = None,
    max_chunk_chars: int = 5000,
    overlap_chars: int = 150,
) -> List[in4_detail]:
    """Đọc PDF rồi parse tất cả trang trong một luồng trạng thái duy nhất."""
    from pypdf import PdfReader

    file_name = os.path.basename(file_path)
    reader = PdfReader(file_path)
    pages = [
        (page_number, page.extract_text() or "")
        for page_number, page in enumerate(reader.pages, start=1)
    ]

    return trich_xuat_theo_trang(
        pages=pages,
        file_name=file_name,
        document_id=document_id,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=overlap_chars,
        remove_bare_page_numbers=True,
    )




def _in_thong_ke(chunks: Sequence[in4_detail]) -> None:
    print(f"Số chunk tạo ra: {len(chunks)}")
    print(f"Số chunk có Chương: {sum(c.chuong is not None for c in chunks)}")
    print(f"Số chunk có Mục: {sum(c.muc is not None for c in chunks)}")
    print(f"Số chunk có Điều: {sum(c.dieu is not None for c in chunks)}")
    print(f"")

    for chunk in chunks[:30]:
        print("---")
        print("chunk_index:", chunk.chunk_index)
        print("file:", chunk.file_name)
        print("pages:", f"{chunk.page_number}-{chunk.page_end_number}")
        print(
            "chuong:",
            chunk.chuong,
            "| muc:",
            chunk.muc,
            "| dieu:",
            chunk.dieu,
        )
        print("char_count:", chunk.char_count, "| token_count:", chunk.token_count)
        print("text:", chunk.text[:300].replace("\n", " "))


if __name__ == "__main__":
    # Thay đường dẫn này bằng file cần kiểm thử.
    file_path = r"/content/drive/MyDrive/Colab Notebooks/Thư viện luật/Luat-thue-thu-nhap-ca-nhan-nam-2025.pdf"
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        result = xu_ly_pdf(file_path)
    else:
        raise ValueError(f"Chưa hỗ trợ file dạng: {extension}")

    _in_thong_ke(result)
  
  #trình tự thực hiện:
  #1: Xử lý pdf: đọc toàn bộ trang và trích xuất text theo trang, trích xuất đầy đủ text trong trang đó
  #2:hàm trích xuất theo trang được gọi-> trong hàm này gọi hàm chia khối theo câu
  #cụ thể:
  #