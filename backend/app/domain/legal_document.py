from __future__ import annotations

import re
import unicodedata
from typing import Iterable


LEGAL_DOCUMENT_TYPES = (
    "hiến pháp",
    "bộ luật",
    "luật",
    "pháp lệnh",
    "nghị quyết",
    "nghị định",
    "quyết định",
    "thông tư",
    "thông tư liên tịch",
    "chỉ thị",
    "văn bản hợp nhất",
)


def _plain(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")


_DOCUMENT_TYPE_BY_PLAIN = {_plain(value): value for value in LEGAL_DOCUMENT_TYPES}


def normalize_document_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip().lower()
    if not cleaned:
        return None
    return _DOCUMENT_TYPE_BY_PLAIN.get(_plain(cleaned), cleaned)


def normalize_document_type_filter(values: Iterable[object] | None) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        values = (values,)
    normalized = (normalize_document_type(value) for value in values or ())
    return tuple(dict.fromkeys(value for value in normalized if value))


#hàm chuẩn hóa tên/số hiệu văn bản về chuỗi token không dấu, ví dụ
#"Nghị định 15/2020/NĐ-CP" thành "nghi dinh 15 2020 nd cp".
def reference_key(value: object) -> str:
    plain = _plain(str(value or "").lower())
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


def contains_reference(haystack: object, needle: object) -> bool:
    """needle nằm trong haystack theo biên token ("15 2020" không khớp "115 2020")."""
    needle_key = reference_key(needle)
    return bool(needle_key) and f" {needle_key} " in f" {reference_key(haystack)} "


def matches_legal_reference(reference: str, document_number: object, *texts: object) -> bool:
    """Khớp tên/số hiệu người dùng nêu với văn bản trong graph, theo 2 chiều:
    - ref nằm trong số hiệu/tiêu đề ("15/2020" ~ "15/2020/NĐ-CP");
    - số hiệu nằm trong ref ("Nghị định 15/2020/NĐ-CP" ~ "15/2020/NĐ-CP"). Chiều
      này chỉ dùng khi số hiệu đủ đặc trưng (có chữ số, >= 2 token) để tránh số
      hiệu cụt như "15" khớp nhầm mọi câu."""
    if contains_reference(" ".join(str(text or "") for text in (document_number, *texts)), reference):
        return True
    number_key = reference_key(document_number)
    return (
        any(char.isdigit() for char in number_key)
        and len(number_key.split()) >= 2
        and contains_reference(reference, document_number)
    )
