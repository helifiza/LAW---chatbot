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
