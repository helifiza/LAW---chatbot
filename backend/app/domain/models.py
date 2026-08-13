from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HistoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"

class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class HistoryRecord:
    id: str
    user_id: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    history_id: str
    file_name: str
    mime_type: str
    size_bytes: int
    status: str
    chunk_count: int
    error_message: str | None
    created_at: datetime


@dataclass(frozen=True)
class MessageRecord:
    id: int
    history_id: str
    role: str
    content: str
    sources: str | None
    created_at: datetime


@dataclass(frozen=True)
class ChunkDetail:
    history_id: str
    document_id: str
    user_id: str
    element_id: str
    file_name: str
    page_number: int
    page_end_number: int
    content_type: str
    chuong: str | None
    muc: str | None
    dieu: str | None
    text: str
    chunk_index: int
    char_count: int
    token_count: int
    created_at: str


@dataclass(frozen=True)
class SearchResult:
    chunk: ChunkDetail
    score: float
