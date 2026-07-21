from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class SessionRecord:
    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    session_id: str
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
    session_id: str
    role: str
    content: str
    created_at: datetime


@dataclass(frozen=True)
class ChunkDetail:
    session_id: str
    document_id: str
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
