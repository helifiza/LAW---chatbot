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


class GraphStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    SKIPPED = "skipped"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class RelationType(StrEnum):
    CAN_CU = "CAN_CU"
    HUONG_DAN_THI_HANH = "HUONG_DAN_THI_HANH"
    QUY_DINH_CHI_TIET = "QUY_DINH_CHI_TIET"
    SUA_DOI = "SUA_DOI"
    BO_SUNG = "BO_SUNG"
    SUA_DOI_BO_SUNG = "SUA_DOI_BO_SUNG"
    THAY_THE = "THAY_THE"
    BAI_BO = "BAI_BO"
    BAI_BO_MOT_PHAN = "BAI_BO_MOT_PHAN"
    DINH_CHI = "DINH_CHI"
    GIA_HAN = "GIA_HAN"
    HOP_NHAT = "HOP_NHAT"


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
    # Mã lĩnh vực (01-19, xem LINH_VUC_TAXONOMY trong query_router_service).
    # None khi chưa phân loại xong (vừa tạo document) hoặc phân loại thất bại.
    linh_vuc: str | None = None
    graph_status: str = GraphStatus.PENDING.value
    graph_error: str | None = None


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
    # Gán cùng giá trị document.linh_vuc cho mọi chunk của document đó, để
    # VectorRepository có thể filter theo lĩnh vực ngay ở tầng Chroma.
    linh_vuc: str | None = None
    document_type: str | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk: ChunkDetail
    score: float


@dataclass(frozen=True)
class DocumentRelationshipRecord:
    id: str
    history_id: str
    source_document_id: str
    source_file_name: str
    target_document_id: str | None
    target_file_name: str | None
    target_reference_text: str
    relation_type: str
    evidence_text: str
    evidence_file_name: str
    evidence_page_number: int
    evidence_page_end_number: int
    confidence: float
    created_at: datetime
