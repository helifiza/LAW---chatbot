from __future__ import annotations

import logging
from datetime import datetime

from app.core.errors import (
    DocumentLimitError,
    DocumentNotFoundError,
    SessionClosedError,
    SessionExpiredError,
    SessionNotFoundError,
)
from app.domain.models import (
    DocumentRecord,
    DocumentStatus,
    MessageRecord,
    MessageRole,
    SessionRecord,
    SessionStatus,
    utc_now,
)
from app.repositories.session_repository import SessionRepository
from app.repositories.vector_repository import VectorRepository


class SessionService:
    def __init__(
        self,
        repository: SessionRepository,
        vector_repository: VectorRepository,
        ttl_minutes: int,
        max_documents: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self.repository = repository
        self.vector_repository = vector_repository
        self.ttl_minutes = ttl_minutes
        self.max_documents = max_documents
        self.logger = logger or logging.getLogger(__name__)

    def create(self) -> SessionRecord:
        self.cleanup_expired()
        session = self.repository.create_session(self.ttl_minutes)
        self.logger.info("Tạo phiên | session=%s", session.id)
        return session

    def require_active(self, session_id: str, touch: bool = True) -> SessionRecord:
        session = self.repository.get_session(session_id)
        if session is None:
            raise SessionNotFoundError("Không tìm thấy phiên làm việc")
        if session.status != SessionStatus.ACTIVE.value:
            raise SessionClosedError("Phiên làm việc đã đóng")
        if session.expires_at <= utc_now():
            self._purge(session_id)
            raise SessionExpiredError("Phiên làm việc đã hết hạn")
        if touch:
            touched = self.repository.touch_session(session_id, self.ttl_minutes)
            if touched is not None:
                session = touched
        return session

    def snapshot(
        self, session_id: str
    ) -> tuple[SessionRecord, list[DocumentRecord], list[MessageRecord]]:
        session = self.require_active(session_id)
        return (
            session,
            self.repository.list_documents(session_id),
            self.repository.list_messages(session_id),
        )

    def start_document(
        self,
        session_id: str,
        file_name: str,
        mime_type: str,
        size_bytes: int,
    ) -> DocumentRecord:
        self.require_active(session_id)
        active_count = self.repository.count_documents(
            session_id,
            statuses=(DocumentStatus.PROCESSING.value, DocumentStatus.READY.value),
        )
        if active_count >= self.max_documents:
            raise DocumentLimitError(
                f"Mỗi phiên chỉ được dùng tối đa {self.max_documents} tài liệu"
            )
        return self.repository.create_document(
            session_id, file_name, mime_type, size_bytes
        )

    def delete_document(self, session_id: str, document_id: str) -> None:
        self.require_active(session_id)
        document = self.repository.get_document(document_id)
        if document is None or document.session_id != session_id:
            raise DocumentNotFoundError("Không tìm thấy tài liệu trong phiên")
        self.vector_repository.delete_document(session_id, document_id)
        self.repository.delete_document(session_id, document_id)
        self.logger.info(
            "Xóa tài liệu | session=%s document=%s", session_id, document_id
        )

    def clear_documents(self, session_id: str) -> None:
        self.require_active(session_id)
        for document in self.repository.list_documents(session_id):
            self.vector_repository.delete_document(session_id, document.id)
            self.repository.delete_document(session_id, document.id)

    def close(self, session_id: str) -> None:
        if self.repository.get_session(session_id) is None:
            return
        self._purge(session_id)
        self.logger.info("Đóng phiên | session=%s", session_id)

    def add_message(
        self, session_id: str, role: MessageRole | str, content: str
    ) -> MessageRecord:
        self.require_active(session_id)
        return self.repository.add_message(session_id, role, content)

    def recent_messages(self, session_id: str, limit: int) -> list[MessageRecord]:
        self.require_active(session_id)
        return self.repository.list_messages(session_id, limit=limit)

    def cleanup_expired(self, at: datetime | None = None) -> int:
        ids = self.repository.list_expired_session_ids(at)
        for session_id in ids:
            self._purge(session_id)
        if ids:
            self.logger.info("Dọn %s phiên hết hạn/đã đóng", len(ids))
        return len(ids)

    def _purge(self, session_id: str) -> None:
        try:
            self.vector_repository.delete_session(session_id)
        finally:
            self.repository.delete_session(session_id)
