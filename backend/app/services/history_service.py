from __future__ import annotations

import logging

from app.core.errors import (
    DocumentLimitError,
    DocumentNotFoundError,
    HistoryNotFoundError,
    HistoryArchivedError,
)
from app.domain.models import (
    DocumentRecord,
    DocumentStatus,
    MessageRecord,
    MessageRole,
    HistoryRecord,
    HistoryStatus,
)
from app.repositories.history_repository import HistoryRepository
from app.repositories.vector_repository import VectorRepository


class HistoryService:
    def __init__(
        self,
        repository: HistoryRepository,
        vector_repository: VectorRepository,
        logger: logging.Logger | None = None,
        *,
        max_documents: int = 5,
    ) -> None:
        self.repository = repository
        self.vector_repository = vector_repository
        self.logger = logger or logging.getLogger(__name__)
        self.max_documents = max_documents

    def create(self,
               user_id: str,
               title: str,
               ) -> HistoryRecord:
        history = self.repository.create_history(user_id, title)
        self.logger.info("Tạo lịch sử | history=%s user=%s", history.id, user_id)
        return history

    def get_any(self, history_id: str) -> HistoryRecord:
        """Lấy trạng thái history để xem nội dung dù nó đã bị đóng hay chưa"""
        history = self.repository.get_history(history_id)
        if history is None:
            raise HistoryNotFoundError("Không tìm thấy lịch sử")
        return history

    def require_active(self, history_id: str) -> HistoryRecord:
        history = self.get_any(history_id)
        if history.status == HistoryStatus.ARCHIVED:
            raise HistoryArchivedError("Lịch sử đã được lưu trữ, vui lòng mở lại trước khi tiếp tục")
        return history

    def list_histories_by_user(self, user_id: str) -> list[HistoryRecord]:
        return self.repository.list_histories_by_user(user_id)

    def snapshot(
        self, history_id: str
    ) -> tuple[HistoryRecord, list[DocumentRecord], list[MessageRecord]]:
        history = self.get_any(history_id)
        return (
            history,
            self.repository.list_documents(history_id),
            self.repository.list_messages(history_id),
        )

    def archive(self, history_id: str) -> HistoryRecord:
        self.require_active(history_id)
        updated = self.repository.set_history_status(history_id, HistoryStatus.ARCHIVED.value)
        self.logger.info("Lưu trữ lịch sử | history=%s", history_id)
        return updated

    def reopen(self, history_id: str) -> HistoryRecord:
        history = self.get_any(history_id)
        if history.status == HistoryStatus.ACTIVE.value:
            return history
        updated = self.repository.set_history_status(history_id, HistoryStatus.ACTIVE.value)
        self.logger.info("Mở lại lịch sử | history=%s", history_id)
        return updated

    def start_document(
        self,
        history_id: str,
        file_name: str,
        mime_type: str,
        size_bytes: int,
    ) -> DocumentRecord:
        self.require_active(history_id)
        active_documents = self.repository.count_documents(
            history_id,
            statuses=(DocumentStatus.PROCESSING.value, DocumentStatus.READY.value),
        )
        if active_documents >= self.max_documents:
            raise DocumentLimitError(
                f"Mỗi lịch sử chỉ được tải tối đa {self.max_documents} tài liệu"
            )
        document = self.repository.create_document(
            history_id, file_name, mime_type, size_bytes
        )
        self.repository.touch_history(history_id)
        return document

    def list_documents(self, history_id: str) -> list[DocumentRecord]:
        self.get_any(history_id)
        return self.repository.list_documents(history_id)

    def delete_document(self, history_id: str, document_id: str) -> None:
        self.require_active(history_id)
        document = self.repository.get_document(document_id)
        if document is None or document.history_id != history_id:
            raise DocumentNotFoundError("Không tìm thấy tài liệu trong phiên")
        self.vector_repository.delete_document(history_id, document_id)
        self.repository.delete_document(history_id, document_id)
        self.logger.info(
            "Xóa tài liệu | history=%s document=%s", history_id, document_id
        )

    def clear_documents(self, history_id: str) -> None:
        self.require_active(history_id)
        for document in self.repository.list_documents(history_id):
            self.vector_repository.delete_document(history_id, document.id)
            self.repository.delete_document(history_id, document.id)

    def delete_history(self, history_id: str) -> None:
        if self.repository.get_history(history_id) is None:
            return

        self.repository.soft_delete_history(history_id)

        self.logger.info(
            "Soft delete lịch sử | history=%s",
            history_id,
        )

    def list_deleted_histories(
        self, *, limit: int = 100, offset: int = 0
    ) -> list[HistoryRecord]:
        return self.repository.list_deleted_histories(limit=limit, offset=offset)

    def restore_history(self, history_id: str) -> HistoryRecord:
        restored = self.repository.restore_history(history_id)
        if restored is None:
            raise HistoryNotFoundError("Không tìm thấy lịch sử đã xóa")
        self.logger.info("Khôi phục lịch sử | history=%s", history_id)
        return restored

    def add_message(
        self, history_id: str, role: MessageRole | str, content: str
    ) -> MessageRecord:
        self.require_active(history_id)
        return self.repository.add_message(history_id, role, content)

    def recent_messages(self, history_id: str, limit: int) -> list[MessageRecord]:
        self.require_active(history_id)
        return self.repository.list_messages(history_id, limit=limit)

    def message_count(self, history_id: str) -> int:
        """Đếm số tin nhắn hiện có, để xác định đây có phải câu hỏi đầu tiên không."""
        self.get_any(history_id)
        return len(self.repository.list_messages(history_id))

    def rename(self, history_id: str, title: str) -> HistoryRecord:
        """Cập nhật title sau khi sinh được title từ câu hỏi đầu tiên."""
        updated = self.repository.touch_history(history_id, title=title)
        self.logger.info("Cập nhật title | history=%s title=%s", history_id, title)
        return updated
