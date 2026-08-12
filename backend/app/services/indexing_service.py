from __future__ import annotations

import logging
from pathlib import Path

from app.core.errors import DocumentIndexingError
from app.domain.models import DocumentRecord, DocumentStatus
from app.repositories.history_repository import HistoryRepository
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import LegalChunkingService
from app.services.document_parser import DocumentParser
from app.services.embedding_service import EmbeddingService
from app.services.history_service import HistoryService


class IndexingService:
    """Điều phối parse -> chunk -> embedding -> Chroma cho một file."""

    def __init__(
        self,
        history_service: HistoryService,
        history_repository: HistoryRepository,
        vector_repository: VectorRepository,
        parser: DocumentParser,
        chunker: LegalChunkingService,
        embedding_service: EmbeddingService,
        logger: logging.Logger | None = None,
    ) -> None:
        self.history_service = history_service
        self.history_repository = history_repository
        self.vector_repository = vector_repository
        self.parser = parser
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.logger = logger or logging.getLogger(__name__)

    def index_file(
        self,
        history_id: str,
        user_id: str,
        temp_path: Path,
        original_file_name: str,
        mime_type: str,
        size_bytes: int,
    ) -> DocumentRecord:
        document = self.history_service.start_document(
            history_id, original_file_name, mime_type, size_bytes
        )
        try:
            self.logger.info(
                "Bắt đầu indexing | history=%s document=%s file=%s",
                history_id,
                document.id,
                original_file_name,
            )
            pages = self.parser.parse(temp_path, original_file_name)
            chunks = self.chunker.chunk_pages(
                history_id=history_id,
                user_id = user_id,
                document_id=document.id,
                file_name=original_file_name,
                pages=pages,
                remove_bare_page_numbers=(
                    Path(original_file_name).suffix.lower() == ".pdf"
                ),
            )
            if not chunks:
                raise DocumentIndexingError("Không tạo được chunk từ tài liệu")
            embeddings = self.embedding_service.embed_texts(
                [chunk.text for chunk in chunks]
            )
            self.vector_repository.upsert(chunks, embeddings)
            ready = self.history_repository.update_document_status(
                document.id, DocumentStatus.READY.value, chunk_count=len(chunks)
            )
            if ready is None:
                raise RuntimeError("Không cập nhật được trạng thái tài liệu")
            self.logger.info(
                "Indexing hoàn tất | history=%s document=%s chunks=%s",
                history_id,
                document.id,
                len(chunks),
            )
            return ready
        except Exception as exc:
            self.vector_repository.delete_document(history_id, document.id)
            self.history_repository.update_document_status(
                document.id,
                DocumentStatus.FAILED.value,
                chunk_count=0,
                error_message=str(exc),
            )
            self.logger.exception(
                "Indexing thất bại | history=%s document=%s",
                history_id,
                document.id,
            )
            if isinstance(exc, DocumentIndexingError):
                raise
            raise DocumentIndexingError(
                f"Không thể lập chỉ mục {original_file_name}: {exc}"
            ) from exc
