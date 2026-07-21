from __future__ import annotations

import logging
from pathlib import Path

from app.core.errors import DocumentIndexingError
from app.domain.models import DocumentRecord, DocumentStatus
from app.repositories.session_repository import SessionRepository
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import LegalChunkingService
from app.services.document_parser import DocumentParser
from app.services.embedding_service import EmbeddingService
from app.services.session_service import SessionService


class IndexingService:
    """Điều phối parse -> chunk -> embedding -> Chroma cho một file."""

    def __init__(
        self,
        session_service: SessionService,
        session_repository: SessionRepository,
        vector_repository: VectorRepository,
        parser: DocumentParser,
        chunker: LegalChunkingService,
        embedding_service: EmbeddingService,
        logger: logging.Logger | None = None,
    ) -> None:
        self.session_service = session_service
        self.session_repository = session_repository
        self.vector_repository = vector_repository
        self.parser = parser
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.logger = logger or logging.getLogger(__name__)

    def index_file(
        self,
        session_id: str,
        temp_path: Path,
        original_file_name: str,
        mime_type: str,
        size_bytes: int,
    ) -> DocumentRecord:
        document = self.session_service.start_document(
            session_id, original_file_name, mime_type, size_bytes
        )
        try:
            self.logger.info(
                "Bắt đầu indexing | session=%s document=%s file=%s",
                session_id,
                document.id,
                original_file_name,
            )
            pages = self.parser.parse(temp_path, original_file_name)
            chunks = self.chunker.chunk_pages(
                session_id=session_id,
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
            ready = self.session_repository.update_document_status(
                document.id, DocumentStatus.READY.value, chunk_count=len(chunks)
            )
            if ready is None:
                raise RuntimeError("Không cập nhật được trạng thái tài liệu")
            self.logger.info(
                "Indexing hoàn tất | session=%s document=%s chunks=%s",
                session_id,
                document.id,
                len(chunks),
            )
            return ready
        except Exception as exc:
            self.vector_repository.delete_document(session_id, document.id)
            self.session_repository.update_document_status(
                document.id,
                DocumentStatus.FAILED.value,
                chunk_count=0,
                error_message=str(exc),
            )
            self.logger.exception(
                "Indexing thất bại | session=%s document=%s",
                session_id,
                document.id,
            )
            if isinstance(exc, DocumentIndexingError):
                raise
            raise DocumentIndexingError(
                f"Không thể lập chỉ mục {original_file_name}: {exc}"
            ) from exc
