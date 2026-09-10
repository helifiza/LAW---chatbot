from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import replace

from app.core.errors import DocumentIndexingError
from app.domain.models import DocumentRecord, DocumentStatus, GraphStatus
from app.repositories.history_repository import HistoryRepository
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import LegalChunkingService
from app.services.document_parser import DocumentParser
from app.services.embedding_service import EmbeddingService
from app.services.history_service import HistoryService
from app.services.classification import LinhVucClassificationService
from app.services.legal_graph_service import LegalGraphService


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
        linh_vuc_classification_service: LinhVucClassificationService | None = None,
        legal_graph_service: LegalGraphService | None = None,
    ) -> None:
        self.history_service = history_service
        self.history_repository = history_repository
        self.vector_repository = vector_repository
        self.parser = parser
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.linh_vuc_classification_service = linh_vuc_classification_service
        self.legal_graph_service = legal_graph_service
        self.logger = logger or logging.getLogger(__name__)

    def index_file(
        self,
        history_id: str,
        user_id: str,
        temp_path: Path,
        original_file_name: str,
        mime_type: str,
        size_bytes: int,
        document: DocumentRecord | None = None,
    ) -> DocumentRecord:
        document = document or self.history_service.start_document(
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
            linh_vuc = None
            if self.linh_vuc_classification_service is not None:
                linh_vuc = self.linh_vuc_classification_service.classify(
                    original_file_name,
                    [chunk.text for chunk in chunks],
                )
                if linh_vuc:
                    chunks = [replace(chunk, linh_vuc=linh_vuc) for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(
                [chunk.text for chunk in chunks]
            )
            self.vector_repository.upsert(chunks, embeddings)
            if linh_vuc is not None:
                self.history_repository.update_document_linh_vuc(
                    document.id,
                    linh_vuc,
                )
            if self.legal_graph_service is not None:
                self.history_repository.update_document_graph_status(
                    document.id, GraphStatus.PROCESSING.value
                )
                try:
                    legal_document_id = self.legal_graph_service.ingest(
                        history_id=history_id, upload_document_id=document.id,
                        file_path=temp_path, file_name=original_file_name, chunks=chunks,
                        primary_linh_vuc_code=linh_vuc,
                    )
                    self._sync_document_type(history_id, document.id, legal_document_id)
                    self.history_repository.update_document_graph_status(
                        document.id, GraphStatus.READY.value
                    )
                except Exception as graph_exc:
                    self.history_repository.update_document_graph_status(
                        document.id, GraphStatus.FAILED.value, str(graph_exc)
                    )
                    self.logger.exception(
                        "Legal graph failed but vector index remains usable | document=%s",
                        document.id,
                    )
            else:
                self.history_repository.update_document_graph_status(
                    document.id, GraphStatus.SKIPPED.value
                )
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
            if self.legal_graph_service is not None:
                self._reconcile_existing_documents(history_id, document.id)
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

    def _reconcile_existing_documents(self, history_id: str, new_document_id: str) -> None:
        if self.legal_graph_service is None:
            return
        for document in self.history_repository.list_documents(history_id):
            if document.id == new_document_id or document.status != DocumentStatus.READY.value:
                continue
            if not self.legal_graph_service.repository.is_latest_version(document.id):
                continue
            chunks = self.vector_repository.list_chunks_by_documents(history_id, [document.id])
            if not chunks:
                continue
            self.history_repository.update_document_graph_status(
                document.id, GraphStatus.PROCESSING.value
            )
            try:
                legal_document_id = self.legal_graph_service.ingest(
                    history_id=history_id, upload_document_id=document.id, file_path=None,
                    file_name=document.file_name, chunks=chunks,
                    primary_linh_vuc_code=document.linh_vuc,
                    file_checksum=self.legal_graph_service.repository.get_version_checksum(document.id),
                )
                self._sync_document_type(history_id, document.id, legal_document_id)
                self.history_repository.update_document_graph_status(
                    document.id, GraphStatus.READY.value
                )
            except Exception as exc:
                self.history_repository.update_document_graph_status(
                    document.id, GraphStatus.FAILED.value, str(exc)
                )
                self.logger.exception("Legal graph reconciliation failed | document=%s", document.id)

    def rebuild_graph(self, history_id: str, document: DocumentRecord) -> None:
        if self.legal_graph_service is None:
            raise DocumentIndexingError("Legal graph service is not configured")
        if not self.legal_graph_service.repository.is_latest_version(document.id):
            raise DocumentIndexingError(
                "Chỉ có thể rebuild phiên bản mới nhất của canonical legal document"
            )
        chunks = self.vector_repository.list_chunks_by_documents(history_id, [document.id])
        if not chunks:
            raise DocumentIndexingError("Document has no indexed chunks")
        self.history_repository.update_document_graph_status(document.id, GraphStatus.PROCESSING.value)
        try:
            legal_document_id = self.legal_graph_service.ingest(
                history_id=history_id, upload_document_id=document.id, file_path=None,
                file_name=document.file_name, chunks=chunks,
                primary_linh_vuc_code=document.linh_vuc,
                file_checksum=self.legal_graph_service.repository.get_version_checksum(document.id),
            )
            self._sync_document_type(history_id, document.id, legal_document_id)
            self.history_repository.update_document_graph_status(document.id, GraphStatus.READY.value)
        except Exception as exc:
            self.history_repository.update_document_graph_status(document.id, GraphStatus.FAILED.value, str(exc))
            raise

    def _sync_document_type(
        self,
        history_id: str,
        upload_document_id: str,
        legal_document_id: str | None,
    ) -> None:
        if self.legal_graph_service is None or legal_document_id is None:
            return
        legal_document = self.legal_graph_service.repository.get_document(legal_document_id)
        self.vector_repository.update_document_metadata(
            history_id,
            upload_document_id,
            document_type=(legal_document or {}).get("document_type"),
        )
