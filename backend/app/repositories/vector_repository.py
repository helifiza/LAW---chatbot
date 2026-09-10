from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import chromadb

from app.domain.models import ChunkDetail, SearchResult
from app.domain.legal_document import normalize_document_type, normalize_document_type_filter


class VectorRepository:
    """Bao toàn bộ thao tác Chroma để dễ thay vector database sau này."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_model: str,
        embedding_provider: str = "gemini",
        embedding_dimension: int | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self.embedding_dimension = embedding_dimension
        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        metadata: dict[str, object] = {
            "hnsw:space": "cosine",
            "embedding_model": embedding_model,
            "embedding_provider": embedding_provider,
        }
        if embedding_dimension is not None:
            metadata["embedding_dimension"] = embedding_dimension
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
        )
        collection_metadata = self._collection.metadata or {}
        stored_model = collection_metadata.get("embedding_model")
        stored_provider = collection_metadata.get("embedding_provider")
        stored_dimension = collection_metadata.get("embedding_dimension")
        if stored_model and stored_model != embedding_model:
            raise ValueError(
                "Collection đang dùng embedding model "
                f"{stored_model}, không thể truy vấn bằng {embedding_model}. "
                "Hãy đổi CHROMA_COLLECTION_NAME hoặc embedding lại dữ liệu."
            )
        if stored_provider and stored_provider != embedding_provider:
            raise ValueError(
                "Collection đang dùng embedding provider "
                f"{stored_provider}, không thể dùng {embedding_provider}. "
                "Hãy đổi CHROMA_COLLECTION_NAME hoặc embedding lại dữ liệu."
            )
        if (
            stored_dimension
            and embedding_dimension is not None
            and int(stored_dimension) != embedding_dimension
        ):
            raise ValueError(
                "Collection đang dùng embedding "
                f"{stored_dimension} chiều, không thể dùng {embedding_dimension} chiều. "
                "Hãy đổi CHROMA_COLLECTION_NAME hoặc embedding lại dữ liệu."
            )

    def close(self) -> None:
        """Release Chroma file handles, especially required on Windows."""
        self._client.close()

    @staticmethod
    def _metadata(chunk: ChunkDetail) -> dict[str, object]:
        return {
            "history_id": chunk.history_id,
            "document_id": chunk.document_id,
            "user_id": chunk.user_id,
            "file_name": chunk.file_name,
            "page_number": chunk.page_number,
            "page_end_number": chunk.page_end_number,
            "content_type": chunk.content_type,
            "chuong": chunk.chuong or "",
            "muc": chunk.muc or "",
            "dieu": chunk.dieu or "",
            "linh_vuc": chunk.linh_vuc or "",
            "document_type": normalize_document_type(chunk.document_type) or "",
            "chunk_index": chunk.chunk_index,
            "char_count": chunk.char_count,
            "token_count": chunk.token_count,
            "created_at": chunk.created_at,
        }

    @staticmethod
    def _chunk_from_result(
        element_id: str,
        text: str,
        metadata: dict[str, object],
    ) -> ChunkDetail:
        return ChunkDetail(
            history_id=str(metadata["history_id"]),
            document_id=str(metadata["document_id"]),
            user_id=str(metadata["user_id"]),
            element_id=element_id,
            file_name=str(metadata["file_name"]),
            page_number=int(metadata["page_number"]),
            page_end_number=int(metadata["page_end_number"]),
            content_type=str(metadata["content_type"]),
            chuong=str(metadata.get("chuong") or "") or None,
            muc=str(metadata.get("muc") or "") or None,
            dieu=str(metadata.get("dieu") or "") or None,
            linh_vuc=str(metadata.get("linh_vuc") or "") or None,
            document_type=str(metadata.get("document_type") or "") or None,
            text=text,
            chunk_index=int(metadata["chunk_index"]),
            char_count=int(metadata.get("char_count") or len(text)),
            token_count=int(metadata.get("token_count") or 0),
            created_at=str(metadata.get("created_at") or ""),
        )

    @staticmethod
    def _build_where(
        history_id: str,
        document_ids: Sequence[str] | None = None,
        linh_vuc_filter: Sequence[str] | None = None,
        document_type_filter: Sequence[str] | None = None,
    ) -> dict[str, object]:
        conditions: list[dict[str, object]] = [{"history_id": history_id}]
        if document_ids:
            conditions.append({"document_id": {"$in": list(document_ids)}})
        if linh_vuc_filter:
            conditions.append({"linh_vuc": {"$in": list(linh_vuc_filter)}})
        normalized_types = normalize_document_type_filter(document_type_filter)
        if normalized_types:
            conditions.append({"document_type": {"$in": list(normalized_types)}})
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def upsert(
        self,
        chunks: Sequence[ChunkDetail],
        embeddings: Sequence[Sequence[float]],
        batch_size: int = 200,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Số chunk ({len(chunks)}) khác số embedding ({len(embeddings)})"
            )
        if not chunks:
            return 0
        with self._lock:
            for start in range(0, len(chunks), batch_size):
                batch_chunks = chunks[start : start + batch_size]
                batch_embeddings = embeddings[start : start + batch_size]
                self._collection.upsert(
                    ids=[chunk.element_id for chunk in batch_chunks],
                    embeddings=[list(vector) for vector in batch_embeddings],
                    documents=[chunk.text for chunk in batch_chunks],
                    metadatas=[self._metadata(chunk) for chunk in batch_chunks],
                )
        return len(chunks)

    def query(
        self,
        history_id: str,
        query_embedding: Sequence[float],
        top_k: int,
        document_ids: Sequence[str] | None = None,
        linh_vuc_filter: Sequence[str] | None = None,
        document_type_filter: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        where = self._build_where(
            history_id, document_ids, linh_vuc_filter, document_type_filter
        )
        with self._lock:
            matching = self._collection.get(where=where, include=[])
            matching_count = len(matching.get("ids") or [])
            if matching_count == 0:
                return []
            result = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=min(max(1, top_k), matching_count),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        search_results: list[SearchResult] = []
        for element_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            if text is None or metadata is None:
                continue
            search_results.append(
                SearchResult(
                    chunk=self._chunk_from_result(element_id, text, metadata),
                    score=1.0 - float(distance),
                )
            )
        return search_results

    def list_chunks(self, history_id: str) -> list[ChunkDetail]:
        with self._lock:
            result = self._collection.get(
                where={"history_id": history_id},
                include=["documents", "metadatas"],
            )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        chunks: list[ChunkDetail] = []
        for element_id, text, metadata in zip(ids, documents, metadatas):
            if text is None or metadata is None:
                continue
            chunks.append(self._chunk_from_result(element_id, text, metadata))
        return chunks

    def list_chunks_by_documents(
        self,
        history_id: str,
        document_ids: Sequence[str],
    ) -> list[ChunkDetail]:
        if not document_ids:
            return []
        where = {
            "$and": [
                {"history_id": history_id},
                {"document_id": {"$in": list(document_ids)}},
            ]
        }
        with self._lock:
            result = self._collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        chunks: list[ChunkDetail] = []
        for element_id, text, metadata in zip(ids, documents, metadatas):
            if text is None or metadata is None:
                continue
            chunks.append(self._chunk_from_result(element_id, text, metadata))
        chunks.sort(key=lambda c: (c.document_id, c.chunk_index))
        return chunks

    def update_document_metadata(
        self,
        history_id: str,
        document_id: str,
        *,
        document_type: str | None,
    ) -> int:
        """Patch chunk metadata without recomputing or replacing embeddings."""
        where = self._build_where(history_id, [document_id])
        with self._lock:
            result = self._collection.get(where=where, include=["metadatas"])
            ids = result.get("ids") or []
            metadatas = result.get("metadatas") or []
            if not ids:
                return 0
            normalized = normalize_document_type(document_type) or ""
            updated = [
                dict(metadata or {}, document_type=normalized)
                for metadata in metadatas
            ]
            self._collection.update(ids=ids, metadatas=updated)
        return len(ids)

    def delete_document(self, history_id: str, document_id: str) -> None:
        with self._lock:
            self._collection.delete(
                where={
                    "$and": [
                        {"history_id": history_id},
                        {"document_id": document_id},
                    ]
                }
            )

    def delete_history(self, history_id: str) -> None:
        with self._lock:
            self._collection.delete(where={"history_id": history_id})

    def count(self) -> int:
        with self._lock:
            return self._collection.count()
