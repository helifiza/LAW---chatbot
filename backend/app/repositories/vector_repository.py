from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import chromadb

from app.domain.models import ChunkDetail, SearchResult


class VectorRepository:
    """Bao toàn bộ thao tác Chroma để dễ thay vector database sau này."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_model: str,
        embedding_provider: str = "ollama",
    ) -> None:
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
        self._lock = threading.RLock()
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": embedding_model,
                "embedding_provider": embedding_provider,
            },
        )
        collection_metadata = self._collection.metadata or {}
        stored_model = collection_metadata.get("embedding_model")
        stored_provider = collection_metadata.get("embedding_provider")
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

    @staticmethod
    def _metadata(chunk: ChunkDetail) -> dict[str, object]:
        return {
            "session_id": chunk.session_id,
            "document_id": chunk.document_id,
            "file_name": chunk.file_name,
            "page_number": chunk.page_number,
            "page_end_number": chunk.page_end_number,
            "content_type": chunk.content_type,
            "chuong": chunk.chuong or "",
            "muc": chunk.muc or "",
            "dieu": chunk.dieu or "",
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
            session_id=str(metadata["session_id"]),
            document_id=str(metadata["document_id"]),
            element_id=element_id,
            file_name=str(metadata["file_name"]),
            page_number=int(metadata["page_number"]),
            page_end_number=int(metadata["page_end_number"]),
            content_type=str(metadata["content_type"]),
            chuong=str(metadata.get("chuong") or "") or None,
            muc=str(metadata.get("muc") or "") or None,
            dieu=str(metadata.get("dieu") or "") or None,
            text=text,
            chunk_index=int(metadata["chunk_index"]),
            char_count=int(metadata.get("char_count") or len(text)),
            token_count=int(metadata.get("token_count") or 0),
            created_at=str(metadata.get("created_at") or ""),
        )

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
        session_id: str,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[SearchResult]:
        where = {"session_id": session_id}
        with self._lock:
            matching = self._collection.get(where=where, include=[])
            matching_count = len(matching.get("ids") or [])
            if matching_count == 0:
                return []
            result = self._collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=min(max(top_k * 2, top_k), matching_count),
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

    def delete_document(self, session_id: str, document_id: str) -> None:
        with self._lock:
            self._collection.delete(
                where={
                    "$and": [
                        {"session_id": session_id},
                        {"document_id": document_id},
                    ]
                }
            )

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._collection.delete(where={"session_id": session_id})

    def count(self) -> int:
        with self._lock:
            return self._collection.count()
