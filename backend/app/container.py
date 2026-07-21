from __future__ import annotations

from dataclasses import dataclass

from app.clients.ollama_client import OllamaClient
from app.core.config import Settings
from app.core.logging import get_logger
from app.repositories.session_repository import SessionRepository
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import LegalChunkingService
from app.services.document_parser import DocumentParser
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.indexing_service import IndexingService
from app.services.rag_service import RagService
from app.services.session_service import SessionService


@dataclass(frozen=True)
class AppContainer:
    settings: Settings
    session_repository: SessionRepository
    vector_repository: VectorRepository
    ollama_client: OllamaClient
    embedding_service: EmbeddingService
    generation_service: GenerationService
    session_service: SessionService
    indexing_service: IndexingService
    rag_service: RagService


def build_container(settings: Settings) -> AppContainer:
    sessions = SessionRepository(settings.sqlite_path)
    vectors = VectorRepository(
        settings.chroma_persist_dir,
        settings.chroma_collection_name,
        settings.embedding_model,
        settings.embedding_provider,
    )
    ollama = OllamaClient(
        settings.ollama_base_url,
        settings.ollama_timeout_seconds,
        settings.ollama_retry_count,
        settings.ollama_retry_delay_seconds,
    )
    embeddings = EmbeddingService(
        ollama,
        settings.embedding_model,
        settings.embedding_batch_size,
    )
    generation = GenerationService(
        ollama,
        settings.generation_model,
        settings.generation_temperature,
        settings.generation_max_tokens,
    )
    session_service = SessionService(
        sessions,
        vectors,
        settings.session_ttl_minutes,
        settings.max_session_documents,
        get_logger("slaw.session"),
    )
    indexing = IndexingService(
        session_service,
        sessions,
        vectors,
        DocumentParser(),
        LegalChunkingService(
            settings.chunk_size_chars,
            settings.chunk_overlap_chars,
            get_logger("slaw.chunking"),
        ),
        embeddings,
        get_logger("slaw.indexing"),
    )
    rag = RagService(
        session_service,
        sessions,
        vectors,
        embeddings,
        generation,
        settings.min_similarity,
        settings.history_message_limit,
    )
    return AppContainer(
        settings=settings,
        session_repository=sessions,
        vector_repository=vectors,
        ollama_client=ollama,
        embedding_service=embeddings,
        generation_service=generation,
        session_service=session_service,
        indexing_service=indexing,
        rag_service=rag,
    )
