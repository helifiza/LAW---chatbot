from __future__ import annotations

from dataclasses import dataclass

from app.clients.gemini_client import GeminiClient
from app.core.config import Settings
from app.core.logging import get_logger
from app.repositories.history_repository import HistoryRepository
from app.repositories.vector_repository import VectorRepository
from app.services.chunking_service import LegalChunkingService
from app.services.document_parser import DocumentParser
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.services.indexing_service import IndexingService
from app.services.query_rewrite_service import QueryRewriteService
from app.services.rag_service import RagService
from app.services.reranking_service import CrossEncoderReranker
from app.services.history_service import HistoryService
from app.services.auth_service import AuthService
from app.repositories.user_repository import UserRepository



@dataclass(frozen=True)
class AppContainer:
    settings: Settings

    history_repository: HistoryRepository
    user_repository: UserRepository
    vector_repository: VectorRepository

    gemini_client: GeminiClient

    embedding_service: EmbeddingService
    generation_service: GenerationService
    query_rewrite_service: QueryRewriteService
    reranker: CrossEncoderReranker
    retrieval_service: HybridRetrievalService
    history_service: HistoryService
    indexing_service: IndexingService
    rag_service: RagService
    auth_service: AuthService


def build_container(settings: Settings) -> AppContainer:
    histories = HistoryRepository(settings.sqlite_path)

    users = UserRepository(settings.sqlite_path)

    auth = AuthService(
        user_repository=users,
        jwt_secret=settings.jwt_secret,
        jwt_algorithm=settings.jwt_algorithm,
    )
    vectors = VectorRepository(
        settings.chroma_persist_dir,
        settings.chroma_collection_name,
        settings.embedding_model,
        settings.embedding_provider,
        settings.embedding_dimension,
    )
    gemini = GeminiClient(
        settings.gemini_api_key,
        settings.gemini_retry_count,
        settings.gemini_retry_delay_seconds,
    )
    embeddings = EmbeddingService(
        gemini,
        settings.embedding_model,
        settings.embedding_batch_size,
        settings.embedding_dimension,
    )
    generation = GenerationService(
        gemini,
        settings.generation_model,
        settings.generation_temperature,
        settings.generation_max_tokens,
    )
    query_rewrite = QueryRewriteService(
        gemini,
        settings.generation_model,
        settings.hyde_enabled,
        settings.hyde_temperature,
        settings.hyde_max_tokens,
        get_logger("slaw.query_rewrite"),
    )
    reranker = CrossEncoderReranker(
        settings.rerank_model,
        settings.rerank_device,
        settings.rerank_batch_size,
        settings.rerank_max_length,
    )
    retrieval = HybridRetrievalService(
        vectors,
        embeddings,
        reranker,
        settings.retrieval_candidate_k,
        settings.rerank_candidate_k,
        settings.rrf_k,
        settings.min_similarity,
        get_logger("slaw.retrieval"),
    )
    history_service = HistoryService(
        histories,
        vectors,
        get_logger("slaw.history"),
    )
    indexing = IndexingService(
        history_service,
        histories,
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
        history_service,
        histories,
        query_rewrite,
        retrieval,
        generation,
        settings.history_message_limit,
    )
    return AppContainer(
        settings=settings,

        history_repository=histories,
        user_repository=users,
        vector_repository=vectors,

        gemini_client=gemini,

        embedding_service=embeddings,
        generation_service=generation,
        query_rewrite_service=query_rewrite,
        reranker=reranker,
        retrieval_service=retrieval,
        history_service=history_service,
        indexing_service=indexing,
        rag_service=rag,
        auth_service=auth,
    )
