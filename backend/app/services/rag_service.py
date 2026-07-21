from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.core.errors import NoReadyDocumentError
from app.domain.models import DocumentStatus, MessageRole, SearchResult
from app.repositories.session_repository import SessionRepository
from app.repositories.vector_repository import VectorRepository
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.session_service import SessionService


NOT_FOUND_ANSWER = (
    "Không tìm thấy thông tin liên quan trong các tài liệu của phiên để trả lời "
    "câu hỏi này."
)


@dataclass(frozen=True)
class RagSource:
    document_id: str
    file_name: str
    page_number: int
    page_end_number: int
    dieu: str | None
    score: float
    excerpt: str


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: tuple[RagSource, ...]
    retrieved_count: int


class RagService:
    def __init__(
        self,
        session_service: SessionService,
        session_repository: SessionRepository,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        generation_service: GenerationService,
        min_similarity: float,
        history_limit: int,
    ) -> None:
        self.session_service = session_service
        self.session_repository = session_repository
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.generation_service = generation_service
        self.min_similarity = min_similarity
        self.history_limit = history_limit

    @staticmethod
    def _jaccard(first: str, second: str) -> float:
        a, b = set(first.lower().split()), set(second.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _deduplicate(
        self, results: Sequence[SearchResult], top_k: int
    ) -> list[SearchResult]:
        kept: list[SearchResult] = []
        for result in sorted(results, key=lambda item: item.score, reverse=True):
            if result.score < self.min_similarity:
                continue
            if any(
                self._jaccard(result.chunk.text, item.chunk.text) >= 0.90
                for item in kept
            ):
                continue
            kept.append(result)
            if len(kept) >= top_k:
                break
        return kept

    @staticmethod
    def _format_context(results: Sequence[SearchResult]) -> str:
        sections: list[str] = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            pages = (
                f"trang {chunk.page_number}"
                if chunk.page_number == chunk.page_end_number
                else f"trang {chunk.page_number}-{chunk.page_end_number}"
            )
            heading = f"[Đoạn {index} | {chunk.file_name} | {pages}"
            if chunk.dieu:
                heading += f" | {chunk.dieu}"
            sections.append(f"{heading}]\n{chunk.text}")
        return "\n\n---\n\n".join(sections)

    def ask(self, session_id: str, question: str, top_k: int) -> RagAnswer:
        self.session_service.require_active(session_id)
        ready_count = self.session_repository.count_documents(
            session_id, statuses=(DocumentStatus.READY.value,)
        )
        if ready_count == 0:
            raise NoReadyDocumentError(
                "Phiên chưa có tài liệu đã lập chỉ mục thành công"
            )
        history_records = self.session_repository.list_messages(
            session_id, limit=self.history_limit
        )
        query_embedding = self.embedding_service.embed_query(question)
        raw_results = self.vector_repository.query(
            session_id, query_embedding, top_k
        )
        results = self._deduplicate(raw_results, top_k)
        if results:
            history = [(message.role, message.content) for message in history_records]
            answer = self.generation_service.generate(
                question, self._format_context(results), history
            )
        else:
            answer = NOT_FOUND_ANSWER
        self.session_repository.add_message(session_id, MessageRole.USER, question)
        self.session_repository.add_message(session_id, MessageRole.ASSISTANT, answer)
        self.session_repository.touch_session(
            session_id, self.session_service.ttl_minutes
        )
        sources = tuple(
            RagSource(
                document_id=result.chunk.document_id,
                file_name=result.chunk.file_name,
                page_number=result.chunk.page_number,
                page_end_number=result.chunk.page_end_number,
                dieu=result.chunk.dieu,
                score=round(result.score, 4),
                excerpt=result.chunk.text[:600],
            )
            for result in results
        )
        return RagAnswer(question, answer, sources, len(results))
