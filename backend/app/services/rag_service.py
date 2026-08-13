from __future__ import annotations

import time
import json
from dataclasses import dataclass
from typing import Sequence

from app.core.errors import NoReadyDocumentError
from app.domain.models import DocumentStatus, MessageRole, SearchResult
from app.repositories.history_repository import HistoryRepository
from app.services.generation_service import GenerationService
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.services.query_rewrite_service import QueryRewriteService
from app.services.history_service import HistoryService


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
    trace: dict[str, object] | None = None


class RagService:
    def __init__(
        self,
        history_service: HistoryService,
        history_repository: HistoryRepository,
        query_rewrite_service: QueryRewriteService,
        retrieval_service: HybridRetrievalService,
        generation_service: GenerationService,
        history_limit: int,
    ) -> None:
        self.history_service = history_service
        self.history_repository = history_repository
        self.query_rewrite_service = query_rewrite_service
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
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

    def ask(
        self,
        history_id: str,
        question: str,
        top_k: int,
        include_trace: bool = False,
    ) -> RagAnswer:
        request_started = time.perf_counter()
        self.history_service.require_active(history_id)
        ready_count = self.history_repository.count_documents(
            history_id, statuses=(DocumentStatus.READY.value,)
        )
        if ready_count == 0:
            raise NoReadyDocumentError(
                "Phiên chưa có tài liệu đã lập chỉ mục thành công"
            )
        history_records = self.history_repository.list_messages(
            history_id, limit=self.history_limit
        )
        history = [
            (message.role, message.content) for message in history_records
        ]

        rewrite_started = time.perf_counter()
        rewrite = self.query_rewrite_service.rewrite(question, history)
        rewrite_ms = (time.perf_counter() - rewrite_started) * 1000

        retrieval = self.retrieval_service.retrieve(
            history_id=history_id,
            original_query=question,
            dense_query=rewrite.retrieval_query,
            top_k=max(top_k * 2, top_k),
        )
        results = self._deduplicate(retrieval.results, top_k)

        generation_ms = 0.0
        if results:
            generation_started = time.perf_counter()
            answer = self.generation_service.generate(
                question, self._format_context(results), history
            )
            generation_ms = (time.perf_counter() - generation_started) * 1000
        else:
            answer = NOT_FOUND_ANSWER

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
        self.history_repository.add_message(history_id, MessageRole.USER, question)
        self.history_repository.add_message(
            history_id,
            MessageRole.ASSISTANT,
            answer,
            sources = json.dumps([s.__dict__ for s in sources], ensure_ascii=False) if sources else None,
        )
        self.history_repository.touch_history(history_id)

        trace: dict[str, object] | None = None
        if include_trace:
            trace = dict(retrieval.trace)
            warnings = list(trace.get("warnings") or [])
            if rewrite.warning:
                warnings.insert(0, rewrite.warning)
            trace["warnings"] = warnings
            trace["rewrite"] = {
                "strategy": "hyde" if rewrite.used_hyde else "original_query",
                "used_hyde": rewrite.used_hyde,
                "model": self.query_rewrite_service.model,
                "time_ms": round(rewrite_ms, 3),
            }
            trace["generation"] = {
                "model": self.generation_service.model,
                "context_element_ids": [
                    result.chunk.element_id for result in results
                ],
                "time_ms": round(generation_ms, 3),
            }
            trace["total_time_ms"] = round(
                (time.perf_counter() - request_started) * 1000, 3
            )
        return RagAnswer(question, answer, sources, len(results), trace)
