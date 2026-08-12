from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

from app.core.errors import RerankerServiceError
from app.domain.models import ChunkDetail, SearchResult
from app.repositories.vector_repository import VectorRepository
from app.services.embedding_service import EmbeddingService
from app.services.reranking_service import CrossEncoderReranker


TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass
class _Candidate:
    chunk: ChunkDetail
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


@dataclass(frozen=True)
class RetrievalOutcome:
    results: tuple[SearchResult, ...]
    trace: dict[str, object]


class HybridRetrievalService:
    """Dense Gemini + BM25, hợp nhất RRF rồi rerank bằng CrossEncoder."""

    def __init__(
        self,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        reranker: CrossEncoderReranker,
        candidate_k: int,
        rerank_candidate_k: int,
        rrf_k: int,
        min_dense_similarity: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self.vector_repository = vector_repository
        self.embedding_service = embedding_service
        self.reranker = reranker
        self.candidate_k = max(1, candidate_k)
        self.rerank_candidate_k = max(1, rerank_candidate_k)
        self.rrf_k = max(1, rrf_k)
        self.min_dense_similarity = min_dense_similarity
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def _bm25(
        self,
        chunks: Sequence[ChunkDetail],
        query: str,
    ) -> list[SearchResult]:
        query_tokens = self.tokenize(query)
        if not chunks or not query_tokens:
            return []
        tokenized_corpus = [self.tokenize(chunk.text) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(chunks)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        results: list[SearchResult] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            results.append(SearchResult(chunks[index], score))
            if len(results) >= self.candidate_k:
                break
        return results

    def _rrf(
        self,
        dense_results: Sequence[SearchResult],
        bm25_results: Sequence[SearchResult],
    ) -> list[_Candidate]:
        candidates: dict[str, _Candidate] = {}
        for rank, result in enumerate(dense_results, start=1):
            item = candidates.setdefault(
                result.chunk.element_id, _Candidate(result.chunk)
            )
            item.dense_score = result.score
            item.rrf_score += 1.0 / (self.rrf_k + rank)
        for rank, result in enumerate(bm25_results, start=1):
            item = candidates.setdefault(
                result.chunk.element_id, _Candidate(result.chunk)
            )
            item.bm25_score = result.score
            item.rrf_score += 1.0 / (self.rrf_k + rank)
        return sorted(
            candidates.values(),
            key=lambda item: item.rrf_score,
            reverse=True,
        )

    def retrieve(
        self,
        history_id: str,
        original_query: str,
        dense_query: str,
        top_k: int,
    ) -> RetrievalOutcome:
        started = time.perf_counter()
        warnings: list[str] = []

        dense_started = time.perf_counter()
        query_embedding = self.embedding_service.embed_query(dense_query)
        dense_results = [
            result
            for result in self.vector_repository.query(
                history_id, query_embedding, self.candidate_k
            )
            if result.score >= self.min_dense_similarity
        ]
        dense_ms = (time.perf_counter() - dense_started) * 1000

        bm25_started = time.perf_counter()
        chunks = self.vector_repository.list_chunks(history_id)
        # Giữ câu hỏi gốc cho BM25 để không làm loãng từ khóa pháp lý chính xác.
        bm25_results = self._bm25(chunks, original_query)
        bm25_ms = (time.perf_counter() - bm25_started) * 1000

        fused_started = time.perf_counter()
        fused = self._rrf(dense_results, bm25_results)
        rerank_pool = fused[: self.rerank_candidate_k]
        fused_results = [
            SearchResult(item.chunk, item.rrf_score) for item in rerank_pool
        ]
        fusion_ms = (time.perf_counter() - fused_started) * 1000

        rerank_started = time.perf_counter()
        try:
            ranked = self.reranker.rerank(
                original_query,
                fused_results,
                top_k=max(1, top_k),
            )
        except RerankerServiceError as exc:
            warning = (
                "CrossEncoder không khả dụng; đã dùng thứ tự RRF: "
                f"{exc.message}"
            )
            warnings.append(warning)
            self.logger.warning(warning)
            ranked = fused_results[: max(1, top_k)]
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

        rerank_scores = {
            result.chunk.element_id: result.score for result in ranked
        }
        candidate_details: list[dict[str, object]] = []
        for item in rerank_pool:
            item.rerank_score = rerank_scores.get(item.chunk.element_id)
            candidate_details.append(
                {
                    "element_id": item.chunk.element_id,
                    "file_name": item.chunk.file_name,
                    "page_number": item.chunk.page_number,
                    "dense_score": (
                        round(item.dense_score, 6)
                        if item.dense_score is not None
                        else None
                    ),
                    "bm25_score": (
                        round(item.bm25_score, 6)
                        if item.bm25_score is not None
                        else None
                    ),
                    "rrf_score": round(item.rrf_score, 6),
                    "rerank_score": (
                        round(item.rerank_score, 6)
                        if item.rerank_score is not None
                        else None
                    ),
                }
            )

        total_ms = (time.perf_counter() - started) * 1000
        trace: dict[str, object] = {
            "queries": {
                "original": original_query,
                "dense_hyde": dense_query,
                "bm25": original_query,
                "rerank": original_query,
            },
            "candidate_counts": {
                "session_chunks": len(chunks),
                "dense": len(dense_results),
                "bm25": len(bm25_results),
                "rrf": len(fused),
                "rerank_pool": len(rerank_pool),
                "selected": len(ranked),
            },
            "timings_ms": {
                "dense": round(dense_ms, 3),
                "bm25": round(bm25_ms, 3),
                "rrf": round(fusion_ms, 3),
                "rerank": round(rerank_ms, 3),
                "retrieval_total": round(total_ms, 3),
            },
            "candidates": candidate_details,
            "warnings": warnings,
        }
        self.logger.info(
            "Hybrid retrieval | session=%s dense=%s bm25=%s fused=%s final=%s",
            history_id,
            len(dense_results),
            len(bm25_results),
            len(fused),
            len(ranked),
        )
        return RetrievalOutcome(tuple(ranked), trace)
