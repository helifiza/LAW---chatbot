from __future__ import annotations

import re
import time
import json
import statistics
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from app.core.errors import NoReadyDocumentError
from app.domain.models import DocumentStatus, MessageRole, SearchResult
from app.repositories.history_repository import HistoryRepository
from app.repositories.vector_repository import VectorRepository
from app.services.generation_service import GenerationService
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.services.query_rewrite_service import QueryRewriteService
from app.services.history_service import HistoryService
from app.services.query_router_service import QueryRouterService, QueryRoute
from app.services.summary_service import SummarizeService

try:
    from rapidfuzz import fuzz

    def _token_similarity(a: str, b: str) -> float:
        return fuzz.token_set_ratio(a, b) / 100
except ImportError:  # fallback nếu chưa cài rapidfuzz
    def _token_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()


NOT_FOUND_ANSWER = (
    "Không tìm thấy thông tin liên quan trong các tài liệu của phiên để trả lời "
    "câu hỏi này."
)

ANTI_HALLUCINATION_ANSWER = (
    "Hệ thống không thể xác thực câu trả lời với nguồn tài liệu đã truy xuất, "
    "nên không thể cung cấp câu trả lời đáng tin cậy cho câu hỏi này. "
    "Vui lòng thử diễn đạt lại câu hỏi hoặc kiểm tra trực tiếp tài liệu."
)

# Chỉ giữ MỘT định nghĩa duy nhất cho các hằng số citation (bản trước có 2 lần
# định nghĩa CITATION_PATTERN, lần sau đè mất lần trước và làm mất regex nới
# rộng hỗ trợ "tr." / dấu gạch en-dash).
#
# Pattern hỗ trợ 2 định dạng citation:
#   - Theo trang:  [tên tài liệu, trang X]  hoặc  [tên tài liệu, trang X-Y]
#   - Theo Điều:   [tên tài liệu, Điều Z]   (chấp nhận cả "Dieu" không dấu)
# Dùng named groups (file/p1/p2/dieu) để phân biệt rõ 2 nhánh khi xử lý ở
# _validate_citations, tránh nhầm lẫn group theo thứ tự.
CITATION_PATTERN = re.compile(
    r"\[(?P<file>[^\[\],]+),\s*"
    r"(?:"
    r"(?:trang|tr\.?)\s*(?P<p1>\d+)\s*(?:[-–]\s*(?P<p2>\d+))?"
    r"|"
    r"(?:[Đđ]i[eề]u|[Dd]ieu)\s*(?P<dieu>\d+[A-Za-z]?)"
    r")\]"
)
CITATION_MATCH_THRESHOLD = 0.70          # ngưỡng khớp tên file (string/token)
CITATION_COVERAGE_THRESHOLD = 0.80       # % citation hợp lệ / tổng citation
SEMANTIC_Z_THRESHOLD = 1.0               # claim phải "nổi bật" >= 1 std so với các nguồn khác
SEMANTIC_MIN_CANDIDATES = 2              # cần ít nhất 2 nguồn mới tính z-score có ý nghĩa

#hàm chuẩn hóa tên file để dễ so sánh tìm kiếm và đối chiếu document, ví dụ Luật đất đai-2024.pdf sẽ thành luật đất đai 2024
def _normalize_filename(name: str) -> str:
    name = unicodedata.normalize("NFC", name)
    name = name.lower().strip()
    name = re.sub(r"\.(pdf|docx?|txt)$", "", name)
    name = re.sub(r"[_\-\.]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


#hàm chuẩn hóa giá trị Điều để so sánh trực tiếp giữa citation trong answer
#(vd "Điều 05", "Dieu 5", "điều 5a") và field `dieu` của RagSource, đưa về
#cùng 1 dạng số (+ hậu tố chữ nếu có), ví dụ "Điều 05" và "Dieu 5" đều thành "5".
def _normalize_dieu(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(\d+)\s*([A-Za-z]?)", value)
    if not match:
        return None
    number = match.group(1).lstrip("0") or "0"
    suffix = match.group(2).lower()
    return f"{number}{suffix}"


#hàm tính cosine similarity, để đo độ tương đồng giữa 2 vector a và b, cosine càng gần 1 thì càng có độ tương đồng cao
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


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
        embedding_client,          
        history_limit: int,
        query_router_service: QueryRouterService,
        vector_repository: VectorRepository,
        summarize_service: SummarizeService,
    ) -> None:
        self.history_service = history_service
        self.history_repository = history_repository
        self.query_rewrite_service = query_rewrite_service
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service
        self.embedding_client = embedding_client
        self.history_limit = history_limit
        self.query_router_service = query_router_service
        self.vector_repository = vector_repository
        self.summarize_service = summarize_service

    # ------------------------------------------------------------------
    # Dedup / format context
    # ------------------------------------------------------------------

#hàm jaccard này lấy giao là những từ xuất hiện ở cả 2 tập a và b, lấy hợp là tất cả các từ xuất hiện trong ít nhất một bên
    @staticmethod
    def _jaccard(first: str, second: str) -> float:#hàm đo độ giống nhau của 2 chunk, dùng Jaccard Similarity: |A giao B|/|A hợp B|
        a, b = set(first.lower().split()), set(second.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
#loại chunk trùng nhau bằng hàm jaccard, nếu độ giống nhau >= 0.90 thì coi là trùng nhau, chỉ giữ lại chunk có điểm cao hơn
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

    # ------------------------------------------------------------------
    # Citation validation (string match và semantic z-score)
    # ------------------------------------------------------------------
    #so tên file bằng hàm token_similarity
    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return _token_similarity(_normalize_filename(a), _normalize_filename(b))

    #hàm cắt câu chứa citattion trong câu trả lời của model để sau đó kiểm tra xem nguồn được citation có thật sự hỗ trợ nội dung mà LLM(gemini) vừa trả lời?
    @staticmethod
    def _extract_claim_sentence(answer: str, citation_start: int) -> str:
        """
        Lấy câu chứa citation, tính từ vị trí bắt đầu của dấu '[' lùi về đến
        dấu kết câu gần nhất trước đó (. ! ? hoặc xuống dòng), và tới dấu kết
        câu gần nhất sau citation. Dùng để so embedding thay vì so cả câu trả
        lời (tránh loãng ngữ nghĩa với các claim khác trong cùng answer).
        """
        left = max(
            answer.rfind(".", 0, citation_start),
            answer.rfind("\n", 0, citation_start),
            answer.rfind("!", 0, citation_start),
            answer.rfind("?", 0, citation_start),
        )
        left = left + 1 if left != -1 else 0

        end_search_start = citation_start
        right_candidates = [
            pos for pos in (
                answer.find(".", end_search_start),
                answer.find("\n", end_search_start),
                answer.find("!", end_search_start),
                answer.find("?", end_search_start),
            ) if pos != -1
        ]
        right = min(right_candidates) if right_candidates else len(answer)

        return answer[left:right].strip()

    def _embed_excerpts(self, sources: Sequence[RagSource]) -> list[list[float]]:
        """
        Embed excerpt của TẤT CẢ sources trong 1 lệnh gọi batch duy nhất cho
        mỗi request (EmbeddingService.embed_texts tự chia theo batch_size),
        để tái sử dụng cho mọi citation trong answer thay vì embed lại từng
        excerpt cho mỗi citation. Dùng task_type RETRIEVAL_DOCUMENT vì excerpt
        là nội dung tài liệu gốc.
        """
        return self.embedding_client.embed_texts(
            [source.excerpt for source in sources]
        )
    #so sánh độ tương đồng ngữ nghĩa bằng hàm cosine
    def _semantic_scores(
        self, claim_vec: Sequence[float], excerpt_vecs: Sequence[Sequence[float]]
    ) -> dict[int, float]:
        return {idx: _cosine(claim_vec, vec) for idx, vec in enumerate(excerpt_vecs)}

    def _validate_citations(
        self, answer: str, sources: Sequence[RagSource]
    ) -> dict[str, object]:
        """
        Trích các citation dạng [file, trang X-Y] hoặc [file, Điều Z] trong
        answer, so khớp MỜ (token-based, sau chuẩn hoá tên file) với tên file
        trong sources.

        - Citation dạng "trang": đối chiếu bằng overlap khoảng trang
          (page_number/page_end_number của source).
        - Citation dạng "Điều": đối chiếu trực tiếp field `dieu` của source
          (sau khi chuẩn hoá qua _normalize_dieu), không dùng page_overlap.

        Sau bước khớp tên/vị trí, xác thực thêm bằng semantic z-score giữa
        câu chứa citation và excerpt của nguồn khớp nhất. Câu trả lời chỉ
        được chấp nhận nếu tỷ lệ citation hợp lệ (matched / total) >=
        CITATION_COVERAGE_THRESHOLD. Logic này dùng chung cho cả nhánh
        SPECIFIC (_ask_specific, chỉ phát sinh citation dạng trang) và nhánh
        SUMMARY (_ask_summary, có thể phát sinh cả 2 dạng) — không cần đổi gì
        ở 2 hàm gọi vì chúng chỉ truyền answer/sources vào đây.
        """
        matched: list[dict[str, object]] = []
        unmatched: list[dict[str, object]] = []

        excerpt_vecs: list[list[float]] | None = None
        if len(sources) >= SEMANTIC_MIN_CANDIDATES:
            excerpt_vecs = self._embed_excerpts(sources)

        for match in CITATION_PATTERN.finditer(answer):
            file_cited = match.group("file")
            p1_str, p2_str = match.group("p1"), match.group("p2")
            dieu_str = match.group("dieu")
            is_dieu_citation = dieu_str is not None

            if is_dieu_citation:
                cited_dieu_norm = _normalize_dieu(dieu_str)
                p1 = p2 = None
            else:
                p1 = int(p1_str)
                p2 = int(p2_str) if p2_str else p1
                cited_dieu_norm = None

            # Check Bước 1: khớp tên file + vị trí (trang hoặc Điều tuỳ loại citation)
            best_name_score = 0.0
            best_source_idx: int | None = None
            for idx, source in enumerate(sources):
                name_score = self._similarity(file_cited, source.file_name)
                if is_dieu_citation:
                    # So trực tiếp field dieu của source, KHÔNG dùng page_overlap.
                    location_ok = (
                        cited_dieu_norm is not None
                        and _normalize_dieu(source.dieu) == cited_dieu_norm
                    )
                else:
                    location_ok = not (p2 < source.page_number or p1 > source.page_end_number)
                # Vị trí không khớp -> phạt điểm similarity một nửa, coi như
                # citation trỏ sai vị trí dù tên file đúng.
                effective_score = name_score if location_ok else name_score * 0.5
                if effective_score > best_name_score:
                    best_name_score = effective_score
                    best_source_idx = idx

            entry: dict[str, object] = {
                "cited_file": file_cited.strip(),
                "citation_type": "dieu" if is_dieu_citation else "trang",
                "cited_location": (
                    f"Điều {dieu_str.strip()}"
                    if is_dieu_citation
                    else (f"{p1}-{p2}" if p2 != p1 else str(p1))
                ),
                "match_score": round(best_name_score, 4),
                "matched_file": sources[best_source_idx].file_name if best_source_idx is not None else None,
                "semantic_score": None,
                "semantic_z": None,
                "semantic_supported": None,
            }

            name_page_ok = best_name_score >= CITATION_MATCH_THRESHOLD #nếu bước 1 độ chính xác >70% thì coi như đạt yêu câu bước 1

            # Check Bước 2: chỉ chạy semantic check nếu name/vị trí đã khớp
            if name_page_ok and best_source_idx is not None and excerpt_vecs is not None:
                claim = self._extract_claim_sentence(answer, match.start())
                # RETRIEVAL_QUERY vì claim đang đóng vai trò "truy vấn" xem
                # có khớp ngữ nghĩa với excerpt (RETRIEVAL_DOCUMENT) hay không.
                claim_vec = self.embedding_client.embed_query(claim)
                sem_scores = self._semantic_scores(claim_vec, excerpt_vecs)#tính điểm cosine similarity giữa claim và tất cả các excerpt

                target_score = sem_scores[best_source_idx]
                other_scores = [s for i, s in sem_scores.items() if i != best_source_idx]

                if other_scores:
                    mean_other = statistics.mean(other_scores)
                    stdev_other = statistics.pstdev(other_scores) or 1e-6
                    z = (target_score - mean_other) / stdev_other
                else:
                    z = float("inf")  # không có nền để so sánh -> không loại trừ được

                entry["semantic_score"] = round(target_score, 4)
                entry["semantic_z"] = round(z, 4)
                entry["semantic_supported"] = z >= SEMANTIC_Z_THRESHOLD

            is_valid = name_page_ok and entry["semantic_supported"] is not False
            (matched if is_valid else unmatched).append(entry)

        total = len(matched) + len(unmatched)
        coverage = (len(matched) / total) if total else 0.0 #tính toán xem số lượng citation answer do model đưa ra có > 80% không, nếu không thì câu trả lời đưa ra khômg đủ tin cậy => ANTI_HALLUCINATION

        return {
            "citation_count": total,
            "matched_citations": matched,
            "unmatched_citations": unmatched,
            "coverage": round(coverage, 4),
            "has_valid_citation": total > 0 and coverage >= CITATION_COVERAGE_THRESHOLD,
        }

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------
    def _ask_specific(
        self,
        history_id: str,
        question: str,
        top_k: int,
    )-> tuple[str, tuple[RagSource, ...], int, dict[str, object]]:
        history_records = self.history_repository.list_messages(
            history_id, limit=self.history_limit
        )
        history = [(message.role, message.content) for message in history_records]
 
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
        generation_ms = 0.0
        citation_validation: dict[str, object] | None = None
        if results:
            generation_started = time.perf_counter()
            raw_answer = self.generation_service.generate(
                question, self._format_context(results), history
            )
            generation_ms = (time.perf_counter() - generation_started) * 1000
            citation_validation = self._validate_citations(raw_answer, sources)
            if citation_validation["has_valid_citation"]:
                answer = raw_answer
            else:
                answer = ANTI_HALLUCINATION_ANSWER
        else:
            answer = NOT_FOUND_ANSWER
        branch_trace: dict[str, object] = dict(retrieval.trace)
        warnings = list(branch_trace.get("warnings") or [])
        if rewrite.warning:
            warnings.insert(0, rewrite.warning)
        branch_trace["warnings"] = warnings
        branch_trace["rewrite"] = {
            "strategy": "hyde" if rewrite.used_hyde else "original_query",
            "used_hyde": rewrite.used_hyde,
            "model": self.query_rewrite_service.model,
            "time_ms": round(rewrite_ms, 3),
        }
        branch_trace["generation"] = {
            "model": self.generation_service.model,
            "context_element_ids": [result.chunk.element_id for result in results],
            "time_ms": round(generation_ms, 3),
        }
        if citation_validation is not None:
            branch_trace["citation_validation"] = citation_validation
 
        return answer, sources, len(results), branch_trace

    def _ask_summary(
            self,
            history_id: str,
            question: str,
    )-> tuple[str, tuple[RagSource, ...], int, dict[str, object]]:
        summary_result = self.summarize_service.run(history_id, question)
        raw_answer = summary_result.answer
        sources = tuple(
            RagSource(
                document_id=h.document_id,
                file_name=h.file_name,
                page_number=h.page_number,
                page_end_number=h.page_end_number,
                dieu=h.dieu,
                score=1.0,  # không có điểm similarity ở nhánh này — lấy full chunk theo document
                excerpt=h.text[:600],
            )
            for h in summary_result.sources
        )

        citation_validation = self._validate_citations(raw_answer, sources)
        if citation_validation["has_valid_citation"]:
            answer = raw_answer
        else:
            answer = ANTI_HALLUCINATION_ANSWER
 
        branch_trace: dict[str, object] = {
            "summary_scope": summary_result.scope_document_ids,
            "citation_validation": citation_validation,
        }
        return answer, sources, len(sources), branch_trace

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
        route_result = self.query_router_service.route(question)
        if route_result.route == QueryRoute.SUMMARY:
            answer, sources, retrieved_count, branch_trace = self._ask_summary(
                history_id, question
            )
        else:
            answer, sources, retrieved_count, branch_trace = self._ask_specific(
                history_id, question, top_k
            )
 
        # Phần lưu lịch sử — DÙNG CHUNG cho cả 2 nhánh, không trùng lặp code
        self.history_repository.add_message(history_id, MessageRole.USER, question)
        self.history_repository.add_message(
            history_id,
            MessageRole.ASSISTANT,
            answer,
            sources=json.dumps([s.__dict__ for s in sources], ensure_ascii=False) if sources else None,
        )
        self.history_repository.touch_history(history_id)
 
        trace: dict[str, object] | None = None
        if include_trace:
            trace = branch_trace
            trace["route"] = route_result.route.value
            trace["total_time_ms"] = round(
                (time.perf_counter() - request_started) * 1000, 3
            )
 
        return RagAnswer(question, answer, sources, retrieved_count, trace)