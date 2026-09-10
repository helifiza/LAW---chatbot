from __future__ import annotations

import re
import time
import json
import inspect
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from app.core.errors import NoReadyDocumentError
from app.domain.models import DocumentStatus, MessageRole, SearchResult
from app.repositories.history_repository import HistoryRepository
from app.repositories.vector_repository import VectorRepository
from app.services.generation_service import GenerationService, GroundedClaim
from app.services.hybrid_retrieval_service import HybridRetrievalService
from app.services.query_rewrite_service import QueryRewriteService
from app.services.history_service import HistoryService
from app.services.query_router_service import QueryCoverage, QueryIntent, QueryRouterService
from app.services.summary_service import SummarizeService
from app.repositories.legal_graph_repository import LegalGraphRepository
from app.services.legal_effect_service import LegalEffectService
from app.domain.legal_document import normalize_document_type_filter

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
SEMANTIC_SUPPORT_THRESHOLD = 0.45

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
        legal_graph_repository: LegalGraphRepository | None = None,
        legal_effect_service: LegalEffectService | None = None,
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
        self.legal_graph_repository = legal_graph_repository
        self.legal_effect_service = legal_effect_service

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
    # Citation validation 
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
        self, answer: str, sources: Sequence[RagSource],
        claims: Sequence[GroundedClaim] | None = None,
        trusted_evidence: bool = False,
    ) -> dict[str, object]:
        """
        Trích các citation dạng [file, trang X-Y] hoặc [file, Điều Z] trong
        answer, so khớp MỜ (token-based, sau chuẩn hoá tên file) với tên file
        trong sources.

        - Citation dạng "trang": đối chiếu bằng overlap khoảng trang
          (page_number/page_end_number của source).
        - Citation dạng "Điều": đối chiếu trực tiếp field `dieu` của source
          (sau khi chuẩn hoá qua _normalize_dieu), không dùng page_overlap.

        Sau bước khớp tên/vị trí, xác thực semantic giữa từng claim và excerpt.
        Coverage được tính theo claim có trọng số, không dựa trên việc tách mọi
        câu văn trong answer.
        """
        matched: list[dict[str, object]] = []
        unmatched: list[dict[str, object]] = []

        excerpt_vecs: list[list[float]] | None = None
        if sources and not trusted_evidence:
            excerpt_vecs = self._embed_excerpts(sources)
        claim_records = tuple(claims or GenerationService._fallback_claims(answer))

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
                "raw_citation": match.group(0),
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
            if (name_page_ok and best_source_idx is not None
                    and (trusted_evidence or excerpt_vecs is not None)):
                structured_claim = next(
                    (item for item in claim_records if match.group(0) in item.citations
                     or match.group(0) in item.text), None
                )
                claim = structured_claim.text if structured_claim else self._extract_claim_sentence(answer, match.start())
                entry["claim_id"] = structured_claim.claim_id if structured_claim else None
                # RETRIEVAL_QUERY vì claim đang đóng vai trò "truy vấn" xem
                # có khớp ngữ nghĩa với excerpt (RETRIEVAL_DOCUMENT) hay không.
                if trusted_evidence:
                    entry["semantic_supported"] = True
                else:
                    claim_vec = self.embedding_client.embed_query(claim)
                    sem_scores = self._semantic_scores(claim_vec, excerpt_vecs)
                    target_score = sem_scores[best_source_idx]
                    entry["semantic_score"] = round(target_score, 4)
                    entry["semantic_z"] = None
                    entry["semantic_supported"] = target_score >= SEMANTIC_SUPPORT_THRESHOLD

            is_valid = name_page_ok and entry["semantic_supported"] is not False
            (matched if is_valid else unmatched).append(entry)

        total = len(matched) + len(unmatched)
        coverage = (len(matched) / total) if total else 0.0 #tính toán xem số lượng citation answer do model đưa ra có > 80% không, nếu không thì câu trả lời đưa ra khômg đủ tin cậy => ANTI_HALLUCINATION
        matched_raw = {str(item["raw_citation"]) for item in matched}
        claim_checks: list[dict[str, object]] = []
        supported_weight = 0.0
        total_weight = 0.0
        for claim in claim_records:
            type_multiplier = 2.0 if claim.claim_type in {
                "legal_conclusion", "condition", "exception", "comparison"
            } else 1.0
            weight = claim.importance * type_multiplier
            valid_citations = [citation for citation in claim.citations if citation in matched_raw]
            minimum = 2 if claim.claim_type == "comparison" and len(sources) >= 2 else 1
            supported = len(valid_citations) >= minimum
            total_weight += weight
            if supported:
                supported_weight += weight
            claim_checks.append({"claim_id": claim.claim_id, "claim_type": claim.claim_type,
                "weight": round(weight, 3), "citation_count": len(claim.citations),
                "valid_citation_count": len(valid_citations), "supported": supported})
        cited_claims = sum(1 for item in claim_checks if item["supported"])
        claim_coverage = supported_weight / total_weight if total_weight else 0.0

        return {
            "citation_count": total,
            "matched_citations": matched,
            "unmatched_citations": unmatched,
            "coverage": round(coverage, 4),
            "claim_count": len(claim_records),
            "cited_claim_count": cited_claims,
            "claim_coverage": round(claim_coverage, 4),
            "claim_checks": claim_checks,
            "has_valid_citation": (
                total > 0
                and coverage >= CITATION_COVERAGE_THRESHOLD
                and claim_coverage >= CITATION_COVERAGE_THRESHOLD
            ),
        }

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------
    def _ask_specific(
        self,
        history_id: str,
        question: str,
        top_k: int,
        linh_vuc_filter: Sequence[str] | None = None,
        document_type_filter: Sequence[str] | None = None,
    ) -> tuple[str, tuple[RagSource, ...], int, dict[str, object]]:
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
            linh_vuc_filter=linh_vuc_filter,
            document_type_filter=document_type_filter,
        )
        results = self._deduplicate(retrieval.results, top_k)

        # Fallback an toàn: nếu lọc theo linh_vuc làm rỗng kết quả (do router
        # phân loại sai câu hỏi, hoặc document bị phân loại sai lúc index),
        # thử lại KHÔNG lọc thay vì trả NOT_FOUND oan — lọc lĩnh vực chỉ nên
        # thu hẹp phạm vi tìm kiếm, không được phép loại bỏ câu trả lời đúng.
        applied_linh_vuc_fallback = False
        if not results and linh_vuc_filter:
            retrieval = self.retrieval_service.retrieve(
                history_id=history_id,
                original_query=question,
                dense_query=rewrite.retrieval_query,
                top_k=max(top_k * 2, top_k),
                document_type_filter=document_type_filter,
            )
            results = self._deduplicate(retrieval.results, top_k)
            applied_linh_vuc_fallback = True

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
            generated_claims: Sequence[GroundedClaim] | None = None
            if hasattr(self.generation_service, "generate_grounded"):
                generated = self.generation_service.generate_grounded(
                    question, self._format_context(results), history
                )
                raw_answer, generated_claims = generated.answer, generated.claims
            else:
                raw_answer = self.generation_service.generate(
                    question, self._format_context(results), history
                )
            generation_ms = (time.perf_counter() - generation_started) * 1000
            citation_validation = self._validate_citations(raw_answer, sources, generated_claims)
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
        if applied_linh_vuc_fallback:
            warnings.insert(
                0,
                f"Lọc theo lĩnh vực {list(linh_vuc_filter)} không có kết quả, "
                "đã tự động thử lại không lọc lĩnh vực.",
            )
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
            target_documents: Sequence[str] | None = None,
            coverage: QueryCoverage = QueryCoverage.MULTI_ASPECT,
            document_type_filter: Sequence[str] | None = None,
    )-> tuple[str, tuple[RagSource, ...], int, dict[str, object]]:
        run_parameters = inspect.signature(self.summarize_service.run).parameters
        run_kwargs: dict[str, object] = {"target_documents": target_documents}
        if "full_enumeration" in run_parameters:
            
            run_kwargs["full_enumeration"] = coverage == QueryCoverage.FULL_ENUMERATION
        if "document_type_filter" in run_parameters:
            run_kwargs["document_type_filter"] = document_type_filter
        summary_result = self.summarize_service.run(history_id, question, **run_kwargs)
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

        citation_validation = self._validate_citations(
            raw_answer, sources, getattr(summary_result, "claims", None)
        )
        if not sources or citation_validation["has_valid_citation"]:
            answer = raw_answer
        else:
            answer = ANTI_HALLUCINATION_ANSWER
 
        branch_trace: dict[str, object] = {
            "summary_scope": summary_result.scope_document_ids,
            "citation_validation": citation_validation,
            "coverage": {
                "documents_considered": len(summary_result.scope_document_ids),
                "documents_matched": len({source.document_id for source in sources}),
                "source_chunks": len(sources),
                "provisions_expected": getattr(summary_result, "total_provisions", len(sources)),
                "provisions_covered": getattr(summary_result, "covered_provisions", len(sources)),
                "failed_map_groups": getattr(summary_result, "failed_map_groups", 0),
            },
        }
        return answer, sources, len(sources), branch_trace

    def _ask_compare(
        self,
        history_id: str,
        question: str,
        target_documents: Sequence[str] | None,
        top_k: int,
        comparison_aspects: Sequence[str] | None = None,
        document_type_filter: Sequence[str] | None = None,
    ) -> tuple[str, tuple[RagSource, ...], int, dict[str, object]]:
        documents = [
            document for document in self.history_repository.list_documents(history_id)
            if document.status == DocumentStatus.READY.value
        ]
        normalized_types = normalize_document_type_filter(document_type_filter)
        if normalized_types:
            if self.legal_graph_repository is None:
                documents = []
            else:
                allowed_upload_ids = self.legal_graph_repository.list_upload_document_ids_by_types(
                    history_id, normalized_types
                )
                documents = [
                    document for document in documents
                    if document.id in allowed_upload_ids
                ]
        selected = documents
        if target_documents:
            graph_upload_ids: set[str] = set()
            if self.legal_graph_repository is not None:
                for legal_document in self.legal_graph_repository.list_documents(history_id):
                    upload_id = legal_document.get("upload_document_id")
                    searchable = " ".join(
                        str(legal_document.get(field) or "")
                        for field in ("document_number", "title", "document_type")
                    )
                    if upload_id and any(
                        _normalize_filename(ref) in _normalize_filename(searchable)
                        for ref in target_documents
                    ):
                        graph_upload_ids.add(str(upload_id))
            selected = [
                document for document in documents
                if document.id in graph_upload_ids or any(
                    _normalize_filename(ref) in _normalize_filename(document.file_name)
                    or _normalize_filename(document.file_name) in _normalize_filename(ref)
                    for ref in target_documents
                )
            ]
        if len(selected) < 2:
            return (
                "Cần xác định ít nhất hai văn bản đã tải lên để so sánh.",
                (), 0,
                {"warnings": ["Không resolve được ít nhất hai văn bản."], "coverage": {"documents_considered": len(selected)}},
            )
        history_records = self.history_repository.list_messages(history_id, limit=self.history_limit)
        history = [(message.role, message.content) for message in history_records]
        context_by_document: dict[str, str] = {}
        all_results: list[SearchResult] = []
        per_document_k = max(2, top_k)
        aspects = list(comparison_aspects or []) or [question]
        aspect_cells = 0
        for document in selected:
            for aspect in aspects:
                retrieval = self.retrieval_service.retrieve(
                    history_id=history_id, original_query=aspect, dense_query=aspect,
                    top_k=per_document_k, document_ids=[document.id],
                    document_type_filter=normalized_types,
                )
                results = self._deduplicate(retrieval.results, per_document_k)
                all_results.extend(results)
                if results:
                    aspect_cells += 1
                context_by_document[f"{document.file_name} — {aspect}"] = (
                    self._format_context(results) or "Không tìm thấy nội dung tương ứng."
                )
        sources = tuple(
            RagSource(
                document_id=result.chunk.document_id,
                file_name=result.chunk.file_name,
                page_number=result.chunk.page_number,
                page_end_number=result.chunk.page_end_number,
                dieu=result.chunk.dieu,
                score=result.score,
                excerpt=result.chunk.text[:600],
            )
            for result in all_results
        )
        comparison_claims: Sequence[GroundedClaim] | None = None
        if hasattr(self.generation_service, "generate_comparison_grounded"):
            generated = self.generation_service.generate_comparison_grounded(
                question, context_by_document, history
            )
            raw_answer, comparison_claims = generated.answer, generated.claims
        else:
            raw_answer = self.generation_service.generate_comparison(
                question, context_by_document, history
            )
        validation = self._validate_citations(raw_answer, sources, comparison_claims)
        answer = raw_answer if validation["has_valid_citation"] else ANTI_HALLUCINATION_ANSWER
        return answer, sources, len(all_results), {
            "citation_validation": validation,
            "coverage": {
                "documents_considered": len(selected),
                "documents_matched": len({source.document_id for source in sources}),
                "aspects": aspects,
                "aspect_cells_expected": len(selected) * len(aspects),
                "aspect_cells_matched": aspect_cells,
            },
        }

    def _ask_current_effect(
        self,
        history_id: str,
        targets: Sequence[str],
        as_of_date: str | None,
        document_type_filter: Sequence[str] | None = None,
    ) -> tuple[str, tuple[RagSource, ...], int, dict[str, object]]:
        if self.legal_effect_service is None:
            return "Chưa cấu hình bộ kiểm tra hiệu lực.", (), 0, {"warnings": ["effect_service_missing"]}
        result = self.legal_effect_service.evaluate(
            history_id, targets, as_of_date, document_type_filter
        )
        sources = tuple(RagSource(**item) for item in result["sources"])
        validation = self._validate_citations(
            result["answer"], sources, trusted_evidence=True
        ) if sources else None
        return result["answer"], sources, len(sources), {
            "effect_status": result["status"], "warnings": result["warnings"],
            "citation_validation": validation,
        }

    def _ask_relationship(
        self,
        history_id: str,
        question: str,
        target_documents: Sequence[str] | None,
        document_type_filter: Sequence[str] | None = None,
    ) -> tuple[str, tuple[RagSource, ...], int, dict[str, object]]:
        if self.legal_graph_repository is None:
            return self._ask_specific(
                history_id,
                question,
                max(1, self.history_limit),
                document_type_filter=document_type_filter,
            )
        details = self.legal_graph_repository.list_relation_details(
            history_id, document_type_filter
        )
        if target_documents:
            normalized_targets = [_normalize_filename(value) for value in target_documents]
            details = [
                item for item in details
                if any(
                    target in _normalize_filename(str(item.get("source_number") or ""))
                    or target in _normalize_filename(str(item.get("target_number") or ""))
                    for target in normalized_targets
                )
            ]
        if not details:
            return "Không tìm thấy quan hệ có chứng cứ trong các tài liệu đã tải lên.", (), 0, {"coverage": {"relations": 0}}
        relation_labels = {
            "SUA_DOI": "sửa đổi", "BO_SUNG": "bổ sung", "SUA_DOI_BO_SUNG": "sửa đổi, bổ sung",
            "BAI_BO": "bãi bỏ", "BAI_BO_MOT_PHAN": "bãi bỏ một phần",
            "THAY_THE": "thay thế", "DINH_CHI": "đình chỉ", "GIA_HAN": "gia hạn",
            "HUONG_DAN_THI_HANH": "hướng dẫn thi hành",
            "QUY_DINH_CHI_TIET": "quy định chi tiết", "CAN_CU": "căn cứ vào",
            "HOP_NHAT": "hợp nhất",
        }
        sources = tuple(
            RagSource(
                document_id=item["upload_document_id"],
                file_name=item["source_file_name"] or "tài liệu đã tải lên",
                page_number=item["page_number"],
                page_end_number=item["page_end_number"],
                dieu=None,
                score=float(item["confidence"]),
                excerpt=item["quote"],
            ) for item in details
        )
        lines = []
        for item, source in zip(details, sources):
            pages = str(source.page_number) if source.page_number == source.page_end_number else f"{source.page_number}-{source.page_end_number}"
            lines.append(
                f"{item['source_number'] or source.file_name} {relation_labels.get(item['relation_type'], item['relation_type'])} "
                f"{item['target_number'] or item['raw_target_reference']} [{source.file_name}, trang {pages}]"
            )
        answer = "\n".join(f"- {line}" for line in lines)
        validation = self._validate_citations(answer, sources, trusted_evidence=True)
        return answer, sources, len(details), {
            "coverage": {"relations": len(details)}, "citation_validation": validation
        }

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
        route_result = self.query_router_service.route(
            question,
            has_chat_history=self.history_service.message_count(history_id) > 0,
        )
        if route_result.intent in (
            QueryIntent.DOCUMENT_SUMMARY,
            QueryIntent.MULTI_DOCUMENT_SYNTHESIS,
        ):
            answer, sources, retrieved_count, branch_trace = self._ask_summary(
                history_id, question, route_result.target_documents, route_result.coverage,
                route_result.loai_van_ban_filter,
            )
        elif route_result.intent == QueryIntent.COMPARE:
            answer, sources, retrieved_count, branch_trace = self._ask_compare(
                history_id, question, route_result.target_documents, top_k,
                route_result.comparison_aspects,
                route_result.loai_van_ban_filter,
            )
        elif route_result.intent == QueryIntent.RELATIONSHIP:
            answer, sources, retrieved_count, branch_trace = self._ask_relationship(
                history_id, question, route_result.target_documents,
                route_result.loai_van_ban_filter,
            )
        elif route_result.intent == QueryIntent.CURRENT_EFFECT_CHECK:
            answer, sources, retrieved_count, branch_trace = self._ask_current_effect(
                history_id, route_result.target_documents, route_result.as_of_date,
                route_result.loai_van_ban_filter,
            )
        else:
            # fact_lookup, current_effect_check: chỉ 2 nhánh này dùng linh_vuc
            # để thu hẹp phạm vi retrieval.
            answer, sources, retrieved_count, branch_trace = self._ask_specific(
                history_id,
                question,
                top_k,
                linh_vuc_filter=route_result.linh_vuc or None,
                document_type_filter=route_result.loai_van_ban_filter or None,
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
            trace["route"] = route_result.intent.value
            trace["scope"] = route_result.scope.value
            trace["coverage"] = route_result.coverage.value
            trace["router_source"] = route_result.source
            trace["document_type_filter"] = (
                route_result.loai_van_ban_filter or None
            )
            trace["total_time_ms"] = round(
                (time.perf_counter() - request_started) * 1000, 3
            )
 
        return RagAnswer(question, answer, sources, retrieved_count, trace)
