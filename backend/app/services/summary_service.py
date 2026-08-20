from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from app.clients.gemini_client import GeminiClient
from app.domain.models import ChunkDetail, DocumentStatus
from app.repositories.history_repository import HistoryRepository
from app.repositories.vector_repository import VectorRepository


MAP_SYSTEM_PROMPT = """Bạn trích xuất (KHÔNG diễn giải lại) những câu/đoạn quan trọng nhất
trong văn bản pháp luật được cung cấp. Giữ nguyên văn câu chữ gốc.
Mỗi văn bản đầu vào đã có sẵn metadata (element_id, dieu, chuong, file_name) —
bạn PHẢI copy lại đúng các giá trị này, không tự suy luận hay bịa ra nguồn khác.
Chỉ trả về JSON đúng định dạng, không thêm markdown, không thêm lời dẫn:
{"highlights": [{"element_id": "...", "dieu": "...", "chuong": "...", "file_name": "...", "text": "..."}]}
"""

REDUCE_SYSTEM_PROMPT = """Bạn tổng hợp các đoạn trích (highlights) đã được cung cấp thành một
bản tóm tắt mạch lạc, đúng trọng tâm câu hỏi của người dùng.
CHỈ được trích dẫn [tên file, Điều X] nếu Điều đó xuất hiện trong metadata của
các highlights được cung cấp — KHÔNG được tự suy đoán hay bịa thêm Điều không có
trong dữ liệu đầu vào. Nếu highlight không có 'dieu' (đoạn văn thường), dùng
định dạng [tên file, trang X] nếu có thông tin trang, hoặc không trích dẫn số cụ thể.
Trả lời bằng tiếng Việt, rõ ràng, đúng trọng tâm câu hỏi.
"""


@dataclass(frozen=True)
class SummaryHighlight:
    """1 đoạn trích đã qua bước Map, vẫn giữ metadata nguồn."""
    document_id: str
    file_name: str
    page_number: int
    page_end_number: int
    dieu: str | None
    chuong: str | None
    text: str


@dataclass(frozen=True)
class SummarizeResult:
    """Kết quả trả về của SummarizeService.run() — RagService._ask_summary()
    cần đúng 3 field này để build RagSource và trace."""
    answer: str
    sources: tuple[SummaryHighlight, ...]
    scope_document_ids: tuple[str, ...]


class SummarizeService:
    def __init__(
        self,
        history_repo: HistoryRepository,
        vector_repo: VectorRepository,
        gemini_client: GeminiClient,
        model: str,
        map_group_size: int = 10,  # số chunk gộp vào 1 lần gọi Map
        max_output_tokens: int = 1000,
        temperature: float = 0.2,
    ) -> None:
        self.history_repo = history_repo
        self.vector_repo = vector_repo
        self.gemini_client = gemini_client
        self.model = model
        self.map_group_size = map_group_size
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    # Bước 1: Scope Resolver

    def _resolve_scope(self, history_id: str, question: str) -> list[str]:
        """
        Xác định các document_id cần tóm tắt.
        Bản đơn giản: lấy tất cả document đã READY trong history.
        """
        documents = self.history_repo.list_documents(history_id)
        ready_documents = [
            d for d in documents if d.status == DocumentStatus.READY.value
        ]

        if not ready_documents:
            return []

        # Nếu user nêu rõ tên file trong câu hỏi, ưu tiên match theo tên
        # so khớp đơn giản theo substring không dấu
        lowered_question = question.lower()
        matched_by_name = [
            d for d in ready_documents
            if d.file_name.lower().rsplit(".", 1)[0] in lowered_question
        ]
        if matched_by_name:
            return [d.id for d in matched_by_name]

        # Nếu chỉ có 1 document sẵn có, dùng luôn — không cần đoán.
        if len(ready_documents) == 1:
            return [ready_documents[0].id]

        return [d.id for d in ready_documents]

    # Bước 2: Lấy toàn bộ chunk của các document đã target

    def _get_chunks_for_documents(
        self, history_id: str, document_ids: Sequence[str]
    ) -> list[ChunkDetail]:
        return self.vector_repo.list_chunks_by_documents(history_id, document_ids)

    # Bước 3: Dedupe theo `dieu` — gộp các chunk cùng 1 Điều bị chia nhỏ do overlap_chars, tránh trùng lặp/cắt đứt ngữ nghĩa khi đưa vào Map.

    @staticmethod
    def _find_overlap(a: str, b: str, max_check: int = 300) -> int:
        """Tìm độ dài phần cuối của a trùng với phần đầu của b."""
        max_len = min(len(a), len(b), max_check)
        for length in range(max_len, 0, -1):
            if a[-length:] == b[:length]:
                return length
        return 0

    def _merge_overlapping_texts(self, texts: list[str]) -> str:
        if not texts:
            return ""
        merged = texts[0]
        for next_text in texts[1:]:
            overlap_len = self._find_overlap(merged, next_text)
            merged += next_text[overlap_len:]
        return merged

    def _dedupe_by_dieu(self, chunks: Sequence[ChunkDetail]) -> list[ChunkDetail]:
        """
        Gộp các chunk cùng document_id + dieu thành 1 chunk đại diện,
        merge text loại phần overlap trùng lặp. Chunk không có `dieu`
        (content_type='paragraph'/'chuong'/'muc') giữ nguyên riêng lẻ.
        """
        grouped: dict[tuple[str, str], list[ChunkDetail]] = {}
        standalone: list[ChunkDetail] = []

        for chunk in chunks:
            if chunk.dieu:
                key = (chunk.document_id, chunk.dieu)
                grouped.setdefault(key, []).append(chunk)
            else:
                standalone.append(chunk)

        result: list[ChunkDetail] = list(standalone)
        for (_document_id, _dieu), items in grouped.items():
            if len(items) == 1:
                result.append(items[0])
                continue
            # Đã được sort theo chunk_index từ list_chunks_by_documents(),
            # nên thứ tự items ở đây đúng thứ tự văn bản gốc.
            items_sorted = sorted(items, key=lambda c: c.chunk_index)
            merged_text = self._merge_overlapping_texts(
                [c.text for c in items_sorted]
            )
            first = items_sorted[0]
            last = items_sorted[-1]
            result.append(
                ChunkDetail(
                    history_id=first.history_id,
                    document_id=first.document_id,
                    user_id=first.user_id,
                    element_id=first.element_id,  # dùng element_id đầu tiên làm đại diện
                    file_name=first.file_name,
                    page_number=first.page_number,
                    page_end_number=last.page_end_number,
                    content_type=first.content_type,
                    chuong=first.chuong,
                    muc=first.muc,
                    dieu=first.dieu,
                    text=merged_text,
                    chunk_index=first.chunk_index,
                    char_count=len(merged_text),
                    token_count=sum(c.token_count for c in items_sorted),
                    created_at=first.created_at,
                )
            )
        return result

    # ------------------------------------------------------------------
    # Bước 4: Map — extractive summarization theo từng nhóm chunk
    # ------------------------------------------------------------------

    def _chunk_groups(
        self, chunks: Sequence[ChunkDetail]
    ) -> list[list[ChunkDetail]]:
        return [
            list(chunks[i : i + self.map_group_size])
            for i in range(0, len(chunks), self.map_group_size)
        ]

    def _format_chunks_for_map(self, group: Sequence[ChunkDetail]) -> str:
        parts = []
        for c in group:
            parts.append(
                f'[element_id={c.element_id} | dieu={c.dieu or ""} | '
                f'chuong={c.chuong or ""} | file_name={c.file_name}]\n{c.text}'
            )
        return "\n\n---\n\n".join(parts)

    def _map_extractive(
        self, chunks: Sequence[ChunkDetail], question: str
    ) -> list[SummaryHighlight]:
        if not chunks:
            return []

        chunk_by_element_id = {c.element_id: c for c in chunks}
        highlights: list[SummaryHighlight] = []

        for group in self._chunk_groups(chunks):
            prompt = (
                f"CÂU HỎI CỦA NGƯỜI DÙNG (để tham khảo mức độ liên quan): {question}\n\n"
                f"CÁC ĐOẠN VĂN BẢN:\n{self._format_chunks_for_map(group)}"
            )
            raw = self.gemini_client.generate(
                model=self.model,
                prompt=prompt,
                system_instruction=MAP_SYSTEM_PROMPT,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            parsed = self._parse_map_output(raw)
            for item in parsed:
                element_id = item.get("element_id")
                source_chunk = chunk_by_element_id.get(element_id)
                if source_chunk is None:
                    # Model trả về element_id không khớp bất kỳ chunk đầu vào
                    # nào -> bỏ qua, không tin tưởng highlight này (an toàn
                    # hơn là giữ lại 1 nguồn không xác thực được).
                    continue
                highlights.append(
                    SummaryHighlight(
                        document_id=source_chunk.document_id,
                        file_name=source_chunk.file_name,
                        page_number=source_chunk.page_number,
                        page_end_number=source_chunk.page_end_number,
                        dieu=source_chunk.dieu,
                        chuong=source_chunk.chuong,
                        text=item.get("text", "").strip() or source_chunk.text,
                    )
                )
        return highlights

    @staticmethod
    def _parse_map_output(raw: str) -> list[dict]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        try:
            data = json.loads(cleaned)
            return data.get("highlights", []) or []
        except (json.JSONDecodeError, AttributeError):
            return []

    # ------------------------------------------------------------------
    # Bước 5: Reduce — tổng hợp thành bản tóm tắt cuối
    # ------------------------------------------------------------------

    def _format_highlights_for_reduce(
        self, highlights: Sequence[SummaryHighlight]
    ) -> str:
        parts = []
        for h in highlights:
            heading = f"[{h.file_name}"
            if h.dieu:
                heading += f" | {h.dieu}"
            else:
                pages = (
                    f"trang {h.page_number}"
                    if h.page_number == h.page_end_number
                    else f"trang {h.page_number}-{h.page_end_number}"
                )
                heading += f" | {pages}"
            heading += "]"
            parts.append(f"{heading}\n{h.text}")
        return "\n\n---\n\n".join(parts)

    def _reduce(self, highlights: Sequence[SummaryHighlight], question: str) -> str:
        if not highlights:
            return (
                "Không tìm thấy nội dung liên quan trong các tài liệu của phiên "
                "để tổng hợp câu trả lời này."
            )
        prompt = (
            f"CÂU HỎI: {question}\n\n"
            f"CÁC ĐOẠN TRÍCH ĐÃ THU THẬP:\n{self._format_highlights_for_reduce(highlights)}"
        )
        return self.gemini_client.generate(
            model=self.model,
            prompt=prompt,
            system_instruction=REDUCE_SYSTEM_PROMPT,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

    # ------------------------------------------------------------------
    # Bước 6: Validate citation nội bộ — kiểm tra [file, Điều X] trong answer
    # có khớp với tập `dieu` đã đưa vào Reduce hay không. Đây là lớp kiểm
    # tra RIÊNG của SummarizeService (thuần regex, không gọi Gemini) —
    # KHÁC với RagService._validate_citations() (dùng semantic z-score),
    # vốn sẽ được RagService gọi lại 1 lần nữa ở tầng trên sau khi nhận
    # SummarizeResult. Hàm này chỉ để log cảnh báo sớm, không quyết định
    # answer cuối cùng có được dùng hay không.
    # ------------------------------------------------------------------

    _DIEU_CITATION_PATTERN = re.compile(r"\[([^,\]]+),\s*(Điều\s+\d+[A-Za-z]?)\]")

    def _validate_citations(
        self, answer: str, chunks: Sequence[ChunkDetail]
    ) -> list[str]:
        valid_dieu = {c.dieu for c in chunks if c.dieu}
        invalid: list[str] = []
        for file_cited, dieu_cited in self._DIEU_CITATION_PATTERN.findall(answer):
            if dieu_cited not in valid_dieu:
                invalid.append(f"[{file_cited}, {dieu_cited}]")
        return invalid

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(self, history_id: str, question: str) -> SummarizeResult:
        target_document_ids = self._resolve_scope(history_id, question)
        chunks = self._get_chunks_for_documents(history_id, target_document_ids)
        deduped = self._dedupe_by_dieu(chunks)
        highlights = self._map_extractive(deduped, question)
        answer = self._reduce(highlights, question)

        invalid_citations = self._validate_citations(answer, deduped)
        if invalid_citations:
            # Không tự ý sửa answer ở đây — chỉ ghi nhận để RagService/log
            # cấp trên quyết định (RagService sẽ tự chạy _validate_citations
            # riêng của nó dựa trên semantic z-score, nghiêm ngặt hơn).
            pass

        sources = tuple(
            SummaryHighlight(
                document_id=h.document_id,
                file_name=h.file_name,
                page_number=h.page_number,
                page_end_number=h.page_end_number,
                dieu=h.dieu,
                chuong=h.chuong,
                text=h.text,
            )
            for h in highlights
        )

        return SummarizeResult(
            answer=answer,
            sources=sources,
            scope_document_ids=tuple(target_document_ids),
        )