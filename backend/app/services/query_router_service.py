from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from app.domain.legal_document import normalize_document_type_filter

logger = logging.getLogger(__name__)

# Trục 1 — Classification (level câu): intent/scope/coverage/history-context.

class QueryIntent(str, Enum):
    FACT_LOOKUP = "fact_lookup"#hỏi 1 thông tin cụ thể
    DOCUMENT_SUMMARY = "document_summary"#tóm tắt một văn bản
    MULTI_DOCUMENT_SYNTHESIS = "multi_document_synthesis"#tổng hợp nội dung từ nhiều văn bản
    COMPARE = "compare"#so sánh >= 2 văn bản
    RELATIONSHIP = "relationship"#tìm quan hệ giữa các văn bản
    CURRENT_EFFECT_CHECK = "current_effect_check"#xác định hiệu lực hiện hành của văn bản


class QueryScope(str, Enum):
    SINGLE_DOCUMENT = "single_document"
    SELECTED_DOCUMENTS = "selected_documents"
    MULTI_DOCUMENT = "multi_document"
    ALL_MATCHING = "all_matching"
    UNKNOWN = "unknown"


class QueryCoverage(str, Enum):
    MOST_RELEVANT_ONLY = "most_relevant_only"#chỉ cần những đoạn phù hợp nhất
    MULTI_ASPECT = "multi_aspect"#cần bao phủ nhiều khía cạnh
    FULL_ENUMERATION = "full_enumeration"#phải cố gắng liệt kê đầy đủ văn bản/quy định


@dataclass(frozen=True)
class IntentClassification:
    """Head phân loại — nhãn ở cấp toàn câu, không gắn với vị trí token nào."""

    intent: QueryIntent
    scope: QueryScope
    coverage: QueryCoverage
    requires_history_context: bool = False


# Head 2 — Slot extraction (cấp span): target_documents/loai_van_ban/linh_vuc.


@dataclass(frozen=True)
class SlotSpan:
    """Một thực thể được trích ra, kèm vị trí trong câu gốc nếu match được.

    `char_start`/`char_end` là None khi Gemini suy luận ra giá trị không xuất
    hiện nguyên văn trong câu hỏi (paraphrase/suy luận ngầm) — trường hợp này
    không dùng được để sinh nhãn BIO cho slot-filling, chỉ giữ lại cho phần
    filter/logic downstream.
    """

    label: str  # "target_document" | "loai_van_ban"
    value: str
    char_start: int | None
    char_end: int | None


@dataclass(frozen=True)
class SlotExtraction:
    target_documents: list[str]
    loai_van_ban_filter: list[str]
    linh_vuc: list[str]
    comparison_aspects: list[str] = field(default_factory=list)
    as_of_date: str | None = None
    spans: list[SlotSpan] = field(default_factory=list)


@dataclass(frozen=True)
class RouteResult:
    classification: IntentClassification
    slots: SlotExtraction
    matched_keyword: str | None = None
    source: str = "keyword"  

    @property
    def intent(self) -> QueryIntent:
        return self.classification.intent

    @property
    def scope(self) -> QueryScope:
        return self.classification.scope

    @property
    def coverage(self) -> QueryCoverage:
        return self.classification.coverage

    @property
    def requires_history_context(self) -> bool:
        return self.classification.requires_history_context

    @property
    def target_documents(self) -> list[str]:
        return self.slots.target_documents

    @property
    def loai_van_ban_filter(self) -> list[str]:
        return self.slots.loai_van_ban_filter

    @property
    def linh_vuc(self) -> list[str]:
        return self.slots.linh_vuc

    @property
    def comparison_aspects(self) -> list[str]:
        return self.slots.comparison_aspects

    @property
    def as_of_date(self) -> str | None:
        return self.slots.as_of_date

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "scope": self.scope.value,
            "coverage": self.coverage.value,
            "requires_history_context": self.requires_history_context,
            "target_documents": self.target_documents,
            "loai_van_ban_filter": self.loai_van_ban_filter,
            "linh_vuc": self.linh_vuc,
            "comparison_aspects": self.comparison_aspects,
            "as_of_date": self.as_of_date,
        }


# =============================================================================
# Taxonomy lĩnh vực — dùng chung cho cả prompt (embed vào system prompt để
# Gemini biết mã 01-19 nghĩa là gì) lẫn validate output (lọc mã model bịa ra
# hoặc trả sai định dạng).
# =============================================================================

LINH_VUC_TAXONOMY: dict[str, str] = {
    "01": "Tổ chức nhà nước và hành chính",
    "02": "Quốc phòng, an ninh và trật tự",
    "03": "Tư pháp",
    "04": "Dân sự và gia đình",
    "05": "Kinh doanh, thương mại và đầu tư",
    "06": "Tài chính, thuế và hải quan",
    "07": "Ngân hàng, bảo hiểm và chứng khoán",
    "08": "Lao động và an sinh xã hội",
    "09": "Đất đai, nhà ở và xây dựng",
    "10": "Nông nghiệp và tài nguyên",
    "11": "Môi trường",
    "12": "Giao thông vận tải",
    "13": "Y tế",
    "14": "Giáo dục và đào tạo",
    "15": "Khoa học, công nghệ và thông tin",
    "16": "Văn hóa, thể thao và du lịch",
    "17": "Quốc tế",
    "18": "Dân tộc và tôn giáo",
    "19": "Liên ngành",
}

_LINH_VUC_PROMPT_BLOCK = "\n".join(
    f"  {code}. {name}" for code, name in LINH_VUC_TAXONOMY.items()
)


# =============================================================================
# Fast-path keyword — chỉ rút gọn case rõ ràng, KHÔNG làm slot extraction vì
# regex/keyword không đáng tin cho việc trích span chính xác.
# =============================================================================

_EFFECT_KEYWORDS = ("còn hiệu lực", "hết hiệu lực", "hiện hành", "đã bị thay thế", "đã bãi bỏ")
_RELATIONSHIP_KEYWORDS = ("hướng dẫn thi hành", "căn cứ vào", "sửa đổi bổ sung", "thay thế cho")
_COMPARE_KEYWORDS = ("so sánh", "khác nhau", "giống nhau")
_SYNTHESIS_KEYWORDS = ("tổng hợp", "liệt kê toàn bộ", "liệt kê tất cả")
_SUMMARY_KEYWORDS = ("tóm tắt", "khái quát", "tổng quan")
_HISTORY_REFERENCE_MARKERS = ("văn bản trên", "văn bản đó", "văn bản vừa", "2 văn bản này", "các văn bản trên")

_EMPTY_SLOTS = SlotExtraction(target_documents=[], loai_van_ban_filter=[], linh_vuc=[])

_FAST_PATH_TABLE: list[tuple[tuple[str, ...], QueryIntent, QueryScope, QueryCoverage]] = [
    (_EFFECT_KEYWORDS, QueryIntent.CURRENT_EFFECT_CHECK, QueryScope.SINGLE_DOCUMENT, QueryCoverage.MOST_RELEVANT_ONLY),
    (_RELATIONSHIP_KEYWORDS, QueryIntent.RELATIONSHIP, QueryScope.MULTI_DOCUMENT, QueryCoverage.MULTI_ASPECT),
    (_COMPARE_KEYWORDS, QueryIntent.COMPARE, QueryScope.MULTI_DOCUMENT, QueryCoverage.MULTI_ASPECT),
    (_SYNTHESIS_KEYWORDS, QueryIntent.MULTI_DOCUMENT_SYNTHESIS, QueryScope.MULTI_DOCUMENT, QueryCoverage.FULL_ENUMERATION),
    (_SUMMARY_KEYWORDS, QueryIntent.DOCUMENT_SUMMARY, QueryScope.SINGLE_DOCUMENT, QueryCoverage.MULTI_ASPECT),
]


_ROUTER_SYSTEM_PROMPT = f"""Bạn là bộ phân loại câu hỏi cho hệ thống tra cứu văn bản pháp luật Việt Nam.
Trả về DUY NHẤT một JSON object, không thêm giải thích, không dùng markdown fence.

Schema bắt buộc:
{{
  "intent": "fact_lookup" | "document_summary" | "multi_document_synthesis" | "compare" | "relationship" | "current_effect_check",
  "scope": "single_document" | "selected_documents" | "multi_document" | "all_matching" | "unknown",
  "coverage": "most_relevant_only" | "multi_aspect" | "full_enumeration",
  "requires_history_context": true | false,
  "target_documents": ["tên/số hiệu văn bản, giữ nguyên văn đúng như trong câu hỏi nếu có"] hoặc [],
  "loai_van_ban_filter": ["Nghị định", "Luật", ...] hoặc [],
  "linh_vuc": ["mã 2 chữ số 01-19, xem danh mục bên dưới"] hoặc [],
  "comparison_aspects": ["tiêu chí cụ thể cần so sánh"] hoặc [],
  "as_of_date": "YYYY-MM-DD" hoặc null
}}

Danh mục lĩnh vực (chỉ dùng đúng mã số trong danh mục này, không tự bịa mã mới):
{_LINH_VUC_PROMPT_BLOCK}

Quy tắc:
- Với "target_documents" và "loai_van_ban_filter": nếu giá trị xuất hiện nguyên văn
  trong câu hỏi, hãy giữ đúng chính tả/dạng viết như trong câu hỏi (để hệ thống có thể
  định vị lại vị trí trong câu). Nếu giá trị là suy luận ngầm (không có chữ đó trong câu),
  vẫn điền nhưng không cần khớp nguyên văn.
- "linh_vuc" dùng suy luận theo NỘI DUNG câu hỏi (không cần khớp nguyên văn trong câu),
  chọn mã phù hợp nhất theo danh mục trên. Chỉ dùng "19" (Liên ngành) khi câu hỏi thực
  sự liên quan rõ rệt tới từ 2 lĩnh vực trở lên; nếu không chắc chắn thuộc lĩnh vực nào,
  để mảng rỗng thay vì đoán bừa hoặc mặc định chọn "19".
- "requires_history_context": true nếu câu hỏi tham chiếu ngầm tới văn bản đã nhắc
  trong đoạn chat trước đó (VD: "văn bản trên", "văn bản đó").
- "current_effect_check" dùng khi hỏi về hiệu lực hiện hành, còn/hết hiệu lực,
  đã bị thay thế/sửa đổi/bãi bỏ chưa.
- "relationship" dùng khi hỏi quan hệ giữa các văn bản (căn cứ, hướng dẫn thi hành,
  sửa đổi, thay thế).
- Nếu không chắc chắn về scope, trả về "unknown", không đoán bừa.
"""


class QueryRouterService:
    def __init__(
        self,
        gemini_client: Any,
        model: str,
        training_log_path: Path | None = None,
    ) -> None:
        self.gemini_client = gemini_client
        self.model = model
        # Khi set path này, mỗi lần LLM trích được slot, ghi (câu hỏi, nhãn
        # BIO) ra jsonl. Đây là cách chuẩn bị dữ liệu cho distillation về sau
        # (PhoBERT/XLM-R + CRF) MÀ KHÔNG cần build model ngay bây giờ — chỉ
        # tích luỹ data trong lúc hệ thống chạy thật với traffic thật.
        self.training_log_path = training_log_path

    def route(self, question: str, has_chat_history: bool = False) -> RouteResult:
        lowered = question.lower()
        needs_history = has_chat_history and any(m in lowered for m in _HISTORY_REFERENCE_MARKERS)

        # Tham chiếu ngầm luôn cần LLM (không đáng tin nếu chỉ dựa keyword).
        # Slot extraction luôn cần LLM (keyword không trích được thực thể).
        # Keyword routing is only a fallback. Summary and comparison queries still
        # need the model to extract target document names.
        fallback = None if needs_history else self._match_keyword_fast_path(lowered)
        return self._route_via_llm(question, needs_history, fallback)

    def _match_keyword_fast_path(self, lowered: str) -> RouteResult | None:
        for keywords, intent, scope, coverage in _FAST_PATH_TABLE:
            for kw in keywords:
                if kw in lowered:
                    return RouteResult(
                        classification=IntentClassification(intent=intent, scope=scope, coverage=coverage),
                        slots=_EMPTY_SLOTS,
                        matched_keyword=kw,
                        source="keyword",
                    )
        return None

    def _route_via_llm(
        self,
        question: str,
        needs_history: bool,
        fallback: RouteResult | None = None,
    ) -> RouteResult:
        try:
            raw_response = self.gemini_client.generate(
                model=self.model,
                prompt=question,
                system_instruction=_ROUTER_SYSTEM_PROMPT,
            )
            data = json.loads(_strip_json_fence(raw_response))

            classification = IntentClassification(
                intent=QueryIntent(data["intent"]),
                scope=QueryScope(data["scope"]),
                coverage=QueryCoverage(data["coverage"]),
                requires_history_context=bool(data.get("requires_history_context", needs_history)),
            )
            slots = self._build_slot_extraction(question, data)

            if self.training_log_path is not None:
                self._log_for_future_distillation(question, classification, slots)

            return RouteResult(classification=classification, slots=slots, source="llm")
        except Exception:
            logger.exception("Router LLM classification failed, dùng default an toàn")
            if fallback is not None:
                return fallback
            return RouteResult(
                classification=IntentClassification(
                    intent=QueryIntent.FACT_LOOKUP,
                    scope=QueryScope.UNKNOWN,
                    coverage=QueryCoverage.MOST_RELEVANT_ONLY,
                    requires_history_context=needs_history,
                ),
                slots=_EMPTY_SLOTS,
                source="llm_fallback_default",
            )

    @staticmethod
    def _build_slot_extraction(question: str, data: dict) -> SlotExtraction:
        target_documents = data.get("target_documents") or []
        raw_document_types = data.get("loai_van_ban_filter") or []
        loai_van_ban_filter = list(
            normalize_document_type_filter(raw_document_types)
        )[:10]
        raw_linh_vuc = data.get("linh_vuc") or []
        # Chỉ giữ mã hợp lệ trong taxonomy — bỏ qua mã model bịa ra hoặc trả
        # sai định dạng (vd trả "5" thay vì "05", hoặc trả thẳng tên lĩnh vực
        # thay vì mã số). An toàn hơn là tin tưởng mù output của LLM.
        linh_vuc = [code for code in raw_linh_vuc if code in LINH_VUC_TAXONOMY]
        comparison_aspects = [
            str(value).strip() for value in data.get("comparison_aspects") or []
            if str(value).strip()
        ][:10]
        as_of_date = data.get("as_of_date")
        if not isinstance(as_of_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of_date):
            as_of_date = None

        spans: list[SlotSpan] = []
        for value in target_documents:
            spans.append(_match_span(question, value, "target_document"))
        for value in raw_document_types:
            spans.append(_match_span(question, value, "loai_van_ban"))
        # linh_vuc thường là mã số/suy luận, hiếm khi xuất hiện nguyên văn
        # nên không cố match span — chỉ giữ cho phần filter downstream.

        return SlotExtraction(
            target_documents=target_documents,
            loai_van_ban_filter=loai_van_ban_filter,
            linh_vuc=linh_vuc,
            comparison_aspects=comparison_aspects,
            as_of_date=as_of_date,
            spans=spans,
        )

    def _log_for_future_distillation(
        self, question: str, classification: IntentClassification, slots: SlotExtraction
    ) -> None:
        """Ghi mẫu (câu hỏi, nhãn) ra jsonl để tích luỹ dữ liệu huấn luyện.

        Chỉ log các span match được nguyên văn (char_start is not None) —
        mẫu suy luận ngầm không dùng được cho slot-filling nên bỏ qua thay
        vì cố ép match sai.
        """
        record = {
            "text": question,
            "intent": classification.intent.value,
            "scope": classification.scope.value,
            "coverage": classification.coverage.value,
            "requires_history_context": classification.requires_history_context,
            "bio_tags": _spans_to_bio(question, [s for s in slots.spans if s.char_start is not None]),
        }
        try:
            with self.training_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("Không ghi được training log tại %s", self.training_log_path)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    return stripped.strip()


def _normalize_for_match(text: str) -> str:
    # So khớp không phân biệt hoa/thường và dấu, vì Gemini có thể trả về
    # giá trị lệch dấu nhẹ so với câu hỏi gốc.
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _match_span(question: str, value: str, label: str) -> SlotSpan:
    """Tìm vị trí `value` trong `question`. Trả về span=None nếu là suy luận
    ngầm (không xuất hiện nguyên văn) — không cố ép match sai lệch.
    """
    pattern = re.escape(_normalize_for_match(value))
    match = re.search(pattern, _normalize_for_match(question))
    if match is None:
        return SlotSpan(label=label, value=value, char_start=None, char_end=None)
    return SlotSpan(label=label, value=value, char_start=match.start(), char_end=match.end())


def _spans_to_bio(question: str, spans: list[SlotSpan]) -> list[dict]:
    """Chuyển span cấp ký tự thành nhãn BIO cấp token (tách token bằng khoảng
    trắng, đủ dùng làm seed data — tokenizer thật của PhoBERT/XLM-R sẽ
    re-tokenize lại khi huấn luyện chính thức).
    """
    tokens: list[str] = []
    token_spans: list[tuple[int, int]] = []
    for m in re.finditer(r"\S+", question):
        tokens.append(m.group())
        token_spans.append((m.start(), m.end()))

    tags = ["O"] * len(tokens)
    for span in sorted(spans, key=lambda s: s.char_start):
        started = False
        for i, (tok_start, tok_end) in enumerate(token_spans):
            if tok_end <= span.char_start or tok_start >= span.char_end:
                continue
            tags[i] = f"I-{span.label}" if started else f"B-{span.label}"
            started = True

    return [{"token": tok, "tag": tag} for tok, tag in zip(tokens, tags)]
