from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class QueryRoute(str, Enum):
    SPECIFIC = "specific"
    SUMMARY = "summary"

@dataclass(frozen = True)
class RouteResult:
    route: QueryRoute
    matched_keyword: str | None = None

SUMMARY_KEYWORDS = ("tóm tắt", "tổng hợp", "liệt kê toàn bộ", "khái quát", "tổng quan")
class QueryRouterService:
    def __init__(self, gemini_client, model: str) -> None:
        self.gemini_client = gemini_client
        self.model = model
    def route(self, question: str) -> RouteResult:
        lowered = question.lower()#chuyển question thành viết thường
        for kw in SUMMARY_KEYWORDS:
            if kw in lowered:
                return RouteResult(QueryRoute.SUMMARY, matched_keyword=kw)
        return RouteResult(QueryRoute.SPECIFIC)
