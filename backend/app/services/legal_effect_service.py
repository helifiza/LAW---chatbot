from __future__ import annotations

from datetime import date
from typing import Sequence

from app.domain.models import GraphStatus, RelationType
from app.repositories.history_repository import HistoryRepository
from app.repositories.legal_graph_repository import LegalGraphRepository


class LegalEffectService:

    def __init__(self, graph: LegalGraphRepository, histories: HistoryRepository) -> None:
        self.graph = graph
        self.histories = histories

    def evaluate(
        self,
        history_id: str,
        targets: Sequence[str],
        as_of: str | None,
        document_type_filter: Sequence[str] | None = None,
    ) -> dict:
        documents = self.graph.find_canonical_documents(
            history_id, targets, document_type_filter
        )
        if len(documents) != 1:
            return {"answer": (
                "Không thể xác định duy nhất văn bản cần kiểm tra hiệu lực. "
                "Hãy nêu rõ số hiệu văn bản."), "sources": [], "status": "INSUFFICIENT_DATA",
                "warnings": [f"Số node canonical phù hợp: {len(documents)}"]}
        document = documents[0]
        upload_id = document.get("upload_document_id")
        upload = self.histories.get_document(str(upload_id)) if upload_id else None
        if upload is None or upload.graph_status != GraphStatus.READY.value:
            return {"answer":"Graph của văn bản chưa hoàn tất nên chưa thể kết luận hiệu lực.",
                    "sources":[],"status":"INSUFFICIENT_DATA","warnings":["graph_not_ready"]}
        try: check_date = date.fromisoformat(as_of) if as_of else date.today()
        except ValueError: check_date = date.today()
        relations = self.graph.list_incoming_relation_details(history_id, document["id"])
        active = []
        for relation in relations:
            relation_date = relation.get("relation_effective_date")
            if relation_date and date.fromisoformat(relation_date) > check_date: continue
            active.append(relation)
        terminal = next((r for r in active if r["relation_type"] in {
            RelationType.BAI_BO.value, RelationType.THAY_THE.value}), None)
        suspended = next((r for r in active if r["relation_type"] == RelationType.DINH_CHI.value), None)
        partial = [r for r in active if r["relation_type"] in {
            RelationType.BAI_BO_MOT_PHAN.value, RelationType.SUA_DOI.value,
            RelationType.BO_SUNG.value, RelationType.SUA_DOI_BO_SUNG.value}]
        effective = self._date(document.get("effective_date")); expiry = self._date(document.get("expiry_date"))
        declared = self._plain(document.get("status_declared"))
        if "het hieu luc" in declared or "bai bo" in declared:
            status, reason = "EXPIRED", "trạng thái hết hiệu lực/bãi bỏ đã được xác thực trong văn bản"
        elif terminal: status, reason = "EXPIRED", f"bị {terminal['relation_type']}"
        elif suspended: status, reason = "SUSPENDED", "bị đình chỉ"
        elif expiry and expiry <= check_date: status, reason = "EXPIRED", f"hết hạn ngày {expiry.isoformat()}"
        elif effective and effective > check_date: status, reason = "NOT_YET_EFFECTIVE", f"có hiệu lực từ {effective.isoformat()}"
        elif not effective:
            status, reason = "INSUFFICIENT_DATA", "chưa có ngày hiệu lực đã xác thực bằng evidence"
        elif partial:
            status, reason = "PARTIALLY_EFFECTIVE", "có quan hệ sửa đổi/bãi bỏ một phần"
        else: status, reason = "EFFECTIVE", "chưa phát hiện quan hệ chấm dứt hiệu lực trong tập tài liệu đã tải lên"
        sources = [self._relation_source(r) for r in active if r["relation_type"] in {
            RelationType.BAI_BO.value,RelationType.THAY_THE.value,RelationType.DINH_CHI.value,
            RelationType.BAI_BO_MOT_PHAN.value,RelationType.SUA_DOI.value,
            RelationType.BO_SUNG.value,RelationType.SUA_DOI_BO_SUNG.value}]
        for evidence in self.graph.list_metadata_evidence(history_id):
            if evidence["document_id"] != document["id"] or evidence["field_name"] not in {
                "effective_date", "expiry_date", "status_declared"
            }:
                continue
            evidence_upload = self.histories.get_document(evidence["upload_document_id"])
            sources.append({"document_id":evidence["upload_document_id"],
                "file_name":evidence_upload.file_name if evidence_upload else "tài liệu đã tải lên",
                "page_number":evidence["page_number"],"page_end_number":evidence["page_end_number"],
                "dieu":None,"score":1.0,"excerpt":evidence["quote"]})
        citations = " ".join(dict.fromkeys(
            f"[{source['file_name']}, trang {source['page_number']}]"
            if source["page_number"] == source["page_end_number"] else
            f"[{source['file_name']}, trang {source['page_number']}-{source['page_end_number']}]"
            for source in sources
        ))
        answer = (f"Kết quả tại ngày {check_date.isoformat()}: {status}; "
                  f"{document.get('document_number') or document.get('title')}: {reason} {citations}. "
                  "Phạm vi dữ liệu: các phiên bản và quan hệ có evidence trong phiên hiện tại.")
        return {"answer":answer,"sources":sources,"status":status,"warnings":[]}

    @staticmethod
    def _date(value: object) -> date | None:
        try: return date.fromisoformat(str(value)) if value else None
        except ValueError: return None

    @staticmethod
    def _plain(value: object) -> str:
        import unicodedata
        return "".join(c for c in unicodedata.normalize("NFD", str(value or "").lower())
                       if unicodedata.category(c) != "Mn").replace("đ", "d")

    @staticmethod
    def _relation_source(item: dict) -> dict:
        return {"document_id":item["upload_document_id"],
                "file_name":item.get("source_file_name") or "tài liệu đã tải lên",
                "page_number":item["page_number"],"page_end_number":item["page_end_number"],
                "dieu":None,"score":float(item["confidence"]),"excerpt":item["quote"]}
