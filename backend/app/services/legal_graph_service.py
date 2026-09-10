from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
import uuid
from datetime import date
from pathlib import Path
from typing import Sequence

from app.domain.models import ChunkDetail, RelationType
from app.domain.legal_document import LEGAL_DOCUMENT_TYPES, normalize_document_type
from app.repositories.legal_graph_repository import LegalGraphRepository, RELATION_MIGRATION
from app.services.query_router_service import LINH_VUC_TAXONOMY


RELATION_TYPES = {item.value for item in RelationType}
RELATION_MARKERS = re.compile(
    r"bãi bỏ|thay thế|sửa đổi|bổ sung|ngưng hiệu lực|đình chỉ|gia hạn|"
    r"hướng dẫn thi hành|quy định chi tiết|căn cứ|có hiệu lực|kể từ ngày", re.I)
DOCUMENT_TYPES = set(LEGAL_DOCUMENT_TYPES)
EVIDENCE_REQUIRED_FIELDS = {
    "document_number", "issuing_authority", "signer", "issued_date",
    "effective_date", "expiry_date", "status_declared",
}
#issuing_authority: cơ quan ban hành văn bản
#issued_date: ngày ban hành văn bản
SYSTEM_PROMPT = """Bạn trích metadata và quan hệ từ văn bản pháp luật Việt Nam.
Chỉ trả về một JSON object, không markdown và không suy đoán. Schema:
{"document":{"document_number":string|null,"document_type":string|null,"title":string|null,
"issuing_authority":string|null,"signer":string|null,"issued_date":"YYYY-MM-DD"|null,
"effective_date":"YYYY-MM-DD"|null,"expiry_date":"YYYY-MM-DD"|null,
"status_declared":string|null,"confidence":number},
"metadata_evidence":[{"field_name":string,"field_value":string,
"evidence_element_id":string,"quote":string}],
"secondary_linh_vuc_codes":["01"],
"relations":[{"relation_type":"CAN_CU"|"HUONG_DAN_THI_HANH"|"QUY_DINH_CHI_TIET"|
"SUA_DOI"|"BO_SUNG"|"SUA_DOI_BO_SUNG"|"THAY_THE"|"BAI_BO"|"BAI_BO_MOT_PHAN"|
"DINH_CHI"|"GIA_HAN"|"HOP_NHAT","target_document_number":string,
"raw_target_reference":string,"relation_effective_date":"YYYY-MM-DD"|null,
"affected_provisions":[string],"evidence_element_id":string,"quote":string,"confidence":number}]}
Mọi evidence quote phải chép nguyên văn từ đúng element. Chỉ tạo relation khi có số hiệu văn bản đích.
affected_provisions phải ghi rõ tới Điều/Khoản/Điểm nếu câu nguồn nêu rõ."""


def normalize_document_number(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value.upper())
    plain = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", "", plain).replace("Đ", "D")


def normalize_quote(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def json_object(raw: str) -> dict:
    value = raw.strip()
    if value.startswith("```"):
        value = value.strip("`").strip()
        if value.lower().startswith("json"): value = value[4:].strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict): raise ValueError("Legal graph extractor must return an object")
    return parsed


def valid_iso_date(value: object) -> str | None:
    if not isinstance(value, str): return None
    try: return date.fromisoformat(value).isoformat()
    except ValueError: return None

#check số hiệu văn bản
def valid_document_number(value: object) -> str | None:
    if not isinstance(value, str): return None
    cleaned = " ".join(value.split()).strip()
    return cleaned if cleaned and len(cleaned) <= 100 and "/" in cleaned and re.search(r"\d", cleaned) else None


def provision_number(label: str | None) -> str | None:
    if not label: return None
    plain = "".join(c for c in unicodedata.normalize("NFD", label) if unicodedata.category(c) != "Mn").replace("Đ", "D").replace("đ", "d")
    match = re.search(r"Dieu\s+([\w.-]+)", plain, re.I)
    return match.group(1) if match else None


class LegalGraphService:
    def __init__(self, repository: LegalGraphRepository, gemini_client, model: str,
                 logger: logging.Logger | None = None) -> None:
        self.repository = repository
        self.gemini_client = gemini_client
        self.model = model
        self.logger = logger or logging.getLogger(__name__)

    def ingest(self, history_id: str, upload_document_id: str, file_path: Path | None,
               file_name: str, chunks: Sequence[ChunkDetail],
               primary_linh_vuc_code: str | None, file_checksum: str | None = None) -> str | None:
        if not chunks: return None
        candidates = [c for c in chunks if RELATION_MARKERS.search(c.text)]
        selected = {c.element_id: c for c in list(chunks[:4]) + candidates[:30]}
        context = "\n\n".join(
            f"[ELEMENT {c.element_id} | trang {c.page_number}-{c.page_end_number}]\n{c.text}"
            for c in selected.values())
        data = json_object(self.gemini_client.generate(
            model=self.model,
            prompt=f"TÊN FILE: {file_name}\nLĨNH VỰC CHÍNH: {primary_linh_vuc_code or 'chưa xác định'}\n\n{context}",
            system_instruction=SYSTEM_PROMPT, temperature=0.0, max_output_tokens=4000))
        document = data.get("document") if isinstance(data.get("document"), dict) else {}
        evidence = self._validate_metadata_evidence(
            upload_document_id, data.get("metadata_evidence") or [], selected)
        evidence = [e for e in evidence if self._metadata_values_match(
            e["field_name"], e["field_value"], document.get(e["field_name"]))]
        verified_fields = {e["field_name"] for e in evidence}

        number = valid_document_number(document.get("document_number")) if "document_number" in verified_fields else None
        document_type = normalize_document_type(document.get("document_type"))
        if document_type not in DOCUMENT_TYPES: document_type = None
        issued = valid_iso_date(document.get("issued_date")) if "issued_date" in verified_fields else None
        effective = valid_iso_date(document.get("effective_date")) if "effective_date" in verified_fields else None
        expiry = valid_iso_date(document.get("expiry_date")) if "expiry_date" in verified_fields else None
        if issued and effective and effective < issued: effective = None
        if effective and expiry and expiry < effective: expiry = None
        checksum = file_checksum or (
            hashlib.sha256(file_path.read_bytes()).hexdigest() if file_path is not None else "")
        metadata = {
            "history_id": history_id, "upload_document_id": upload_document_id,
            "file_name": file_name, "document_number": number,
            "normalized_document_number": normalize_document_number(number) if number else None,
            "document_type": document_type, "title": document.get("title"),
            "issuing_authority": document.get("issuing_authority") if "issuing_authority" in verified_fields else None,
            "signer": document.get("signer") if "signer" in verified_fields else None,
            "issued_date": issued, "effective_date": effective, "expiry_date": expiry,
            "primary_linh_vuc_code": primary_linh_vuc_code,
            "status_declared": document.get("status_declared") if "status_declared" in verified_fields else None,
            "file_checksum": checksum,
            "extraction_confidence": max(0.0,min(1.0,float(document.get("confidence") or 0))),
            "verification_status": "evidence_validated" if verified_fields & EVIDENCE_REQUIRED_FIELDS else "partially_verified",
        }
        legal_id = self.repository.upsert_uploaded_document(metadata)
        self.repository.replace_metadata_evidence(legal_id, evidence)
        codes = ([primary_linh_vuc_code] if primary_linh_vuc_code else []) + [
            c for c in data.get("secondary_linh_vuc_codes") or [] if c in LINH_VUC_TAXONOMY]
        codes = list(dict.fromkeys(codes))
        self.repository.replace_fields(legal_id, [
            (c,c == primary_linh_vuc_code,1.0 if c == primary_linh_vuc_code else 0.7) for c in codes])
        self.repository.replace_provisions(legal_id, self._extract_provisions(legal_id, chunks))
        relations = []
        for raw in data.get("relations") or []:
            validated = self._validate_relation(history_id, upload_document_id, raw, selected)
            if validated: relations.append(validated)
        self.repository.replace_outgoing_relations(history_id, legal_id, relations)
        return legal_id

    def _validate_metadata_evidence(self, upload_id: str, raw_items: Sequence[dict],
                                    chunks: dict[str,ChunkDetail]) -> list[dict]:
        result = []
        for item in raw_items:
            field = str(item.get("field_name") or "")
            value = str(item.get("field_value") or "").strip()
            element = str(item.get("evidence_element_id") or "")
            quote = str(item.get("quote") or "").strip(); chunk = chunks.get(element)
            if field not in EVIDENCE_REQUIRED_FIELDS or not value or not quote or chunk is None: continue
            if normalize_quote(quote) not in normalize_quote(chunk.text): continue
            if normalize_quote(value).casefold() not in normalize_quote(quote).casefold(): continue
            result.append({"upload_document_id":upload_id,"field_name":field,"field_value":value,
                "element_id":element,"page_number":chunk.page_number,"page_end_number":chunk.page_end_number,
                "quote":quote,"quote_hash":hashlib.sha256(quote.encode()).hexdigest()})
        return result

    @staticmethod
    def _metadata_values_match(field: str, evidence_value: str, extracted_value: object) -> bool:
        if extracted_value is None:
            return False
        if field == "document_number":
            return normalize_document_number(evidence_value) == normalize_document_number(str(extracted_value))
        return normalize_quote(evidence_value).casefold() == normalize_quote(str(extracted_value)).casefold()

    def _validate_relation(self, history_id: str, upload_id: str, relation: dict,
                           chunks: dict[str,ChunkDetail]) -> dict | None:
        relation_type = RELATION_MIGRATION.get(str(relation.get("relation_type")), str(relation.get("relation_type")))
        target = str(relation.get("target_document_number") or "").strip()
        element = str(relation.get("evidence_element_id") or "")
        quote = str(relation.get("quote") or "").strip(); chunk = chunks.get(element)
        if relation_type not in RELATION_TYPES or not valid_document_number(target) or chunk is None or not quote: return None
        if normalize_quote(quote) not in normalize_quote(chunk.text): return None
        normalized = normalize_document_number(target)
        if normalized not in normalize_document_number(quote): return None
        target_id = self.repository.resolve_or_create_reference(history_id,target,normalized)
        affected = [str(v) for v in relation.get("affected_provisions") or []]
        return {"target_document_id":target_id,"relation_type":relation_type,
            "relation_effective_date":valid_iso_date(relation.get("relation_effective_date")),
            "affected_provisions":json.dumps(affected,ensure_ascii=False),
            "target_provision_ids":self.repository.resolve_provision_ids(target_id,affected),
            "raw_target_reference":str(relation.get("raw_target_reference") or target),
            "confidence":max(0.0,min(1.0,float(relation.get("confidence") or 0))),
            "evidence":{"upload_document_id":upload_id,"element_id":element,
                "page_number":chunk.page_number,"page_end_number":chunk.page_end_number,"quote":quote,
                "quote_hash":hashlib.sha256(quote.encode()).hexdigest(),"extractor":"gemini","model":self.model}}

    @staticmethod
    def _extract_provisions(legal_id: str, chunks: Sequence[ChunkDetail]) -> list[dict]:
        namespace = uuid.UUID(legal_id)
        grouped: dict[str,list[ChunkDetail]] = {}
        for chunk in chunks:
            number = provision_number(chunk.dieu)
            if number: grouped.setdefault(number,[]).append(chunk)
        provisions: list[dict] = []
        for article, items in grouped.items():
            items.sort(key=lambda c:c.chunk_index); first,last=items[0],items[-1]
            text = LegalGraphService._merge_overlap([c.text for c in items])
            article_id=str(uuid.uuid5(namespace,f"article:{article.lower()}"))
            provisions.append(LegalGraphService._node(article_id,None,"article",article,None,None,
                first.dieu,first,last,text))
            lines=text.splitlines(); clause_starts=[]
            for index,line in enumerate(lines):
                match=re.match(r"^\s*(\d+)\.\s+\S",line)#nhận diện khoản theo từng dòng
                if match: clause_starts.append((index,match.group(1)))
            for pos,(start,clause) in enumerate(clause_starts):
                end=clause_starts[pos+1][0] if pos+1<len(clause_starts) else len(lines)
                clause_lines=lines[start:end]; clause_text="\n".join(clause_lines).strip()
                clause_id=str(uuid.uuid5(namespace,f"clause:{article.lower()}:{clause.lower()}"))
                provisions.append(LegalGraphService._node(clause_id,article_id,"clause",article,clause,None,
                    None,first,last,clause_text))
                point_starts=[]
                for relative,line in enumerate(clause_lines):
                    match=re.match(r"^\s*([a-zđ])\)\s+\S",line,re.I)#nhận diện điểm
                    if match: point_starts.append((relative,match.group(1).lower()))
                for pidx,(pstart,point) in enumerate(point_starts):
                    pend=point_starts[pidx+1][0] if pidx+1<len(point_starts) else len(clause_lines)
                    point_text="\n".join(clause_lines[pstart:pend]).strip()
                    point_id=str(uuid.uuid5(namespace,f"point:{article.lower()}:{clause.lower()}:{point}"))
                    provisions.append(LegalGraphService._node(point_id,clause_id,"point",article,clause,point,
                        None,first,last,point_text))
        return provisions

    @staticmethod
    def _node(node_id: str,parent: str|None,kind: str,article: str,clause: str|None,
              point: str|None,heading: str|None,first: ChunkDetail,last: ChunkDetail,text: str) -> dict:
        return {"id":node_id,"parent_provision_id":parent,"provision_type":kind,
            "article_number":article,"clause_number":clause,"point_number":point,"heading":heading,
            "element_id":first.element_id,"page_number":first.page_number,"page_end_number":last.page_end_number,
            "text":text,"text_hash":hashlib.sha256(text.encode()).hexdigest()}

    @staticmethod
    def _merge_overlap(texts: Sequence[str]) -> str:
        if not texts: return ""
        merged=texts[0]
        for text in texts[1:]:
            overlap=0
            for length in range(min(500,len(merged),len(text)),0,-1):
                if merged[-length:]==text[:length]: overlap=length; break
            merged += text[overlap:]
        return merged
