from __future__ import annotations

import re
import sqlite3
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from app.domain.models import utc_now
from app.domain.legal_document import normalize_document_type_filter


RELATION_MIGRATION = {
    "AMENDS": "SUA_DOI", "SUPPLEMENTS": "BO_SUNG", "REPEALS": "BAI_BO",
    "PARTIALLY_REPEALS": "BAI_BO_MOT_PHAN", "REPLACES": "THAY_THE",
    "SUSPENDS": "DINH_CHI", "EXTENDS": "GIA_HAN", "GUIDES": "HUONG_DAN_THI_HANH",
    "DETAILS": "QUY_DINH_CHI_TIET", "BASED_ON": "CAN_CU", "CONSOLIDATES": "HOP_NHAT",
    "can_cu": "CAN_CU", "thay_the": "THAY_THE", "bai_bo": "BAI_BO",
    "sua_doi_bo_sung": "SUA_DOI_BO_SUNG", "huong_dan_thi_hanh": "HUONG_DAN_THI_HANH",
}


class LegalGraphRepository:
    """Persistent graph: one canonical legal document can have many uploaded versions."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        with self._connection() as connection:
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS legal_documents (
              id TEXT PRIMARY KEY, history_id TEXT NOT NULL, upload_document_id TEXT UNIQUE,
              document_number TEXT, normalized_document_number TEXT, document_type TEXT,
              title TEXT, issuing_authority TEXT, signer TEXT, issued_date TEXT,
              effective_date TEXT, expiry_date TEXT, primary_linh_vuc_code TEXT,
              status_declared TEXT, source_type TEXT NOT NULL, file_checksum TEXT,
              extraction_confidence REAL NOT NULL DEFAULT 0,
              verification_status TEXT NOT NULL DEFAULT 'extracted',
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
              FOREIGN KEY(history_id) REFERENCES history(id) ON DELETE CASCADE,
              FOREIGN KEY(upload_document_id) REFERENCES documents(id) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS idx_legal_documents_history_number
              ON legal_documents(history_id, normalized_document_number);
            CREATE TABLE IF NOT EXISTS legal_document_versions (
              id TEXT PRIMARY KEY, legal_document_id TEXT NOT NULL,
              upload_document_id TEXT NOT NULL UNIQUE, file_name TEXT NOT NULL,
              file_checksum TEXT NOT NULL, is_latest INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL,
              FOREIGN KEY(legal_document_id) REFERENCES legal_documents(id) ON DELETE CASCADE,
              FOREIGN KEY(upload_document_id) REFERENCES documents(id) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS idx_legal_versions_document
              ON legal_document_versions(legal_document_id, created_at);
            CREATE TABLE IF NOT EXISTS legal_document_fields (
              document_id TEXT NOT NULL, linh_vuc_code TEXT NOT NULL,
              is_primary INTEGER NOT NULL DEFAULT 0, confidence REAL NOT NULL DEFAULT 0,
              PRIMARY KEY(document_id, linh_vuc_code),
              FOREIGN KEY(document_id) REFERENCES legal_documents(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS legal_metadata_evidence (
              id TEXT PRIMARY KEY, document_id TEXT NOT NULL, upload_document_id TEXT NOT NULL,
              field_name TEXT NOT NULL, field_value TEXT NOT NULL, element_id TEXT NOT NULL,
              page_number INTEGER NOT NULL, page_end_number INTEGER NOT NULL,
              quote TEXT NOT NULL, quote_hash TEXT NOT NULL,
              validated_exact_match INTEGER NOT NULL, created_at TEXT NOT NULL,
              FOREIGN KEY(document_id) REFERENCES legal_documents(id) ON DELETE CASCADE,
              FOREIGN KEY(upload_document_id) REFERENCES documents(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS legal_provisions (
              id TEXT PRIMARY KEY, document_id TEXT NOT NULL, parent_provision_id TEXT,
              provision_type TEXT NOT NULL, article_number TEXT, clause_number TEXT,
              point_number TEXT, heading TEXT, element_id TEXT NOT NULL,
              page_number INTEGER NOT NULL, page_end_number INTEGER NOT NULL,
              text TEXT NOT NULL, text_hash TEXT NOT NULL, created_at TEXT NOT NULL,
              FOREIGN KEY(document_id) REFERENCES legal_documents(id) ON DELETE CASCADE,
              FOREIGN KEY(parent_provision_id) REFERENCES legal_provisions(id) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS idx_legal_provisions_lookup
              ON legal_provisions(document_id,article_number,clause_number,point_number);
            CREATE TABLE IF NOT EXISTS legal_relations (
              id TEXT PRIMARY KEY, history_id TEXT NOT NULL, source_document_id TEXT NOT NULL,
              target_document_id TEXT NOT NULL, relation_type TEXT NOT NULL, relation_effective_date TEXT,
              affected_provisions TEXT, raw_target_reference TEXT NOT NULL,
              confidence REAL NOT NULL DEFAULT 0,
              verification_status TEXT NOT NULL DEFAULT 'evidence_validated', created_at TEXT NOT NULL,
              FOREIGN KEY(history_id) REFERENCES history(id) ON DELETE CASCADE,
              FOREIGN KEY(source_document_id) REFERENCES legal_documents(id) ON DELETE CASCADE,
              FOREIGN KEY(target_document_id) REFERENCES legal_documents(id) ON DELETE CASCADE);
            CREATE INDEX IF NOT EXISTS idx_legal_relations_target
              ON legal_relations(history_id,target_document_id,relation_type);
            CREATE TABLE IF NOT EXISTS legal_relation_evidence (
              id TEXT PRIMARY KEY, relation_id TEXT NOT NULL, upload_document_id TEXT NOT NULL,
              element_id TEXT NOT NULL, page_number INTEGER NOT NULL, page_end_number INTEGER NOT NULL,
              quote TEXT NOT NULL, quote_hash TEXT NOT NULL, extractor TEXT NOT NULL, model TEXT,
              validated_exact_match INTEGER NOT NULL, created_at TEXT NOT NULL,
              FOREIGN KEY(relation_id) REFERENCES legal_relations(id) ON DELETE CASCADE,
              FOREIGN KEY(upload_document_id) REFERENCES documents(id) ON DELETE CASCADE);
            CREATE TABLE IF NOT EXISTS legal_relation_provisions (
              relation_id TEXT NOT NULL, target_provision_id TEXT NOT NULL, effect_type TEXT NOT NULL,
              PRIMARY KEY(relation_id,target_provision_id),
              FOREIGN KEY(relation_id) REFERENCES legal_relations(id) ON DELETE CASCADE,
              FOREIGN KEY(target_provision_id) REFERENCES legal_provisions(id) ON DELETE CASCADE);
            """)
            relation_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(legal_relations)").fetchall()
            }
            if "effective_date" in relation_columns and "relation_effective_date" not in relation_columns:
                connection.execute(
                    "ALTER TABLE legal_relations RENAME COLUMN effective_date TO relation_effective_date"
                )
            connection.execute("""INSERT OR IGNORE INTO legal_document_versions
              (id,legal_document_id,upload_document_id,file_name,file_checksum,is_latest,created_at)
              SELECT lower(hex(randomblob(16))),ld.id,ld.upload_document_id,d.file_name,
                     COALESCE(ld.file_checksum,''),1,ld.created_at
              FROM legal_documents ld JOIN documents d ON d.id=ld.upload_document_id
              WHERE ld.upload_document_id IS NOT NULL""")
            connection.execute("UPDATE legal_documents SET upload_document_id=NULL")
            for old, new in RELATION_MIGRATION.items():
                connection.execute("UPDATE legal_relations SET relation_type=? WHERE relation_type=?", (new, old))

    def upsert_uploaded_document(self, metadata: dict) -> str:
        now = utc_now().isoformat()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT legal_document_id id FROM legal_document_versions WHERE upload_document_id=?",
                (metadata["upload_document_id"],)).fetchone()
            if row is None and metadata.get("normalized_document_number"):
                row = connection.execute(
                    "SELECT id FROM legal_documents WHERE history_id=? AND normalized_document_number=? ORDER BY created_at LIMIT 1",
                    (metadata["history_id"], metadata["normalized_document_number"])).fetchone()
            legal_id = str(row["id"]) if row else str(uuid.uuid4())
            fields = (
                metadata.get("document_number"), metadata.get("normalized_document_number"),
                metadata.get("document_type"), metadata.get("title"), metadata.get("issuing_authority"),
                metadata.get("signer"), metadata.get("issued_date"), metadata.get("effective_date"),
                metadata.get("expiry_date"), metadata.get("primary_linh_vuc_code"),
                metadata.get("status_declared"), metadata.get("file_checksum"),
                float(metadata.get("extraction_confidence") or 0),
                metadata.get("verification_status", "extracted"), now,
            )
            if row:
                connection.execute("""UPDATE legal_documents SET document_number=?,normalized_document_number=?,
                  document_type=?,title=?,issuing_authority=?,signer=?,issued_date=?,effective_date=?,expiry_date=?,
                  primary_linh_vuc_code=?,status_declared=?,source_type='user_upload',file_checksum=?,
                  extraction_confidence=?,verification_status=?,updated_at=? WHERE id=?""", (*fields, legal_id))
            else:
                connection.execute("""INSERT INTO legal_documents
                  (id,history_id,document_number,normalized_document_number,document_type,title,
                   issuing_authority,signer,issued_date,effective_date,expiry_date,primary_linh_vuc_code,
                   status_declared,source_type,file_checksum,extraction_confidence,verification_status,created_at,updated_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,'user_upload',?,?,?,?,?)""",
                  (legal_id, metadata["history_id"], *fields[:-1], now, now))
            connection.execute("UPDATE legal_document_versions SET is_latest=0 WHERE legal_document_id=?", (legal_id,))
            connection.execute("""INSERT INTO legal_document_versions
              (id,legal_document_id,upload_document_id,file_name,file_checksum,is_latest,created_at)
              VALUES(?,?,?,?,?,1,?) ON CONFLICT(upload_document_id) DO UPDATE SET
              legal_document_id=excluded.legal_document_id,file_name=excluded.file_name,
              file_checksum=excluded.file_checksum,is_latest=1""",
              (str(uuid.uuid4()),legal_id,metadata["upload_document_id"],metadata.get("file_name") or "",
               metadata.get("file_checksum") or "",now))
        return legal_id

    def replace_fields(self, document_id: str, fields: Sequence[tuple[str,bool,float]]) -> None:
        with self._connection() as c:
            c.execute("DELETE FROM legal_document_fields WHERE document_id=?", (document_id,))
            c.executemany("INSERT INTO legal_document_fields VALUES(?,?,?,?)",
                          [(document_id,code,int(primary),confidence) for code,primary,confidence in fields])

    def replace_metadata_evidence(self, document_id: str, evidence: Sequence[dict]) -> None:
        with self._connection() as c:
            c.execute("DELETE FROM legal_metadata_evidence WHERE document_id=?", (document_id,))
            now = utc_now().isoformat()
            c.executemany("""INSERT INTO legal_metadata_evidence VALUES(?,?,?,?,?,?,?,?,?,?,1,?)""",
                [(str(uuid.uuid4()),document_id,e["upload_document_id"],e["field_name"],e["field_value"],
                  e["element_id"],e["page_number"],e["page_end_number"],e["quote"],e["quote_hash"],now)
                 for e in evidence])

    def replace_provisions(self, document_id: str, provisions: Sequence[dict]) -> None:
        with self._connection() as c:
            c.execute("DELETE FROM legal_provisions WHERE document_id=?", (document_id,))
            now = utc_now().isoformat()
            for p in provisions:
                c.execute("""INSERT INTO legal_provisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (p["id"],document_id,p.get("parent_provision_id"),p["provision_type"],p.get("article_number"),
                   p.get("clause_number"),p.get("point_number"),p.get("heading"),p["element_id"],p["page_number"],
                   p["page_end_number"],p["text"],p["text_hash"],now))

    def list_provisions(self, document_id: str, limit: int|None=None, offset: int=0) -> list[dict]:
        return self._paged("SELECT * FROM legal_provisions WHERE document_id=? ORDER BY page_number,article_number,clause_number,point_number",(document_id,),limit,offset)

    @staticmethod
    def _part(value: str, pattern: str) -> str|None:
        value = "".join(c for c in unicodedata.normalize("NFD",value.lower()) if unicodedata.category(c)!="Mn").replace("đ", "d")
        match = re.search(pattern,value)
        return match.group(1) if match else None

    def resolve_provision_ids(self, document_id: str, references: Sequence[str]) -> list[str]:
        result: list[str] = []
        with self._connection() as c:
            for ref in references:
                article=self._part(ref,r"dieu\s+(\d+[a-z]?)"); clause=self._part(ref,r"khoan\s+(\d+)"); point=self._part(ref,r"diem\s+([a-z])")
                if not article: continue
                sql="SELECT id FROM legal_provisions WHERE document_id=? AND lower(article_number)=lower(?)"; params=[document_id,article]
                if clause: sql+=" AND lower(clause_number)=lower(?)"; params.append(clause)
                if point: sql+=" AND lower(point_number)=lower(?)"; params.append(point)
                row=c.execute(sql+" ORDER BY provision_type DESC LIMIT 1",params).fetchone()
                if row: result.append(str(row["id"]))
        return list(dict.fromkeys(result))

    def resolve_or_create_reference(self, history_id: str, document_number: str, normalized_number: str) -> str:
        with self._connection() as c:
            row=c.execute("SELECT id FROM legal_documents WHERE history_id=? AND normalized_document_number=? ORDER BY created_at LIMIT 1",(history_id,normalized_number)).fetchone()
            if row: return str(row["id"])
            legal_id,now=str(uuid.uuid4()),utc_now().isoformat()
            c.execute("""INSERT INTO legal_documents(id,history_id,document_number,normalized_document_number,
              source_type,extraction_confidence,verification_status,created_at,updated_at)
              VALUES(?,?,?,?,'referenced_only',1,'referenced_only',?,?)""",(legal_id,history_id,document_number,normalized_number,now,now))
            return legal_id

    def replace_outgoing_relations(self, history_id: str, source_document_id: str, relations: Sequence[dict]) -> None:
        with self._connection() as c:
            c.execute("DELETE FROM legal_relations WHERE history_id=? AND source_document_id=?",(history_id,source_document_id)); now=utc_now().isoformat()
            for item in relations:
                relation_id=str(uuid.uuid4())
                c.execute("""INSERT INTO legal_relations(id,history_id,source_document_id,target_document_id,relation_type,
                  relation_effective_date,affected_provisions,raw_target_reference,confidence,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                  (relation_id,history_id,source_document_id,item["target_document_id"],item["relation_type"],item.get("relation_effective_date"),
                   item.get("affected_provisions"),item["raw_target_reference"],item["confidence"],now))
                e=item["evidence"]
                c.execute("""INSERT INTO legal_relation_evidence VALUES(?,?,?,?,?,?,?,?,?,?,1,?)""",
                  (str(uuid.uuid4()),relation_id,e["upload_document_id"],e["element_id"],e["page_number"],e["page_end_number"],
                   e["quote"],e["quote_hash"],e["extractor"],e.get("model"),now))
                c.executemany("INSERT OR IGNORE INTO legal_relation_provisions VALUES(?,?,?)",
                              [(relation_id,pid,item["relation_type"]) for pid in item.get("target_provision_ids",[])])

    def _paged(self, sql: str, params: Sequence[object], limit: int|None, offset: int) -> list[dict]:
        values=list(params)
        if limit is not None: sql+=" LIMIT ? OFFSET ?"; values.extend((limit,offset))
        with self._connection() as c: return [dict(r) for r in c.execute(sql,values).fetchall()]

    def list_documents(self, history_id: str, limit: int|None=None, offset: int=0) -> list[dict]:
        rows = self._paged("""SELECT ld.*,(SELECT v.upload_document_id FROM legal_document_versions v WHERE v.legal_document_id=ld.id
          ORDER BY v.is_latest DESC,v.created_at DESC LIMIT 1) latest_upload_document_id FROM legal_documents ld
          WHERE ld.history_id=? ORDER BY ld.created_at""",(history_id,),limit,offset)
        for row in rows:
            row["upload_document_id"] = row.pop("latest_upload_document_id")
        return rows
    def list_versions(self, document_id: str) -> list[dict]:
        return self._paged("SELECT * FROM legal_document_versions WHERE legal_document_id=? ORDER BY created_at",(document_id,),None,0)
    def get_document(self, document_id: str) -> dict | None:
        rows = self._paged("SELECT * FROM legal_documents WHERE id=?", (document_id,), 1, 0)
        return rows[0] if rows else None
    def list_document_type_assignments(self) -> list[dict]:
        return self._paged(
            """SELECT ld.history_id,v.upload_document_id,ld.document_type
               FROM legal_documents ld
               JOIN legal_document_versions v ON v.legal_document_id=ld.id""",
            (), None, 0,
        )
    def list_upload_document_ids_by_types(
        self,
        history_id: str,
        document_type_filter: Sequence[str],
    ) -> set[str]:
        allowed = set(normalize_document_type_filter(document_type_filter))
        if not allowed:
            return set()
        rows = self._paged(
            """SELECT v.upload_document_id,ld.document_type
               FROM legal_documents ld
               JOIN legal_document_versions v ON v.legal_document_id=ld.id
               WHERE ld.history_id=?""",
            (history_id,), None, 0,
        )
        return {
            str(row["upload_document_id"])
            for row in rows
            if row.get("document_type") in allowed
        }
    def get_version_checksum(self, upload_id: str) -> str|None:
        rows=self._paged("SELECT file_checksum FROM legal_document_versions WHERE upload_document_id=?",(upload_id,),1,0)
        return str(rows[0]["file_checksum"]) if rows else None
    def is_latest_version(self, upload_id: str) -> bool:
        rows=self._paged("SELECT is_latest FROM legal_document_versions WHERE upload_document_id=?",(upload_id,),1,0)
        return bool(rows and rows[0]["is_latest"])
    def list_relations(self, history_id: str, limit: int|None=None, offset: int=0) -> list[dict]:
        return self._paged("SELECT * FROM legal_relations WHERE history_id=? ORDER BY created_at",(history_id,),limit,offset)
    def list_fields(self, history_id: str, limit: int|None=None, offset: int=0) -> list[dict]:
        return self._paged("SELECT f.* FROM legal_document_fields f JOIN legal_documents d ON d.id=f.document_id WHERE d.history_id=? ORDER BY f.document_id,f.is_primary DESC",(history_id,),limit,offset)
    def list_all_provisions(self, history_id: str, limit: int|None=None, offset: int=0) -> list[dict]:
        return self._paged("SELECT p.* FROM legal_provisions p JOIN legal_documents d ON d.id=p.document_id WHERE d.history_id=? ORDER BY p.document_id,p.page_number",(history_id,),limit,offset)
    def list_evidence(self, history_id: str, limit: int|None=None, offset: int=0) -> list[dict]:
        return self._paged("SELECT e.* FROM legal_relation_evidence e JOIN legal_relations r ON r.id=e.relation_id WHERE r.history_id=? ORDER BY e.created_at",(history_id,),limit,offset)
    def list_metadata_evidence(self, history_id: str, limit: int|None=None, offset: int=0) -> list[dict]:
        return self._paged("SELECT e.* FROM legal_metadata_evidence e JOIN legal_documents d ON d.id=e.document_id WHERE d.history_id=? ORDER BY e.created_at",(history_id,),limit,offset)
    def list_relation_details(
        self,
        history_id: str,
        document_type_filter: Sequence[str] | None = None,
    ) -> list[dict]:
        details = self._paged("""SELECT r.*,sd.document_number source_number,td.document_number target_number,
          sd.document_type source_document_type,td.document_type target_document_type,v.file_name source_file_name,
          e.upload_document_id,e.element_id,e.page_number,e.page_end_number,e.quote FROM legal_relations r
          JOIN legal_documents sd ON sd.id=r.source_document_id JOIN legal_documents td ON td.id=r.target_document_id
          JOIN legal_relation_evidence e ON e.relation_id=r.id LEFT JOIN legal_document_versions v ON v.upload_document_id=e.upload_document_id
          WHERE r.history_id=? ORDER BY r.created_at""",(history_id,),None,0)
        allowed = set(normalize_document_type_filter(document_type_filter))
        if not allowed:
            return details
        return [
            item for item in details
            if item.get("source_document_type") in allowed
            or item.get("target_document_type") in allowed
        ]
    def find_canonical_documents(
        self,
        history_id: str,
        references: Sequence[str],
        document_type_filter: Sequence[str] | None = None,
    ) -> list[dict]:
        def compact(value: object) -> str:
            plain = "".join(c for c in unicodedata.normalize("NFD", str(value or "").lower())
                            if unicodedata.category(c) != "Mn").replace("đ", "d")
            return "".join(ch for ch in plain if ch.isalnum())
        needles=[compact(r) for r in references if compact(r)]
        docs=self.list_documents(history_id)
        allowed = set(normalize_document_type_filter(document_type_filter))
        if allowed:
            docs = [d for d in docs if d.get("document_type") in allowed]
        return docs if not needles else [d for d in docs if any(n in compact(f"{d.get('document_number','')} {d.get('title','')}") for n in needles)]
    def list_incoming_relation_details(self, history_id: str, target_id: str) -> list[dict]:
        return [r for r in self.list_relation_details(history_id) if r["target_document_id"]==target_id]
    def list_relation_provisions(self, history_id: str, limit: int|None=None, offset: int=0) -> list[dict]:
        return self._paged("""SELECT rp.* FROM legal_relation_provisions rp
          JOIN legal_relations r ON r.id=rp.relation_id WHERE r.history_id=?
          ORDER BY rp.relation_id,rp.target_provision_id""",(history_id,),limit,offset)
    def count(self, table: str, history_id: str) -> int:
        queries={
          "legal_documents":"SELECT count(*) FROM legal_documents WHERE history_id=?",
          "legal_relations":"SELECT count(*) FROM legal_relations WHERE history_id=?",
          "legal_provisions":"SELECT count(*) FROM legal_provisions p JOIN legal_documents d ON d.id=p.document_id WHERE d.history_id=?"}
        if table not in queries: raise ValueError("Unsupported table")
        with self._connection() as c: return int(c.execute(queries[table],(history_id,)).fetchone()[0])
