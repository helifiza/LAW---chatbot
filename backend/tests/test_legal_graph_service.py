import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app.domain.models import ChunkDetail
from app.repositories.history_repository import HistoryRepository
from app.repositories.legal_graph_repository import LegalGraphRepository
from app.repositories.user_repository import UserRepository
from app.services.legal_graph_service import LegalGraphService
from app.services.legal_effect_service import LegalEffectService


class FakeGeminiClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def generate(self, **_kwargs) -> str:
        return json.dumps(self.payload, ensure_ascii=False)


class LegalGraphServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        root = Path(self.temp_directory.name)
        self.file_path = root / "nghi_dinh.pdf"
        self.file_path.write_bytes(b"fixture")
        database = root / "state.db"
        self.database = database
        user = UserRepository(database).create_user("Test", "graph@example.com", "hash")
        self.histories = HistoryRepository(database)
        self.history = self.histories.create_history(user["id"], "Graph")
        self.upload = self.histories.create_document(
            self.history.id, "nghi_dinh.pdf", "application/pdf", 7
        )
        self.graph = LegalGraphRepository(database)
        self.chunk = ChunkDetail(
            history_id=self.history.id,
            document_id=self.upload.id,
            user_id=user["id"],
            element_id="element-1",
            file_name="nghi_dinh.pdf",
            page_number=3,
            page_end_number=3,
            content_type="dieu",
            chuong=None,
            muc=None,
            dieu="Điều 10",
            text=(
                "Nghị định số 50/2026/NĐ-CP ban hành ngày 2026-01-01, có hiệu lực ngày 2026-02-01.\n"
                "Điều 10. Điều khoản thi hành\n"
                "1. Nghị định này thay thế Nghị định số 12/2020/NĐ-CP.\n"
                "a) Cơ quan có trách nhiệm thi hành."
            ),
            chunk_index=0,
            char_count=55,
            token_count=12,
            created_at="2026-01-01T00:00:00+00:00",
            linh_vuc="09",
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def _payload(self, quote: str) -> dict:
        return {
            "document": {
                "document_number": "50/2026/NĐ-CP",
                "document_type": "nghị định",
                "title": "Nghị định thử nghiệm",
                "issuing_authority": "Chính phủ",
                "signer": None,
                "issued_date": "2026-01-01",
                "effective_date": "2026-02-01",
                "expiry_date": None,
                "status_declared": None,
                "confidence": 0.95,
            },
            "secondary_linh_vuc_codes": ["11", "99"],
            "metadata_evidence": [
                {"field_name": "document_number", "field_value": "50/2026/NĐ-CP",
                 "evidence_element_id": "element-1", "quote": "Nghị định số 50/2026/NĐ-CP"},
                {"field_name": "issued_date", "field_value": "2026-01-01",
                 "evidence_element_id": "element-1", "quote": "ban hành ngày 2026-01-01"},
                {"field_name": "effective_date", "field_value": "2026-02-01",
                 "evidence_element_id": "element-1", "quote": "có hiệu lực ngày 2026-02-01"},
            ],
            "relations": [{
                "relation_type": "REPLACES",
                "target_document_number": "12/2020/NĐ-CP",
                "raw_target_reference": "Nghị định số 12/2020/NĐ-CP",
                "relation_effective_date": "2026-02-01",
                "affected_provisions": [],
                "evidence_element_id": "element-1",
                "quote": quote,
                "confidence": 0.94,
            }],
        }

    def test_persists_uploaded_node_placeholder_and_validated_relation(self) -> None:
        service = LegalGraphService(self.graph, FakeGeminiClient(self._payload(self.chunk.text)), "test")
        service.ingest(
            self.history.id, self.upload.id, self.file_path,
            "nghi_dinh.pdf", [self.chunk], "09",
        )

        documents = self.graph.list_documents(self.history.id)
        relations = self.graph.list_relations(self.history.id)
        self.assertEqual({item["source_type"] for item in documents}, {"user_upload", "referenced_only"})
        self.assertEqual(relations[0]["relation_type"], "THAY_THE")
        self.assertEqual(relations[0]["relation_effective_date"], "2026-02-01")
        self.assertEqual(relations[0]["raw_target_reference"], "Nghị định số 12/2020/NĐ-CP")
        uploaded = next(item for item in documents if item["source_type"] == "user_upload")
        provision_types = {
            item["provision_type"] for item in self.graph.list_provisions(uploaded["id"])
        }
        self.assertEqual(provision_types, {"article", "clause", "point"})
        self.assertEqual(
            len(self.graph.list_relation_details(self.history.id, ["NGHI DINH"])),
            1,
        )
        self.assertEqual(
            self.graph.list_relation_details(self.history.id, ["luật"]),
            [],
        )
        self.assertEqual(
            [item["id"] for item in self.graph.find_canonical_documents(
                self.history.id, [], ["nghị định"]
            )],
            [uploaded["id"]],
        )

    def test_rejects_relation_when_quote_is_not_in_chunk(self) -> None:
        service = LegalGraphService(self.graph, FakeGeminiClient(self._payload("Câu do model tự tạo")), "test")
        service.ingest(
            self.history.id, self.upload.id, self.file_path,
            "nghi_dinh.pdf", [self.chunk], "09",
        )

        self.assertEqual(self.graph.list_relations(self.history.id), [])

    def test_invalid_number_and_dates_are_not_persisted(self) -> None:
        payload = self._payload(self.chunk.text)
        payload["document"]["document_number"] = "không có số hiệu"
        payload["document"]["issued_date"] = "2026-99-99"
        payload["document"]["effective_date"] = "2025-01-01"
        service = LegalGraphService(self.graph, FakeGeminiClient(payload), "test")
        service.ingest(
            self.history.id, self.upload.id, self.file_path,
            "nghi_dinh.pdf", [self.chunk], "09",
        )

        uploaded = next(
            item for item in self.graph.list_documents(self.history.id)
            if item["source_type"] == "user_upload"
        )
        self.assertIsNone(uploaded["document_number"])
        self.assertIsNone(uploaded["issued_date"])

    def test_two_uploads_share_one_canonical_document_with_two_versions(self) -> None:
        service = LegalGraphService(self.graph, FakeGeminiClient(self._payload(self.chunk.text)), "test")
        service.ingest(self.history.id, self.upload.id, self.file_path, "v1.pdf", [self.chunk], "09")
        second = self.histories.create_document(self.history.id, "v2.pdf", "application/pdf", 7)
        chunk_v2 = ChunkDetail(**{**self.chunk.__dict__, "document_id": second.id, "file_name": "v2.pdf"})
        service.ingest(self.history.id, second.id, self.file_path, "v2.pdf", [chunk_v2], "09")

        canonical = [d for d in self.graph.list_documents(self.history.id) if d["source_type"] == "user_upload"]
        self.assertEqual(len(canonical), 1)
        self.assertEqual(len(self.graph.list_versions(canonical[0]["id"])), 2)

    def test_graph_lists_are_paginated(self) -> None:
        service = LegalGraphService(self.graph, FakeGeminiClient(self._payload(self.chunk.text)), "test")
        service.ingest(self.history.id, self.upload.id, self.file_path, "v1.pdf", [self.chunk], "09")

        first_page = self.graph.list_documents(self.history.id, limit=1, offset=0)
        second_page = self.graph.list_documents(self.history.id, limit=1, offset=1)
        self.assertEqual(len(first_page), 1)
        self.assertEqual(len(second_page), 1)
        self.assertNotEqual(first_page[0]["id"], second_page[0]["id"])
        self.assertEqual(self.graph.count("legal_documents", self.history.id), 2)

    def test_migrates_legacy_relation_effective_date_column(self) -> None:
        service = LegalGraphService(self.graph, FakeGeminiClient(self._payload(self.chunk.text)), "test")
        service.ingest(self.history.id, self.upload.id, self.file_path, "v1.pdf", [self.chunk], "09")
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "ALTER TABLE legal_relations RENAME COLUMN relation_effective_date TO effective_date"
            )
            connection.commit()

        migrated = LegalGraphRepository(self.database)

        with closing(sqlite3.connect(self.database)) as connection:
            columns = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info(legal_relations)"
                ).fetchall()
            }
        self.assertIn("relation_effective_date", columns)
        self.assertNotIn("effective_date", columns)
        self.assertEqual(
            migrated.list_relations(self.history.id)[0]["relation_effective_date"],
            "2026-02-01",
        )

    def test_metadata_value_is_rejected_when_evidence_disagrees(self) -> None:
        payload = self._payload(self.chunk.text)
        payload["metadata_evidence"][0]["field_value"] = "99/2026/NĐ-CP"
        service = LegalGraphService(self.graph, FakeGeminiClient(payload), "test")
        service.ingest(self.history.id, self.upload.id, self.file_path, "v1.pdf", [self.chunk], "09")

        uploaded = next(
            d for d in self.graph.list_documents(self.history.id)
            if d["source_type"] == "user_upload"
        )
        self.assertIsNone(uploaded["document_number"])

    def test_relation_targets_the_exact_point_node(self) -> None:
        target_service = LegalGraphService(self.graph, FakeGeminiClient(self._payload(self.chunk.text)), "test")
        target_service.ingest(self.history.id, self.upload.id, self.file_path, "target.pdf", [self.chunk], "09")
        source = self.histories.create_document(self.history.id, "source.pdf", "application/pdf", 7)
        source_text = (
            "Nghị định số 60/2026/NĐ-CP sửa đổi điểm a khoản 1 Điều 10 "
            "Nghị định số 50/2026/NĐ-CP."
        )
        source_chunk = ChunkDetail(**{
            **self.chunk.__dict__, "document_id": source.id, "file_name": "source.pdf",
            "element_id": "source-element", "dieu": "Điều 1", "text": source_text,
        })
        payload = {
            "document": {"document_number":"60/2026/NĐ-CP","document_type":"nghị định",
                "title":"Văn bản sửa đổi","confidence":0.9},
            "metadata_evidence":[{"field_name":"document_number","field_value":"60/2026/NĐ-CP",
                "evidence_element_id":"source-element","quote":"Nghị định số 60/2026/NĐ-CP"}],
            "relations":[{"relation_type":"SUA_DOI","target_document_number":"50/2026/NĐ-CP",
                "raw_target_reference":"Nghị định số 50/2026/NĐ-CP","relation_effective_date":None,
                "affected_provisions":["điểm a khoản 1 Điều 10"],
                "evidence_element_id":"source-element","quote":source_text,"confidence":0.9}],
        }
        LegalGraphService(self.graph, FakeGeminiClient(payload), "test").ingest(
            self.history.id, source.id, self.file_path, "source.pdf", [source_chunk], "09")

        links = self.graph.list_relation_provisions(self.history.id)
        self.assertEqual(len(links), 1)
        point = next(p for p in self.graph.list_all_provisions(self.history.id)
                     if p["point_number"] == "a" and p["article_number"] == "10")
        self.assertEqual(links[0]["target_provision_id"], point["id"])
        self.histories.update_document_graph_status(self.upload.id, "ready")
        effect = LegalEffectService(self.graph, self.histories).evaluate(
            self.history.id, ["50/2026/NĐ-CP"], "2026-09-09"
        )
        self.assertEqual(effect["status"], "PARTIALLY_EFFECTIVE", effect)
        self.assertEqual(len(effect["sources"]), 2)


if __name__ == "__main__":
    unittest.main()
