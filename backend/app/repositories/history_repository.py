from __future__ import annotations
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from app.domain.models import (
    DocumentRecord,
    DocumentStatus,
    HistoryStatus,
    MessageRecord,
    MessageRole,
    HistoryRecord,
    utc_now,
)


def _to_iso(value: datetime) -> str:
    return value.isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class HistoryRepository:
    """Lưu lịch sử hội thoại, tài liệu và tin nhắn trong SQLite."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS history (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id)
                        REFERENCES users(id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_history_user ON history(user_id);

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    history_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    linh_vuc TEXT,
                    graph_status TEXT NOT NULL DEFAULT 'pending',
                    graph_error TEXT,
                    FOREIGN KEY(history_id) REFERENCES history(id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_documents_history
                    ON documents(history_id);

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(history_id) REFERENCES history(id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_history
                    ON messages(history_id, id);
                """
            )
            # Migration cho csdl đã tồn tại từ trước khi có cột linh_vuc — CREATE
            # TABLE IF NOT EXISTS ở trên không thêm cột vào bảng đã có sẵn.
            try:
                connection.execute("ALTER TABLE documents ADD COLUMN linh_vuc TEXT")
            except sqlite3.OperationalError:
                pass  # cột đã tồn tại

            for statement in (
                "ALTER TABLE documents ADD COLUMN graph_status TEXT NOT NULL DEFAULT 'pending'",
                "ALTER TABLE documents ADD COLUMN graph_error TEXT",
            ):
                try:
                    connection.execute(statement)
                except sqlite3.OperationalError:
                    pass

    @staticmethod
    def _history_from_row(row: sqlite3.Row) -> HistoryRecord:
        return HistoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            status=row["status"], 
            created_at=_from_iso(row["created_at"]),
            updated_at=_from_iso(row["updated_at"]),
        )

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            history_id=row["history_id"],
            file_name=row["file_name"],
            mime_type=row["mime_type"],
            size_bytes=int(row["size_bytes"]),
            status=row["status"],
            chunk_count=int(row["chunk_count"]),
            error_message=row["error_message"],
            created_at=_from_iso(row["created_at"]),
            linh_vuc=row["linh_vuc"],
            graph_status=row["graph_status"],
            graph_error=row["graph_error"],
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> MessageRecord:
        return MessageRecord(
            id=int(row["id"]),
            history_id=row["history_id"],
            role=row["role"],
            content=row["content"],
            sources=row["sources"],
            created_at=_from_iso(row["created_at"]),
        )

    def create_history(self, user_id: str, title: str) -> HistoryRecord:
        now = utc_now()
        history = HistoryRecord(
            id=str(uuid.uuid4()),
            user_id = user_id,
            title=title,
            status=HistoryStatus.ACTIVE.value,
            created_at=now,
            updated_at=now,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO history(id, user_id, title,status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    history.id,
                    history.user_id,
                    history.title,
                    history.status,
                    _to_iso(history.created_at),
                    _to_iso(history.updated_at),
                ),
            )
        return history

    def get_history(self, history_id: str) -> HistoryRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM history WHERE id = ?", (history_id,)
            ).fetchone()
        return self._history_from_row(row) if row else None

    def set_history_status(self, history_id: str, status: str) -> HistoryRecord | None:
        now = utc_now()
        with self._connection() as connection:
            connection.execute(
                "UPDATE history SET status = ?, updated_at = ? WHERE id = ?",
                (status, _to_iso(now), history_id),
            )
        return self.get_history(history_id)

    def list_histories_by_user(self, user_id: str, status: str | None = None) -> list[HistoryRecord]:
        query = "SELECT * FROM history WHERE user_id = ?"
        params: list[object] = [user_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC"
        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._history_from_row(row) for row in rows]

    def touch_history(self, history_id: str, title: str |None = None) -> HistoryRecord | None:
        now = utc_now()
        with self._connection() as connection:
            if title is not None:
                connection.execute(
                    """
                    UPDATE history
                    SET updated_at = ?, title = ?
                    WHERE id = ?
                    """,
                    (_to_iso(now), title, history_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE history
                    SET updated_at = ?
                    WHERE id = ?
                    """,
                    (_to_iso(now), history_id),
                )
        return self.get_history(history_id)

    def delete_history(self, history_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM history WHERE id = ?", (history_id,)
            )
        return cursor.rowcount > 0

    def create_document(
        self,
        history_id: str,
        file_name: str,
        mime_type: str,
        size_bytes: int,
    ) -> DocumentRecord:
        now = utc_now()
        document = DocumentRecord(
            id=str(uuid.uuid4()),
            history_id=history_id,
            file_name=file_name,
            mime_type=mime_type,
            size_bytes=size_bytes,
            status=DocumentStatus.PROCESSING.value,
            chunk_count=0,
            error_message=None,
            created_at=now,
            linh_vuc=None,
            graph_status="pending",
            graph_error=None,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO documents(
                    id, history_id, file_name, mime_type, size_bytes,
                    status, chunk_count, error_message, created_at, linh_vuc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.history_id,
                    document.file_name,
                    document.mime_type,
                    document.size_bytes,
                    document.status,
                    document.chunk_count,
                    document.error_message,
                    _to_iso(document.created_at),
                    document.linh_vuc,
                ),
            )
        return document

    def update_document_status(
        self,
        document_id: str,
        status: str,
        chunk_count: int = 0,
        error_message: str | None = None,
    ) -> DocumentRecord | None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, chunk_count = ?, error_message = ?
                WHERE id = ?
                """,
                (status, chunk_count, error_message, document_id),
            )
        return self.get_document(document_id)

    def update_document_linh_vuc(
        self,
        document_id: str,
        linh_vuc: str | None,
    ) -> DocumentRecord | None:
        """Gọi sau khi LinhVucClassificationService phân loại xong. Cho phép ghi None (giữ nguyên "chưa xác định") nếu phân
        loại thất bại hoặc model không chắc chắn."""
        with self._connection() as connection:
            connection.execute(
                "UPDATE documents SET linh_vuc = ? WHERE id = ?",
                (linh_vuc, document_id),
            )
        return self.get_document(document_id)

    def update_document_graph_status(
        self, document_id: str, status: str, error_message: str | None = None
    ) -> DocumentRecord | None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE documents SET graph_status = ?, graph_error = ? WHERE id = ?",
                (status, error_message, document_id),
            )
        return self.get_document(document_id)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(self, history_id: str) -> list[DocumentRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                WHERE history_id = ?
                ORDER BY created_at ASC
                """,
                (history_id,),
            ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def count_documents(
        self,
        history_id: str,
        statuses: Sequence[str] | None = None,
    ) -> int:
        query = "SELECT COUNT(*) FROM documents WHERE history_id = ?"
        params: list[object] = [history_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        with self._connection() as connection:
            return int(connection.execute(query, params).fetchone()[0])

    def delete_document(self, history_id: str, document_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE id = ? AND history_id = ?",
                (document_id, history_id),
            )
        return cursor.rowcount > 0

    def add_message(
        self,
        history_id: str,
        role: MessageRole | str,
        content: str,
        sources: str | None = None,
    ) -> MessageRecord:
        now = utc_now()
        role_value = role.value if isinstance(role, MessageRole) else role
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages(history_id, role, content, sources, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (history_id, role_value, content,sources, _to_iso(now)),
            )
            message_id = int(cursor.lastrowid)
        return MessageRecord(
            id=message_id,
            history_id=history_id,
            role=role_value,
            content=content,
            sources=sources,
            created_at=now,
        )

    def list_messages(
        self,
        history_id: str,
        limit: int | None = None,
    ) -> list[MessageRecord]:
        with self._connection() as connection:
            if limit is None:
                rows = connection.execute(
                    """
                    SELECT * FROM messages
                    WHERE history_id = ? ORDER BY id ASC
                    """,
                    (history_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM messages
                        WHERE history_id = ?
                        ORDER BY id DESC LIMIT ?
                    ) ORDER BY id ASC
                    """,
                    (history_id, limit),
                ).fetchall()
        return [self._message_from_row(row) for row in rows]
