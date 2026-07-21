from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Sequence

from app.domain.models import (
    DocumentRecord,
    DocumentStatus,
    MessageRecord,
    MessageRole,
    SessionRecord,
    SessionStatus,
    utc_now,
)


def _to_iso(value: datetime) -> str:
    return value.isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SessionRepository:
    """Lưu trạng thái phiên, tài liệu và lịch sử chat trong SQLite."""

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
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_documents_session
                    ON documents(session_id);

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
                """
            )

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            id=row["id"],
            status=row["status"],
            created_at=_from_iso(row["created_at"]),
            updated_at=_from_iso(row["updated_at"]),
            expires_at=_from_iso(row["expires_at"]),
        )

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            session_id=row["session_id"],
            file_name=row["file_name"],
            mime_type=row["mime_type"],
            size_bytes=int(row["size_bytes"]),
            status=row["status"],
            chunk_count=int(row["chunk_count"]),
            error_message=row["error_message"],
            created_at=_from_iso(row["created_at"]),
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> MessageRecord:
        return MessageRecord(
            id=int(row["id"]),
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=_from_iso(row["created_at"]),
        )

    def create_session(self, ttl_minutes: int) -> SessionRecord:
        now = utc_now()
        session = SessionRecord(
            id=str(uuid.uuid4()),
            status=SessionStatus.ACTIVE.value,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(minutes=ttl_minutes),
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions(id, status, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.status,
                    _to_iso(session.created_at),
                    _to_iso(session.updated_at),
                    _to_iso(session.expires_at),
                ),
            )
        return session

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._session_from_row(row) if row else None

    def touch_session(self, session_id: str, ttl_minutes: int) -> SessionRecord | None:
        now = utc_now()
        expires_at = now + timedelta(minutes=ttl_minutes)
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE sessions
                SET updated_at = ?, expires_at = ?
                WHERE id = ? AND status = ?
                """,
                (
                    _to_iso(now),
                    _to_iso(expires_at),
                    session_id,
                    SessionStatus.ACTIVE.value,
                ),
            )
        return self.get_session(session_id)

    def list_expired_session_ids(self, at: datetime | None = None) -> list[str]:
        moment = at or utc_now()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id FROM sessions
                WHERE expires_at <= ? OR status != ?
                """,
                (_to_iso(moment), SessionStatus.ACTIVE.value),
            ).fetchall()
        return [row["id"] for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM sessions WHERE id = ?", (session_id,)
            )
        return cursor.rowcount > 0

    def create_document(
        self,
        session_id: str,
        file_name: str,
        mime_type: str,
        size_bytes: int,
    ) -> DocumentRecord:
        now = utc_now()
        document = DocumentRecord(
            id=str(uuid.uuid4()),
            session_id=session_id,
            file_name=file_name,
            mime_type=mime_type,
            size_bytes=size_bytes,
            status=DocumentStatus.PROCESSING.value,
            chunk_count=0,
            error_message=None,
            created_at=now,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO documents(
                    id, session_id, file_name, mime_type, size_bytes,
                    status, chunk_count, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.session_id,
                    document.file_name,
                    document.mime_type,
                    document.size_bytes,
                    document.status,
                    document.chunk_count,
                    document.error_message,
                    _to_iso(document.created_at),
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

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(self, session_id: str) -> list[DocumentRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM documents
                WHERE session_id = ?
                ORDER BY created_at ASC
                """,
                (session_id,),
            ).fetchall()
        return [self._document_from_row(row) for row in rows]

    def count_documents(
        self,
        session_id: str,
        statuses: Sequence[str] | None = None,
    ) -> int:
        query = "SELECT COUNT(*) FROM documents WHERE session_id = ?"
        params: list[object] = [session_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        with self._connection() as connection:
            return int(connection.execute(query, params).fetchone()[0])

    def delete_document(self, session_id: str, document_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE id = ? AND session_id = ?",
                (document_id, session_id),
            )
        return cursor.rowcount > 0

    def add_message(
        self,
        session_id: str,
        role: MessageRole | str,
        content: str,
    ) -> MessageRecord:
        now = utc_now()
        role_value = role.value if isinstance(role, MessageRole) else role
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO messages(session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role_value, content, _to_iso(now)),
            )
            message_id = int(cursor.lastrowid)
        return MessageRecord(
            id=message_id,
            session_id=session_id,
            role=role_value,
            content=content,
            created_at=now,
        )

    def list_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[MessageRecord]:
        with self._connection() as connection:
            if limit is None:
                rows = connection.execute(
                    """
                    SELECT * FROM messages
                    WHERE session_id = ? ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM (
                        SELECT * FROM messages
                        WHERE session_id = ?
                        ORDER BY id DESC LIMIT ?
                    ) ORDER BY id ASC
                    """,
                    (session_id, limit),
                ).fetchall()
        return [self._message_from_row(row) for row in rows]
