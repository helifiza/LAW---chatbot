from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class UserRepository:
    """Lưu và truy xuất tài khoản người dùng trong SQLite."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)

        # Tự tạo thư mục chứa database nếu chưa tồn tại.
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Tự tạo bảng users khi repository được khởi tạo.
        self._initialize_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """
        Mở kết nối SQLite.

        - Tự commit khi thao tác thành công.
        - Rollback nếu có lỗi.
        - Luôn đóng kết nối sau khi sử dụng.
        """
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )

        # Cho phép truy cập dữ liệu bằng tên cột:
        # row["email"] thay vì row[2].
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
        """Tạo bảng users nếu bảng chưa tồn tại."""
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _normalize_email(email: str) -> str:
        """
        Chuẩn hóa email.

        Ví dụ:
        '  Test@Example.com ' -> 'test@example.com'
        """
        return email.strip().lower()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Chuyển một dòng SQLite thành dictionary."""
        return {
            "id": row["id"],
            "full_name": row["full_name"],
            "email": row["email"],
            "password_hash": row["password_hash"],
            "role": row["role"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_user(
        self,
        full_name: str,
        email: str,
        password_hash: str,
        role: str = "user",
    ) -> dict[str, str]:
        """
        Tạo tài khoản mới.

        Lưu ý:
        - password_hash phải là mật khẩu đã được AuthService mã hóa.
        - Không truyền mật khẩu gốc vào hàm này.
        - Kết quả trả về không chứa password_hash.
        """
        normalized_name = full_name.strip()
        normalized_email = self._normalize_email(email)

        if not normalized_name:
            raise ValueError("Họ và tên không được để trống.")

        if not normalized_email:
            raise ValueError("Email không được để trống.")

        if not password_hash:
            raise ValueError("Mật khẩu chưa được mã hóa.")

        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO users (
                        id,
                        full_name,
                        email,
                        password_hash,
                        role,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        normalized_name,
                        normalized_email,
                        password_hash,
                        role,
                        now,
                        now,
                    ),
                )

        except sqlite3.IntegrityError as error:
            # Trường hợp email đã tồn tại.
            if (
                "users.email" in str(error)
                or "UNIQUE constraint failed" in str(error)
            ):
                raise ValueError(
                    "Email này đã được đăng ký."
                ) from error

            raise

        # Không trả password_hash về frontend.
        return {
            "id": user_id,
            "full_name": normalized_name,
            "email": normalized_email,
            "role": role,
            "created_at": now,
            "updated_at": now,
        }

    def find_by_email(
        self,
        email: str,
    ) -> dict[str, Any] | None:
        """
        Tìm tài khoản theo email.

        Hàm này có trả password_hash vì AuthService cần
        dùng nó để kiểm tra mật khẩu khi đăng nhập.
        """
        normalized_email = self._normalize_email(email)

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    full_name,
                    email,
                    password_hash,
                    role,
                    created_at,
                    updated_at
                FROM users
                WHERE email = ?
                LIMIT 1
                """,
                (normalized_email,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def find_by_id(
        self,
        user_id: str,
    ) -> dict[str, str] | None:
        """
        Tìm tài khoản theo ID.

        Hàm này không trả password_hash, phù hợp để dùng
        cho API /auth/me sau này.
        """
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    id,
                    full_name,
                    email,
                    role,
                    created_at,
                    updated_at
                FROM users
                WHERE id = ?
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "id": row["id"],
            "full_name": row["full_name"],
            "email": row["email"],
            "role": row["role"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }