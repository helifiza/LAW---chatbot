import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from app.api.dependencies import get_current_admin


class _AuthService:
    def decode_access_token(self, _token: str) -> str:
        return "user-1"


class _UserRepository:
    def __init__(self, role: str) -> None:
        self.role = role

    def find_by_id(self, user_id: str) -> dict[str, str] | None:
        return {"id": user_id, "role": self.role}


def _request(role: str) -> SimpleNamespace:
    container = SimpleNamespace(
        auth_service=_AuthService(),
        user_repository=_UserRepository(role),
    )
    return SimpleNamespace(
        cookies={"access_token": "token"},
        headers={},
        app=SimpleNamespace(state=SimpleNamespace(container=container)),
    )


class AdminDependencyTests(unittest.TestCase):
    def test_admin_is_authorized(self) -> None:
        self.assertEqual(get_current_admin(_request("admin")), "user-1")

    def test_regular_user_is_forbidden(self) -> None:
        with self.assertRaises(HTTPException) as context:
            get_current_admin(_request("user"))
        self.assertEqual(context.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
