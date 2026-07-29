from datetime import datetime, timedelta, timezone

from jose import jwt
from pwdlib import PasswordHash


class AuthService:
    def __init__(
        self,
        user_repository,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
    ):
        self.user_repository = user_repository
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.password_hasher = PasswordHash.recommended()

    def register(
        self,
        full_name: str,
        email: str,
        password: str,
    ):
        normalized_email = email.lower().strip()

        existing_user = self.user_repository.find_by_email(
            normalized_email
        )

        if existing_user:
            raise ValueError("Email này đã được đăng ký.")

        password_hash = self.password_hasher.hash(password)

        return self.user_repository.create_user(
            full_name=full_name,
            email=normalized_email,
            password_hash=password_hash,
        )

    def login(self, email: str, password: str):
        normalized_email = email.lower().strip()

        user = self.user_repository.find_by_email(normalized_email)

        if not user:
            raise ValueError("Email hoặc mật khẩu không chính xác.")

        password_is_valid = self.password_hasher.verify(
            password,
            user["password_hash"],
        )

        if not password_is_valid:
            raise ValueError("Email hoặc mật khẩu không chính xác.")

        token = self._create_access_token(
            user_id=user["id"],
            role=user["role"],
        )

        public_user = {
            "id": user["id"],
            "full_name": user["full_name"],
            "email": user["email"],
            "role": user["role"],
        }

        return token, public_user

    def _create_access_token(
        self,
        user_id: str,
        role: str,
    ) -> str:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

        payload = {
            "sub": user_id,
            "role": role,
            "exp": expires_at,
        }

        return jwt.encode(
            payload,
            self.jwt_secret,
            algorithm=self.jwt_algorithm,
        )