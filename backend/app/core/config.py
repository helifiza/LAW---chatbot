from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str | None, default: str) -> Path:
    path = Path(value or default)
    if not path.is_absolute():
        path = BACKEND_DIR / path
    return path.resolve()


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    debug: bool
    log_level: str
    api_prefix: str
    cors_origins: tuple[str, ...]

    ollama_base_url: str
    ollama_timeout_seconds: float
    ollama_retry_count: int
    ollama_retry_delay_seconds: float

    embedding_model: str
    generation_model: str
    embedding_batch_size: int
    generation_max_tokens: int
    generation_temperature: float

    sqlite_path: Path
    chroma_persist_dir: Path
    chroma_collection_name: str
    upload_temp_dir: Path

    session_ttl_minutes: int
    session_cleanup_interval_seconds: int
    max_session_documents: int
    max_file_size_bytes: int
    chunk_size_chars: int
    chunk_overlap_chars: int
    default_top_k: int
    max_top_k: int
    min_similarity: float
    history_message_limit: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(BACKEND_DIR / ".env")
        origins = tuple(
            item.strip()
            for item in os.getenv(
                "CORS_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if item.strip()
        )
        settings = cls(
            app_name=os.getenv("APP_NAME", "SLaw RAG API"),
            app_version=os.getenv("APP_VERSION", "1.1.0"),
            debug=_to_bool(os.getenv("DEBUG"), True),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            api_prefix=os.getenv("API_PREFIX", "/api/v1"),
            cors_origins=origins,
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            ollama_timeout_seconds=float(
                os.getenv("OLLAMA_TIMEOUT_SECONDS", "300")
            ),
            ollama_retry_count=int(os.getenv("OLLAMA_RETRY_COUNT", "3")),
            ollama_retry_delay_seconds=float(
                os.getenv("OLLAMA_RETRY_DELAY_SECONDS", "2")
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
            generation_model=os.getenv(
                "OLLAMA_GENERATION_MODEL", "qwen3:4b"
            ),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
            generation_max_tokens=int(os.getenv("GENERATION_MAX_TOKENS", "1000")),
            generation_temperature=float(os.getenv("GENERATION_TEMPERATURE", "0.2")),
            sqlite_path=_resolve_path(
                os.getenv("SQLITE_PATH"), "data/slaw_ollama.db"
            ),
            chroma_persist_dir=_resolve_path(
                os.getenv("CHROMA_PERSIST_DIR"), "data/chroma_ollama"
            ),
            chroma_collection_name=os.getenv(
                "CHROMA_COLLECTION_NAME", "slaw_documents_bge_m3_v1"
            ),
            upload_temp_dir=_resolve_path(os.getenv("UPLOAD_TEMP_DIR"), "data/tmp"),
            session_ttl_minutes=int(os.getenv("SESSION_TTL_MINUTES", "1440")),
            session_cleanup_interval_seconds=max(
                30, int(os.getenv("SESSION_CLEANUP_INTERVAL_SECONDS", "300"))
            ),
            max_session_documents=int(os.getenv("MAX_SESSION_DOCUMENTS", "5")),
            max_file_size_bytes=int(os.getenv("MAX_FILE_SIZE_MB", "10"))
            * 1024
            * 1024,
            chunk_size_chars=int(os.getenv("CHUNK_SIZE_CHARS", "2500")),
            chunk_overlap_chars=int(os.getenv("CHUNK_OVERLAP_CHARS", "250")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
            max_top_k=int(os.getenv("MAX_TOP_K", "20")),
            min_similarity=float(os.getenv("MIN_SIMILARITY", "0.20")),
            history_message_limit=int(os.getenv("HISTORY_MESSAGE_LIMIT", "8")),
        )
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_temp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embedding_provider(self) -> str:
        return "ollama"

    @property
    def generation_provider(self) -> str:
        return "ollama"
