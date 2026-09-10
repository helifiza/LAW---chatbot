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

    gemini_api_key: str
    gemini_retry_count: int
    gemini_retry_delay_seconds: float

    embedding_model: str
    embedding_dimension: int
    generation_model: str
    embedding_batch_size: int
    generation_max_tokens: int
    generation_temperature: float
    hyde_enabled: bool
    hyde_max_tokens: int
    hyde_temperature: float

    retrieval_candidate_k: int
    rerank_candidate_k: int
    rrf_k: int
    rerank_model: str
    rerank_device: str
    rerank_batch_size: int
    rerank_max_length: int

    sqlite_path: Path
    chroma_persist_dir: Path
    chroma_collection_name: str
    upload_temp_dir: Path

    max_file_size_bytes: int
    chunk_size_chars: int
    chunk_overlap_chars: int
    default_top_k: int
    max_top_k: int
    min_similarity: float
    history_message_limit: int
    max_history_documents: int

    jwt_secret: str
    jwt_algorithm: str = "HS256"

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
            app_version=os.getenv("APP_VERSION", "2.0.0"),
            debug=_to_bool(os.getenv("DEBUG"), True),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            api_prefix=os.getenv("API_PREFIX", "/api/v1"),
            cors_origins=origins,
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            gemini_retry_count=max(
                1, int(os.getenv("GEMINI_RETRY_COUNT", "5"))
            ),
            gemini_retry_delay_seconds=max(
                0.0, float(os.getenv("GEMINI_RETRY_DELAY_SECONDS", "2"))
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "gemini-embedding-001"
            ),
            embedding_dimension=max(
                1, int(os.getenv("EMBEDDING_DIMENSION", "768"))
            ),
            generation_model=os.getenv(
                "GEMINI_GENERATION_MODEL", "gemini-3.5-flash-lite"
            ),
            embedding_batch_size=max(
                1, int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
            ),
            generation_max_tokens=max(
                1, int(os.getenv("GENERATION_MAX_TOKENS", "1000"))
            ),
            generation_temperature=float(
                os.getenv("GENERATION_TEMPERATURE", "0.2")
            ),
            hyde_enabled=_to_bool(os.getenv("HYDE_ENABLED"), True),
            hyde_max_tokens=max(
                1, int(os.getenv("HYDE_MAX_TOKENS", "350"))
            ),
            hyde_temperature=float(os.getenv("HYDE_TEMPERATURE", "0.4")),
            retrieval_candidate_k=max(
                1, int(os.getenv("RETRIEVAL_CANDIDATE_K", "30"))
            ),
            rerank_candidate_k=max(
                1, int(os.getenv("RERANK_CANDIDATE_K", "30"))
            ),
            rrf_k=max(1, int(os.getenv("RRF_K", "60"))),
            rerank_model=os.getenv(
                "RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
            ),
            rerank_device=os.getenv("RERANK_DEVICE", "auto").strip() or "auto",
            rerank_batch_size=max(
                1, int(os.getenv("RERANK_BATCH_SIZE", "8"))
            ),
            rerank_max_length=max(
                128, int(os.getenv("RERANK_MAX_LENGTH", "1024"))
            ),
            sqlite_path=_resolve_path(
                os.getenv("SQLITE_PATH"), "data/slaw_gemini.db"
            ),
            chroma_persist_dir=_resolve_path(
                os.getenv("CHROMA_PERSIST_DIR"), "data/chroma_gemini"
            ),
            chroma_collection_name=os.getenv(
                "CHROMA_COLLECTION_NAME",
                "slaw_documents_gemini_embedding_001_768_v1",
            ),
            upload_temp_dir=_resolve_path(os.getenv("UPLOAD_TEMP_DIR"), "data/tmp"),
            max_file_size_bytes=int(os.getenv("MAX_FILE_SIZE_MB", "10"))
            * 1024
            * 1024,
            chunk_size_chars=int(os.getenv("CHUNK_SIZE_CHARS", "2500")),
            chunk_overlap_chars=int(os.getenv("CHUNK_OVERLAP_CHARS", "250")),
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "8")),
            max_top_k=int(os.getenv("MAX_TOP_K", "20")),
            min_similarity=float(os.getenv("MIN_SIMILARITY", "0.20")),
            history_message_limit=int(os.getenv("HISTORY_MESSAGE_LIMIT", "8")),
            max_history_documents=max(
                1, int(os.getenv("MAX_SESSION_DOCUMENTS", "5"))
            ),
            jwt_secret=os.getenv(
                "JWT_SECRET",
                "dev-secret-change-this",
            ),
            jwt_algorithm=os.getenv(
                "JWT_ALGORITHM",
                "HS256",
            ),
        )
        settings.ensure_directories()
        return settings

    def ensure_directories(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_temp_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embedding_provider(self) -> str:
        return "gemini"

    @property
    def generation_provider(self) -> str:
        return "gemini"
