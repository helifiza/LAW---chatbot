from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator, EmailStr,field_validator

from app.domain.models import DocumentRecord, MessageRecord, HistoryRecord
from app.services.rag_service import RagAnswer, RagSource
import json

class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)

    @model_validator(mode="after")
    def validate_passwords(self):
        if self.password != self.confirm_password:
            raise ValueError("Mật khẩu xác nhận không khớp.")

        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    id: str
    full_name: str
    email: EmailStr
    role: str

    @classmethod
    def from_dict(cls, value: dict[str, str]) -> "UserOut":
        return cls(
            id=value["id"],
            full_name=value["full_name"],
            email=value["email"],
            role=value["role"],
        )


class AuthResponse(BaseModel):
    message: str
    user: UserOut


class DocumentOut(BaseModel):
    id: str
    file_name: str
    mime_type: str
    size_bytes: int
    status: str
    chunk_count: int
    error_message: str | None
    created_at: datetime
    graph_status: str
    graph_error: str | None

    @classmethod
    def from_record(cls, value: DocumentRecord) -> "DocumentOut":
        return cls(
            id=value.id,
            file_name=value.file_name,
            mime_type=value.mime_type,
            size_bytes=value.size_bytes,
            status=value.status,
            chunk_count=value.chunk_count,
            error_message=value.error_message,
            created_at=value.created_at,
            graph_status=value.graph_status,
            graph_error=value.graph_error,
        )

class SourceOut(BaseModel):
    document_id: str
    file_name: str
    page_number: int
    page_end_number: int
    dieu: str | None
    score: float
    excerpt: str

    @classmethod
    def from_source(cls, source: RagSource) -> "SourceOut":
        return cls(**source.__dict__)


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[SourceOut] | None = None
    created_at: datetime

    @classmethod
    def from_record(cls, value: MessageRecord) -> "MessageOut":
        parsed_sources = None
        if value.sources:
            raw = json.loads(value.sources)
            parsed_sources = [SourceOut(**item) for item in raw]
        return cls(
            id=value.id,
            role=value.role,
            content=value.content,
            sources = parsed_sources,
            created_at=value.created_at,
        )


class CreateHistoryRequest(BaseModel):
    question: str | None = Field(default=None, min_length=1, max_length=4000)


class HistoryOut(BaseModel):
    history_id: str
    title: str
    status: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, value: HistoryRecord) -> "HistoryOut":
        return cls(
            history_id=value.id,
            title=value.title,
            status=value.status,
            created_at=value.created_at,
            updated_at=value.updated_at,
        )


class HistorySnapshotOut(HistoryOut):
    documents: list[DocumentOut]
    messages: list[MessageOut]

    @classmethod
    def from_records(
        cls,
        history: HistoryRecord,
        documents: list[DocumentRecord],
        messages: list[MessageRecord],
    ) -> "HistorySnapshotOut":
        return cls(
            **HistoryOut.from_record(history).model_dump(),
            documents=[DocumentOut.from_record(item) for item in documents],
            messages=[MessageOut.from_record(item) for item in messages],
        )


class UploadErrorOut(BaseModel):
    file_name: str
    message: str


class UploadResponse(BaseModel):
    documents: list[DocumentOut]
    errors: list[UploadErrorOut]


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )

    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
    )
    debug: bool = Field(
        default=False,
        description="Trả thêm trace truy hồi; không bao gồm suy luận nội bộ.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(
        cls,
        value: str,
    ) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Câu hỏi không được để trống"
            )

        return cleaned


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    retrieved_count: int
    trace: dict[str, object] | None = None

    @classmethod
    def from_answer(cls, answer: RagAnswer) -> "QuestionResponse":
        return cls(
            question=answer.question,
            answer=answer.answer,
            sources=[SourceOut.from_source(item) for item in answer.sources],
            retrieved_count=answer.retrieved_count,
            trace=answer.trace,
        )


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    generation_provider: str
    generation_model: str
    gemini_configured: bool
    hyde_enabled: bool
    reranker_model: str
    reranker_loaded: bool
    vector_count: int
