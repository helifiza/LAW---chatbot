from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.domain.models import DocumentRecord, MessageRecord, SessionRecord
from app.services.rag_service import RagAnswer, RagSource


class DocumentOut(BaseModel):
    id: str
    file_name: str
    mime_type: str
    size_bytes: int
    status: str
    chunk_count: int
    error_message: str | None
    created_at: datetime

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
        )


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: datetime

    @classmethod
    def from_record(cls, value: MessageRecord) -> "MessageOut":
        return cls(
            id=value.id,
            role=value.role,
            content=value.content,
            created_at=value.created_at,
        )


class SessionOut(BaseModel):
    session_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime

    @classmethod
    def from_record(cls, value: SessionRecord) -> "SessionOut":
        return cls(
            session_id=value.id,
            status=value.status,
            created_at=value.created_at,
            updated_at=value.updated_at,
            expires_at=value.expires_at,
        )


class SessionSnapshotOut(SessionOut):
    documents: list[DocumentOut]
    messages: list[MessageOut]

    @classmethod
    def from_records(
        cls,
        session: SessionRecord,
        documents: list[DocumentRecord],
        messages: list[MessageRecord],
    ) -> "SessionSnapshotOut":
        return cls(
            **SessionOut.from_record(session).model_dump(),
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
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Câu hỏi không được để trống")
        return cleaned


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


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    retrieved_count: int

    @classmethod
    def from_answer(cls, answer: RagAnswer) -> "QuestionResponse":
        return cls(
            question=answer.question,
            answer=answer.answer,
            sources=[SourceOut.from_source(item) for item in answer.sources],
            retrieved_count=answer.retrieved_count,
        )


class HealthResponse(BaseModel):
    status: str
    version: str
    embedding_provider: str
    embedding_model: str
    ollama_available: bool
    embedding_model_available: bool
    generation_provider: str
    generation_model: str
    generation_model_available: bool
    vector_count: int
