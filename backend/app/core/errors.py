from __future__ import annotations


class AppError(Exception):
    status_code = 400
    code = "APP_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"


class SessionNotFoundError(NotFoundError):
    code = "SESSION_NOT_FOUND"


class SessionExpiredError(AppError):
    status_code = 410
    code = "SESSION_EXPIRED"


class SessionClosedError(AppError):
    status_code = 410
    code = "SESSION_CLOSED"


class DocumentNotFoundError(NotFoundError):
    code = "DOCUMENT_NOT_FOUND"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "UNSUPPORTED_FILE_TYPE"


class FileTooLargeError(AppError):
    status_code = 413
    code = "FILE_TOO_LARGE"


class DocumentLimitError(AppError):
    status_code = 409
    code = "DOCUMENT_LIMIT_REACHED"


class DocumentParseError(AppError):
    status_code = 422
    code = "DOCUMENT_PARSE_ERROR"


class DocumentIndexingError(AppError):
    status_code = 422
    code = "DOCUMENT_INDEXING_ERROR"


class NoReadyDocumentError(AppError):
    status_code = 409
    code = "NO_READY_DOCUMENT"


class OllamaServiceError(AppError):
    status_code = 502
    code = "OLLAMA_REQUEST_FAILED"
