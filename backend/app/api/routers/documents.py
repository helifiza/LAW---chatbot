from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile, status

from app.api.dependencies import get_container, get_current_user
from app.api.schemas import DocumentOut, UploadErrorOut, UploadResponse
from app.container import AppContainer
from app.core.errors import AppError, FileTooLargeError, UnsupportedFileTypeError
from app.services.document_parser import SUPPORTED_EXTENSIONS


router = APIRouter(prefix="/histories/{history_id}/documents", tags=["documents"])


def _safe_file_name(value: str | None) -> str:
    name = Path(value or "").name.strip()
    if not name:
        raise UnsupportedFileTypeError("Tên file không hợp lệ")
    return name


def _save_temporary_upload(
    upload: UploadFile, container: AppContainer
) -> tuple[Path, str, int]:
    file_name = _safe_file_name(upload.filename)
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(f"{file_name}: chỉ hỗ trợ {supported}")
    temporary = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=extension,
        dir=container.settings.upload_temp_dir,
        delete=False,
    )
    total = 0
    try:
        while True:
            block = upload.file.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > container.settings.max_file_size_bytes:
                raise FileTooLargeError(
                    f"{file_name}: vượt giới hạn "
                    f"{container.settings.max_file_size_bytes // (1024 * 1024)} MB"
                )
            temporary.write(block)
    except Exception:
        temporary.close()
        Path(temporary.name).unlink(missing_ok=True)
        raise
    finally:
        if not temporary.closed:
            temporary.close()
    return Path(temporary.name), file_name, total


@router.post("", response_model=UploadResponse)
def upload_documents(
    history_id: str,
    files: list[UploadFile] = File(...),
    user_id: str = Depends(get_current_user),
    container: AppContainer = Depends(get_container),
) -> UploadResponse:
    history = container.history_service.require_active(history_id)
    if history.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy cuộc trò chuyện.",
        )
    errors: list[UploadErrorOut] = []
    for upload in files:
        temporary_path: Path | None = None
        display_name = Path(upload.filename or "file").name
        try:
            temporary_path, file_name, size_bytes = _save_temporary_upload(
                upload, container
            )
            container.indexing_service.index_file(
                history_id,
                user_id,
                temporary_path,
                file_name,
                upload.content_type or "application/octet-stream",
                size_bytes,
            )
        except AppError as exc:
            errors.append(UploadErrorOut(file_name=display_name, message=exc.message))
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            upload.file.close()
    documents = container.history_service.list_documents(history_id)
    return UploadResponse(
        documents=[DocumentOut.from_record(item) for item in documents],
        errors=errors,
    )


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def clear_documents(
    history_id: str,
    user_id: str = Depends(get_current_user),
    container: AppContainer = Depends(get_container),
) -> Response:
    history = container.history_service.require_active(history_id)
    if history.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy cuộc trò chuyện.",
        )
    container.history_service.clear_documents(history_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    history_id: str,
    document_id: str,
    user_id: str = Depends(get_current_user),
    container: AppContainer = Depends(get_container),
) -> Response:
    history = container.history_service.require_active(history_id)
    if history.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy cuộc trò chuyện.",
        )
    container.history_service.delete_document(history_id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
