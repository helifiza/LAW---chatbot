import logging 

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.dependencies import get_container, get_current_user
from app.api.schemas import CreateHistoryRequest, HistoryOut, HistorySnapshotOut
from app.container import AppContainer

logger = logging.getLogger(__name__)  
router = APIRouter(prefix="/histories", tags=["histories"])


@router.post("", response_model=HistoryOut, status_code=status.HTTP_201_CREATED)
def create_history(
    request: CreateHistoryRequest | None = None,
    user_id: str = Depends(get_current_user),
    container: AppContainer = Depends(get_container),
) -> HistoryOut:
    return HistoryOut.from_record(container.history_service.create(user_id, "Cuộc trò chuyện mới")
                                  )


@router.get("/{history_id}", response_model=HistorySnapshotOut)
def get_history(
    history_id: str,
    user_id: str = Depends(get_current_user),
    container: AppContainer = Depends(get_container),
) -> HistorySnapshotOut:
    history = container.history_service.get_any(history_id)
    if history.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy cuộc trò chuyện.",
        )
    _, documents, messages = container.history_service.snapshot(history_id)
    return HistorySnapshotOut.from_records(history, documents, messages)

@router.get("", response_model=list[HistoryOut])
def list_histories(
    user_id: str = Depends(get_current_user),
    container: AppContainer = Depends(get_container),
) -> list[HistoryOut]:
    return [
        HistoryOut.from_record(item)
        for item in container.history_service.list_histories_by_user(user_id)
    ]

@router.delete("/{history_id}", status_code=status.HTTP_204_NO_CONTENT)
def close_history(
    history_id: str,
    user_id: str = Depends(get_current_user),
    container: AppContainer = Depends(get_container),
) -> Response:
    history = container.history_service.get_any(history_id)
    if history.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy cuộc trò chuyện.",
        )
    container.history_service.delete_history(history_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.patch("/{history_id}/archive", response_model=HistoryOut)
def archive_history(
    history_id: str,
    user_id = Depends(get_current_user),
    container = Depends(get_container)
):
    history = container.history_service.get_any(history_id)
    if history.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy cuộc trò chuyện.",
        )
    updated = container.history_service.archive(history_id)
    return HistoryOut.from_record(updated)

@router.patch("/{history_id}/reopen", response_model=HistoryOut)
def reopen_history(
    history_id: str,
    user_id = Depends(get_current_user),
    container = Depends(get_container)
):
    history = container.history_service.get_any(history_id)
    if history.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy cuộc trò chuyện.",
        )
    updated = container.history_service.reopen(history_id)
    return HistoryOut.from_record(updated)