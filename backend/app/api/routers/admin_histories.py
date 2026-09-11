from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_container, get_current_admin
from app.api.schemas import DeletedHistoryOut, HistoryOut
from app.container import AppContainer


router = APIRouter(prefix="/admin/histories", tags=["admin-histories"])


@router.get("/deleted", response_model=list[DeletedHistoryOut])
def list_deleted_histories(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _admin_id: str = Depends(get_current_admin),
    container: AppContainer = Depends(get_container),
) -> list[DeletedHistoryOut]:
    return [
        DeletedHistoryOut.from_record(item)
        for item in container.history_service.list_deleted_histories(
            limit=limit, offset=offset
        )
    ]


@router.patch("/{history_id}/restore", response_model=HistoryOut)
def restore_history(
    history_id: str,
    _admin_id: str = Depends(get_current_admin),
    container: AppContainer = Depends(get_container),
) -> HistoryOut:
    return HistoryOut.from_record(container.history_service.restore_history(history_id))
