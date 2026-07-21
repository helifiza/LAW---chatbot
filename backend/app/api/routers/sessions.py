from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_container
from app.api.schemas import SessionOut, SessionSnapshotOut
from app.container import AppContainer


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
def create_session(container: AppContainer = Depends(get_container)) -> SessionOut:
    return SessionOut.from_record(container.session_service.create())


@router.get("/{session_id}", response_model=SessionSnapshotOut)
def get_session(
    session_id: str, container: AppContainer = Depends(get_container)
) -> SessionSnapshotOut:
    session, documents, messages = container.session_service.snapshot(session_id)
    return SessionSnapshotOut.from_records(session, documents, messages)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def close_session(
    session_id: str, container: AppContainer = Depends(get_container)
) -> Response:
    container.session_service.close(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
