from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from app.api.dependencies import get_container, get_current_user
from app.container import AppContainer


router = APIRouter(prefix="/histories/{history_id}/legal-graph", tags=["legal-graph"])


def _require_owner(history_id: str, user_id: str, container: AppContainer) -> None:
    history = container.history_service.get_any(history_id)
    if history.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy cuộc trò chuyện.",
        )


@router.get("")
def get_legal_graph(
    history_id: str,
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user_id: str = Depends(get_current_user),
    container: AppContainer = Depends(get_container),
) -> dict[str, object]:
    _require_owner(history_id, user_id, container)
    repository = container.legal_graph_repository
    return {
        "documents": repository.list_documents(history_id, limit, offset),
        "fields": repository.list_fields(history_id, limit, offset),
        "provisions": repository.list_all_provisions(history_id, limit, offset),
        "relations": repository.list_relations(history_id, limit, offset),
        "relation_provisions": repository.list_relation_provisions(history_id, limit, offset),
        "evidence": repository.list_evidence(history_id, limit, offset),
        "metadata_evidence": repository.list_metadata_evidence(history_id, limit, offset),
        "pagination": {"limit": limit, "offset": offset,
            "document_total": repository.count("legal_documents", history_id),
            "relation_total": repository.count("legal_relations", history_id),
            "provision_total": repository.count("legal_provisions", history_id)},
    }


@router.get("/documents")
def get_graph_documents(history_id: str, limit: int=Query(50,ge=1,le=200),
                        offset: int=Query(0,ge=0), user_id: str=Depends(get_current_user),
                        container: AppContainer=Depends(get_container)) -> dict:
    _require_owner(history_id,user_id,container)
    return {"items":container.legal_graph_repository.list_documents(history_id,limit,offset),
            "total":container.legal_graph_repository.count("legal_documents",history_id),"limit":limit,"offset":offset}


@router.get("/relations")
def get_graph_relations(history_id: str, limit: int=Query(50,ge=1,le=200),
                        offset: int=Query(0,ge=0), user_id: str=Depends(get_current_user),
                        container: AppContainer=Depends(get_container)) -> dict:
    _require_owner(history_id,user_id,container)
    return {"items":container.legal_graph_repository.list_relations(history_id,limit,offset),
            "total":container.legal_graph_repository.count("legal_relations",history_id),"limit":limit,"offset":offset}


@router.get("/documents/{legal_document_id}/provisions")
def get_graph_provisions(history_id: str, legal_document_id: str,
                         limit: int=Query(100,ge=1,le=200), offset: int=Query(0,ge=0),
                         user_id: str=Depends(get_current_user),
                         container: AppContainer=Depends(get_container)) -> dict:
    _require_owner(history_id,user_id,container)
    legal_ids={item["id"] for item in container.legal_graph_repository.list_documents(history_id)}
    if legal_document_id not in legal_ids:
        raise HTTPException(status_code=404,detail="Không tìm thấy văn bản pháp lý.")
    return {"items":container.legal_graph_repository.list_provisions(legal_document_id,limit,offset),
            "versions":container.legal_graph_repository.list_versions(legal_document_id),
            "limit":limit,"offset":offset}


@router.post("/uploads/{document_id}/rebuild", status_code=status.HTTP_202_ACCEPTED)
def rebuild_graph(history_id: str, document_id: str, background_tasks: BackgroundTasks,
                  user_id: str=Depends(get_current_user),
                  container: AppContainer=Depends(get_container)) -> dict:
    _require_owner(history_id,user_id,container)
    document=container.history_repository.get_document(document_id)
    if document is None or document.history_id != history_id:
        raise HTTPException(status_code=404,detail="Không tìm thấy tài liệu.")
    if not container.legal_graph_repository.is_latest_version(document_id):
        raise HTTPException(status_code=409,detail="Chỉ rebuild phiên bản canonical mới nhất.")
    container.history_repository.update_document_graph_status(document_id, "processing")
    background_tasks.add_task(container.indexing_service.rebuild_graph,history_id,document)
    return {"document_id":document_id,"graph_status":"processing"}
