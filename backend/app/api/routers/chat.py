from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_container, get_current_user
from app.api.schemas import QuestionRequest, QuestionResponse
from app.container import AppContainer


router = APIRouter(prefix="/histories/{history_id}/questions", tags=["chat"])


@router.post("", response_model=QuestionResponse)
def ask_question(
    history_id: str,
    request: QuestionRequest,
    user_id: str = Depends(get_current_user),
    container: AppContainer = Depends(get_container),
) -> QuestionResponse:
    history = container.history_service.require_active(history_id)
    if history.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy cuộc trò chuyện.",
        )
    top_k = min(
        request.top_k or container.settings.default_top_k,
        container.settings.max_top_k,
    )
    answer = container.rag_service.ask(
        history_id,
        request.question,
        top_k,
        include_trace=request.debug,
    )
    return QuestionResponse.from_answer(answer)
