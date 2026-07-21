from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.api.schemas import QuestionRequest, QuestionResponse
from app.container import AppContainer


router = APIRouter(prefix="/sessions/{session_id}/questions", tags=["chat"])


@router.post("", response_model=QuestionResponse)
def ask_question(
    session_id: str,
    request: QuestionRequest,
    container: AppContainer = Depends(get_container),
) -> QuestionResponse:
    top_k = min(
        request.top_k or container.settings.default_top_k,
        container.settings.max_top_k,
    )
    answer = container.rag_service.ask(session_id, request.question, top_k)
    return QuestionResponse.from_answer(answer)
