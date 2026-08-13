from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_container, get_current_user
from app.api.schemas import QuestionRequest, QuestionResponse
from app.container import AppContainer

logger = logging.getLogger(__name__)
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
    check_first_message = container.history_service.message_count(history_id) == 0

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

    if check_first_message:
        try:
            context = "\n\n".join(source.excerpt for source in answer.sources)
            generated_title, _ = container.generation_service.generate_title(
                request.question,
                context,
            )
            container.history_service.rename(history_id, generated_title)
        except Exception:
            logger.exception(
                "Không sinh được title cho history=%s, giữ title mặc định",
                history_id,
            )
    return QuestionResponse.from_answer(answer)
