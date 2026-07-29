from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.api.schemas import HealthResponse
from app.container import AppContainer


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(container: AppContainer = Depends(get_container)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=container.settings.app_version,
        embedding_provider=container.settings.embedding_provider,
        embedding_model=container.settings.embedding_model,
        embedding_dimension=container.settings.embedding_dimension,
        generation_provider=container.settings.generation_provider,
        generation_model=container.settings.generation_model,
        gemini_configured=container.gemini_client.is_configured,
        hyde_enabled=container.settings.hyde_enabled,
        reranker_model=container.reranker.model_name,
        reranker_loaded=container.reranker.is_loaded,
        vector_count=container.vector_repository.count(),
    )
