from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.api.schemas import HealthResponse
from app.container import AppContainer


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(container: AppContainer = Depends(get_container)) -> HealthResponse:
    required_models = [
        container.settings.embedding_model,
        container.settings.generation_model,
    ]
    ollama_available, model_availability = container.ollama_client.status(
        required_models
    )
    return HealthResponse(
        status="ok",
        version=container.settings.app_version,
        embedding_provider=container.settings.embedding_provider,
        embedding_model=container.settings.embedding_model,
        ollama_available=ollama_available,
        embedding_model_available=model_availability[
            container.settings.embedding_model
        ],
        generation_provider=container.settings.generation_provider,
        generation_model=container.settings.generation_model,
        generation_model_available=model_availability[
            container.settings.generation_model
        ],
        vector_count=container.vector_repository.count(),
    )
