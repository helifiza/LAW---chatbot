from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routers import chat, documents, health, sessions
from app.container import AppContainer, build_container
from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import configure_logging


settings = Settings.from_env()
configure_logging(settings.log_level)
logger = logging.getLogger("slaw.api")


async def cleanup_expired_sessions(container: AppContainer) -> None:
    while True:
        await asyncio.sleep(settings.session_cleanup_interval_seconds)
        try:
            await asyncio.to_thread(container.session_service.cleanup_expired)
        except Exception:
            logger.exception("Không dọn được phiên hết hạn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = build_container(settings)
    app.state.container = container
    removed = container.session_service.cleanup_expired()
    logger.info(
        "SLaw API khởi động | version=%s expired_sessions_removed=%s",
        settings.app_version,
        removed,
    )
    cleanup_task = asyncio.create_task(cleanup_expired_sessions(container))
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        container.ollama_client.close()
        logger.info("SLaw API dừng")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Dữ liệu gửi lên không hợp lệ",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Lỗi chưa được xử lý")
    message = str(exc) if settings.debug else "Lỗi nội bộ máy chủ"
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": message}},
    )


app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(sessions.router, prefix=settings.api_prefix)
app.include_router(documents.router, prefix=settings.api_prefix)
app.include_router(chat.router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }
