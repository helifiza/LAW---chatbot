from fastapi import Depends, HTTPException, Request, status

from app.container import AppContainer


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_current_user(request: Request) -> str:
    access_token = request.cookies.get("access_token")
    if not access_token:
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            access_token = authorization[7:].strip()

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yêu cầu xác thực. Vui lòng đăng nhập.",
        )

    auth_service = request.app.state.container.auth_service
    try:
        return auth_service.decode_access_token(access_token)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(error),
        ) from error
