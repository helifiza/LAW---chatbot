from fastapi import APIRouter, HTTPException, Request, Response

from app.api.schemas import AuthResponse, LoginRequest, RegisterRequest, UserOut


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
def register(
    body: RegisterRequest,
    request: Request,
):
    auth_service = request.app.state.container.auth_service

    try:
        user = auth_service.register(
            full_name=body.full_name,
            email=body.email,
            password=body.password,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    return {
        "message": "Đăng ký thành công.",
        "user": UserOut.from_dict(user),
    }


@router.post("/login", response_model=AuthResponse)
def login(
    body: LoginRequest,
    response: Response,
    request: Request,
):
    auth_service = request.app.state.container.auth_service

    try:
        token, user = auth_service.login(
            email=body.email,
            password=body.password,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=401,
            detail=str(error),
        ) from error

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=24 * 60 * 60,
        path="/",
    )

    return {
        "message": "Đăng nhập thành công.",
        "user": UserOut.from_dict(user),
    }


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(
        key="access_token",
        httponly=True,
        samesite="lax",
        secure=False,
    )

    return {
        "message": "Đăng xuất thành công.",
    }
#hiện tại đang chạy local bằng http, đổi secure=True khi deploy https