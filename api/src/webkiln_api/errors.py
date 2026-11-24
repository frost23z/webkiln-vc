import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

HTTP_CODES = {
    status.HTTP_400_BAD_REQUEST: "bad_request",
    status.HTTP_401_UNAUTHORIZED: "unauthorized",
    status.HTTP_403_FORBIDDEN: "forbidden",
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
    status.HTTP_409_CONFLICT: "conflict",
    status.HTTP_410_GONE: "gone",
    status.HTTP_413_CONTENT_TOO_LARGE: "payload_too_large",
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: "unsupported_media_type",
    status.HTTP_422_UNPROCESSABLE_CONTENT: "validation_error",
    status.HTTP_429_TOO_MANY_REQUESTS: "too_many_requests",
}


def error_response(
    status_code: int, code: str, message: str, details: list[dict[str, str]] | None = None
) -> JSONResponse:
    """Error response in a consistent format, with optional details."""
    error: dict = {"code": code, "message": message}
    if details:
        error["details"] = details
    return JSONResponse(status_code=status_code, content={"error": error})


async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = HTTP_CODES.get(exc.status_code, "http_error")
    return error_response(exc.status_code, code, exc.detail)


async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Field and message only. The submitted value is never echoed back.
    details = [
        {"field": ".".join(str(part) for part in err["loc"][1:]), "message": err["msg"]}
        for err in exc.errors()
    ]
    return error_response(422, "validation_error", "The request is invalid.", details)


async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error", exc_info=exc)  # the traceback stays in the server log
    return error_response(500, "internal_error", "Something went wrong.")


def register_error_handlers(app: FastAPI) -> None:
    app.exception_handler(StarletteHTTPException)(http_error)
    app.exception_handler(RequestValidationError)(validation_error)
    app.exception_handler(Exception)(unexpected_error)
