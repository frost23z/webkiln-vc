import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from webkiln_api.models import ErrorBody, ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


HTTP_CODES = {
    status.HTTP_400_BAD_REQUEST: (
        "bad_request",
        "Bad request.",
    ),
    status.HTTP_401_UNAUTHORIZED: (
        "unauthorized",
        "Authentication required.",
    ),
    status.HTTP_403_FORBIDDEN: (
        "forbidden",
        "Permission denied.",
    ),
    status.HTTP_404_NOT_FOUND: (
        "not_found",
        "Resource not found.",
    ),
    status.HTTP_405_METHOD_NOT_ALLOWED: (
        "method_not_allowed",
        "Method not allowed.",
    ),
    status.HTTP_409_CONFLICT: (
        "conflict",
        "Resource conflict.",
    ),
    status.HTTP_410_GONE: (
        "gone",
        "Resource is no longer available.",
    ),
    status.HTTP_413_CONTENT_TOO_LARGE: (
        "payload_too_large",
        "Request payload is too large.",
    ),
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: (
        "unsupported_media_type",
        "Unsupported media type.",
    ),
    status.HTTP_422_UNPROCESSABLE_CONTENT: (
        "validation_error",
        "Request validation failed.",
    ),
    status.HTTP_429_TOO_MANY_REQUESTS: (
        "too_many_requests",
        "Too many requests.",
    ),
    status.HTTP_500_INTERNAL_SERVER_ERROR: (
        "internal_error",
        "Internal server error.",
    ),
}


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: list[ErrorDetail] | None = None,
) -> JSONResponse:
    """Error response in a consistent format, with optional details."""

    response = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            details=details,
        )
    )

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(
            mode="json",
            exclude_none=True,
        ),
    )


async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code, message = HTTP_CODES.get(
        exc.status_code,
        ("http_error", "An HTTP error occurred."),
    )

    if isinstance(exc.detail, str):
        message = exc.detail

    return error_response(
        status_code=exc.status_code,
        code=code,
        message=message,
    )


async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    # Field and message only. The submitted value is never echoed back.
    details = [
        ErrorDetail(
            field=".".join(str(part) for part in error["loc"][1:]),
            message=error["msg"],
        )
        for error in exc.errors()
    ]

    return error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code=HTTP_CODES[status.HTTP_422_UNPROCESSABLE_CONTENT][0],
        message=HTTP_CODES[status.HTTP_422_UNPROCESSABLE_CONTENT][1],
        details=details,
    )


async def unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled error", exc_info=exc)  # the traceback stays in the server log

    return error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code=HTTP_CODES[status.HTTP_500_INTERNAL_SERVER_ERROR][0],
        message=HTTP_CODES[status.HTTP_500_INTERNAL_SERVER_ERROR][1],
    )


def register_error_handlers(app: FastAPI) -> None:
    app.exception_handler(StarletteHTTPException)(http_error)
    app.exception_handler(RequestValidationError)(validation_error)
    app.exception_handler(Exception)(unexpected_error)
