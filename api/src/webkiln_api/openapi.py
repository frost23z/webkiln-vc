from typing import Any

from webkiln_api.errors import HTTP_CODES
from webkiln_api.models import ErrorResponse

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status_code: {
        "model": ErrorResponse,
        "description": message,
    }
    for status_code, (_, message) in HTTP_CODES.items()
}
