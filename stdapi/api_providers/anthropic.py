"""Anthropic API."""

from typing import TYPE_CHECKING

from stdapi.api_providers import (
    FORMATTER_BY_TAG,
    REQUEST_ID_HEADER_BY_TAG,
    SET_LOG_FIELDS_BY_TAG,
    SET_RESPONSE_HEADERS_BY_TAG,
)
from stdapi.monitoring import REQUEST_ID

if TYPE_CHECKING:
    from stdapi.types import JsonMapping

#: Route tag
TAG_ANTHROPIC: str = "Anthropic"

#: Mapping from status code to Anthropic error types, default to "invalid_request_error"
_STATUS = {
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    529: "overloaded_error",
}


def _format_error(
    status: int,
    message: str,
    param: str | None = None,  # noqa: ARG001
    code: str | None = None,  # noqa: ARG001
) -> tuple[JsonMapping, int]:
    """Format an error as an Anthropic-compatible JSON envelope.

    Args:
        status: HTTP status code used to derive overloaded_error for 529.
        message: Human-readable error message.
        param: Unused (OpenAI-specific), kept for signature symmetry.
        code: Unused (OpenAI-specific), kept for signature symmetry.

    Returns:
        A tuple containing the error response in JSON format and the corresponding HTTP status code.
    """
    return {
        "type": "error",
        "error": {
            "type": _STATUS.get(status, "invalid_request_error"),
            "message": message,
        },
        "request_id": REQUEST_ID.get(""),
    }, 529 if status == 503 else status


REQUEST_ID_HEADER_BY_TAG[TAG_ANTHROPIC] = "request-id"
FORMATTER_BY_TAG[TAG_ANTHROPIC] = _format_error
SET_RESPONSE_HEADERS_BY_TAG[TAG_ANTHROPIC] = lambda *_: None
SET_LOG_FIELDS_BY_TAG[TAG_ANTHROPIC] = lambda *_: None
