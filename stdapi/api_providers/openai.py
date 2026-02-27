"""OpenAI API."""

from typing import TYPE_CHECKING

from stdapi.api_providers import FORMATTER_BY_TAG

if TYPE_CHECKING:
    from stdapi.types import JsonMapping

#: Route tag
TAG_OPENAI: str = "OpenAI"

#: Mapping from status code to OpenAI error types, default to "invalid_request_error"
_STATUS = {
    401: "authentication_error",
    403: "permission_error",
    409: "conflict_error",
    429: "rate_limit_error",
}


def _format_error(
    status: int, message: str, param: str | None = None, code: str | None = None
) -> tuple[JsonMapping, int]:
    """Format an error as an OpenAI-compatible JSON envelope.

    Args:
        status: HTTP status code (unused in body but kept for signature symmetry).
        message: Human-readable error message.
        param: Optional parameter name that caused the error.
        code: Optional machine-readable error code.

    Returns:
        A tuple containing the error response in JSON format and the corresponding HTTP status code.
    """
    return {
        "error": {
            "message": message,
            "type": _STATUS.get(status, "invalid_request_error"),
            "param": param,
            "code": code,
        }
    }, status


FORMATTER_BY_TAG[TAG_OPENAI] = _format_error
