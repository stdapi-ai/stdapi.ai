"""OpenAI API."""

from typing import TYPE_CHECKING

from stdapi.api_providers import (
    FORMATTER_BY_TAG,
    SET_LOG_FIELDS_BY_TAG,
    SET_RESPONSE_HEADERS_BY_TAG,
)

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import Response

    from stdapi.monitoring import EventLog
    from stdapi.types import JsonMapping

#: OpenAI request API header
OPENAI_ORGANIZATION_HEADER = "OpenAI-Organization"

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


def set_openai_headers(
    request: Request, response: Response, processing_ms: int
) -> None:
    """Attach OpenAI-compatible headers to all responses.

    Adds:
    - openai-processing-ms: processing time in milliseconds
    - openai-version: OpenAI API version header.
    - openai-organization: echo of incoming OpenAI-Organization header, if present

    Args:
        request: Incoming HTTP request.
        response: Outgoing response object.
        processing_ms: Processing time in milliseconds.
    """
    response.headers["openai-processing-ms"] = str(processing_ms)
    response.headers["openai-version"] = "2020-10-01"
    if org_id := request.headers.get(OPENAI_ORGANIZATION_HEADER):
        response.headers["openai-organization"] = org_id


def set_openai_log_fields(request: Request, log: EventLog) -> None:
    """Sets the OpenAI-specific log fields based on the incoming request headers.

    Args:
        request: The incoming HTTP request object.
        log: The event log dictionary.
    """
    if org_id := request.headers.get(OPENAI_ORGANIZATION_HEADER):
        log["request_org_id"] = org_id


SET_RESPONSE_HEADERS_BY_TAG[TAG_OPENAI] = set_openai_headers
SET_LOG_FIELDS_BY_TAG[TAG_OPENAI] = set_openai_log_fields
