"""OpenAI API common."""

from contextlib import suppress

from fastapi import Request
from fastapi.responses import Response

from stdapi.monitoring import REQUEST_LOG

#: OpenAI request API header
OPENAI_ORGANIZATION_HEADER = "OpenAI-Organization"


def set_openai_headers(
    request: Request, response: Response, request_id: str, processing_ms: int
) -> None:
    """Attach OpenAI-compatible headers to all responses.

    Adds:
    - openai-processing-ms: processing time in milliseconds
    - openai-version: OpenAI API version header.
    - openai-organization: echo of incoming OpenAI-Organization header, if present

    Args:
        request: Incoming HTTP request.
        response: Outgoing response object.
        request_id: Unique identifier.
        processing_ms: Processing time in milliseconds.
    """
    response.headers["x-request-id"] = request_id
    response.headers["openai-processing-ms"] = str(processing_ms)
    response.headers["openai-version"] = "2020-10-01"
    log = REQUEST_LOG.get()
    with suppress(KeyError):
        log["request_org_id"] = response.headers["openai-organization"] = (
            request.headers[OPENAI_ORGANIZATION_HEADER]
        )
