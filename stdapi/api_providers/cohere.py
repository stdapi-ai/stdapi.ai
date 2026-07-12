"""Cohere API."""

from typing import TYPE_CHECKING

from stdapi.api_providers import (
    FORMATTER_BY_TAG,
    SET_LOG_FIELDS_BY_TAG,
    SET_RESPONSE_HEADERS_BY_TAG,
)

if TYPE_CHECKING:
    from stdapi.types import JsonMapping

#: Route tag
TAG_COHERE: str = "Cohere"


def _format_error(
    status: int,
    message: str,
    param: str | None = None,  # noqa: ARG001
    code: str | None = None,  # noqa: ARG001
) -> tuple[JsonMapping, int]:
    """Format an error as a Cohere-compatible JSON envelope.

    Args:
        status: HTTP status code, returned unchanged.
        message: Human-readable error message.
        param: Unused (OpenAI-specific), kept for signature symmetry.
        code: Unused (OpenAI-specific), kept for signature symmetry.

    Returns:
        A tuple containing the error response in JSON format and the corresponding HTTP status code.
    """
    return {"message": message}, status


FORMATTER_BY_TAG[TAG_COHERE] = _format_error
SET_RESPONSE_HEADERS_BY_TAG[TAG_COHERE] = lambda *_: None
SET_LOG_FIELDS_BY_TAG[TAG_COHERE] = lambda *_: None
