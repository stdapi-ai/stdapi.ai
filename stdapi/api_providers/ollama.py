"""Ollama API.

Every operation of this dialect duplicates a tool an agent already has on
another dialect, and a redundant tool degrades tool choice, so none of them is
published as an MCP tool by default; an operator re-exposes one by naming it in
``mcp_include_tools``.
"""

from typing import TYPE_CHECKING, Final

from stdapi.api_providers import (
    FORMATTER_BY_TAG,
    SET_LOG_FIELDS_BY_TAG,
    SET_RESPONSE_HEADERS_BY_TAG,
)

if TYPE_CHECKING:
    from stdapi.types import JsonMapping

#: Route tag
TAG_OLLAMA: str = "Ollama"

#: Media type of the newline-delimited JSON streams the Ollama API answers with.
NDJSON_MEDIA_TYPE: Final = "application/x-ndjson"

#: Ollama release whose published API contract this surface targets.
OLLAMA_API_VERSION: Final = "0.33.1"

#: Operation IDs mounted but not published as MCP tools by default.
MCP_EXCLUDED_OPERATIONS: Final = frozenset(
    {
        "ollama_chat",
        "ollama_generate",
        "ollama_embed",
        "ollama_embeddings",
        "ollama_tags",
        "ollama_show",
        "ollama_ps",
        "ollama_version",
        "ollama_pull",
        "ollama_create",
        "ollama_copy",
        "ollama_push",
        "ollama_delete",
    }
)


def _format_error(
    status: int,
    message: str,
    param: str | None = None,  # noqa: ARG001
    code: str | None = None,  # noqa: ARG001
) -> tuple[JsonMapping, int]:
    """Format an error as an Ollama-compatible JSON envelope.

    Args:
        status: HTTP status code, returned unchanged.
        message: Human-readable error message.
        param: Unused (OpenAI-specific), kept for signature symmetry.
        code: Unused (OpenAI-specific), kept for signature symmetry.

    Returns:
        A tuple containing the error response in JSON format and the corresponding HTTP status code.
    """
    return {"error": message}, status


FORMATTER_BY_TAG[TAG_OLLAMA] = _format_error
SET_RESPONSE_HEADERS_BY_TAG[TAG_OLLAMA] = lambda *_: None
SET_LOG_FIELDS_BY_TAG[TAG_OLLAMA] = lambda *_: None
