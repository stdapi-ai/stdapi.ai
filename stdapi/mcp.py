"""MCP (Model Context Protocol) server setup.

Registers FastApiMCP, mounts the enabled transports, and wires up structured-logging
and request-ID propagation so that internal MCP → API calls share the same log entry.
"""

from contextlib import suppress
from logging import ERROR, Handler, LogRecord, getLogger
from traceback import format_exception
from typing import TYPE_CHECKING

from fastapi import Depends
from fastapi_mcp import AuthConfig, FastApiMCP  # type: ignore[import-untyped]
from httpx import ASGITransport, AsyncClient
from httpx import Request as HttpxRequest

from stdapi import server
from stdapi.auth import authenticate
from stdapi.config import SETTINGS, LogLevel
from stdapi.metering import EDITION_TITLE
from stdapi.monitoring import REQUEST, REQUEST_ID, log_error_details
from stdapi.routes.core_root import MCP_SERVER_CARD

if TYPE_CHECKING:
    from fastapi import FastAPI
    from mcp.types import Tool

#: Marker fastapi_mcp always prepends to the auto-generated response/example block.
_RESPONSES_MARKER = "\n\n### Responses:"


def _strip_response_docs(tools: list[Tool]) -> None:
    """Drop the auto-generated response/example section from each tool's description.

    Args:
        tools: MCP tools to mutate in place.
    """
    for tool in tools:
        if tool.description and _RESPONSES_MARKER in tool.description:
            tool.description = tool.description.split(_RESPONSES_MARKER, 1)[0]


def is_mcp() -> bool:
    """Check if the current request originates from an MCP client.

    Detects MCP calls by checking both the User-Agent header and the presence
    of the internal request ID header (to avoid spoofing). Uses the cached
    REQUEST ContextVar from stdapi.monitoring for header access.

    Returns:
        True if the current request is from an MCP client, False otherwise.

    Raises:
        LookupError: If called outside a request context.
    """
    headers = REQUEST.get().headers
    return (
        headers.get("user-agent") == server.MCP_USER_AGENT
        and server.INTERNAL_REQUEST_ID_HEADER in headers
    )


async def _inject_request_id(request: HttpxRequest) -> None:
    """Stamp the active request ID on outgoing MCP → API calls for log correlation.

    Args:
        request: Outgoing httpx request to mutate.
    """
    with suppress(LookupError):
        request.headers[server.INTERNAL_REQUEST_ID_HEADER] = REQUEST_ID.get()


class _McpLogHandler(Handler):
    """Captures ``fastapi_mcp.server`` records into the structured JSON logger instead of stderr."""

    def emit(self, record: LogRecord) -> None:
        """Forward the record to :func:`~stdapi.monitoring.log_error_details`.

        Appends the exception traceback when ``exc_info`` is present.
        Silently drops the record when called outside a request context.

        Args:
            record: Log record from ``fastapi_mcp.server``.
        """
        try:
            level: LogLevel = "error" if record.levelno >= ERROR else "warning"
            details: list[str] = [record.getMessage()]
            if record.exc_info and record.exc_info[0] is not None:
                details.append("".join(format_exception(*record.exc_info)))
            log_error_details(*details, level=level)
        except LookupError:
            pass


def _make_stateless(mcp: FastApiMCP) -> None:
    """Switch the mounted Streamable HTTP transport to stateless mode.

    ``fastapi_mcp`` hard-codes ``stateless=False`` on the session manager it
    builds, and exposes no way to configure it. The flag is only read when a
    request is dispatched, so flipping it on the manager the transport creates
    is equivalent to having constructed it stateless.

    Args:
        mcp: The FastApiMCP instance whose HTTP transport was just mounted.
    """
    transport = mcp._http_transport  # noqa: SLF001
    start = transport._ensure_session_manager_started  # noqa: SLF001

    async def start_stateless() -> None:
        await start()
        if (manager := transport._session_manager) is not None:  # noqa: SLF001
            manager.stateless = True

    transport._ensure_session_manager_started = start_stateless  # noqa: SLF001


def mount_mcp(app: FastAPI) -> None:
    """Attach FastApiMCP to *app* and mount the enabled transports.

    Installs :class:`_McpLogHandler` on the ``fastapi_mcp.server`` logger so
    tool errors appear in the structured JSON log rather than as stderr tracebacks.
    The internal HTTP client uses the ASGI transport (no TCP) with
    ``MCP_USER_AGENT`` and injects the parent request ID for log correlation.
    Populates ``MCP_SERVER_CARD["tools"]`` in :mod:`stdapi.routes.core_root` with
    the discovered tool list so the server card reflects real tools at startup.
    Strips the auto-generated response/example section from each tool description
    (see :func:`_strip_response_docs`) to reduce MCP context cost.
    With ``mcp_stateless_http`` the Streamable HTTP transport is switched to
    stateless mode (see :func:`_make_stateless`).

    Args:
        app: FastAPI application to attach MCP to.
    """
    _mcp_logger = getLogger("fastapi_mcp.server")
    _mcp_logger.addHandler(_McpLogHandler())
    _mcp_logger.propagate = False

    mcp = FastApiMCP(
        app,
        name=EDITION_TITLE,
        description="AWS standardized AI API",
        include_operations=SETTINGS.mcp_include_tools,
        exclude_operations=SETTINGS.mcp_exclude_tools,
        auth_config=AuthConfig(dependencies=[Depends(authenticate)]),
        http_client=AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://apiserver",
            headers={"User-Agent": server.MCP_USER_AGENT},
            timeout=SETTINGS.ai_response_timeout,
            event_hooks={"request": [_inject_request_id]},
        ),
    )
    _strip_response_docs(mcp.tools)

    if SETTINGS.enable_mcp_streamable_http:
        mcp.mount_http()
        if SETTINGS.mcp_stateless_http:
            _make_stateless(mcp)
    if SETTINGS.enable_mcp_sse:
        mcp.mount_sse()

    if (card := MCP_SERVER_CARD) is not None:
        card["tools"] = [
            {
                "name": t.name,
                **({"description": t.description} if t.description else {}),
                "inputSchema": t.inputSchema,
            }
            for t in mcp.tools
        ]
