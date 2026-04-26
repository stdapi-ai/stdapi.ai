"""MCP (Model Context Protocol) server setup.

Registers FastApiMCP, mounts the enabled transports, and wires up structured-logging
and request-ID propagation so that internal MCP → API calls share the same log entry.
"""

from contextlib import suppress
from logging import ERROR, Handler, LogRecord, getLogger
from traceback import format_exception
from typing import TYPE_CHECKING

from fastapi_mcp import FastApiMCP  # type: ignore[import-untyped]
from httpx import ASGITransport, AsyncClient
from httpx import Request as HttpxRequest

from stdapi import server
from stdapi.config import SETTINGS, LogLevel
from stdapi.metering import EDITION_TITLE
from stdapi.monitoring import REQUEST_ID, log_error_details
from stdapi.routes.core_root import MCP_SERVER_CARD

if TYPE_CHECKING:
    from fastapi import FastAPI


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


def mount_mcp(app: FastAPI) -> None:
    """Attach FastApiMCP to *app* and mount the enabled transports.

    Installs :class:`_McpLogHandler` on the ``fastapi_mcp.server`` logger so
    tool errors appear in the structured JSON log rather than as stderr tracebacks.
    The internal HTTP client uses the ASGI transport (no TCP) with
    ``MCP_USER_AGENT`` and injects the parent request ID for log correlation.
    Populates ``MCP_SERVER_CARD["tools"]`` in :mod:`stdapi.routes.core_root` with
    the discovered tool list so the server card reflects real tools at startup.

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
        http_client=AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://apiserver",
            headers={"User-Agent": server.MCP_USER_AGENT},
            timeout=SETTINGS.ai_response_timeout,
            event_hooks={"request": [_inject_request_id]},
        ),
    )
    if SETTINGS.enable_mcp_streamable_http:
        mcp.mount_http()
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
