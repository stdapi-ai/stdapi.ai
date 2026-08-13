"""MCP (Model Context Protocol) server setup.

Registers FastApiMCP, mounts the enabled transports, and wires up structured-logging
and request-ID propagation so that internal MCP → API calls share the same log entry.
"""

from contextlib import suppress
from json import JSONDecodeError
from logging import ERROR, Handler, LogRecord, getLogger
from traceback import format_exception
from typing import TYPE_CHECKING

import fastapi_mcp.server  # type: ignore[import-untyped]
from fastapi import Depends
from fastapi_mcp import AuthConfig, FastApiMCP
from httpx import ASGITransport, AsyncClient
from httpx import Request as HttpxRequest
from pydantic_core import to_json

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

#: Streamable HTTP body ceiling when no input file size is configured.
_MCP_MAX_BODY_SIZE = 64 * 1024 * 1024

#: Request parameters an MCP client cannot use: streaming modes (tool results are
#: single messages), token-level tuning, and caller-identity/routing identifiers.
_HIDDEN_TOOL_PARAMS: frozenset[str] = frozenset(
    {
        "stream",
        "stream_options",
        "user",
        "safety_identifier",
        "prompt_cache_key",
        "logit_bias",
        "logprobs",
        "top_logprobs",
    }
)


class _CompactJson:
    """``json`` façade for ``fastapi_mcp.server`` rendering tool results compactly.

    ``fastapi_mcp`` re-serializes every tool result with ``indent=2``, inflating
    the payload the MCP client feeds to its LLM by roughly a third. Swapping the
    module's ``json`` attribute for this façade keeps the parse-failure handling
    (``JSONDecodeError``) while rendering with the native serializer instead.
    """

    #: Exception ``fastapi_mcp`` catches when a response body is not JSON.
    JSONDecodeError = JSONDecodeError

    @staticmethod
    def dumps(obj: object, **_kwargs: object) -> str:
        """Serialize *obj* compactly, ignoring formatting keyword arguments.

        Args:
            obj: Parsed tool result to serialize.
            _kwargs: Formatting options from the caller, all ignored.

        Returns:
            Compact JSON text.
        """
        return to_json(obj).decode()


def _collect_refs(node: object, refs: set[str]) -> None:
    """Accumulate every ``$ref`` target name reachable from *node*.

    Args:
        node: JSON schema fragment to walk.
        refs: Output set receiving ``$defs`` entry names.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                refs.add(value.rpartition("/")[2])
            else:
                _collect_refs(value, refs)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, refs)


def _prune_hidden_params(tools: list[Tool]) -> None:
    """Drop LLM-unusable parameters and newly unreferenced ``$defs`` from tool schemas.

    Hidden parameters remain accepted by the API routes; they are only removed
    from the advertised input schemas so they stop costing MCP client context.

    Args:
        tools: MCP tools to mutate in place.
    """
    for tool in tools:
        schema = tool.inputSchema
        properties = schema.get("properties")
        if not properties or not _HIDDEN_TOOL_PARAMS.intersection(properties):
            continue
        for name in _HIDDEN_TOOL_PARAMS.intersection(properties):
            del properties[name]
        if required := schema.get("required"):
            schema["required"] = [
                name for name in required if name not in _HIDDEN_TOOL_PARAMS
            ]
        if defs := schema.get("$defs"):
            reachable: set[str] = set()
            _collect_refs(
                {key: schema[key] for key in schema if key != "$defs"}, reachable
            )
            frontier = set(reachable)
            while frontier:
                previous = set(reachable)
                for name in frontier:
                    _collect_refs(defs.get(name), reachable)
                frontier = reachable - previous
            for name in set(defs) - reachable:
                del defs[name]


def _fix_union_param_types(tools: list[Tool]) -> None:
    """Drop the contradictory ``type`` key fastapi_mcp adds beside ``anyOf``.

    ``fastapi_mcp`` stamps every tool parameter with a single ``type`` picked
    from an unordered set of the union's member types, so a ``str | list``
    parameter randomly advertises ``"type": "array"`` depending on the process
    hash seed — and the MCP SDK's server-side schema validation then rejects
    perfectly valid string arguments. The ``anyOf`` already constrains the
    parameter fully, so the sibling ``type`` is removed.

    Args:
        tools: MCP tools to mutate in place.
    """
    for tool in tools:
        for prop in (tool.inputSchema.get("properties") or {}).values():
            if isinstance(prop, dict) and "anyOf" in prop and "type" in prop:
                del prop["type"]


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

    Requires both the User-Agent header and the internal request ID header, so
    a client cannot claim to be MCP on its own.

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


def _lift_body_limit(mcp: FastApiMCP) -> None:
    """Stop the MCP transport rejecting media the API itself accepts.

    The MCP SDK caps a Streamable HTTP request body at 4 MiB and answers 413
    before parsing, while the same tool called over HTTP is bounded only by
    ``max_input_file_size`` (unlimited by default). Base64 inflates media by a
    third, so an agent could not edit an image much over 3 MB through a tool it
    could edit directly. The limit follows the API's own where one is set, and
    otherwise rises to a ceiling wide enough that the transport is never the
    binding constraint while a hostile body is still bounded.

    Args:
        mcp: The FastApiMCP instance whose HTTP transport was just mounted.
    """
    transport = mcp._http_transport  # noqa: SLF001
    start = transport._ensure_session_manager_started  # noqa: SLF001

    async def start_with_limit() -> None:
        await start()
        if (manager := transport._session_manager) is not None:  # noqa: SLF001
            manager.asgi_app.max_body_size = (
                SETTINGS.max_input_file_size or _MCP_MAX_BODY_SIZE
            )

    transport._ensure_session_manager_started = start_with_limit  # noqa: SLF001


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

    Tool errors are routed to the structured JSON log instead of stderr, and the
    internal HTTP client reaches the API over the ASGI transport (no TCP) while
    injecting the parent request ID for log correlation. The discovered tools are
    trimmed and repaired for MCP clients, then published in
    ``MCP_SERVER_CARD["tools"]`` so the server card reflects them at startup.

    Args:
        app: FastAPI application to attach MCP to.
    """
    _mcp_logger = getLogger("fastapi_mcp.server")
    _mcp_logger.addHandler(_McpLogHandler())
    _mcp_logger.propagate = False
    fastapi_mcp.server.json = _CompactJson

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
    _prune_hidden_params(mcp.tools)
    _fix_union_param_types(mcp.tools)

    if SETTINGS.enable_mcp_streamable_http:
        mcp.mount_http()
        _lift_body_limit(mcp)
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
