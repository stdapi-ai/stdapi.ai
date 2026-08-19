"""MCP (Model Context Protocol) server setup.

Registers FastApiMCP, mounts the enabled transports, and wires up structured-logging
and request-ID propagation so that internal MCP → API calls share the same log entry.
"""

from contextlib import suppress
from contextvars import ContextVar
from json import JSONDecodeError
from logging import ERROR, Handler, LogRecord, getLogger
from traceback import format_exception
from typing import TYPE_CHECKING, Any

import fastapi_mcp.server  # type: ignore[import-untyped]
from fastapi import Depends
from fastapi_mcp import AuthConfig, FastApiMCP
from httpx import ASGITransport, AsyncClient, QueryParams
from httpx import Request as HttpxRequest
from httpx import Response as HttpxResponse
from mcp.types import AudioContent, ImageContent
from pydantic_core import to_json

from stdapi import server
from stdapi.auth import authenticate
from stdapi.config import SETTINGS, LogLevel
from stdapi.metering import EDITION_TITLE
from stdapi.monitoring import REQUEST, REQUEST_ID, log_error_details
from stdapi.routes.core_root import MCP_SERVER_CARD
from stdapi.utils import b64encode

if TYPE_CHECKING:
    from collections.abc import Buffer

    from fastapi import FastAPI
    from mcp.types import Tool

#: Marker fastapi_mcp always prepends to the auto-generated response/example block.
_RESPONSES_MARKER = "\n\n### Responses:"

#: Streamable HTTP body ceiling when no input file size is configured.
_MCP_MAX_BODY_SIZE = 64 * 1024 * 1024

#: Largest response body returned inline as MCP media; base64 inflates it by a third.
_MCP_MAX_INLINE_BYTES = 3 * 1024 * 1024

#: Media types outside ``text/*`` whose body a text tool result still carries intact.
_TEXT_MEDIA_TYPES: frozenset[str] = frozenset(
    {
        "application/javascript",
        "application/json",
        "application/x-json-stream",
        "application/x-jsonlines",
        "application/x-ndjson",
        "application/x-www-form-urlencoded",
        "application/xml",
    }
)

#: Media returned by the tool call in flight, read back once its result is built.
_INLINE_MEDIA: ContextVar[AudioContent | ImageContent | None] = ContextVar(
    "_INLINE_MEDIA", default=None
)

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


def _is_text_body(content_type: str) -> bool:
    """Tell whether a response body survives being carried as tool result text.

    Args:
        content_type: Value of the response's ``Content-Type`` header.

    Returns:
        True for text and text-based structured formats, False for binary ones.
    """
    media_type = content_type.partition(";")[0].strip().lower()
    return (
        not media_type
        or media_type.startswith("text/")
        or media_type in _TEXT_MEDIA_TYPES
        or media_type.endswith(("+json", "+xml"))
    )


async def _as_media(media_type: str, payload: Buffer) -> AudioContent | ImageContent:
    """Wrap a binary payload in the MCP content type that carries it.

    Args:
        media_type: Media type of the payload, an ``image/`` or ``audio/`` one.
        payload: Raw response body.

    Returns:
        The image or audio content block for the payload.
    """
    data = await b64encode(payload)
    if media_type.startswith("image/"):
        return ImageContent(type="image", data=data, mimeType=media_type)
    return AudioContent(type="audio", data=data, mimeType=media_type)


def _as_download_reference(
    response: HttpxResponse, media_type: str, path: str, query: dict[str, Any]
) -> HttpxResponse:
    """Build the JSON body standing in for a body that cannot travel inline.

    Args:
        response: The answered request, already closed.
        media_type: Media type of the body being replaced.
        path: Request path, with its path parameters substituted.
        query: Request query parameters.

    Returns:
        A response carrying the reference as JSON.
    """
    reference: dict[str, Any] = {
        "content_type": media_type or "application/octet-stream",
        "url": f"{path}?{params}" if (params := str(QueryParams(query))) else path,
        "message": "Content is not returned inline; download it from 'url'.",
    }
    if size := response.headers.get("content-length"):
        reference["size_bytes"] = int(size)
    return HttpxResponse(
        response.status_code,
        headers={"content-type": "application/json"},
        content=to_json(reference),
        request=response.request,
    )


def _bind_media_results(mcp: FastApiMCP) -> None:
    """Make the tools whose route answers with bytes usable by an MCP client.

    ``fastapi_mcp`` parses every response as JSON and falls back to its decoded
    text, so a binary body either raises while being decoded — an MP4 starts
    with the null bytes ``json.loads`` reads as a UTF-32 signature — or reaches
    the agent as mojibake. Images and audio small enough to travel come back as
    the MCP content types built for them; everything else, video above all,
    comes back as a JSON reference to the URL it downloads from, and its body is
    never read, so a large asset is not buffered to be discarded.

    Args:
        mcp: The FastApiMCP instance whose tool calls are being wrapped.
    """
    execute = mcp._execute_api_tool  # noqa: SLF001

    async def request(
        client: AsyncClient,
        method: str,
        path: str,
        query: dict[str, Any],
        headers: dict[str, str],
        body: Any,  # noqa: ANN401
    ) -> HttpxResponse:
        """Answer the tool's API call, holding a binary body out of the result."""
        verb = method.upper()
        response = await client.send(
            client.build_request(
                verb,
                path,
                params=query,
                headers=headers,
                json=body if verb in {"POST", "PUT", "PATCH"} else None,
            ),
            stream=True,
        )
        content_type = response.headers.get("content-type", "")
        if response.status_code >= 400 or _is_text_body(content_type):
            await response.aread()
            return response
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type.startswith(("image/", "audio/")):
            payload = bytearray()
            async for chunk in response.aiter_bytes():
                payload += chunk
                if len(payload) > _MCP_MAX_INLINE_BYTES:
                    break
            else:
                _INLINE_MEDIA.set(await _as_media(media_type, payload))
        await response.aclose()
        return _as_download_reference(response, media_type, path, query)

    async def execute_api_tool(**kwargs: Any) -> list[Any]:  # noqa: ANN401
        """Return the media the response carried, or the text result as built."""
        token = _INLINE_MEDIA.set(None)
        try:
            contents = await execute(**kwargs)
            media = _INLINE_MEDIA.get()
        finally:
            _INLINE_MEDIA.reset(token)
        return [media] if media is not None else contents

    mcp._request = request  # noqa: SLF001
    mcp._execute_api_tool = execute_api_tool  # noqa: SLF001


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
    _bind_media_results(mcp)

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
