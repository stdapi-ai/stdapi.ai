"""AWS Bedrock Mantle endpoint HTTP client.

The ``bedrock-mantle`` endpoint serves OpenAI-compatible (Chat Completions,
Responses) and Anthropic-compatible (Messages) APIs over plain HTTPS and is not
covered by botocore. This module provides the transport: short-term bearer
tokens derived from the shared botocore credential chain, a pooled HTTP
session, request/stream helpers with Mantle-specific error mapping, and pure
usage-extraction helpers for the three wire formats.

Mantle exposes two disjoint OpenAI routing surfaces: legacy-catalog models
answer under ``/v1`` while newer Mantle-only models answer under
``/openai/v1``. The Anthropic Messages API lives under ``/anthropic/v1``.
"""

from asyncio import sleep
from base64 import b32decode, b32encode, b64encode
from binascii import Error as Base32Error
from binascii import crc32
from collections.abc import Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from random import uniform
from re import compile as compile_regex
from time import monotonic
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict
from urllib.parse import urlsplit
from weakref import finalize

from aiohttp import (
    ClientConnectorDNSError,
    ClientSession,
    ClientTimeout,
    SocketTimeoutError,
)
from aiohttp import ClientError as AiohttpClientError
from aiohttp.http_exceptions import HttpProcessingError
from botocore.auth import SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from pydantic import ValidationError
from pydantic_core import from_json

from stdapi import server
from stdapi.api_errors import ApiError
from stdapi.config import AWS_SESSION, SETTINGS
from stdapi.utils import to_json_bytes, to_json_str

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator, Sequence

    from aiohttp import ClientResponse
    from starlette.datastructures import Headers
    from types_aiobotocore_bedrock.literals import RegionName

#: OpenAI-shaped routing surface: legacy catalog vs newer Mantle-only models.
type Surface = Literal["/v1", "/openai/v1"]

#: Mantle upstream API used to serve a request.
type MantleApi = Literal["chat_completions", "responses", "messages"]

#: Parsed server-sent event: (event name or None, raw data payload).
type SseEvent = tuple[str | None, str]


class UsageKwargs(TypedDict, total=False):
    """Token-usage keyword arguments for ``record_bedrock_usage``."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    cache_write_tokens: int


#: Mantle upstream API to request path (OpenAI paths relative to a surface).
API_PATHS: dict[MantleApi, str] = {
    "chat_completions": "/chat/completions",
    "responses": "/responses",
    "messages": "/anthropic/v1/messages",
}

#: Anthropic version header required by the Mantle Messages API.
MESSAGES_API_HEADERS = {"anthropic-version": "2023-06-01"}

#: OpenAI-compatible header selecting the Mantle project (cost/usage attribution).
_OPENAI_PROJECT_HEADER = "OpenAI-Project"

#: Anthropic Messages header selecting the Mantle workspace (same project ID).
_ANTHROPIC_WORKSPACE_HEADER = "anthropic-workspace"

#: Selected Mantle project/workspace ID for the current request.
MANTLE_PROJECT_VAR: ContextVar[str] = ContextVar("mantle_project", default="")

#: Valid Mantle project/workspace identifier ("default" or "proj_...").
_PROJECT_ID_RE = compile_regex(r"^(?:default|proj_[A-Za-z0-9]+)$")


def set_mantle_project(headers: Headers) -> None:
    """Select the Mantle project/workspace for the current request.

    A per-request ``OpenAI-Project`` or ``anthropic-workspace`` header is honored
    when ``aws_bedrock_allow_mantle_project_override`` is enabled, or whenever no
    default project is configured; otherwise the configured default is used. The
    project applies only to models served by the Bedrock Mantle endpoint.

    Args:
        headers: Incoming request headers.

    Raises:
        ApiError: If a request-supplied project identifier is malformed.
    """
    default = SETTINGS.aws_bedrock_mantle_project
    value = (
        headers.get(_OPENAI_PROJECT_HEADER)
        or headers.get(_ANTHROPIC_WORKSPACE_HEADER)
        or ""
    ).strip()
    if value and (SETTINGS.aws_bedrock_allow_mantle_project_override or not default):
        if not _PROJECT_ID_RE.match(value):
            msg = f"Invalid Bedrock Mantle project identifier: {value!r}"
            raise ApiError(msg)
        MANTLE_PROJECT_VAR.set(value)
    elif default:
        MANTLE_PROJECT_VAR.set(default)


def mantle_request_headers(api: MantleApi) -> dict[str, str] | None:
    """Build outbound headers for a Mantle API call.

    Combines the fixed Messages API version header with the selected project
    header (``anthropic-workspace`` on Messages, ``OpenAI-Project`` otherwise).

    Args:
        api: Target Mantle API.

    Returns:
        Header mapping, or ``None`` when no headers are required.
    """
    headers = dict(MESSAGES_API_HEADERS) if api == "messages" else {}
    if project := MANTLE_PROJECT_VAR.get():
        header = (
            _ANTHROPIC_WORKSPACE_HEADER if api == "messages" else _OPENAI_PROJECT_HEADER
        )
        headers[header] = project
    return headers or None


#: SigV4 service and host used to presign bearer tokens (shared with bedrock-runtime).
_TOKEN_HOST = "bedrock.amazonaws.com"  # noqa: S105

#: Bearer token prefix defined by the Bedrock API-key format.
_TOKEN_PREFIX = "bedrock-api-key-"  # noqa: S105

#: Presigned token validity in seconds (kept short; regeneration is local HMAC work).
_TOKEN_EXPIRY = 3600

#: Cached token refresh interval in seconds (below temporary-credential lifetimes).
_TOKEN_TTL = 300

#: Cached bearer tokens: region -> (token, monotonic expiry).
_TOKENS: dict[str, tuple[str, float]] = {}

#: Shared HTTP session toward Mantle endpoints (opened at startup when enabled).
_SESSION: ClientSession | None = None

#: Upstream error message pattern marking a model/API binding mismatch.
_UNSUPPORTED_API_RE = compile_regex(r"does not support the '[^']*' API")

#: Upstream error message marking a model/surface routing mismatch (any status).
_UNSUPPORTED_SURFACE_MARKER = "isn't supported on this route"

#: Misleading upstream 401 message returned when hitting the wrong surface.
_UNSUPPORTED_SURFACE_MARKER_401 = "is not enabled"

#: Safe charset for a decoded native Mantle response ID (interpolated into request URLs).
_NATIVE_RESPONSE_ID_RE = compile_regex(r"[A-Za-z0-9._-]+")

#: Multiple of the connect timeout allowed for one region's model catalog fetch.
_CATALOG_TIMEOUT_FACTOR: Final = 2


class MantleError(ApiError):
    """Mantle upstream error mapped to an API error.

    Attributes:
        failover: Whether the error indicates a region-level issue worth
            retrying in another region (throttling, capacity, 5xx).
    """

    failover: bool = False

    def __init__(
        self, message: str, *, status: int | None = None, failover: bool = False
    ) -> None:
        """Create a Mantle error.

        Args:
            message: Human-readable error message.
            status: Optional HTTP status code override.
            failover: Whether another region may serve the request.
        """
        super().__init__(message, status=status)
        self.failover = failover


class MantleApiUnsupportedError(MantleError):
    """The model does not support the requested Mantle API (binding mismatch)."""


class MantleSurfaceUnsupportedError(MantleError):
    """The model is not served on the requested routing surface."""


def endpoint_url(region: RegionName) -> str:
    """Return the Mantle endpoint base URL for *region*.

    Args:
        region: AWS region name.

    Returns:
        Base URL without a trailing slash.
    """
    if template := SETTINGS.aws_bedrock_mantle_endpoint_url:
        return template.format(region=region).rstrip("/")
    return f"https://bedrock-mantle.{region}.api.aws"


def _exception_causes(error: BaseException) -> Iterator[BaseException]:
    """Yield *error* then every exception it was raised from, outermost first.

    Follows ``__cause__`` and, in its absence, the implicit ``__context__``,
    exactly as a printed traceback would; a cycle stops the walk.

    Args:
        error: Exception to walk.

    Yields:
        The exception and each of its causes.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )


def format_exception_chain(error: BaseException) -> str:
    """Render an exception and its cause chain as one operator-facing log line.

    The message an upstream failure is mapped to is deliberately generic, so on
    its own it cannot tell a missing endpoint from a blocked route, a refused
    connection, a TLS failure or a read timeout: the causes carry that.

    Args:
        error: Exception to render.

    Returns:
        Single-line ``"outermost <- ... <- root cause"`` rendering.
    """
    return " <- ".join(
        f"{type(cause).__name__}: {cause}" if str(cause) else type(cause).__name__
        for cause in _exception_causes(error)
    )


def endpoint_unresolved(error: BaseException, region: RegionName) -> bool:
    """Whether *error* was caused by the Mantle endpoint hostname not resolving.

    A region that does not serve Amazon Bedrock Mantle has no
    ``bedrock-mantle`` DNS record at all, which is a permanent configuration
    fact rather than the transient failure every other connection error is.
    The failing hostname must be the endpoint's own: through a proxy, the name
    that did not resolve is the proxy's, and blaming the region for it would
    send the operator to drop a perfectly good one.

    Args:
        error: Exception raised by a Mantle request.
        region: Region the request was sent to.

    Returns:
        ``True`` when the endpoint hostname does not resolve.
    """
    host = urlsplit(endpoint_url(region)).hostname
    return any(
        isinstance(cause, ClientConnectorDNSError) and cause.host == host
        for cause in _exception_causes(error)
    )


def catalog_timeout() -> float:
    """Return the time budget for fetching one region's Mantle model catalog.

    The shared session grants a response up to ``ai_response_timeout`` to be
    read, which a region that accepts connections and never answers would spend
    of the server's startup. Listing a catalog is a short call, so its budget
    follows the connect timeout instead.

    Returns:
        Budget in seconds.
    """
    return SETTINGS.aws_connect_timeout * _CATALOG_TIMEOUT_FACTOR


async def bearer_token(region: RegionName) -> str:
    """Return a short-term Mantle bearer token for *region*.

    Tokens are presigned locally from the shared botocore credential chain
    (no static secret involved) and cached briefly so refreshed session
    credentials are picked up transparently.

    Args:
        region: AWS region the token is scoped to.

    Returns:
        Bearer token string.

    Raises:
        ApiError: When no AWS credentials are available.
    """
    if (cached := _TOKENS.get(region)) and cached[1] > monotonic():
        return cached[0]
    if (credentials := await AWS_SESSION.get_credentials()) is None:
        msg = "No AWS credentials available to authenticate with Bedrock Mantle."
        raise ApiError(msg, status=500)
    frozen = await credentials.get_frozen_credentials()
    if not (frozen.access_key and frozen.secret_key):  # pragma: no cover
        msg = "No AWS credentials available to authenticate with Bedrock Mantle."
        raise ApiError(msg, status=500)
    request = AWSRequest(
        method="POST",
        url=f"https://{_TOKEN_HOST}/",
        headers={"host": _TOKEN_HOST},
        params={"Action": "CallWithBearerToken"},
    )
    SigV4QueryAuth(
        Credentials(frozen.access_key, frozen.secret_key, frozen.token),
        "bedrock",
        region,
        expires=_TOKEN_EXPIRY,
    ).add_auth(request)
    token = _TOKEN_PREFIX + b64encode(
        f"{str(request.url).removeprefix('https://')}&Version=1".encode()
    ).decode("ascii")
    _TOKENS[region] = (token, monotonic() + _TOKEN_TTL)
    return token


@asynccontextmanager
async def mantle_http_session() -> AsyncGenerator[ClientSession]:
    """Open the shared Mantle HTTP session for the server's lifetime.

    Only the first opener owns the module-level session: when one is already
    live (a secondary connection manager, or a warmup attempt cleaned up after
    a concurrent client failure), it is reused and left untouched on exit, so
    the server's session is never closed from under it.

    Yields:
        The shared :class:`aiohttp.ClientSession`.
    """
    global _SESSION  # noqa: PLW0603
    if _SESSION is not None:
        yield _SESSION
        return
    session = ClientSession(
        headers=server.HTTP_CLIENT_HEADERS,
        timeout=ClientTimeout(
            total=None,
            connect=SETTINGS.aws_connect_timeout,
            sock_read=SETTINGS.ai_response_timeout,
        ),
        # Large SSE lines: a single event can carry the whole response JSON.
        read_bufsize=2**22,
        # Same proxy environment the AWS SDK already honours unconditionally.
        trust_env=True,
    )
    _SESSION = session
    try:
        yield session
    finally:
        _SESSION = None
        _TOKENS.clear()
        await session.close()


def _map_error(status: int, body: str, region: RegionName) -> MantleError:
    """Map a Mantle HTTP error response to a :class:`MantleError`.

    Args:
        status: HTTP status code.
        body: Raw response body.
        region: Region the request was sent to.

    Returns:
        The mapped error, ready to raise.
    """
    try:
        details: Any = from_json(body).get("error") or {}
    except ValueError, AttributeError:
        details = {}
    if not isinstance(details, Mapping):
        # Some upstream errors carry a bare string in the "error" field.
        details = {"message": str(details)}
    raw_message = details.get("message")
    if raw_message is not None and not isinstance(raw_message, str):
        # Some upstream errors nest structured content in the message field.
        raw_message = to_json_str(raw_message)
    message = raw_message or f"The request could not be completed (HTTP {status})."
    error: MantleError
    if _UNSUPPORTED_API_RE.search(message):
        error = MantleApiUnsupportedError(message, status=400)
    elif _UNSUPPORTED_SURFACE_MARKER in message or (
        # Gated on 401 so genuine 403 permission errors ("Model access is
        # not enabled ...") still map to the server-side branch below.
        status == 401 and _UNSUPPORTED_SURFACE_MARKER_401 in message
    ):
        error = MantleSurfaceUnsupportedError(message, status=400)
    elif status == 429 or status >= 500:
        error = MantleError(message, status=status, failover=True)
    elif status in (401, 403):
        # Server-side credential/permission issue: evict the cached bearer token
        # so a rotated credential self-heals on the next request instead of
        # failing for up to _TOKEN_TTL seconds. The upstream message may
        # disclose IAM ARNs/permissions, so log it and forward a generic one.
        _TOKENS.pop(region, None)
        # Imported here: stdapi.monitoring transitively imports this module.
        from stdapi.monitoring import log_error_details  # noqa: PLC0415

        log_error_details(message, level="warning")
        # 503 feature_unavailable, as every other backend answers a denial of
        # the server's own role: a 500 tells the caller to retry something no
        # retry can fix, and "Retry the request" is the opposite of true. The
        # message stays generic -- the operator reads the real cause, IAM ARN
        # and action, in the warning logged just above.
        error = MantleError(
            "This feature is not available on this deployment.", status=503
        )
        error.code = "feature_unavailable"
        # Upstream's own code must not displace it below.
        details = {}
    else:
        error = MantleError(message, status=status)
    if code := details.get("code"):
        error.code = code
    if param := details.get("param"):
        error.param = param
    return error


async def _request(
    region: RegionName,
    path: str,
    body: bytes | None,
    headers: Mapping[str, str] | None,
    method: str = "POST",
) -> ClientResponse:
    """Send one request to Mantle and validate its status.

    Args:
        region: Target AWS region.
        path: Request path, absolute from the endpoint root.
        body: Serialized JSON request body, or ``None`` for a body-less call.
        headers: Extra HTTP headers.
        method: HTTP method.

    Returns:
        The open :class:`aiohttp.ClientResponse` (status already checked).

    Raises:
        MantleError: On upstream errors or connection failures; a post-send
            read timeout is never flagged for failover since the invocation
            already reached Mantle and is billed regardless.
    """
    if _SESSION is None:  # pragma: no cover
        msg = "Bedrock Mantle support is not initialized on this server."
        raise ApiError(msg, status=503)
    try:
        response = await _SESSION.request(
            method,
            f"{endpoint_url(region)}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {await bearer_token(region)}",
                **({"Content-Type": "application/json"} if body is not None else {}),
                **(headers or {}),
            },
        )
    except SocketTimeoutError as error:
        # Imported here: stdapi.monitoring transitively imports this module.
        from stdapi.monitoring import log_error_details  # noqa: PLC0415

        log_error_details(f"Timed out reading the Bedrock Mantle response in {region}.")
        # No failover: the invocation reached Mantle and is billed regardless.
        msg = "The service is temporarily unavailable. Retry the request."
        raise MantleError(msg, status=503) from error
    except (AiohttpClientError, TimeoutError) as error:
        # Imported here: stdapi.monitoring transitively imports this module.
        from stdapi.monitoring import log_error_details  # noqa: PLC0415

        log_error_details(f"Unable to reach the Bedrock Mantle endpoint in {region}.")
        msg = "The service is temporarily unavailable. Retry the request."
        raise MantleError(msg, status=503, failover=True) from error
    if response.status >= 400:
        try:
            error_body = await response.text()
        except AiohttpClientError, TimeoutError:
            error_body = ""
        finally:
            response.release()
        raise _map_error(response.status, error_body, region)
    return response


async def _request_with_retry(
    region: RegionName,
    path: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None,
    *,
    single_region: bool,
) -> ClientResponse:
    """Send a POST request, retrying failover-class errors in-region.

    Mirrors the botocore client split: when the region router can cycle
    regions (*single_region* is False) no in-region retry is performed.

    Args:
        region: Target AWS region.
        path: Request path, absolute from the endpoint root.
        payload: JSON request body.
        headers: Extra HTTP headers.
        single_region: Whether this region is the only candidate.

    Returns:
        The open :class:`aiohttp.ClientResponse`.

    Raises:
        MantleError: When every attempt fails.
    """
    body = to_json_bytes(payload)
    retries = SETTINGS.aws_bedrock_max_retries if single_region else 0
    for attempt in range(retries):
        try:
            return await _request(region, path, body, headers)
        except MantleError as error:
            if not error.failover:
                raise
            # Exponential backoff with full jitter, capped at 20 seconds.
            await sleep(uniform(0, min(2**attempt, 20)))  # noqa: S311
    return await _request(region, path, body, headers)


async def invoke(
    region: RegionName,
    path: str,
    payload: Mapping[str, Any],
    *,
    single_region: bool,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Invoke a Mantle API and return the parsed JSON response.

    Args:
        region: Target AWS region.
        path: Request path, absolute from the endpoint root.
        payload: JSON request body.
        single_region: Whether this region is the only candidate.
        headers: Extra HTTP headers.

    Returns:
        Parsed JSON response body.

    Raises:
        MantleError: On upstream errors.
    """
    return await _read_json(
        await _request_with_retry(
            region, path, payload, headers, single_region=single_region
        )
    )


async def request_json(
    region: RegionName,
    method: str,
    path: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Send a body-less Mantle request (GET/POST/DELETE) and return the JSON response.

    Args:
        region: Target AWS region.
        method: HTTP method.
        path: Request path, absolute from the endpoint root.
        headers: Extra HTTP headers.

    Returns:
        Parsed JSON response body.

    Raises:
        MantleError: On upstream errors or connection failures.
    """
    return await _read_json(await _request(region, path, None, headers, method))


async def _read_json(response: ClientResponse) -> dict[str, Any]:
    """Read and parse an open Mantle JSON response, then release it.

    Args:
        response: Open HTTP response (status already checked).

    Returns:
        Parsed JSON response body.

    Raises:
        MantleError: When the body cannot be read or parsed.
    """
    async with response:
        try:
            return await response.json(  # type: ignore[no-any-return]
                content_type=None, loads=from_json
            )
        except (AiohttpClientError, TimeoutError, ValueError) as error:
            msg = "The Bedrock Mantle response could not be read."
            raise MantleError(msg, status=502) from error


async def invoke_stream(
    region: RegionName,
    path: str,
    payload: Mapping[str, Any],
    *,
    single_region: bool,
    headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[SseEvent]:
    """Invoke a streaming Mantle API and return its server-sent event stream.

    Region failover remains possible until this coroutine returns; once the
    stream is open the region is locked for the duration.

    Args:
        region: Target AWS region.
        path: Request path, absolute from the endpoint root.
        payload: JSON request body.
        single_region: Whether this region is the only candidate.
        headers: Extra HTTP headers.

    Returns:
        Async generator of ``(event name or None, raw data)`` tuples,
        terminating before any ``[DONE]`` sentinel. Its ``async with`` only
        runs once iteration starts, so a GC-tied fallback closes the response
        for a generator dropped without ever being iterated (e.g. an immediate
        client disconnect).

    Raises:
        MantleError: On upstream errors before the stream opens.
    """
    response = await _request_with_retry(
        region, path, payload, headers, single_region=single_region
    )
    generator = _iter_sse(response)
    finalize(generator, response.close)
    return generator


async def _iter_sse(response: ClientResponse) -> AsyncGenerator[SseEvent]:
    """Yield parsed server-sent events from an open response.

    Args:
        response: Open streaming HTTP response.

    Yields:
        ``(event name or None, raw data)`` tuples.

    Raises:
        MantleError: When the connection drops mid-stream.
    """
    event: str | None = None
    data: list[str] = []
    try:
        async with response:
            async for raw_line in response.content:
                match raw_line.decode().rstrip("\r\n"):
                    case "":
                        if data and (joined := "\n".join(data)) != "[DONE]":
                            yield event, joined
                        event, data = None, []
                    case line if line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    case line if line.startswith("event:"):
                        event = line[6:].lstrip(" ")
                    case _:
                        pass  # Comments and unknown fields are ignored.
        if data and (joined := "\n".join(data)) != "[DONE]":  # pragma: no cover
            yield event, joined
    except (
        AiohttpClientError,
        TimeoutError,
        HttpProcessingError,
        UnicodeDecodeError,
    ) as error:
        # HttpProcessingError covers LineTooLong, which aiohttp raises outside
        # its ClientError hierarchy for oversized SSE lines; UnicodeDecodeError
        # covers a non-UTF-8 line mid-stream.
        msg = "The Bedrock Mantle response stream was interrupted."
        raise MantleError(msg, status=502) from error


#: Maximum pruning rounds when validating an upstream passthrough response.
_MAX_PRUNE_ROUNDS = 5


def validate_pruning_extras[ModelT](
    model_cls: type[ModelT], raw: dict[str, Any]
) -> ModelT:
    """Validate an upstream payload, pruning unknown extra fields on mismatch.

    Mantle responses may carry provider extensions ahead of the server's
    schemas (e.g. ``billing``); passthrough must tolerate them rather than
    fail, so fields rejected as ``extra_forbidden`` are dropped in-place and
    validation retried. Any other validation error propagates.

    Args:
        model_cls: Pydantic response model to validate against.
        raw: Upstream JSON payload (mutated in-place when pruning).

    Returns:
        The validated model instance.

    Raises:
        MantleError: When the payload is invalid beyond extra fields.
    """
    for _ in range(_MAX_PRUNE_ROUNDS):
        try:
            return model_cls.model_validate(raw)  # type: ignore[attr-defined,no-any-return]
        except ValidationError as error:
            pruned = False
            for issue in error.errors():
                if issue["type"] != "extra_forbidden":
                    continue
                *path, leaf = issue["loc"]
                target = _resolve_error_loc(raw, path)
                if isinstance(target, dict) and leaf in target:
                    del target[leaf]
                    pruned = True
            if not pruned:
                # Surface a shaped 502 instead of pydantic's plain-text error.
                msg = "The Bedrock Mantle response failed validation."
                raise MantleError(msg, status=502) from error
    try:
        return model_cls.model_validate(raw)  # type: ignore[attr-defined,no-any-return]
    except ValidationError as error:
        msg = "The Bedrock Mantle response failed validation."
        raise MantleError(msg, status=502) from error


def _resolve_error_loc(raw: dict[str, Any], path: Sequence[int | str]) -> object:
    """Resolve a pydantic error location path inside a raw payload.

    Union member locs interleave tag labels that are not payload keys;
    segments missing from a dict are skipped so traversal stays on the
    current container. Traversal never raises: unresolvable list or scalar
    segments yield ``None`` (the caller then treats the leaf as not pruned).

    Args:
        raw: Payload the error location refers to.
        path: Error location segments, without the leaf.

    Returns:
        The container holding the leaf, or ``None`` when unresolvable.
    """
    target: Any = raw
    for key in path:
        if isinstance(target, dict):
            if key in target:
                target = target[key]
        elif isinstance(target, list):
            if not isinstance(key, int) or not -len(target) <= key < len(target):
                return None
            target = target[key]
        else:
            return None
    return target


def encode_mantle_response_id(region: RegionName, native_id: str) -> str:
    """Tag a native Mantle response ID with its serving region.

    Mantle stored responses are region-local: the public ID embeds a CRC32
    fingerprint of the region so chained requests can be pinned back to it.
    The resulting ID is not reusable against the Mantle API directly.

    Args:
        region: Region that stores the response.
        native_id: Native Mantle response ID.

    Returns:
        Public ``resp_``-prefixed identifier.
    """
    payload = crc32(region.encode()).to_bytes(4, "big") + native_id.encode()
    return "resp_" + b32encode(payload).decode("ascii").lower().rstrip("=")


#: Cached (regions object, fingerprint -> region mapping); rebuilt when the
#: configured regions object changes (startup reconfiguration, or tests
#: monkeypatching ``SETTINGS.aws_bedrock_mantle_regions``).
_REGION_FINGERPRINTS_CACHE: tuple[Any, dict[int, RegionName]] | None = None


def _region_fingerprints() -> dict[int, RegionName]:
    """Return the crc32-fingerprint -> region mapping for the configured Mantle regions.

    Lazily built and cached by the identity of
    ``SETTINGS.aws_bedrock_mantle_regions`` so it stays in sync whenever that
    setting is replaced (regions are otherwise final after config load).

    Returns:
        Mapping of each configured region's crc32 fingerprint to the region.
    """
    global _REGION_FINGERPRINTS_CACHE  # noqa: PLW0603
    regions = SETTINGS.aws_bedrock_mantle_regions
    if (
        _REGION_FINGERPRINTS_CACHE is None
        or _REGION_FINGERPRINTS_CACHE[0] is not regions
    ):
        _REGION_FINGERPRINTS_CACHE = (
            regions,
            {crc32(region.encode()): region for region in regions},
        )
    return _REGION_FINGERPRINTS_CACHE[1]


def decode_mantle_response_id(public_id: str) -> tuple[RegionName, str] | None:
    """Decode a region-tagged public response ID.

    Args:
        public_id: Public response identifier.

    Returns:
        Tuple of (region, native Mantle response ID), or ``None`` when the ID
        is not a region-tagged Mantle ID (e.g. a native stdapi stored ID) or
        decodes to a native ID unsafe to interpolate into a request URL.
    """
    if not public_id.startswith("resp_"):
        return None
    encoded = public_id.removeprefix("resp_").upper()
    try:
        raw = b32decode(encoded + "=" * (-len(encoded) % 8))
    except Base32Error:
        return None
    fingerprint = int.from_bytes(raw[:4], "big")
    region = _region_fingerprints().get(fingerprint)
    if region is None:
        return None
    try:
        native_id = raw[4:].decode("ascii")
    except UnicodeDecodeError:
        return None
    # The native ID is interpolated verbatim into upstream request URLs: reject
    # anything outside a safe charset to prevent path/query/fragment injection.
    if not _NATIVE_RESPONSE_ID_RE.fullmatch(native_id):
        return None
    return region, native_id


def _openai_usage(
    usage: Mapping[str, Any], input_key: str, output_key: str, details_key: str
) -> UsageKwargs:
    """Extract usage-recording keyword arguments from an OpenAI ``usage`` block.

    OpenAI input token counts include cached tokens; Bedrock accounting keeps
    fresh and cached input separate, so the cached share is subtracted.

    Args:
        usage: OpenAI-shaped usage object.
        input_key: Key holding the input token count.
        output_key: Key holding the output token count.
        details_key: Key holding the input token details object.

    Returns:
        Keyword arguments for ``record_bedrock_usage``.
    """
    details = usage.get(details_key) or {}
    cached = details.get("cached_tokens") or 0
    cache_written = details.get("cache_write_tokens") or 0
    return {
        "input_tokens": max(0, (usage.get(input_key) or 0) - cached - cache_written),
        "output_tokens": usage.get(output_key) or 0,
        "total_tokens": usage.get("total_tokens") or 0,
        "cached_tokens": cached,
        "cache_write_tokens": cache_written,
    }


def usage_from_chat_completion(usage: Mapping[str, Any]) -> UsageKwargs:
    """Extract usage-recording keyword arguments from a Chat Completions ``usage``.

    Args:
        usage: OpenAI Chat Completions usage object.

    Returns:
        Keyword arguments for ``record_bedrock_usage``.
    """
    return _openai_usage(
        usage, "prompt_tokens", "completion_tokens", "prompt_tokens_details"
    )


def usage_from_response(usage: Mapping[str, Any]) -> UsageKwargs:
    """Extract usage-recording keyword arguments from a Responses API ``usage``.

    Args:
        usage: OpenAI Responses usage object.

    Returns:
        Keyword arguments for ``record_bedrock_usage``.
    """
    return _openai_usage(usage, "input_tokens", "output_tokens", "input_tokens_details")


def usage_from_message(usage: Mapping[str, Any]) -> UsageKwargs:
    """Extract usage-recording keyword arguments from an Anthropic ``usage``.

    Anthropic ``input_tokens`` already excludes cache reads and writes.

    Args:
        usage: Anthropic Messages usage object.

    Returns:
        Keyword arguments for ``record_bedrock_usage``.
    """
    return {
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "cached_tokens": usage.get("cache_read_input_tokens") or 0,
        "cache_write_tokens": usage.get("cache_creation_input_tokens") or 0,
    }


def web_search_queries(item: Mapping[str, Any]) -> int:
    """Count the billed web-search queries one response output item performed.

    AWS meters the built-in web search per query, and one search call may run
    several at once. Page reads (``open_page``/``find_in_page``) are a separate
    unmetered operation, so only search actions count.

    Args:
        item: One Responses API output item.

    Returns:
        Number of billed queries, or 0 for any other output item.
    """
    if item.get("type") != "web_search_call":
        return 0
    action = item.get("action") or {}
    if action.get("type") != "search":
        return 0
    if queries := action.get("queries"):
        return len(queries) if isinstance(queries, list) else 1
    return 1 if action.get("query") else 0


def response_web_search_queries(response: Mapping[str, Any]) -> int:
    """Count the billed web-search queries a complete Responses payload performed.

    Args:
        response: Complete Responses API response.

    Returns:
        Total number of billed queries across every output item.
    """
    return sum(web_search_queries(item) for item in response.get("output") or ())


#: Stored-response native ID -> serving OpenAI routing surface (bounded LRU).
_SURFACE_CACHE: dict[str, Surface] = {}

#: Maximum entries retained in the stored-response surface cache.
_SURFACE_CACHE_MAX = 4096


def cache_response_surface(native_id: str, surface: Surface) -> None:
    """Remember the routing surface serving a stored response.

    Args:
        native_id: Native Mantle response ID.
        surface: Surface the response is reachable on.
    """
    if native_id not in _SURFACE_CACHE and len(_SURFACE_CACHE) >= _SURFACE_CACHE_MAX:
        del _SURFACE_CACHE[next(iter(_SURFACE_CACHE))]
    _SURFACE_CACHE[native_id] = surface


def cached_response_surface(native_id: str) -> Surface | None:
    """Return the remembered surface for a stored response, refreshing its age.

    Args:
        native_id: Native Mantle response ID.

    Returns:
        The cached surface, or ``None`` when unknown.
    """
    if (surface := _SURFACE_CACHE.pop(native_id, None)) is not None:
        _SURFACE_CACHE[native_id] = surface
    return surface
