"""Monitoring."""

from contextlib import contextmanager, suppress
from contextvars import ContextVar
from re import compile as re_compile
from time import perf_counter_ns
from traceback import format_exception
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, TypeVar

from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from fastapi import Request  # noqa: TC002
from pydantic import AwareDatetime, BaseModel, JsonValue
from sse_starlette import JSONServerSentEvent, ServerSentEvent

from stdapi import server
from stdapi.api_errors import ApiError
from stdapi.api_providers import format_http_error
from stdapi.aws_bedrock import AWS_ERROR_MAP
from stdapi.config import SETTINGS, LogLevel
from stdapi.metering import SERVER_FULL_VERSION
from stdapi.utils import hide_security_details, stdout_write, webuuid

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Mapping

    from pydantic.main import IncEx
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_meteringmarketplace.type_defs import (
        RegisterUsageResultTypeDef,
    )

    from stdapi.monitoring_otel import OpenTelemetryManager
    from stdapi.types import JsonList, JsonMappingOrList

if not SETTINGS.otel_enabled:
    from stdapi.monitoring_otel_base import (  # type: ignore[assignment]
        OpenTelemetryManager,
    )

else:
    from opentelemetry.trace import Status, StatusCode

    from stdapi.monitoring_otel import OpenTelemetryManager

otel_manager = OpenTelemetryManager()

T = TypeVar("T")
RegionLatenciesStatsKeys = Literal["latency_ms", "stddev_ms"]


class EventLog(TypedDict):
    """Event log fields."""

    type: Literal["request", "start", "stop", "background", "request_stream"]
    level: LogLevel
    date: AwareDatetime
    error_detail: NotRequired[JsonList]
    server_id: str
    server_version: str

    # "start" type
    server_start_time_ms: NotRequired[int]
    server_warnings: NotRequired[JsonList]
    register_usage_response: NotRequired[RegisterUsageResultTypeDef]
    region_latencies: NotRequired[
        dict[RegionName, dict[RegionLatenciesStatsKeys, float]]
    ]

    # "stop" type
    server_uptime_ms: NotRequired[int]

    # "request" + "request_stream" + "background" type
    execution_time_ms: NotRequired[int]

    # "background" type
    event: NotRequired[str]

    # "request" type
    client_ip: NotRequired[str]
    client_user_agent: NotRequired[str]
    method: NotRequired[Literal["DELETE", "GET", "OPTIONS", "PATCH", "POST", "PUT"]]
    path: NotRequired[str]
    id: NotRequired[str]
    status_code: NotRequired[int]

    model_id: NotRequired[str]
    model_regions: NotRequired[set[RegionName]]
    voice_id: NotRequired[str]  # TTS voice

    request_user_id: NotRequired[str]  # User ID passed from request
    request_org_id: NotRequired[str]  # Org ID passed from request

    request_params: NotRequired[
        JsonMappingOrList
    ]  # Request params (Body, form, query, ...)
    request_response: NotRequired[JsonMappingOrList]  # Request response


ParamsT = TypeVar("ParamsT", bound="BaseModel | dict[str, Any] | list[Any] | None")

#: Request ID (x-request-id header)
REQUEST_ID: ContextVar[str] = ContextVar("request_id")

# Request TZ aware datetime
REQUEST_TIME: ContextVar[AwareDatetime] = ContextVar("request_time")

#: Request log dict
REQUEST_LOG: ContextVar[EventLog] = ContextVar("request_log")

#: HTTP request object
REQUEST: ContextVar[Request] = ContextVar("request")

#: Paths to ignore in logging
LOGGING_PATHS_IGNORE: frozenset[str] = frozenset(
    {
        "/",
        "/docs",
        "/favicon.ico",
        "/health",
        "/openapi.json",
        "/redoc",
        "/.well-known/api-catalog",
        "/.well-known/mcp/server-card.json",
        "/robots.txt",
    }
)

#: Sorted log levels
_SORTED_LOG_LEVELS: tuple[LogLevel, ...] = ("info", "warning", "error", "critical")

#: stdapi.ai custom prefix for metadata and tags
_STDAPI_METADATA_PREFIX = "stdapi-ai."

#: Strips characters not allowed in stdapi metadata values.
_METADATA_VALUE_STRIP_RE = re_compile(r"[^a-zA-Z0-9\s:_@$#=/+,.\-]")


def _init_log_levels() -> set[LogLevel]:
    """Initializes a set of log levels based on the current application setting.

    This function generates a set of log levels, starting from the highest
    log level in the configuration and including all levels up to and
    including the configured log level. The log levels are considered in
    reverse order of severity.

    Returns:
        set[LogLevel]: A set containing log levels lower or equal to the
        configured log level.
    """
    levels: set[LogLevel] = set()
    for level in reversed(_SORTED_LOG_LEVELS):
        levels.add(level)
        if level == SETTINGS.log_level:
            break
    return levels


#: Log levels to publish
_PUBLISHED_LOG_LEVELS = _init_log_levels()
del _init_log_levels


def add_server_warning(start_event: EventLog, warning: JsonValue) -> None:
    """Append a warning to a "start" event log and raise its level to ``warning``.

    Args:
        start_event: Startup event log to update.
        warning: Warning message or structured detail to record.
    """
    start_event.setdefault("server_warnings", []).append(warning)
    start_event["level"] = "warning"


def write_log_event(log: EventLog) -> None:
    """Writes a log event to the standard output in JSON format.

    This function converts the given log event to a JSON representation, encodes it,
    and writes the resulting data to the standard output with a newline appended.

    Args:
        log: The log event to be written, represented as an `EventLog` object.
    """
    if log["level"] in _PUBLISHED_LOG_LEVELS:
        stdout_write(log)  # type: ignore[arg-type]


@contextmanager
def log_request_event(request: Request) -> Generator[EventLog]:
    """Log a request event with OpenTelemetry tracing.

    Reuses the parent request ID for internal MCP → API calls (identified by
    ``MCP_USER_AGENT`` + ``INTERNAL_REQUEST_ID_HEADER``) so both legs share
    the same correlation ID in structured output.  All ``ContextVar`` tokens
    are reset in the ``finally`` block so nested calls restore the parent
    context correctly on exit.

    Args:
        request: Incoming HTTP request.

    Yields:
        Mutable event log dict populated during the request lifetime.
    """
    request_id = (
        parent_id
        if (
            request.headers.get("user-agent") == server.MCP_USER_AGENT
            and (parent_id := request.headers.get(server.INTERNAL_REQUEST_ID_HEADER))
        )
        else webuuid()
    )
    request_id_token = REQUEST_ID.set(request_id)
    request_time_token = REQUEST_TIME.set(request_time := SETTINGS.now())
    request_log_token = REQUEST_LOG.set(
        log := EventLog(
            type="request",
            level="info",
            date=request_time,
            server_id=server.SERVER_NAME,
            server_version=SERVER_FULL_VERSION,
            id=request_id,
            method=request.method,  # type: ignore[typeddict-item]
            path=request.url.path,
        )
    )
    request_token = REQUEST.set(request)
    span_context = otel_manager.start_span(
        f"{request.method} {request.url.path}",
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
            "http.scheme": request.url.scheme,
            "http.host": request.url.hostname or "localhost",
            "http.target": request.url.path,
            "request.id": request_id,
            "server.id": server.SERVER_NAME,
        },
    )
    with suppress(KeyError):
        log["client_user_agent"] = request.headers["User-Agent"]
        if span_context:
            span_context.set_attribute("http.user_agent", request.headers["User-Agent"])
    if SETTINGS.log_client_ip and request.client:
        log["client_ip"] = request.client.host
        if span_context:
            span_context.set_attribute("client.address", request.client.host)
            if request.client.port:
                span_context.set_attribute("client.port", request.client.port)
    start = perf_counter_ns()

    try:
        with otel_manager.use_span(span_context):
            yield log
    except Exception as exc:
        log["level"] = "critical"
        log["status_code"] = 500
        log.setdefault("error_detail", []).append("\n".join(format_exception(exc)))
        if span_context:
            span_context.set_status(Status(StatusCode.ERROR, str(exc)))
            span_context.set_attribute("error", value=True)
            span_context.set_attribute("error.message", str(exc))
        raise
    finally:
        log["execution_time_ms"] = (perf_counter_ns() - start) // 1000000
        REQUEST_ID.reset(request_id_token)
        REQUEST.reset(request_token)
        REQUEST_LOG.reset(request_log_token)
        REQUEST_TIME.reset(request_time_token)
        if span_context:
            span_context.set_attribute("http.status_code", log.get("status_code", 200))
            span_context.set_attribute("duration_ms", log["execution_time_ms"])
            if log.get("status_code", 200) >= 400:
                span_context.set_status(Status(StatusCode.ERROR))
            span_context.end()
        write_log_event(log)


def log_request_params[ParamsT: "BaseModel | dict[str, Any] | list[Any] | None"](
    request: ParamsT, exclude: IncEx | None = None, user_id: str | None = None
) -> ParamsT:
    """Logs the request and response parameters if the respective setting is enabled.

    Args:
        request: The request data to be logged. Must be JSON serializable.
        exclude: An iterable of keys to exclude from the log.
        user_id: The user ID associated with the request, if available.

    Returns:
        Unmodified request.
    """
    if SETTINGS.log_request_params:
        _format_params(
            REQUEST_LOG.get(), "request_params", request, exclude, exclude_unset=True
        )
    if user_id:
        REQUEST_LOG.get()["request_user_id"] = user_id
    return request


def log_response_params[ParamsT: "BaseModel | dict[str, Any] | list[Any] | None"](
    response: ParamsT, exclude: IncEx | None = None
) -> ParamsT:
    """Logs the request and response parameters if the respective setting is enabled.

    Args:
        response: The response data to be logged. Must be JSON serializable.
        exclude: An iterable of keys to exclude from the log.

    Returns:
        Unmodified response.
    """
    if SETTINGS.log_request_params:
        _format_params(REQUEST_LOG.get(), "request_response", response, exclude)
    return response


def log_error_details(
    *error_detail: JsonValue, level: LogLevel | None = None, status: int | None = None
) -> None:
    """Logs error details into the current request context.

    Args:
        *error_detail: Variable length argument list of error details to be
            logged. Each item should be a JSON-compatible value.
        level: Optional. Logging level to specify the severity of the error.
        status: Optional. HTTP status code associated with the error.
    """
    level = level or (
        ("warning" if status < 500 else "error") if status else "critical"
    )
    log = REQUEST_LOG.get()
    log.setdefault("error_detail", []).extend(error_detail)
    if level and _SORTED_LOG_LEVELS.index(level) > _SORTED_LOG_LEVELS.index(
        log["level"]
    ):
        log["level"] = level


def _format_params(
    log: EventLog,
    key: Literal["request_params", "request_response"],
    value: BaseModel | dict[str, Any] | list[Any] | None,
    exclude: IncEx | None = None,
    *,
    exclude_unset: bool = False,
) -> None:
    """Formats and updates the log with the specified key and value.

    Args:
        log: The log object where the key-value pair should be updated.
        key: The key in the log to be updated.
        value: The value to be assigned to the specified key in the log.
        exclude: An iterable of keys to exclude from the log.
        exclude_unset: Exclude unset keys.
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(
            mode="json", exclude_none=True, exclude_unset=exclude_unset, exclude=exclude
        )
    elif exclude and isinstance(value, dict):
        value = value.copy()
        for name in exclude:
            value.pop(name, None)  # type:ignore[arg-type]
    if value:
        log[key] = value


@contextmanager
def log_background_event(event: str, request_id: str) -> Generator[EventLog]:
    """Context manager to log a background event.

    Args:
        event: Event type label.
        request_id: Unique identifier for the associated request.

    Yields:
        Mutable event log dict populated during execution.
    """
    span_context = otel_manager.start_span(
        "background",
        attributes={"request.id": request_id, "server.id": server.SERVER_NAME},
    )
    log = EventLog(
        type="background",
        level="info",
        date=SETTINGS.now(),
        server_id=server.SERVER_NAME,
        server_version=SERVER_FULL_VERSION,
        id=request_id,
        event=event,
    )
    start = perf_counter_ns()
    try:
        with otel_manager.use_span(span_context):
            yield log
    except Exception as exc:
        log["level"] = "critical"
        log.setdefault("error_detail", []).append("\n".join(format_exception(exc)))
        raise
    finally:
        log["execution_time_ms"] = (perf_counter_ns() - start) // 1000000
        write_log_event(log)


async def _rebuild_and_log_stream[T](
    first_chunk: T, stream: AsyncGenerator[T]
) -> AsyncGenerator[T]:
    """Rebuilds a given asynchronous generator stream while logging its execution details.

    This function processes an asynchronous generator stream by injecting logging and
    monitoring functionalities. It yields the first chunk immediately, starts a tracing span,
    and logs the performance metrics and errors encountered during the execution.
    The stream is closed automatically upon completion or in case of an exception.

    Args:
        first_chunk: Initial chunk to be yielded before consuming the stream.
        stream: An asynchronous generator representing the stream to process and log.

    Yields:
        Yields items from the input stream including the first_chunk.
    """
    try:
        request_id = REQUEST_ID.get()
        yield first_chunk

        span_context = otel_manager.start_span(
            "request_stream",
            attributes={"request.id": request_id, "server.id": server.SERVER_NAME},
        )
        log = EventLog(
            type="request_stream",
            level="info",
            date=SETTINGS.now(),
            server_id=server.SERVER_NAME,
            server_version=SERVER_FULL_VERSION,
            id=request_id,
        )
        start = perf_counter_ns()
        try:
            with otel_manager.use_span(span_context):
                async for chunk in stream:
                    yield chunk

        except ApiError, ClientError:
            raise
        except Exception as exc:
            log["level"] = "critical"
            log.setdefault("error_detail", []).append("\n".join(format_exception(exc)))
            raise
        finally:
            log["execution_time_ms"] = (perf_counter_ns() - start) // 1000000
            write_log_event(log)
    finally:
        await stream.aclose()


async def log_request_stream_event[T](stream: AsyncGenerator[T]) -> AsyncGenerator[T]:
    """Logs and processes events of a stream while preserving the original structure.

    This function takes the first yielded element of the stream, processes it
    by re-logging or modifying it as needed, and then combines it with the remaining original
    events of the input stream for consumption.

    Args:
        stream:
            An asynchronous generator stream producing events of type T.

    Yields:
        Items from the input asynchronous generator in their modified or original form.
    """
    return _rebuild_and_log_stream(await stream.__anext__(), stream)


async def log_request_sse_stream_event(
    stream: AsyncGenerator[ServerSentEvent],
) -> AsyncGenerator[ServerSentEvent]:
    """Log, monitor, and error-guard an SSE stream for use with ``EventSourceResponse``.

    Combines :func:`log_request_stream_event` and an SSE error boundary into a
    single step.  After the HTTP response headers are sent, any exception that
    escapes the underlying generator cannot be turned into an HTTP error response
    (Starlette raises ``RuntimeError: Caught handled exception, but response
    already started``).  This wrapper catches such exceptions, logs them via
    :func:`log_error_details`, and yields a terminal ``error`` SSE event
    formatted for the matched API provider so that ``EventSourceResponse`` can
    close the connection cleanly.

    Args:
        stream: Raw SSE async generator (e.g. from an adapter's ``format_stream``).

    Yields:
        Items from ``stream`` (after monitoring setup), followed by a provider-
        formatted ``error`` SSE event on failure.
    """
    try:
        async for chunk in _rebuild_and_log_stream(await stream.__anext__(), stream):
            yield chunk
    except ApiError as exc:
        status = exc.status
        log_error_details(exc.args[0], status=status)
        yield JSONServerSentEvent(
            data=format_http_error(
                REQUEST.get(),
                status,
                hide_security_details(status, exc.args[0]),
                exc.param,
                exc.code,
            )[0],
            event="error",
        )
    except ClientError as exc:
        error = exc.response["Error"]
        status = AWS_ERROR_MAP.get(error["Code"], (502, "server_error"))[0]
        log_error_details(error["Message"], status=status)
        yield JSONServerSentEvent(
            data=format_http_error(
                REQUEST.get(), status, hide_security_details(status, error["Message"])
            )[0],
            event="error",
        )
    except (HTTPClientError, BotocoreConnectionError) as exc:
        message = str(exc)
        status = AWS_ERROR_MAP.get(exc.__class__.__name__, (503, "server_error"))[0]
        log_error_details(message, status=status)
        yield JSONServerSentEvent(
            data=format_http_error(
                REQUEST.get(), status, hide_security_details(status, message)
            )[0],
            event="error",
        )
    except Exception as exc:  # noqa: BLE001
        log_error_details("\n".join(format_exception(exc)), level="critical")
        yield JSONServerSentEvent(
            data=format_http_error(REQUEST.get(), 500, "Internal Server Error")[0],
            event="error",
        )


def build_metadata(
    existing: Mapping[str, str] | None = None, *, apn: bool = False
) -> dict[str, str]:
    """Build request metadata with ``stdapi-ai.`` keys injected.

    Drops any key starting with ``stdapi-ai.`` from *existing* to prevent
    caller spoofing, then injects the current request context.

    Args:
        existing: Caller-supplied metadata from the request body, if any.
        apn: When ``True``, add ``aws-apn-id`` tag.
            Only set this when the result is used as resource tags, not as
            request-level metadata.

    Returns:
        Merged metadata dict with ``stdapi-ai.*`` keys always set.
    """
    metadata = {
        k: v
        for k, v in (existing or {}).items()
        if not k.startswith(_STDAPI_METADATA_PREFIX)
    }
    metadata["stdapi-ai.request_id"] = REQUEST_ID.get()
    metadata["stdapi-ai.server_id"] = server.SERVER_NAME
    if (user_id := REQUEST_LOG.get().get("request_user_id")) and (
        user_id := _METADATA_VALUE_STRIP_RE.sub("", user_id)[:256]
    ):
        metadata["stdapi-ai.user_id"] = user_id
    if apn:
        metadata["aws-apn-id"] = server.AWS_APN_ID
    return metadata
