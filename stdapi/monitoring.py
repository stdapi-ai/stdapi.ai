"""Monitoring."""

from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from time import perf_counter_ns
from traceback import format_exception
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, TypeVar

from fastapi import Request
from pydantic import AwareDatetime, BaseModel, JsonValue

from stdapi.config import SETTINGS, LogLevel
from stdapi.metering import SERVER_FULL_VERSION
from stdapi.server import SERVER_NAME
from stdapi.utils import stdout_write, webuuid

if TYPE_CHECKING:
    from pydantic.main import IncEx

    from stdapi.monitoring_otel import OpenTelemetryManager

if not SETTINGS.otel_enabled:
    from stdapi.monitoring_otel_base import (  # type: ignore[assignment]
        OpenTelemetryManager,
    )

else:
    from opentelemetry.trace import Status, StatusCode

    from stdapi.monitoring_otel import OpenTelemetryManager

otel_manager = OpenTelemetryManager()

T = TypeVar("T")


class EventLog(TypedDict):
    """Event log fields."""

    type: Literal["request", "start", "stop", "background", "request_stream"]
    level: LogLevel
    date: AwareDatetime
    error_detail: NotRequired[list[JsonValue]]
    server_id: str
    server_version: str

    # "start" type
    server_start_time_ms: NotRequired[int]
    server_warnings: NotRequired[list[JsonValue]]

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
    voice_id: NotRequired[str]  # TTS voice

    request_user_id: NotRequired[str]  # User ID passed from request
    request_org_id: NotRequired[str]  # Org ID passed from request

    request_params: NotRequired[
        dict[str, JsonValue] | list[JsonValue]
    ]  # Request params (Body, form, query, ...)
    request_response: NotRequired[
        dict[str, JsonValue] | list[JsonValue]
    ]  # Request response


ParamsT = TypeVar("ParamsT", bound="BaseModel | dict[str, Any] | list[Any] | None")

#: Request ID (x-request-id header)
REQUEST_ID: ContextVar[str] = ContextVar("request_id")

# Request TZ aware datetime
REQUEST_TIME: ContextVar[AwareDatetime] = ContextVar("request_time")

#: Request log dict
REQUEST_LOG: ContextVar[EventLog] = ContextVar("request_log")

#: Paths to ignore in logging
LOGGING_PATHS_IGNORE = {"/docs", "/favicon.ico", "/health", "/openapi.json", "/redoc"}

#: Sorted log levels
_SORTED_LOG_LEVELS: tuple[LogLevel, ...] = ("info", "warning", "error", "critical")


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
    """Context manager to log a request event with OpenTelemetry tracing.

    Args:
        request: A `Request` object representing the HTTP request.
    """
    request_id = webuuid()
    REQUEST_ID.set(request_id)
    request_time = SETTINGS.now()
    REQUEST_TIME.set(request_time)
    log = EventLog(
        type="request",
        level="info",
        date=request_time,
        server_id=SERVER_NAME,
        server_version=SERVER_FULL_VERSION,
        id=request_id,
        method=request.method,  # type: ignore[typeddict-item]
        path=request.url.path,
    )
    REQUEST_LOG.set(log)
    span_context = otel_manager.start_span(
        f"{request.method} {request.url.path}",
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
            "http.scheme": request.url.scheme,
            "http.host": request.url.hostname or "localhost",
            "http.target": request.url.path,
            "request.id": request_id,
            "server.id": SERVER_NAME,
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
        if span_context:
            span_context.set_attribute("http.status_code", log.get("status_code", 200))
            span_context.set_attribute("duration_ms", log["execution_time_ms"])
            if log.get("status_code", 200) >= 400:
                span_context.set_status(Status(StatusCode.ERROR))
            span_context.end()
        write_log_event(log)


def log_request_params[ParamsT: "BaseModel | dict[str, Any] | list[Any] | None"](
    request: ParamsT, exclude: "IncEx | None" = None
) -> ParamsT:
    """Logs the request and response parameters if the respective setting is enabled.

    Args:
        request: The request data to be logged. Must be JSON serializable.
        exclude: An iterable of keys to exclude from the log.

    Returns:
        Unmodified request.
    """
    if SETTINGS.log_request_params:
        log = REQUEST_LOG.get()
        _format_params(log, "request_params", request, exclude, exclude_unset=True)
    return request


def log_response_params[ParamsT: "BaseModel | dict[str, Any] | list[Any] | None"](
    response: ParamsT, exclude: "IncEx | None" = None
) -> ParamsT:
    """Logs the request and response parameters if the respective setting is enabled.

    Args:
        response: The response data to be logged. Must be JSON serializable.
        exclude: An iterable of keys to exclude from the log.

    Returns:
        Unmodified response.
    """
    if SETTINGS.log_request_params:
        log = REQUEST_LOG.get()
        _format_params(log, "request_response", response, exclude)
    return response


def log_error_details(*error_detail: JsonValue, level: LogLevel | None = None) -> None:
    """Logs error details into the current request context.

    Args:
        *error_detail: Variable length argument list of error details to be
            logged. Each item should be a JSON-compatible value.
        level: Optional. Logging level to specify the severity of the error.
    """
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
    exclude: "IncEx | None" = None,
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
        event: Event type.
        request_id: A string uniquely identifying the request.
    """
    span_context = otel_manager.start_span(
        "background", attributes={"request.id": request_id, "server.id": SERVER_NAME}
    )
    log = EventLog(
        type="background",
        level="info",
        date=SETTINGS.now(),
        server_id=SERVER_NAME,
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


async def _rebuild_and_log_stream(
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
            attributes={"request.id": request_id, "server.id": SERVER_NAME},
        )
        log = EventLog(
            type="request_stream",
            level="info",
            date=SETTINGS.now(),
            server_id=SERVER_NAME,
            server_version=SERVER_FULL_VERSION,
            id=request_id,
        )
        start = perf_counter_ns()
        try:
            with otel_manager.use_span(span_context):
                async for chunk in stream:
                    yield chunk

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
