"""Monitoring."""

from contextlib import contextmanager, suppress
from contextvars import ContextVar
from re import compile as re_compile
from time import perf_counter_ns
from traceback import format_exception
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, get_args

from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from fastapi import Request  # noqa: TC002
from pydantic import AwareDatetime, BaseModel, JsonValue
from sse_starlette import ServerSentEvent

from stdapi import server
from stdapi.api_errors import ApiError
from stdapi.api_providers import format_http_error
from stdapi.aws_bedrock import AWS_ERROR_MAP
from stdapi.config import SETTINGS, LogLevel
from stdapi.metering import SERVER_FULL_VERSION
from stdapi.usage import (
    IMAGE_SPEC,
    MODEL_STATE,
    OPERATION,
    USAGE,
    UsageLogEntry,
    compute_costs,
    emit_usage_metrics,
    format_cost,
    init_model_state,
    init_usage,
    total_costs_by_currency,
    usage_log_entries,
)
from stdapi.utils import hide_security_details, stdout_write, to_json_str, webuuid

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterable, Mapping
    from contextvars import Token

    from opentelemetry.trace.span import Span
    from pydantic.main import IncEx
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_meteringmarketplace.type_defs import (
        RegisterUsageResultTypeDef,
    )

    from stdapi.monitoring_otel import OpenTelemetryManager
    from stdapi.types import JsonList, JsonMappingOrList
    from stdapi.usage import ModelInvocationState, UsageKey, UsageRecord

if not SETTINGS.otel_enabled:
    from stdapi.monitoring_otel_base import (  # type: ignore[assignment]
        OpenTelemetryManager,
    )

else:
    from opentelemetry.trace import Status, StatusCode

    from stdapi.monitoring_otel import OpenTelemetryManager

otel_manager = OpenTelemetryManager()

#: Per-region latency stat keys reported in the "start" event's region_latencies.
RegionLatenciesStatsKeys = Literal["latency_ms", "stddev_ms"]


class AwsApiCallLog(TypedDict):
    """Correlation identifiers of one downstream AWS API call."""

    service: str  # AWS service (client pool key, e.g. "bedrock-runtime")
    operation: str  # AWS API operation name (e.g. "Converse")
    request_id: str  # AWS-side request ID (ResponseMetadata.RequestId)
    error: NotRequired[str]  # AWS error code, failed calls only


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

    # Edge correlation headers from the incoming request, when present
    amzn_trace_id: NotRequired[str]  # X-Amzn-Trace-Id (ALB / X-Ray)
    apigw_request_id: NotRequired[str]  # x-amz-apigw-id (API Gateway)
    cloudfront_request_id: NotRequired[str]  # X-Amz-Cf-Id (CloudFront)

    # AWS-side request IDs of downstream AWS API calls (capped, oldest dropped)
    aws_requests: NotRequired[list[AwsApiCallLog]]

    request_params: NotRequired[
        JsonMappingOrList
    ]  # Request params (Body, form, query, ...)
    request_response: NotRequired[JsonMappingOrList]  # Request response

    # Usage metrics (real AWS-billed usage, per service+model+operation)
    usage: NotRequired[list[UsageLogEntry]]

    # Request-level cost total per currency, as exact plain-decimal text
    cost: NotRequired[dict[str, str]]


#: Request ID (x-request-id header)
REQUEST_ID: ContextVar[str] = ContextVar("request_id")

#: Request-scoped, timezone-aware start timestamp.
REQUEST_TIME: ContextVar[AwareDatetime] = ContextVar("request_time")

#: Request log dict
REQUEST_LOG: ContextVar[EventLog] = ContextVar("request_log")

#: HTTP request object
REQUEST: ContextVar[Request] = ContextVar("request")

#: Per-request accumulator of downstream AWS API call correlation entries
_AWS_API_CALLS: ContextVar[list[AwsApiCallLog]] = ContextVar("aws_api_calls")

#: Maximum AWS API call entries kept per log event (oldest dropped first)
_AWS_REQUESTS_MAX = 50

#: Maximum length kept from an edge correlation header value
_EDGE_HEADER_MAX_LENGTH = 256

#: Strips non-printable-ASCII characters from edge correlation header values.
_EDGE_HEADER_STRIP_RE = re_compile(r"[^\x20-\x7e]")

#: Paths to ignore in logging
LOGGING_PATHS_IGNORE: frozenset[str] = frozenset(
    {
        "/",
        "/docs",
        "/favicon.ico",
        "/health",
        "/openapi.json",
        "/ping",
        "/redoc",
        "/.well-known/api-catalog",
        "/.well-known/mcp/server-card.json",
        "/robots.txt",
    }
)

#: Log levels, least to most severe -- derived from LogLevel so the two can't drift apart.
_SORTED_LOG_LEVELS: tuple[LogLevel, ...] = get_args(LogLevel)

#: stdapi.ai custom prefix for metadata and tags
_STDAPI_METADATA_PREFIX = "stdapi-ai."

#: Strips characters not allowed in stdapi metadata values.
_METADATA_VALUE_STRIP_RE = re_compile(r"[^a-zA-Z0-9\s:_@$#=/+,.\-]")


def _published_log_levels(level: LogLevel | Literal["disabled"]) -> set[LogLevel]:
    """Return the log levels to publish to stdout for the configured level.

    Args:
        level: The minimum severity to publish, or "disabled" for none.

    Returns:
        The levels at or above *level*'s severity.
    """
    if level == "disabled":
        return set()
    return set(_SORTED_LOG_LEVELS[_SORTED_LOG_LEVELS.index(level) :])


#: Levels at or above SETTINGS.log_level's severity -- these get published to stdout.
_PUBLISHED_LOG_LEVELS = _published_log_levels(SETTINGS.log_level)


def _add_warnings(
    log: EventLog, warnings: Iterable[JsonValue], level: LogLevel = "warning"
) -> None:
    """Append *warnings* to *log*'s ``error_detail`` and raise its level to *level*.

    Args:
        log: The event log to update in place.
        warnings: Messages or structured details to append.
        level: Severity to raise the log's level to (never lowered).
    """
    warnings = list(warnings)
    if not warnings:
        return
    log.setdefault("error_detail", []).extend(warnings)
    if _SORTED_LOG_LEVELS.index(level) > _SORTED_LOG_LEVELS.index(log["level"]):
        log["level"] = level


def _finalize_usage(log: EventLog) -> None:
    """Compute costs, attach usage entries/totals to *log*, emit metrics, then drain.

    Multi-currency cost warnings are folded into *log*'s ``error_detail``/
    ``level`` instead of being logged separately.

    Args:
        log: The event log to populate in place.
    """
    try:
        if SETTINGS.cost_tracking:
            _add_warnings(log, compute_costs())
        if entries := usage_log_entries():
            log["usage"] = entries
            if SETTINGS.cost_tracking and (
                cost_by_currency := total_costs_by_currency(entries)
            ):
                log["cost"] = {c: format_cost(s) for c, s in cost_by_currency.items()}
        emit_usage_metrics()
    finally:
        # Drain synchronously (atomic on the event loop), even on failure so
        # a later stream finalize can't re-log the same records: tasks
        # spawned before a stream's log scope keep recording into this same
        # dict, so the stream finalize picks up exactly the later records.
        if (records := USAGE.get(None)) is not None:
            records.clear()


def _finalize_usage_safely(log: EventLog) -> None:
    """Run :func:`_finalize_usage`, folding any failure into *log* instead of raising.

    Args:
        log: The event log to finalize and update in place.
    """
    try:
        _finalize_usage(log)
    except Exception as exc:  # noqa: BLE001
        _add_warnings(log, ["\n".join(format_exception(exc))], level="error")


def record_aws_api_call(
    service: str, operation: str, request_id: str, error_code: str | None = None
) -> None:
    """Record a downstream AWS API call's request ID into the current request scope.

    No-op outside a request scope. The accumulator is capped at
    ``_AWS_REQUESTS_MAX`` entries, dropping the oldest first.

    Args:
        service: AWS service name (e.g. ``bedrock-runtime``).
        operation: AWS API operation name (e.g. ``Converse``).
        request_id: AWS-side request ID of the call.
        error_code: AWS error code when the call failed.
    """
    if (calls := _AWS_API_CALLS.get(None)) is None:
        return
    if len(calls) >= _AWS_REQUESTS_MAX:
        del calls[0]
    entry = AwsApiCallLog(service=service, operation=operation, request_id=request_id)
    if error_code:
        entry["error"] = error_code
    calls.append(entry)


def _attach_aws_api_calls(log: EventLog) -> None:
    """Move accumulated AWS API call entries onto *log* and clear the accumulator.

    Like the ``USAGE`` drain, the accumulator is shared with tasks spawned in
    the request scope: whichever event finalizes next picks up the entries
    recorded since the previous drain.

    Args:
        log: The event log to populate in place.
    """
    if calls := _AWS_API_CALLS.get(None):
        log["aws_requests"] = calls.copy()
        calls.clear()


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


def _reset_request_context(
    request_id_token: Token[str],
    request_token: Token[Request],
    operation_token: Token[str],
    request_log_token: Token[EventLog],
    request_time_token: Token[AwareDatetime],
    usage_token: Token[dict[UsageKey, UsageRecord]],
    model_state_token: Token[dict[str, ModelInvocationState]],
    aws_api_calls_token: Token[list[AwsApiCallLog]],
) -> None:
    """Reset every per-request ContextVar.

    Args:
        request_id_token: Token restoring ``REQUEST_ID``.
        request_token: Token restoring ``REQUEST``.
        operation_token: Token restoring ``OPERATION``.
        request_log_token: Token restoring ``REQUEST_LOG``.
        request_time_token: Token restoring ``REQUEST_TIME``.
        usage_token: Token restoring ``USAGE``.
        model_state_token: Token restoring ``MODEL_STATE``.
        aws_api_calls_token: Token restoring ``_AWS_API_CALLS``.
    """
    REQUEST_ID.reset(request_id_token)
    REQUEST.reset(request_token)
    OPERATION.reset(operation_token)
    REQUEST_LOG.reset(request_log_token)
    REQUEST_TIME.reset(request_time_token)
    USAGE.reset(usage_token)
    MODEL_STATE.reset(model_state_token)
    _AWS_API_CALLS.reset(aws_api_calls_token)
    # Token-less: set mid-request by image jobs; cleared defensively so a
    # failed invoke can't leak a stale spec into a later request context.
    IMAGE_SPEC.set("")


def _edge_header_value(request: Request, header: str) -> str:
    """Return a sanitized edge correlation header value.

    The value is client-supplied input: non-printable-ASCII characters are
    stripped and the result is truncated before it reaches the log.

    Args:
        request: Incoming HTTP request.
        header: Header name to read.

    Returns:
        The sanitized value, or "" when the header is absent or empty.
    """
    if value := request.headers.get(header, ""):
        value = _EDGE_HEADER_STRIP_RE.sub("", value)[:_EDGE_HEADER_MAX_LENGTH]
    return value


def _record_edge_headers(
    log: EventLog, request: Request, span_context: Span | None
) -> None:
    """Record edge correlation headers from *request*, when present.

    Args:
        log: The event log to populate in place.
        request: Incoming HTTP request.
        span_context: Active span to annotate, if any.
    """
    if trace_id := _edge_header_value(request, "x-amzn-trace-id"):
        log["amzn_trace_id"] = trace_id
        if span_context:
            span_context.set_attribute("aws.amzn_trace_id", trace_id)
    if apigw_id := _edge_header_value(request, "x-amz-apigw-id"):
        log["apigw_request_id"] = apigw_id
        if span_context:
            span_context.set_attribute("aws.apigw_request_id", apigw_id)
    if cf_id := _edge_header_value(request, "x-amz-cf-id"):
        log["cloudfront_request_id"] = cf_id
        if span_context:
            span_context.set_attribute("aws.cloudfront_request_id", cf_id)


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
    usage_token = init_usage()
    model_state_token = init_model_state()
    aws_api_calls_token = _AWS_API_CALLS.set([])
    operation_token = OPERATION.set(request.url.path)
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
    _record_edge_headers(log, request, span_context)
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
        _finalize_usage_safely(log)
        _attach_aws_api_calls(log)
        _reset_request_context(
            request_id_token,
            request_token,
            operation_token,
            request_log_token,
            request_time_token,
            usage_token,
            model_state_token,
            aws_api_calls_token,
        )
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


def _error_level(level: LogLevel | None, status: int | None) -> LogLevel:
    """Resolve an error's severity: an explicit *level* wins over *status*.

    Args:
        level: Explicit level override, if any.
        status: HTTP status code associated with the error, if any.

    Returns:
        *level* if given, else "warning" for status < 500, "error" for
        status >= 500, or "critical" when neither is given.
    """
    return level or (("warning" if status < 500 else "error") if status else "critical")


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
    _add_warnings(REQUEST_LOG.get(), error_detail, level=_error_level(level, status))


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
        _attach_aws_api_calls(log)
        write_log_event(log)


class SseHandledStreamError(Exception):
    """Mid-stream error already reported to the client as spec SSE events.

    Raised by SSE adapters (e.g. the Responses ``format_stream``) after they
    emitted their own protocol-compliant error events.
    :func:`log_request_sse_stream_event` records it in the request log but
    does not emit the legacy REST-envelope ``error`` event.
    """

    def __init__(
        self, message: str, *, status: int | None = None, level: LogLevel | None = None
    ) -> None:
        """Initialize the marker exception.

        Args:
            message: Error detail to record in the request log.
            status: HTTP status code associated with the error, if any.
            level: Log level override; defaults to "warning" for 4xx statuses
                and "error" otherwise (never "critical": already handled).
        """
        super().__init__(message)
        self.status = status
        self.level: LogLevel = level or (
            "warning" if status and status < 500 else "error"
        )


def _stream_exception_detail(
    exc: ApiError | ClientError | SseHandledStreamError,
) -> tuple[JsonValue, LogLevel]:
    """Extract the log message and severity for a mid-stream error.

    Args:
        exc: The mid-stream exception caught by :func:`_rebuild_and_log_stream`.

    Returns:
        A ``(message, level)`` pair, using the same level logic as
        :func:`log_error_details`/:class:`SseHandledStreamError`.
    """
    if isinstance(exc, SseHandledStreamError):
        return exc.args[0], exc.level
    if isinstance(exc, ApiError):
        return exc.args[0], _error_level(None, exc.status)
    error = exc.response["Error"]
    status = AWS_ERROR_MAP.get(error["Code"], (502, "server_error"))[0]
    return error["Message"], _error_level(None, status)


async def _rebuild_and_log_stream[T](
    first_chunk: T, stream: AsyncGenerator[T]
) -> AsyncGenerator[T]:
    """Log a "request_stream" event for streamed response portions.

    Yields ``first_chunk`` immediately (before any logging setup runs),
    then logs the remaining stream's performance and usage as its own
    separate log entry.  The ``USAGE`` accumulator is shared with the
    request scope: the request finalize drains what it logged, and tasks
    spawned before this point (which captured the request context) keep
    recording into the same dict, so their usage lands here instead of
    being lost.

    Args:
        first_chunk: Initial chunk to be yielded before consuming the stream.
        stream: An asynchronous generator representing the stream to process and log.

    Yields:
        Every item from the input stream, including ``first_chunk``.
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

        except (ApiError, ClientError, SseHandledStreamError) as exc:
            message, level = _stream_exception_detail(exc)
            _add_warnings(log, [message], level=level)
            raise
        except Exception as exc:
            log["level"] = "critical"
            log.setdefault("error_detail", []).append("\n".join(format_exception(exc)))
            raise
        finally:
            log["execution_time_ms"] = (perf_counter_ns() - start) // 1000000
            _finalize_usage_safely(log)
            _attach_aws_api_calls(log)
            write_log_event(log)
    finally:
        await stream.aclose()


async def log_request_stream_event[T](stream: AsyncGenerator[T]) -> AsyncGenerator[T]:
    """Wrap a plain async-generator stream with a "request_stream" log entry.

    Consumes the stream's first item up front and delegates the rest to
    :func:`_rebuild_and_log_stream`.

    Args:
        stream: An asynchronous generator stream producing events of type T.

    Yields:
        Every item from *stream*, unmodified.
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
    except SseHandledStreamError as exc:
        # The adapter already emitted spec-compliant error events; log only.
        log_error_details(exc.args[0], status=exc.status, level=exc.level)
    except ApiError as exc:
        status = exc.status
        log_error_details(exc.args[0], status=status)
        yield ServerSentEvent(
            data=to_json_str(
                format_http_error(
                    REQUEST.get(),
                    status,
                    hide_security_details(status, exc.args[0]),
                    exc.param,
                    exc.code,
                )[0]
            ),
            event="error",
        )
    except ClientError as exc:
        error = exc.response["Error"]
        status = AWS_ERROR_MAP.get(error["Code"], (502, "server_error"))[0]
        log_error_details(error["Message"], status=status)
        message = (
            "The request could not be completed. Retry the request."
            if status >= 500
            else hide_security_details(status, error["Message"])
        )
        yield ServerSentEvent(
            data=to_json_str(format_http_error(REQUEST.get(), status, message)[0]),
            event="error",
        )
    except (HTTPClientError, BotocoreConnectionError) as exc:
        message = str(exc)
        status = AWS_ERROR_MAP.get(exc.__class__.__name__, (503, "server_error"))[0]
        log_error_details(message, status=status)
        yield ServerSentEvent(
            data=to_json_str(
                format_http_error(
                    REQUEST.get(),
                    status,
                    "The service is temporarily unavailable. Retry the request.",
                )[0]
            ),
            event="error",
        )
    except Exception as exc:  # noqa: BLE001
        log_error_details("\n".join(format_exception(exc)), level="critical")
        yield ServerSentEvent(
            data=to_json_str(
                format_http_error(REQUEST.get(), 500, "Internal Server Error")[0]
            ),
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
