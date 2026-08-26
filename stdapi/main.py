"""FastAPI application main module for AWS-based OpenAI compatible API.

This module sets up the FastAPI application with middleware, exception handlers,
and AWS service integrations for providing OpenAI-compatible endpoints.
"""

from asyncio import gather
from contextlib import asynccontextmanager
from re import compile as compile_regex
from time import time_ns
from traceback import format_exception
from typing import TYPE_CHECKING, Final

from botocore.exceptions import BotoCoreError, ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.routing import iter_route_contexts
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException
from starlette.routing import Match

from stdapi import server
from stdapi.api_errors import ApiError, denied_feature_unavailable
from stdapi.api_providers import (
    format_http_error,
    get_request_id_header,
    set_log_fields,
    set_response_headers,
)
from stdapi.auth import initialize_authentication
from stdapi.aws import (
    AWSConnectionManager,
    initialize_aws_account_info,
    raise_first_exception,
    service_regions,
    verify_user_role_access,
)
from stdapi.aws_bedrock import (
    AWS_ERROR_MAP,
    set_guardrail_configuration,
    set_performance_configuration,
)
from stdapi.aws_bedrock_mantle import set_mantle_project
from stdapi.aws_bidi import drain_stream_closes, initialize_bidi_clients
from stdapi.aws_dynamodb import table_client_specs, verify_table
from stdapi.cleanup import (
    CLEANUPS,
    drain_cleanups,
    run_cleanups_detached,
    run_scheduled_cleanups,
)
from stdapi.config import SETTINGS
from stdapi.exceptions import ServerError
from stdapi.input_file import reset_current_input_files
from stdapi.metering import EDITION_TITLE, LICENCE_INFO, SERVER_FULL_VERSION, register
from stdapi.models import (
    drain_model_refresh,
    initialize_bedrock_models,
    update_unified_models_collections,
)
from stdapi.models.audio.amazon_polly import initialize_polly_models
from stdapi.models.audio.amazon_transcribe import (
    initialize_transcribe_models,
    transcribe_job_candidates,
)
from stdapi.models.moderation import initialize_moderation_models
from stdapi.monitoring import (
    LOGGING_PATHS_IGNORE,
    EventLog,
    add_server_warning,
    log_error_details,
    log_request_event,
    otel_manager,
    write_log_event,
)
from stdapi.pricing import (
    pricing_endpoint_region,
    start_price_catalog,
    stop_price_catalog,
)
from stdapi.realtime import (
    close_realtime_sessions,
    drain_session_stops,
    open_realtime_sessions,
)
from stdapi.region_routing import measure_region_latencies, quota_retry_after
from stdapi.routes import discover_routers
from stdapi.routes.core_root import WWW_AUTHENTICATE_CHALLENGE
from stdapi.server import SERVER_VERSION
from stdapi.tenant_keys import (
    close_tenant_key_reconciliation,
    initialize_tenant_keys,
    open_tenant_key_reconciliation,
    tenant_key_client_specs,
)
from stdapi.utils import JSONResponse, hide_security_details
from stdapi.vector_stores.engine import drain_indexing
from stdapi.vector_stores.jobs import (
    close_job_consumer,
    drain_indexing_jobs,
    initialize_job_queue,
    open_job_consumer,
    queue_region,
)
from stdapi.vector_stores.knowledge_base import verify_knowledge_bases

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
    from typing import Any

    from types_aiobotocore_bedrock.literals import RegionName

#: Detached background registries drained at shutdown, in the order awaited.
_DRAINED_REGISTRIES: Final = (
    "cleanups",
    "stream_closes",
    "realtime_readers",
    "file_indexing",
    "indexing_jobs",
    "model_refresh",
)


async def drain_background_tasks() -> dict[str, int]:
    """Wait for the background work requests left running, and report the losses.

    Best effort, never a durability guarantee: the platform sends ``SIGKILL`` a
    fixed delay after ``SIGTERM`` (30 seconds on Amazon ECS by default) and a
    deployment may not use the published module at all, so a hard kill is a
    normal event. Nothing whose correctness depends on finishing may rely on
    this; it only converts the common case from a silent loss into a completed
    one, and the rest into a counted one.

    Every registry drains concurrently, so ``shutdown_drain_timeout`` is one
    wall-clock deadline rather than a budget each of them spends in turn.

    Returns:
        Number of tasks abandoned per registry, empty when everything finished.
    """
    timeout = SETTINGS.shutdown_drain_timeout
    return {
        name: abandoned
        for name, abandoned in zip(
            _DRAINED_REGISTRIES,
            await gather(
                drain_cleanups(timeout),
                drain_stream_closes(timeout),
                drain_session_stops(timeout),
                drain_indexing(timeout),
                drain_indexing_jobs(timeout),
                drain_model_refresh(timeout),
            ),
            strict=True,
        )
        if abandoned
    }


def write_stop_event(start: int, abandoned: dict[str, int]) -> None:
    """Report the process stopping, and whatever background work it dropped.

    Args:
        start: Startup timestamp, in nanoseconds.
        abandoned: Tasks cancelled unfinished at the drain deadline, per registry.
    """
    stop_event = EventLog(
        type="stop",
        # Raised so work lost to the deadline reaches an operator at a level
        # they watch, with the count that was dropped.
        level="warning" if abandoned else "info",
        date=SETTINGS.now(),
        server_id=server.SERVER_NAME,
        server_version=SERVER_FULL_VERSION,
        server_uptime_ms=(time_ns() - start) // 1000000,
    )
    if abandoned:
        stop_event["abandoned_background_tasks"] = abandoned
    write_log_event(stop_event)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:  # noqa: PLR0915 - one linear start/stop sequence, a statement per service
    """Manage FastAPI application lifespan: start AWS connections and initialize services.

    Args:
        _: The FastAPI application instance (unused).

    Yields:
        None once startup is complete; shutdown runs after the yield.
    """
    start = time_ns()
    abandoned: dict[str, int] = {}
    # Fall back to the configured/default region so a bucket-less deployment
    # still warms one Transcribe client.
    transcribe_regions: list[RegionName | None] = [
        region for region, _ in transcribe_job_candidates()
    ] or [SETTINGS.aws_transcribe_region]
    try:
        async with AWSConnectionManager(
            *(
                *(
                    ("polly", region)
                    for region in service_regions(SETTINGS.aws_polly_region)
                ),
                *(
                    ("comprehend", region)
                    for region in service_regions(SETTINGS.aws_comprehend_region)
                ),
                *(("bedrock", region) for region in SETTINGS.aws_bedrock_regions),
                *(
                    ("bedrock-runtime", region)
                    for region in SETTINGS.aws_bedrock_regions
                ),
                *(
                    ("bedrock-agent-runtime", region)
                    for region in SETTINGS.aws_bedrock_regions
                ),
                # Only warmed when prompt ARNs are allowed or a knowledge base is
                # served: the Prompt Management lookup and the knowledge base
                # documents are its only consumers.
                *(
                    (
                        ("bedrock-agent", region)
                        for region in SETTINGS.aws_bedrock_regions
                    )
                    if SETTINGS.aws_bedrock_allow_prompt_arn
                    or SETTINGS.aws_bedrock_knowledge_base_ids
                    else ()
                ),
                *(("transcribe", region) for region in transcribe_regions),
                *(
                    ("translate", region)
                    for region in service_regions(SETTINGS.aws_translate_region)
                ),
                # Only warmed when cost tracking is enabled: its sole consumer,
                # the price catalog loader, otherwise no-ops. A None region
                # (GovCloud, which has no Price List endpoint) warms the default
                # client, which the never-loading catalog then never uses.
                *(
                    (("pricing", pricing_endpoint_region()),)
                    if SETTINGS.cost_tracking
                    else ()
                ),
                *(("s3", region) for region in transcribe_regions),
                ("s3", SETTINGS.aws_bedrock_regions[0]),
                ("s3.accelerate", SETTINGS.aws_bedrock_regions[0]),
                *(("s3", region) for region in SETTINGS.aws_s3_regional_buckets),
                # Externally owned read-only buckets live in their own regions,
                # which no other setting warms.
                *(
                    ("s3", region)
                    for region in SETTINGS.aws_s3_accepted_buckets.values()
                ),
                # A vector bucket's indexes are regional: one client, no failover.
                *(
                    (("s3vectors", SETTINGS.aws_s3_vectors_region),)
                    if SETTINGS.aws_s3_vectors_bucket
                    else ()
                ),
                # The indexing queue lives in the one region its URL names.
                *((("sqs", region),) if (region := queue_region()) else ()),
                # Nothing at all until a table is configured: the features
                # sharing it are opt-in, and none is enabled by default.
                *table_client_specs(),
                # Delivers minted tenant keys; nothing until the feature is on.
                *tenant_key_client_specs(),
            )
        ):
            # Not botocore clients, but they target the endpoints just resolved above.
            initialize_bidi_clients()
            span_context = otel_manager.start_span(
                "Application start", attributes={"server.id": server.SERVER_NAME}
            )
            try:
                with otel_manager.use_span(span_context):
                    # Latency probes run alone so no concurrent AWS traffic
                    # (catalog load included) can skew their measurements.
                    region_latencies = await measure_region_latencies()
                    start_price_catalog()
                    start_event = EventLog(
                        type="start",
                        level="info",
                        date=SETTINGS.now(),
                        server_id=server.SERVER_NAME,
                        server_version=SERVER_FULL_VERSION,
                    )
                    # Runs alone: the ECS metadata endpoint answers on the task
                    # ENI and times out when the startup fan-out below saturates
                    # the task CPU.
                    if account_warning := await initialize_aws_account_info():
                        add_server_warning(start_event, account_warning)
                    raise_first_exception(
                        await gather(
                            initialize_authentication(start_event),
                            initialize_bedrock_models(start_event),
                            initialize_polly_models(start_event),
                            initialize_transcribe_models(),
                            initialize_moderation_models(),
                            verify_user_role_access(start_event),
                            verify_knowledge_bases(start_event),
                            verify_table(start_event),
                            initialize_tenant_keys(start_event),
                            initialize_job_queue(start_event),
                            register(start_event),
                            return_exceptions=True,
                        )
                    )
                    update_unified_models_collections()
                if region_latencies:
                    start_event["region_latencies"] = region_latencies
                if not SETTINGS.aws_s3_bucket:
                    add_server_warning(
                        start_event,
                        "S3 bucket not configured ('aws_s3_bucket' not set): "
                        "some features are disabled",
                    )
                elif not SETTINGS.aws_bedrock_batch_role_arn:
                    add_server_warning(
                        start_event,
                        "Batch service role not configured "
                        "('aws_bedrock_batch_role_arn' not set): "
                        "the Batch APIs are disabled",
                    )
                if deprecated := SETTINGS.deprecated():
                    add_server_warning(
                        start_event,
                        f"Ignored deprecated settings: {', '.join(deprecated)}",
                    )
                start_event["server_start_time_ms"] = (time_ns() - start) // 1000000
                write_log_event(start_event)
                open_realtime_sessions()
                open_job_consumer()
                open_tenant_key_reconciliation()
                try:
                    yield
                finally:
                    # No graceful drain: a signal kills sockets without a close frame.
                    close_realtime_sessions()
                    # Stop taking new jobs before draining what is running:
                    # anything not finished here is redelivered elsewhere.
                    close_job_consumer()
                    await close_tenant_key_reconciliation()
                    # Drained here, and nowhere else: asking the sessions to
                    # close is what enqueues their teardown, while the AWS
                    # clients and the price catalog that the pending work still
                    # calls are only torn down below. Telemetry is flushed after
                    # this, so the drain's own report is exported rather than
                    # lost.
                    abandoned = await drain_background_tasks()
            finally:
                await stop_price_catalog()
    except (BotoCoreError, ClientError, ServerError) as exception:
        write_log_event(
            EventLog(
                type="start",
                level="error",
                date=SETTINGS.now(),
                server_id=server.SERVER_NAME,
                server_version=SERVER_FULL_VERSION,
                error_detail=[
                    f"{type(exception).__name__}: {exception}",
                    *getattr(exception, "__notes__", []),
                ],
            )
        )
        raise
    except Exception as exception:
        write_log_event(
            EventLog(
                type="start",
                level="critical",
                date=SETTINGS.now(),
                server_id=server.SERVER_NAME,
                server_version=SERVER_FULL_VERSION,
                error_detail=["\n".join(format_exception(exception))],
            )
        )
        raise
    finally:
        write_stop_event(start, abandoned)
        otel_manager.flush()


app = FastAPI(
    title=EDITION_TITLE,
    description="AWS standardized AI API",
    version=SERVER_VERSION,
    lifespan=lifespan,
    contact={"name": "stdapi.ai", "url": "https://stdapi.ai"},
    license_info=LICENCE_INFO,
    # The built-in pages load their icon from fastapi.tiangolo.com; the routes in
    # stdapi/routes/core_docs.py serve the same pages from the gateway alone.
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json" if SETTINGS.enable_openapi_json else None,
    # pydantic_core-rendered responses on every route (stdlib-identical wire format).
    default_response_class=JSONResponse,
)
otel_manager.instrument(app)
discover_routers(app)

if SETTINGS.enable_mcp_streamable_http or SETTINGS.enable_mcp_sse:
    from stdapi.mcp import mount_mcp

    mount_mcp(app)

if SETTINGS.enable_gzip:
    from fastapi.middleware.gzip import GZipMiddleware

    app.add_middleware(GZipMiddleware, minimum_size=1024)

if SETTINGS.trusted_hosts:
    from fastapi.middleware.trustedhost import TrustedHostMiddleware

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=SETTINGS.trusted_hosts)

if SETTINGS.cors_allow_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=SETTINGS.cors_allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        # Not CORS-safelisted: a browser client sees the challenge only if exposed.
        expose_headers=["www-authenticate"],
    )


def set_retry_after_header(request: Request, response: Response) -> None:
    """Attach ``retry-after`` to a rate-limited response when a delay is known.

    Advertises the region router's own quota backoff so client SDKs wait the
    server-driven delay instead of a blind exponential backoff; omitted when no
    region was put on a quota backoff while serving the request.

    Args:
        request: Incoming HTTP request.
        response: Outgoing response object.
    """
    if response.status_code == 429 and (seconds := quota_retry_after(request)):
        response.headers["retry-after"] = str(seconds)


def set_www_authenticate_header(response: Response) -> None:
    """Attach the ``www-authenticate`` challenge to an unauthenticated response.

    A 401 without a challenge leaves a client no way to learn which credential
    the API expects, which is what an agent needs to authenticate on its own.
    The challenge is the same on every 401, so it never reveals whether a
    credential was missing or merely wrong.

    Args:
        response: Outgoing response object.
    """
    if response.status_code == 401:
        response.headers["www-authenticate"] = WWW_AUTHENTICATE_CHALLENGE


@app.middleware("http")
async def _middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Main middleware to customize responses.

    Args:
        request: Incoming HTTP request.
        call_next: ASGI handler to execute the next middleware/app.

    Returns:
        Response.
    """
    if request.url.path in LOGGING_PATHS_IGNORE:
        response = await call_next(request)
    else:
        CLEANUPS.set([])
        reset_current_input_files()
        with log_request_event(request) as log:
            try:
                try:
                    set_guardrail_configuration(request.headers)
                    set_performance_configuration(request.headers)
                    set_mantle_project(request.headers)
                    response = await call_next(request)
                except ApiError as exc:
                    response = await _handle_request_setup_error(request, exc)
                except RuntimeError as exc:
                    if await request.is_disconnected():
                        log["status_code"] = 499
                        log_error_details(f"Client disconnected: {exc}", status=499)
                        run_cleanups_detached(log["id"])
                        return Response(status_code=499)
                    raise
            except BaseException:
                # No response will be sent (unhandled error or cancellation),
                # so the post-response background drain never runs.
                run_cleanups_detached(log["id"])
                raise
            log["status_code"] = response.status_code
            response.headers[get_request_id_header(request)] = log["id"]
            set_log_fields(request, log)
        set_response_headers(request, response, log["execution_time_ms"])
        set_retry_after_header(request, response)
        set_www_authenticate_header(response)
        # Attached unconditionally: a streamed body has produced nothing yet,
        # so what it defers is scheduled long after this point. The drain runs
        # once the body is complete, and is a no-op when nothing was scheduled.
        response.background = BackgroundTask(run_scheduled_cleanups, log["id"])
    response.headers["server"] = "stdapi.ai"
    return response


# Added last so it wraps `_middleware`, which reads the client on entry.
if SETTINGS.enable_proxy_headers:
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    app.add_middleware(
        ProxyHeadersMiddleware, trusted_hosts=SETTINGS.proxy_trusted_hosts
    )


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert HTTPException to the correct API error envelope.

    Registered on the base ``starlette.exceptions.HTTPException`` so it also
    catches the router's no-route/method-not-allowed errors, which raise that
    base class directly instead of FastAPI's subclass.

    Args:
        request: The current request.
        exc: The HTTPException raised by the router, a route, or a dependency.

    Returns:
        JSONResponse formatted in the appropriate error schema.
    """
    status_code = exc.status_code
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    log_error_details(message, status=status_code)
    return JSONResponse(
        *format_http_error(
            request, status_code, hide_security_details(status_code, message)
        )
    )


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    """Convert ApiError to the correct API error envelope.

    Args:
        request: The current request.
        exc: The ApiError raised by low-level code.

    Returns:
        JSONResponse formatted in the appropriate error schema.
    """
    status_code = exc.status
    log_error_details(exc.args[0], status=status_code)
    return JSONResponse(
        *format_http_error(
            request,
            status_code,
            hide_security_details(status_code, exc.args[0], disclosed=exc.disclosed),
            exc.param,
            exc.code,
        )
    )


async def _handle_request_setup_error(request: Request, exc: ApiError) -> JSONResponse:
    """Format an error raised before routing with the target route's envelope.

    The middleware runs above the ``ExceptionMiddleware`` holding the registered
    handlers, so an ``ApiError`` raised while preparing the request context (the
    per-request header configuration) reaches ``ServerErrorMiddleware`` as a bare
    500 instead. Nothing has matched a route yet either, so the route is resolved
    here to pick the envelope and headers of the API the client called.

    Args:
        request: The current request.
        exc: The ApiError raised while preparing the request context.

    Returns:
        JSONResponse formatted in the appropriate error schema.
    """
    if "route" not in request.scope:
        for context in iter_route_contexts(app.routes):
            if context.matches(request.scope)[0] is not Match.NONE:
                request.scope["route"] = context.original_route
                break
    return await handle_api_error(request, exc)


#: Pydantic's own container and union tags in an error location, e.g. ``list[union[A,B]]``.
_PYDANTIC_TYPE_TAG = compile_regex(r"[a-z-]+\[.+\]")


def _validation_error_path(loc: Iterable[Any]) -> str:
    """Join a Pydantic error location into a field path a client can act on.

    Drops the union and list wrappers Pydantic descended through, which bury the
    failing field: only those carry a parameterized type name, while a field the
    client sent, such as the multipart ``image[]``, keeps its empty brackets.

    Args:
        loc: Location parts of one Pydantic error.

    Returns:
        The dotted field path, empty when nothing addressable remains.
    """
    return ".".join(
        part for part in map(str, loc) if not _PYDANTIC_TYPE_TAG.fullmatch(part)
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Format Pydantic/FastAPI validation errors as invalid_request_error.

    Args:
        request: The current request.
        exc: The RequestValidationError raised by FastAPI/Pydantic.

    Returns:
        JSONResponse with status 400 and the appropriate error schema.
    """
    errors = exc.errors()

    # Report the deepest location: a union-typed field reports one error per
    # branch, and the shallowest blames the whole field instead of the single
    # item inside it that actually failed.
    match max(errors, key=lambda error: len(error.get("loc", ())), default=None):
        case {"loc": loc, "msg": msg} if path := _validation_error_path(loc):
            message = f"Validation error at {path}: {msg}"
        case {"msg": msg}:
            message = f"Validation error: {msg}"
        case _:
            message = "Validation error"
    # The whole error list stays server-side: it is the only place the other
    # union branches, and any further faults, remain visible for debugging.
    log_error_details(
        [message, *(str(error) for error in errors)] if len(errors) > 1 else message,
        level="warning",
    )
    return JSONResponse(*format_http_error(request, 400, message))


@app.exception_handler(ClientError)
async def handle_botocore_client_error(
    request: Request, exc: ClientError
) -> JSONResponse:
    """Format AWS botocore ClientError using the correct API error envelope.

    Maps common AWS error codes to appropriate HTTP statuses. Region-routing
    bookkeeping (marking a region blocked for retryable errors) already
    happened upstream in ``route_and_execute``, if applicable; this handler
    only formats the terminal error and never touches ``REGION_ROUTER``.

    Args:
        request: The current request.
        exc: The AWS botocore ClientError raised by SDK calls.

    Returns:
        JSONResponse with mapped HTTP status and the appropriate error schema.
    """
    if (denied := denied_feature_unavailable(exc)) is not None:
        return await handle_api_error(request, denied)
    error = exc.response["Error"]
    aws_code = error["Code"]
    status = AWS_ERROR_MAP.get(aws_code, (502, "server_error"))[0]
    log_error_details(error["Message"], status=status)
    message = (
        "The request could not be completed. Retry the request."
        if status >= 500
        else hide_security_details(status, error["Message"])
    )
    return JSONResponse(*format_http_error(request, status, message))


@app.exception_handler(BotocoreConnectionError)
@app.exception_handler(HTTPClientError)
async def handle_botocore_connection_error(
    request: Request, exc: BotocoreConnectionError | HTTPClientError
) -> JSONResponse:
    """Format botocore connection errors (timeouts, unreachable endpoints) as 503.

    Args:
        request: The current request.
        exc: The botocore ConnectionError or HTTPClientError raised by SDK calls.

    Returns:
        JSONResponse with 503 status.
    """
    log_error_details(str(exc), status=503)
    return JSONResponse(
        *format_http_error(
            request,
            503,
            "The service is temporarily unavailable. Retry the request.",
            "server_error",
        )
    )


#: All exception handlers
_EXCEPTION_HANDLERS: dict[
    type[Exception], Callable[[Request, Any], Awaitable[JSONResponse]]
] = {
    ApiError: handle_api_error,
    ClientError: handle_botocore_client_error,
    BotocoreConnectionError: handle_botocore_connection_error,
    HTTPClientError: handle_botocore_connection_error,
    HTTPException: handle_http_exception,
}


@app.exception_handler(ExceptionGroup)
async def handle_exception_group(request: Request, exc: ExceptionGroup) -> JSONResponse:
    """Unwrap ExceptionGroup (from TaskGroup) and handle known sub-exceptions.

    Re-raises unchanged when any sub-exception is unhandled, to fall back on the
    default 500 path with critical logging.

    Args:
        request: The current request.
        exc: The ExceptionGroup raised by a TaskGroup.

    Returns:
        JSONResponse formatted in the appropriate error schema.
    """
    handled_exceptions, unhandled_exceptions = exc.split(tuple(_EXCEPTION_HANDLERS))
    if handled_exceptions and not unhandled_exceptions:
        first = handled_exceptions.exceptions[0]
        while isinstance(first, BaseExceptionGroup):
            first = first.exceptions[0]
        # First matching handler wins: exc.split() matches subclasses too
        # (e.g. botocore's ClientError subclasses), so an exact-class lookup
        # would miss them.
        handler = next(
            handler
            for cls, handler in _EXCEPTION_HANDLERS.items()
            if isinstance(first, cls)
        )
        return await handler(request, first)
    raise exc
