"""FastAPI application main module for AWS-based OpenAI compatible API.

This module sets up the FastAPI application with middleware, exception handlers,
and AWS service integrations for providing OpenAI-compatible endpoints.
"""

from asyncio import gather
from contextlib import asynccontextmanager
from time import time_ns
from traceback import format_exception
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from botocore.exceptions import BotoCoreError, ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.background import BackgroundTask

from stdapi import server
from stdapi.api_errors import ApiError
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
)
from stdapi.aws_bedrock import (
    AWS_ERROR_MAP,
    set_guardrail_configuration,
    set_performance_configuration,
)
from stdapi.cleanup import CLEANUPS, run_scheduled_cleanups
from stdapi.config import SETTINGS
from stdapi.exceptions import ServerError
from stdapi.input_file import reset_current_input_files
from stdapi.metering import EDITION_TITLE, LICENCE_INFO, SERVER_FULL_VERSION, register
from stdapi.models import initialize_bedrock_models, update_unified_models_collections
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
from stdapi.region_routing import measure_region_latencies
from stdapi.routes import discover_routers
from stdapi.server import SERVER_VERSION
from stdapi.utils import hide_security_details

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from typing import Any

    from types_aiobotocore_bedrock.literals import RegionName


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Manage FastAPI application lifespan: start AWS connections and initialize services.

    Args:
        _: The FastAPI application instance (unused).

    Yields:
        None once startup is complete; shutdown runs after the yield.
    """
    start = time_ns()
    # Fall back to the configured/default region so a bucket-less deployment
    # still warms one Transcribe client (requests then 404 as before).
    transcribe_regions: list[RegionName | None] = [
        region for region, _ in transcribe_job_candidates()
    ] or [SETTINGS.aws_transcribe_region]
    try:
        # Prepare AWS clients list
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
                *(("transcribe", region) for region in transcribe_regions),
                *(
                    ("translate", region)
                    for region in service_regions(SETTINGS.aws_translate_region)
                ),
                # Only warmed when cost tracking is enabled: its sole
                # consumer (the price catalog loader) otherwise no-ops.
                # None (GovCloud, no Price List endpoint) warms the default
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
            )
        ):
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
                    raise_first_exception(
                        await gather(
                            initialize_authentication(start_event),
                            initialize_bedrock_models(start_event),
                            initialize_polly_models(start_event),
                            initialize_transcribe_models(),
                            initialize_moderation_models(),
                            register(start_event),
                            initialize_aws_account_info(),
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
                if deprecated := SETTINGS.deprecated():
                    add_server_warning(
                        start_event,
                        f"Ignored deprecated settings: {', '.join(deprecated)}",
                    )
                start_event["server_start_time_ms"] = (time_ns() - start) // 1000000
                write_log_event(start_event)
                yield
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
                error_detail=[f"{type(exception).__name__}: {exception}"],
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
        write_log_event(
            EventLog(
                type="stop",
                level="info",
                date=SETTINGS.now(),
                server_id=server.SERVER_NAME,
                server_version=SERVER_FULL_VERSION,
                server_uptime_ms=(time_ns() - start) // 1000000,
            )
        )
        otel_manager.flush()


app = FastAPI(
    title=EDITION_TITLE,
    description="AWS standardized AI API",
    version=SERVER_VERSION,
    lifespan=lifespan,
    contact={"name": "stdapi.ai", "url": "https://stdapi.ai"},
    license_info=LICENCE_INFO,
    docs_url="/docs" if SETTINGS.enable_docs else None,
    redoc_url="/redoc" if SETTINGS.enable_redoc else None,
    openapi_url="/openapi.json" if SETTINGS.enable_openapi_json else None,
)
otel_manager.instrument(app)
discover_routers(app)

if SETTINGS.enable_mcp_streamable_http or SETTINGS.enable_mcp_sse:
    from stdapi.mcp import mount_mcp

    mount_mcp(app)

if SETTINGS.enable_gzip:
    from fastapi.middleware.gzip import GZipMiddleware

    app.add_middleware(GZipMiddleware, minimum_size=1024)

if SETTINGS.enable_proxy_headers:
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    app.add_middleware(
        ProxyHeadersMiddleware, trusted_hosts=SETTINGS.proxy_trusted_hosts
    )

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
    )


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
            set_guardrail_configuration(request.headers)
            set_performance_configuration(request.headers)
            try:
                response = await call_next(request)
            except RuntimeError as exc:
                if await request.is_disconnected():
                    log["status_code"] = 499
                    log_error_details(f"Client disconnected: {exc}", status=499)
                    return Response(status_code=499)
                raise
            log["status_code"] = response.status_code
            response.headers[get_request_id_header(request)] = log["id"]
            set_log_fields(request, log)
        set_response_headers(request, response, log["execution_time_ms"])
        if CLEANUPS.get():
            response.background = BackgroundTask(run_scheduled_cleanups, log["id"])
    response.headers["server"] = "stdapi.ai"
    return response


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """Convert standard FastAPI HTTPException to the correct API error envelope.

    Args:
        request: The current request.
        exc: The HTTPException raised by a route or dependency.

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
            hide_security_details(status_code, exc.args[0]),
            exc.param,
            exc.code,
        )
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
    match exc.errors():
        case [{"loc": loc, "msg": msg}, *_]:
            loc_str = ".".join(str(x) for x in loc)
            message = (
                f"Validation error at {loc_str}: {msg}"
                if loc_str
                else f"Validation error: {msg}"
            )
        case [{"msg": msg}, *_]:
            message = f"Validation error: {msg}"
        case _:
            message = "Validation error"
    log_error_details(message, level="warning")
    return JSONResponse(
        *format_http_error(request, 400, message, "invalid_request_error")
    )


@app.exception_handler(ClientError)
async def handle_botocore_client_error(
    request: Request, exc: ClientError
) -> JSONResponse:
    """Format AWS botocore ClientError using the correct API error envelope.

    Maps common AWS error codes to appropriate HTTP statuses.
    When region routing is enabled, marks the region as blocked for retryable errors.

    Args:
        request: The current request.
        exc: The AWS botocore ClientError raised by SDK calls.

    Returns:
        JSONResponse with mapped HTTP status and the appropriate error schema.
    """
    error = exc.response["Error"]
    aws_code = error["Code"]
    status, err_type = AWS_ERROR_MAP.get(aws_code, (502, "server_error"))
    log_error_details(error["Message"], status=status)
    return JSONResponse(
        *format_http_error(
            request,
            status,
            hide_security_details(status, error["Message"]),
            err_type,
            code=aws_code,
        )
    )


#: AWS service host tokens mapped to human-friendly names for error messages.
_AWS_SERVICE_NAMES: dict[str, str] = {
    "bedrock": "Bedrock",
    "bedrock-runtime": "Bedrock",
    "s3": "S3",
    "polly": "Polly",
    "comprehend": "Comprehend",
    "transcribe": "Transcribe",
    "translate": "Translate",
    "ssm": "SSM Parameter Store",
    "secretsmanager": "Secrets Manager",
    "sts": "STS",
}


def _upstream_service_name(exc: BotocoreConnectionError | HTTPClientError) -> str:
    """Return a friendly upstream AWS service name from a connection error.

    Derives the service from the endpoint host embedded in the botocore
    exception, mapping only recognised AWS service tokens so custom or VPC
    endpoint identifiers are never disclosed to clients.

    Args:
        exc: The botocore connection error.

    Returns:
        A friendly service name, or "The upstream service" when unknown.
    """
    endpoint = getattr(exc, "kwargs", {}).get("endpoint_url", "") or ""
    host = urlparse(endpoint).hostname or ""
    for token in host.split("."):
        if name := _AWS_SERVICE_NAMES.get(token):
            return name
    return "The upstream service"


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
            f"{_upstream_service_name(exc)} is temporarily unavailable.",
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

    If all sub-exceptions are of a known handled type, returns the appropriate
    error response for the first match. If any sub-exception is unhandled,
    logs all sub-exceptions and re-raises to trigger the default 500 path with
    critical logging.

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
