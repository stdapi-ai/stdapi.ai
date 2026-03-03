"""FastAPI application main module for AWS-based OpenAI compatible API.

This module sets up the FastAPI application with middleware, exception handlers,
and AWS service integrations for providing OpenAI-compatible endpoints.
"""

from asyncio import gather
from contextlib import asynccontextmanager
from time import time_ns
from traceback import format_exception
from typing import TYPE_CHECKING

from botocore.exceptions import BotoCoreError, ClientError
from botocore.exceptions import ConnectionError as BotocoreConnectionError
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.background import BackgroundTask

from stdapi.api_errors import ApiError
from stdapi.api_providers import (
    format_http_error,
    get_request_id_header,
    set_log_fields,
    set_response_headers,
)
from stdapi.auth import initialize_authentication
from stdapi.aws import AWSConnectionManager, initialize_aws_account_info
from stdapi.aws_bedrock import (
    AWS_ERROR_MAP,
    set_guardrail_configuration,
    set_performance_configuration,
)
from stdapi.cleanup import CLEANUPS, run_scheduled_cleanups
from stdapi.config import SETTINGS
from stdapi.exceptions import ServerError
from stdapi.metering import EDITION_TITLE, LICENCE_INFO, SERVER_FULL_VERSION, register
from stdapi.models import initialize_bedrock_models, update_unified_models_collections
from stdapi.models.audio.amazon_polly import initialize_polly_models
from stdapi.models.audio.amazon_transcribe import initialize_transcribe_models
from stdapi.monitoring import (
    LOGGING_PATHS_IGNORE,
    EventLog,
    log_error_details,
    log_request_event,
    otel_manager,
    write_log_event,
)
from stdapi.region_routing import measure_region_latencies
from stdapi.routes import discover_routers
from stdapi.server import SERVER_NAME, SERVER_VERSION
from stdapi.utils import hide_security_details

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from typing import Any


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Manage FastAPI application lifespan with AWS connections.

    Args:
        _: The FastAPI application instance (unused).

    Returns:
        Async generator managing application startup and shutdown.
    """
    start = time_ns()
    try:
        # Prepare AWS clients list
        async with AWSConnectionManager(
            *(
                ("polly", SETTINGS.aws_polly_region),
                ("comprehend", SETTINGS.aws_comprehend_region),
                *(("bedrock", region) for region in SETTINGS.aws_bedrock_regions),
                *(
                    ("bedrock-runtime", region)
                    for region in SETTINGS.aws_bedrock_regions
                ),
                ("transcribe", SETTINGS.aws_transcribe_region),
                ("translate", SETTINGS.aws_translate_region),
                ("s3", SETTINGS.aws_transcribe_region),
                ("s3", SETTINGS.aws_bedrock_regions[0]),
                ("s3.accelerate", SETTINGS.aws_bedrock_regions[0]),
                *(("s3", region) for region in SETTINGS.aws_s3_regional_buckets),
            )
        ):
            span_context = otel_manager.start_span(
                "Application start", attributes={"server.id": SERVER_NAME}
            )
            with otel_manager.use_span(span_context):
                region_latencies = await measure_region_latencies()
                results = await gather(
                    initialize_authentication(),
                    initialize_bedrock_models(),
                    initialize_polly_models(),
                    initialize_transcribe_models(),
                    register(),
                    initialize_aws_account_info(),
                )
                auth_enabled = results[0]
                register_usage_response = results[-1]
                unavailable_models = results[1][1]
                invalid_arn_mappings = results[1][2]
                unmatched_restrict_keys = results[1][3]
                update_unified_models_collections()
            start_event = EventLog(
                type="start",
                level="info",
                date=SETTINGS.now(),
                server_id=SERVER_NAME,
                server_version=SERVER_FULL_VERSION,
                server_start_time_ms=(time_ns() - start) // 1000000,
            )
            if register_usage_response:
                start_event["register_usage_response"] = register_usage_response
            if region_latencies:
                start_event["region_latencies"] = region_latencies
            if not auth_enabled:
                start_event.setdefault("server_warnings", []).append(
                    "SECURITY risk: Authentication is not enabled "
                    "('api_key', 'api_key_ssm_parameter', 'api_key_secretsmanager_secret' not set)"
                )
                start_event["level"] = "warning"
            if not SETTINGS.aws_s3_bucket:
                start_event.setdefault("server_warnings", []).append(
                    "S3 bucket not configured ('aws_s3_bucket' not set): some features are disabled"
                )
                start_event["level"] = "warning"
            if unavailable_models:
                start_event.setdefault("server_warnings", []).append(
                    {"unavailable_bedrock_models": unavailable_models}  # type: ignore[dict-item]
                )
                start_event["level"] = "warning"
            if invalid_arn_mappings:
                start_event.setdefault("server_warnings", []).append(
                    {"invalid_bedrock_model_arn_mappings": invalid_arn_mappings}  # type: ignore[dict-item]
                )
                start_event["level"] = "warning"
            if unmatched_restrict_keys:
                start_event.setdefault("server_warnings", []).append(
                    f"'aws_bedrock_model_region_restrict' has no matching available model for: {', '.join(sorted(unmatched_restrict_keys))}. "
                    f"Check for unknown model IDs/prefixes or models not available in the configured regions."
                )
                start_event["level"] = "warning"
            write_log_event(start_event)
            yield
    except (BotoCoreError, ClientError, ServerError) as exception:
        write_log_event(
            EventLog(
                type="start",
                level="error",
                date=SETTINGS.now(),
                server_id=SERVER_NAME,
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
                server_id=SERVER_NAME,
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
                server_id=SERVER_NAME,
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

if SETTINGS.enable_gzip:
    from fastapi.middleware.gzip import GZipMiddleware

    app.add_middleware(GZipMiddleware, minimum_size=1024)

if SETTINGS.enable_proxy_headers:
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")

if SETTINGS.trusted_hosts:
    from fastapi.middleware.trustedhost import TrustedHostMiddleware

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=SETTINGS.trusted_hosts)

if SETTINGS.cors_allow_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=SETTINGS.cors_allow_origins,
        allow_credentials=True,
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
        with log_request_event(request) as log:
            set_guardrail_configuration(request.headers)
            set_performance_configuration(request.headers)
            response = await call_next(request)
            log["status_code"] = response.status_code
            response.headers[get_request_id_header(request)] = log["id"]
            set_log_fields(request, log)
        set_response_headers(request, response, log["execution_time_ms"])
        if CLEANUPS.get():
            response.background = BackgroundTask(run_scheduled_cleanups)
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


@app.exception_handler(BotocoreConnectionError)
async def handle_botocore_connection_error(
    request: Request, exc: BotocoreConnectionError
) -> JSONResponse:
    """Format botocore connection errors (timeouts, unreachable endpoints) as 503.

    Args:
        request: The current request.
        exc: The botocore ConnectionError raised by SDK calls.

    Returns:
        JSONResponse with 503 status.
    """
    message = str(exc)
    log_error_details(message, status=503)
    return JSONResponse(*format_http_error(request, 503, message, "server_error"))


#: All exception handlers
_EXCEPTION_HANDLERS: dict[
    type[Exception], Callable[[Request, Any], Awaitable[JSONResponse]]
] = {
    ApiError: handle_api_error,
    ClientError: handle_botocore_client_error,
    BotocoreConnectionError: handle_botocore_connection_error,
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
        return await _EXCEPTION_HANDLERS[first.__class__](request, first)
    raise exc
