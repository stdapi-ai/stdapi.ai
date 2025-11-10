"""FastAPI application main module for AWS-based OpenAI compatible API.

This module sets up the FastAPI application with middleware, exception handlers,
and AWS service integrations for providing OpenAI-compatible endpoints.
"""

from asyncio import gather
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import time_ns
from traceback import format_exception

from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from stdapi.auth import initialize_authentication
from stdapi.aws import AWSConnectionManager
from stdapi.aws_bedrock import set_guardrail_configuration
from stdapi.config import SETTINGS, LogLevel
from stdapi.exceptions import ServerError
from stdapi.metering import EDITION_TITLE, LICENCE_INFO, SERVER_FULL_VERSION, register
from stdapi.models import initialize_bedrock_models, update_unified_models_collections
from stdapi.monitoring import (
    LOGGING_PATHS_IGNORE,
    EventLog,
    log_error_details,
    log_request_event,
    otel_manager,
    write_log_event,
)
from stdapi.openai import set_openai_headers
from stdapi.openai_exceptions import OpenaiError
from stdapi.routes import discover_routers
from stdapi.routes.openai_audio_speech import initialize_polly_models
from stdapi.routes.openai_audio_transcriptions import initialize_transcribe_models
from stdapi.server import SERVER_NAME, SERVER_VERSION
from stdapi.utils import hide_security_details


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
                results = await gather(
                    initialize_authentication(),
                    initialize_bedrock_models(),
                    initialize_polly_models(),
                    initialize_transcribe_models(),
                    register(),
                )
                auth_enabled = results[0]
                unavailable_models = results[1][1]
                update_unified_models_collections()
            start_event = EventLog(
                type="start",
                level="info",
                date=SETTINGS.now(),
                server_id=SERVER_NAME,
                server_version=SERVER_FULL_VERSION,
                server_start_time_ms=(time_ns() - start) // 1000000,
            )
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
    except Exception as exception:  # noqa: BLE001
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
        with log_request_event(request) as log:
            set_guardrail_configuration(request.headers)
            response = await call_next(request)
            log["status_code"] = response.status_code
            response.headers["x-request-id"] = log["id"]
        set_openai_headers(request, response, log["id"], log["execution_time_ms"])
    response.headers["server"] = "stdapi.ai"
    return response


#: Status codes to OpenAI error codes
_STATUS_ERROR_MAP = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
}


@app.exception_handler(HTTPException)
async def handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
    """Convert standard FastAPI HTTPException using OpenAI error envelope.

    The response body matches: {"error": {"message", "type", "param", "code"}}.

    Args:
        _: The current request (unused).
        exc: The HTTPException raised by a route or dependency.

    Returns:
        JSONResponse formatted in OpenAI error schema with the appropriate status.
    """
    status_code = exc.status_code
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    if 500 <= status_code <= 599:
        level: LogLevel = "error"
        error_type = "server_error"
    else:
        level = "warning"
        error_type = _STATUS_ERROR_MAP.get(status_code, "api_error")
    log_error_details(message, level=level)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": hide_security_details(status_code, message),
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


@app.exception_handler(OpenaiError)
async def handle_openai_exception(_: Request, exc: OpenaiError) -> JSONResponse:
    """Raise FastAPI HTTPException using OpenAI error envelope.

    The response body matches: {"error": {"message", "type", "param", "code"}}.

    Args:
        _: The current request (unused).
        exc: The HTTPException raised by a route or dependency.

    Returns:
        JSONResponse formatted in OpenAI error schema with the appropriate status.
    """
    log_error_details(exc.args[0], level="warning")
    return JSONResponse(
        status_code=exc.status,
        content={
            "error": {
                "message": hide_security_details(exc.status, exc.args[0]),
                "type": exc.type,
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Format Pydantic/FastAPI validation errors as OpenAI invalid_request_error (422).

    Args:
        _: The current request (unused).
        exc: The RequestValidationError raised by FastAPI/Pydantic.

    Returns:
        JSONResponse with status 400 and OpenAI error schema content.
    """
    # Build a concise message summarizing the first error to align with OpenAI style
    code = None
    param = None

    if exc.errors():
        first = exc.errors()[0]
        loc = ".".join(str(x) for x in first.get("loc", []))
        msg = first.get("msg", "validation error")
        message = (
            f"Validation error at {loc}: {msg}" if loc else f"Validation error: {msg}"
        )
    else:
        message = "Validation error"

    log_error_details(message, level="warning")
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": code,
            }
        },
    )


#: AWS error codes to OpenAI error codes
_AWS_ERROR_MAP: dict[str, tuple[int, str]] = {
    **dict.fromkeys(
        {
            "ThrottlingException",
            "TooManyRequestsException",
            "ServiceQuotaExceededException",
        },
        (429, "rate_limit_error"),
    ),
    **dict.fromkeys({"AccessDeniedException"}, (403, "permission_error")),
    **dict.fromkeys(
        {
            "UnrecognizedClientException",
            "InvalidSignatureException",
            "ExpiredTokenException",
        },
        (401, "authentication_error"),
    ),
    **dict.fromkeys({"ResourceNotFoundException"}, (404, "not_found_error")),
    **dict.fromkeys(
        {"ValidationException", "BadRequestException"}, (400, "invalid_request_error")
    ),
    **dict.fromkeys(
        {
            "ServiceUnavailableException",
            "InternalServerException",
            "ServiceFailureException",
        },
        (503, "server_error"),
    ),
}


@app.exception_handler(ClientError)
async def handle_botocore_client_error(_: Request, exc: ClientError) -> JSONResponse:
    """Format AWS botocore ClientError using OpenAI error envelope.

    Maps common AWS error codes to appropriate HTTP statuses.

    Args:
        _: The current request (unused).
        exc: The AWS botocore ClientError raised by SDK calls.

    Returns:
        JSONResponse with mapped HTTP status and OpenAI error schema content.
    """
    error = exc.response["Error"]
    aws_code = error["Code"]
    status, err_type = _AWS_ERROR_MAP.get(aws_code, (502, "server_error"))
    log_error_details(error["Message"], level="warning" if status < 500 else "error")
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": hide_security_details(status, error["Message"]),
                "type": err_type,
                "param": None,
                "code": aws_code,
            }
        },
    )


discover_routers(app)
