"""AWS client management and connection pooling."""

from asyncio import gather, sleep
from contextlib import AsyncExitStack, suppress
from datetime import datetime
from json import dumps as _std_dumps
from logging import getLogger
from os import environ
from typing import TYPE_CHECKING, Any, Final, NotRequired, Self, TypedDict

from aiobotocore.config import AioConfig
from aiohttp import ClientError as HttpClientError
from aiohttp import ClientSession, ClientTimeout
from botocore import serialize as botocore_serialize
from botocore.exceptions import BotoCoreError, ClientError
from botocore.utils import parse_timestamp
from pydantic_core import to_json

from stdapi import server
from stdapi.aws_bedrock_mantle import mantle_http_session
from stdapi.config import AWS_REGION, AWS_SESSION, SETTINGS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from types import TracebackType

    from botocore.model import OperationModel
    from types_aiobotocore_bedrock.literals import RegionName

    class AwsEnvironment(TypedDict):
        """AWS environment."""

        account_id: NotRequired[str]


#: AWS environment information (populated during startup)
AWS_ENVIRONMENT: AwsEnvironment = {}

#: Cached AWS service clients keyed by (service, region)
_CLIENTS: dict[str, dict[RegionName, Any]] = {}

#: Attempts allowed to read the ECS container metadata endpoint at startup
_ECS_METADATA_ATTEMPTS: Final = 3

#: Delay between two ECS container metadata attempts, in seconds
_ECS_METADATA_RETRY_DELAY: Final = 1.0

#: Total timeout of a single ECS container metadata request, in seconds
_ECS_METADATA_TIMEOUT: Final = 10

#: Connection timeout of a single ECS container metadata request, in seconds
_ECS_METADATA_CONNECT_TIMEOUT: Final = 5

#: Region-rotated Bedrock services that get single-attempt ".no-retry" client pools
_NO_RETRY_SERVICES: Final = ("bedrock-runtime", "bedrock-agent-runtime")

#: Default retry configuration (configurable attempts and retry mode)
_RETRIES = {
    "max_attempts": SETTINGS.aws_bedrock_max_retries + 1,
    "mode": "adaptive" if SETTINGS.aws_adaptive_retry else "standard",
}

#: Default configuration — used by all services including bedrock-runtime
CONFIG = AioConfig(
    user_agent=server.USER_AGENT,
    retries=_RETRIES,
    max_pool_connections=SETTINGS.aws_max_pool_connections,
    parameter_validation=False,
    connect_timeout=SETTINGS.aws_connect_timeout,
    read_timeout=SETTINGS.ai_response_timeout,
)


getLogger("aiobotocore").setLevel("CRITICAL")


class PydanticRestJSONSerializer(botocore_serialize.RestJSONSerializer):
    """botocore rest-json serializer encoding request bodies with pydantic_core.

    3.5x faster than the stdlib encoder on large Bedrock ``Converse`` bodies;
    output is semantically identical JSON (compact separators, raw UTF-8
    instead of ASCII escapes). The stdlib encoder remains as fallback for
    input pydantic_core rejects, such as strings carrying lone surrogates.
    """

    def _serialize_body_params(self, params: Any, shape: Any) -> bytes:  # noqa: ANN401
        serialized_body = self.MAP_TYPE()
        self._serialize(serialized_body, params, shape)  # type: ignore[attr-defined]
        try:
            return to_json(serialized_body)
        except ValueError:
            return _std_dumps(serialized_body).encode(self.DEFAULT_ENCODING)


# Registry-level install: every rest-json client created afterwards (all pooled
# Bedrock/Polly/S3-control clients and ad-hoc ones) gets the fast serializer.
botocore_serialize.SERIALIZERS["rest-json"] = PydanticRestJSONSerializer  # type: ignore[assignment]


def parse_aws_timestamp(value: Any) -> datetime:  # noqa: ANN401
    """Parse an AWS response timestamp, ``fromisoformat`` fast path first.

    13x faster than botocore's dateutil-based default on timestamp-dense
    control-plane listings; non-ISO strings (RFC 822 headers) and numeric
    epochs fall back to botocore's parser, whose output (including naive
    datetimes for offset-less strings) this matches exactly.

    Args:
        value: Raw timestamp value from a parsed AWS response.

    Returns:
        Parsed datetime.
    """
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return parse_timestamp(value)
    return parse_timestamp(value)


def _aws_request_id(response: dict[str, Any]) -> str:
    """Extract the AWS-side request ID from a parsed AWS response.

    Args:
        response: Parsed AWS response, including ``ResponseMetadata``.

    Returns:
        The request ID, or "" when the response carries none.
    """
    metadata = response.get("ResponseMetadata") or {}
    request_id: str = metadata.get("RequestId") or (
        metadata.get("HTTPHeaders") or {}
    ).get("x-amzn-requestid", "")
    return request_id


def _record_after_call(
    parsed: dict[str, Any], model: OperationModel, **_kwargs: object
) -> None:
    """Record a completed AWS API call's request ID (``after-call`` hook).

    Also fires for HTTP error responses, before botocore raises the matching
    ``ClientError``, so failed calls are captured with their error code.

    Args:
        parsed: Parsed AWS response, including ``ResponseMetadata``.
        model: Operation model of the call.
        **_kwargs: Unused botocore event arguments.
    """
    if not (request_id := _aws_request_id(parsed)):
        return
    # Imported here: stdapi.monitoring transitively imports this module.
    from stdapi.monitoring import record_aws_api_call  # noqa: PLC0415

    record_aws_api_call(
        model.service_model.service_id.hyphenize(),
        model.name,
        request_id,
        (parsed.get("Error") or {}).get("Code"),
    )


def _record_after_call_error(
    event_name: str, exception: Exception, **_kwargs: object
) -> None:
    """Record a failed AWS API call's request ID (``after-call-error`` hook).

    Only errors carrying a parsed AWS response (``ClientError`` subclasses)
    expose a request ID; transport errors without a response are skipped.

    Args:
        event_name: Full event name, ``after-call-error.<service>.<operation>``.
        exception: The exception that aborted the call.
        **_kwargs: Unused botocore event arguments.
    """
    response = getattr(exception, "response", None)
    if not isinstance(response, dict) or not (request_id := _aws_request_id(response)):
        return
    from stdapi.monitoring import record_aws_api_call  # noqa: PLC0415

    _, service, operation = event_name.split(".", 2)
    record_aws_api_call(
        service, operation, request_id, (response.get("Error") or {}).get("Code")
    )


# Registered on the shared session so every client created from it (including
# per-region pools and ad-hoc STS clients) inherits the hooks at creation.
AWS_SESSION.register("after-call", _record_after_call, unique_id="stdapi-request-id")
AWS_SESSION.register(
    "after-call-error", _record_after_call_error, unique_id="stdapi-request-id-error"
)
# Session-level install: every client created from the shared session parses
# response timestamps through the fromisoformat fast path.
AWS_SESSION.get_component("response_parser_factory").set_parser_defaults(
    timestamp_parser=parse_aws_timestamp
)


def raise_first_exception(results: Sequence[Any]) -> None:
    """Re-raise the first exception found in a ``gather(return_exceptions=True)`` result.

    Any additional exceptions are recorded as a PEP 678 note on the first one so
    concurrent sibling failures are not silently discarded.

    Args:
        results: Results from a ``gather(..., return_exceptions=True)`` call.

    Raises:
        BaseException: The first exception found in *results*, if any.
    """
    exceptions = [result for result in results if isinstance(result, BaseException)]
    if not exceptions:
        return
    first, *others = exceptions
    if others:
        first.add_note(
            f"{len(others)} concurrent failure(s) suppressed: "
            + "; ".join(f"{type(e).__name__}: {e}" for e in others)
        )
    raise first


class AWSConnectionManager:
    """Manages persistent AWS client connections."""

    __slots__ = ("_client_specs", "_exit_stack")

    def __init__(self, *clients: tuple[str, RegionName | None]) -> None:
        """Initialize AWS connection manager with client specifications.

        Args:
            *clients: Variable number of tuples containing service name and optional region.
                Each tuple contains (service_name, region_name or None).
        """
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._client_specs = clients

    async def __aenter__(self) -> Self:
        """Initialize AWS clients.

        Returns:
            AWSConnectionManager: The initialized connection manager.

        Raises:
            BotoCoreError: A client failed to initialize; every client
                already entered is closed before the exception propagates.
            ClientError: Same as above.
        """
        await self._exit_stack.__aenter__()
        try:
            services_configs: dict[str, AioConfig] = {
                "s3.accelerate": AioConfig(
                    user_agent=server.USER_AGENT,
                    retries=_RETRIES,
                    max_pool_connections=SETTINGS.aws_max_pool_connections,
                    parameter_validation=False,
                    connect_timeout=SETTINGS.aws_connect_timeout,
                    s3={"use_accelerate_endpoint": SETTINGS.aws_s3_accelerate},
                ),
                "pricing": AioConfig(
                    user_agent=server.USER_AGENT,
                    retries={
                        # No dedicated pricing setting; reuses the Bedrock retry count.
                        "max_attempts": SETTINGS.aws_bedrock_max_retries + 1,
                        # Always adaptive: the Pricing API rate quota is very low.
                        "mode": "adaptive",
                    },
                    max_pool_connections=SETTINGS.aws_max_pool_connections,
                    parameter_validation=False,
                    connect_timeout=SETTINGS.aws_connect_timeout,
                ),
            }
            # Failover services trade deep in-region retries for fast
            # cross-region failover when several regions are candidates.
            failover_config = AioConfig(
                user_agent=server.USER_AGENT,
                retries={
                    "max_attempts": SETTINGS.aws_failover_max_retries + 1,
                    "mode": "adaptive" if SETTINGS.aws_adaptive_retry else "standard",
                },
                max_pool_connections=SETTINGS.aws_max_pool_connections,
                parameter_validation=False,
                connect_timeout=SETTINGS.aws_connect_timeout,
                read_timeout=SETTINGS.ai_response_timeout,
            )
            services_configs.update(
                {
                    service: failover_config
                    for service, region_setting in (
                        ("polly", SETTINGS.aws_polly_region),
                        ("comprehend", SETTINGS.aws_comprehend_region),
                        ("translate", SETTINGS.aws_translate_region),
                        ("transcribe", SETTINGS.aws_transcribe_region),
                    )
                    if len(service_regions(region_setting)) > 1
                }
            )

            specs = [
                *{
                    (service, region or SETTINGS.aws_bedrock_regions[0])
                    for service, region in self._client_specs
                }
            ]
            results = await gather(
                *(
                    self._exit_stack.enter_async_context(
                        # New service names must join the botocore/data allowlist
                        # pruned in the Dockerfile, or the image fails at runtime.
                        AWS_SESSION.create_client(  # type: ignore[call-overload]
                            service.split(".", 1)[0],
                            region_name=region,
                            config=services_configs.get(service, CONFIG),
                        )
                    )
                    for service, region in specs
                ),
                return_exceptions=True,
            )
            raise_first_exception(results)
            for (service, region), client in zip(specs, results, strict=True):
                _CLIENTS.setdefault(service, {})[region] = client

            if SETTINGS.aws_bedrock_region_routing != "disabled":
                no_retry = AioConfig(
                    user_agent=server.USER_AGENT,
                    retries={
                        "max_attempts": 1,
                        "mode": "adaptive"
                        if SETTINGS.aws_adaptive_retry
                        else "standard",
                    },
                    max_pool_connections=SETTINGS.aws_max_pool_connections,
                    parameter_validation=False,
                    connect_timeout=SETTINGS.aws_connect_timeout,
                    read_timeout=SETTINGS.ai_response_timeout,
                )
                for service in _NO_RETRY_SERVICES:
                    if service not in _CLIENTS:
                        continue
                    regions = [*_CLIENTS[service]]
                    no_retry_results = await gather(
                        *(
                            self._exit_stack.enter_async_context(
                                AWS_SESSION.create_client(  # type: ignore[call-overload]
                                    service, region_name=region, config=no_retry
                                )
                            )
                            for region in regions
                        ),
                        return_exceptions=True,
                    )
                    raise_first_exception(no_retry_results)
                    _CLIENTS[f"{service}.no-retry"] = dict(
                        zip(regions, no_retry_results, strict=True)
                    )

            if SETTINGS.aws_bedrock_mantle_enabled:
                await self._exit_stack.enter_async_context(mantle_http_session())
        except BaseException as exception:
            # __aenter__ raising skips __aexit__ under the context-manager
            # protocol: close whatever this attempt already entered so no
            # client leaks, then let the caller see the original error.
            try:
                await self._exit_stack.aclose()
            except Exception as cleanup_error:  # noqa: BLE001
                exception.add_note(
                    "Client cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            _CLIENTS.clear()
            raise
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Cleanup all AWS clients.

        Args:
            exc_type: Exception type if an error occurred within the context.
            exc_val: Exception instance if an error occurred within the context.
            exc_tb: Traceback object if an error occurred within the context.
        """
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        _CLIENTS.clear()


#: ClientError codes indicating a region-level issue worth failing over.
_FAILOVER_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "InternalFailure",
        "InternalServerError",
        "InternalServerException",
        # Transcribe's per-region concurrent-job quota: the reason multi-region
        # failover exists.
        "LimitExceededException",
        "RequestTimeout",
        "ServiceQuotaExceededException",
        "ServiceUnavailable",
        "ServiceUnavailableException",
        "ThrottlingException",
        "TooManyRequestsException",
    }
)

#: ClientError codes answered by a service absent from the called region
_REGION_UNAVAILABLE_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {"NotAuthorizedException"}
)

#: Error message prefix of an operation unavailable in the called region
_UNSUPPORTED_OPERATION_PREFIX: Final = "UNSUPPORTED_OPERATION"


def service_regions(region: RegionName | None) -> list[RegionName]:
    """Return the candidate regions for an auxiliary AWS service.

    Args:
        region: The service-specific region setting, if configured.

    Returns:
        The configured region alone, or a copy of every Bedrock region otherwise.
    """
    return [region] if region else [*SETTINGS.aws_bedrock_regions]


def is_failover_error(exception: BotoCoreError | ClientError) -> bool:
    """Whether an AWS error indicates a region-level issue worth failing over.

    Args:
        exception: The AWS error.

    Returns:
        True for network/availability/throttling/5xx errors and for a service
        or operation the region does not offer, False for caller errors
        (validation, bad input) that would fail everywhere.
    """
    if isinstance(exception, BotoCoreError):
        return True
    error = exception.response.get("Error", {})
    code: str = error.get("Code", "")
    status = exception.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
    if code in _FAILOVER_ERROR_CODES or status >= 500:
        return True
    # A service the region does not host answers a 4xx the next region serves:
    # Comprehend reports NotAuthorizedException outside its regions, and
    # UNSUPPORTED_OPERATION where only that operation is unavailable.
    message: str = error.get("Message", "")
    return code in _REGION_UNAVAILABLE_ERROR_CODES or message.startswith(
        _UNSUPPORTED_OPERATION_PREFIX
    )


async def call_with_region_failover[ResultT](
    service: str,
    regions: Sequence[RegionName],
    call: Callable[[Any, RegionName], Awaitable[ResultT]],
    on_failed_region: Callable[[Any, RegionName], Awaitable[None]] | None = None,
) -> tuple[ResultT, RegionName]:
    """Run an AWS call, failing over across candidate regions.

    Args:
        service: AWS service name (client pool key).
        regions: Candidate regions, in priority order (at least one).
        call: Coroutine factory receiving the region's client and the region.
        on_failed_region: Optional best-effort cleanup run with the failed
            region's client after a failover-class failure there (its own AWS
            errors are ignored); for calls whose server-side effect may have
            been applied even though the call errored.

    Returns:
        The first successful result and the region that served it.

    Raises:
        BotoCoreError: The last candidate region's error when every region
            fails, or the first caller error (never worth retrying).
        ClientError: Same as above.
    """
    *fallible, last_region = regions
    for region in fallible:
        try:
            return await call(get_client(service, region), region), region
        except (BotoCoreError, ClientError) as exception:
            if not is_failover_error(exception):
                raise
            await _cleanup_failed_region(service, region, on_failed_region)
            # Imported here: stdapi.monitoring transitively imports this module.
            from stdapi.monitoring import (  # noqa: PLC0415
                REQUEST_LOG,
                log_error_details,
            )

            if REQUEST_LOG.get(None) is not None:
                log_error_details(
                    f"AWS {service} error in region {region} "
                    f"({type(exception).__name__}); failing over to the next region.",
                    level="warning",
                )
    try:
        return await call(get_client(service, last_region), last_region), last_region
    except (BotoCoreError, ClientError) as exception:
        if is_failover_error(exception):
            await _cleanup_failed_region(service, last_region, on_failed_region)
        raise


async def _cleanup_failed_region(
    service: str,
    region: RegionName,
    on_failed_region: Callable[[Any, RegionName], Awaitable[None]] | None,
) -> None:
    """Run the failed-region cleanup hook, ignoring its own AWS errors.

    Args:
        service: AWS service name (client pool key).
        region: The region whose call failed.
        on_failed_region: The cleanup hook, if any.
    """
    if on_failed_region is None:
        return
    with suppress(BotoCoreError, ClientError):
        await on_failed_region(get_client(service, region), region)


def get_client(service: str, region_name: RegionName | None = None) -> Any:  # noqa:ANN401
    """Get AWS client.

    Args:
        service: AWS service name.
        region_name: Optional specific region,
            use default region if not specified.

    Returns:
        AWS client instance.

    Raises:
        KeyError: If multiple regional clients exist and the requested region
            is not available in the pool.
    """
    clients = _CLIENTS[service]
    try:
        return clients[region_name or SETTINGS.aws_bedrock_regions[0]]
    except KeyError:
        if len(clients) == 1:
            return next(iter(clients.values()))
        raise


async def _set_account_id_from_sts() -> None:
    """Set the account ID from the STS caller identity."""
    async with AWS_SESSION.create_client(
        "sts",
        config=AioConfig(user_agent=server.USER_AGENT, parameter_validation=False),
        region_name=AWS_REGION,
    ) as sts_client:
        AWS_ENVIRONMENT["account_id"] = (await sts_client.get_caller_identity())[
            "Account"
        ]


async def _set_account_info_from_ecs(metadata_path: str) -> None:
    """Set the account ID and the task-qualified server name from container metadata.

    Args:
        metadata_path: ECS task metadata endpoint URI.
    """
    async with ClientSession(
        headers=server.HTTP_CLIENT_HEADERS,
        timeout=ClientTimeout(
            total=_ECS_METADATA_TIMEOUT, connect=_ECS_METADATA_CONNECT_TIMEOUT
        ),
    ) as session:
        async with session.get(metadata_path) as resp:
            resp.raise_for_status()
            container_name = (await resp.json())["Name"]
        async with session.get(f"{metadata_path}/task") as resp:
            resp.raise_for_status()
            parts = (await resp.json())["TaskARN"].split(":")
            AWS_ENVIRONMENT["account_id"] = parts[4]
            task_id = parts[5].split("/")[-1]
    server.SERVER_NAME = f"{task_id}-{container_name}-{server.SERVER_ID}"


async def initialize_aws_account_info() -> str | None:
    """Initialize AWS account information at server startup.

    Retrieves the AWS account ID from the ECS container metadata endpoint, which
    also names the server after its task, and falls back to the STS API outside
    ECS or when that endpoint stays unreachable. The endpoint is served by the
    ECS agent over the task ENI and can answer slowly while the rest of the
    startup sequence competes for the task CPU, hence the retries.

    Returns:
        A warning message when the metadata endpoint could not be reached,
        ``None`` otherwise.
    """
    try:
        metadata_path = environ["ECS_CONTAINER_METADATA_URI_V4"]
    except KeyError:
        await _set_account_id_from_sts()
        return None
    warning = None
    for attempt in range(_ECS_METADATA_ATTEMPTS):
        if attempt:
            await sleep(_ECS_METADATA_RETRY_DELAY)
        try:
            await _set_account_info_from_ecs(metadata_path)
        except (HttpClientError, TimeoutError) as exception:
            warning = (
                "ECS container metadata endpoint unreachable after "
                f"{_ECS_METADATA_ATTEMPTS} attempts ({type(exception).__name__}): "
                "the server name does not identify the ECS task"
            )
        else:
            return None
    await _set_account_id_from_sts()
    return warning
