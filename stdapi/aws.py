"""AWS client management and connection pooling."""

import re
from asyncio import gather, sleep
from contextlib import AsyncExitStack, suppress
from datetime import datetime
from hashlib import blake2b
from json import dumps as _std_dumps
from logging import getLogger
from os import environ
from sys import modules
from time import monotonic
from typing import TYPE_CHECKING, Any, Final, NotRequired, Self, TypedDict

from aiobotocore.config import AioConfig
from aiobotocore.credentials import AioDeferredRefreshableCredentials
from aiohttp import ClientError as HttpClientError
from aiohttp import ClientSession, ClientTimeout
from botocore import serialize as botocore_serialize
from botocore.exceptions import BotoCoreError, ClientError
from botocore.utils import parse_timestamp
from pydantic_core import to_json

from stdapi import server
from stdapi.api_errors import (
    ACCESS_DENIED_CODES,
    DENIED_CALL_KEY,
    ApiError,
    TenantCredentialError,
    iam_action,
)
from stdapi.aws_bedrock_mantle import mantle_http_session
from stdapi.aws_sagemaker import sagemaker_http_session
from stdapi.config import AWS_REGION, AWS_SESSION, SETTINGS

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from types import TracebackType

    from botocore.model import OperationModel
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.monitoring import EventLog, TenantAwsCredential

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

#: Duration above which a successful ECS container metadata read is reported, in seconds
_ECS_METADATA_SLOW_SECONDS: Final = 1.0

#: Region of the pooled AWS STS client opening the end user role sessions
_STS_REGION: RegionName = AWS_REGION  # type: ignore[assignment]

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

    3.5x faster than the stdlib encoder on large Bedrock ``Converse`` bodies,
    for semantically identical JSON. The stdlib encoder stays as the fallback
    for input pydantic_core rejects, such as strings carrying lone surrogates.
    """

    def _serialize_body_params(self, params: Any, shape: Any) -> bytes:  # noqa: ANN401
        serialized_body = self.MAP_TYPE()
        self._serialize(serialized_body, params, shape)  # type: ignore[attr-defined]
        try:
            return to_json(serialized_body)
        except ValueError:
            return _std_dumps(serialized_body).encode(self.DEFAULT_ENCODING)


# Registry-level install: every rest-json client created afterwards gets it.
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
    parsed: dict[str, Any],
    model: OperationModel,
    context: dict[str, Any] | None = None,
    **_kwargs: object,
) -> None:
    """Record a completed AWS API call's request ID (``after-call`` hook).

    Also fires for HTTP error responses, before botocore raises the matching
    ``ClientError``, so failed calls are captured with their error code. A
    denial additionally gets the IAM action it needed and the region it was
    made in recorded on the response botocore is about to raise from, which is
    the only place both are still known: a ``ClientError`` carries neither.

    Args:
        parsed: Parsed AWS response, including ``ResponseMetadata``.
        model: Operation model of the call.
        context: Botocore request context, whose ``client_region`` is the
            region the call was made in.
        **_kwargs: Unused botocore event arguments.
    """
    if (parsed.get("Error") or {}).get("Code") in ACCESS_DENIED_CODES:
        parsed[DENIED_CALL_KEY] = {
            "action": iam_action(model),
            "region": (context or {}).get("client_region"),
        }
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
# Likewise, every client created from the session gets the fast timestamp parser.
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
            *clients: ``(service_name, region_name or None)`` tuples.
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
                },
                # Consumers are the per-end-user and per-tenant role sessions.
                *(
                    (("sts", _STS_REGION),)
                    if SETTINGS.aws_bedrock_user_role_arn
                    or SETTINGS.tenant_aws_credentials
                    else ()
                ),
            ]

            # Region-rotated services also get a single-attempt ".no-retry" pool
            # per region, so routed calls fail over without sitting through
            # botocore's own retries first.
            no_retry_specs: list[tuple[str, RegionName]] = []
            if SETTINGS.aws_bedrock_region_routing != "disabled":
                no_retry_config = AioConfig(
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
                no_retry_specs = [
                    (service, region)
                    for service in _NO_RETRY_SERVICES
                    for svc, region in specs
                    if svc == service
                ]

            # Every client pool (base, no-retry, and the Mantle HTTP session) is
            # warmed in a single wave, so startup latency is bounded by the
            # slowest client rather than by the sum of every batch.
            client_cms = [
                # New service names must join the botocore/data allowlist
                # pruned in the Dockerfile, or the image fails at runtime.
                self._exit_stack.enter_async_context(
                    AWS_SESSION.create_client(
                        service.split(".", 1)[0],
                        region_name=region,
                        config=services_configs.get(service, CONFIG),
                    )
                )
                for service, region in specs
            ] + [
                self._exit_stack.enter_async_context(
                    AWS_SESSION.create_client(  # type: ignore[call-overload]
                        service, region_name=region, config=no_retry_config
                    )
                )
                for service, region in no_retry_specs
            ]
            if SETTINGS.aws_bedrock_mantle_enabled:
                client_cms.append(
                    self._exit_stack.enter_async_context(mantle_http_session())
                )
            if SETTINGS.aws_sagemaker_endpoints:
                client_cms.append(
                    self._exit_stack.enter_async_context(sagemaker_http_session())
                )

            results = await gather(*client_cms, return_exceptions=True)
            raise_first_exception(results)

            for (service, region), client in zip(
                specs, results[: len(specs)], strict=True
            ):
                _CLIENTS.setdefault(service, {})[region] = client
            no_retry_results = results[len(specs) : len(specs) + len(no_retry_specs)]
            for (service, region), client in zip(
                no_retry_specs, no_retry_results, strict=True
            ):
                _CLIENTS.setdefault(f"{service}.no-retry", {})[region] = client
        except BaseException as exception:
            # __aenter__ raising skips __aexit__ under the context-manager
            # protocol: close whatever this attempt already entered so no
            # client leaks.
            try:
                await self._exit_stack.aclose()
            except Exception as cleanup_error:  # noqa: BLE001
                exception.add_note(
                    "Client cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            _CLIENTS.clear()
            _close_bidi_clients()
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
        _close_bidi_clients()


def _close_bidi_clients() -> None:
    """Drop the bidirectional clients built from this pool, if any were.

    Looked up in the loaded modules rather than imported: ``stdapi.aws_bidi``
    imports this one, and a teardown has nothing to drop when the pool was never
    built.
    """
    if (bidi := modules.get("stdapi.aws_bidi")) is not None:
        bidi.close_bidi_clients()


def pooled_clients(service: str) -> dict[RegionName, Any]:
    """Return every pooled client of *service*, keyed by region.

    Args:
        service: AWS service name (client pool key).

    Returns:
        The service's clients, empty when the pool holds none.
    """
    return _CLIENTS.get(service, {})


#: AWS error codes indicating a region-level issue worth failing over.
FAILOVER_ERROR_CODES: Final[frozenset[str]] = frozenset(
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
    if code in FAILOVER_ERROR_CODES or status >= 500:
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
        KeyError: If *service* has no pool at all -- the pool is built at
            start-up and cleared if any client fails, so this is a deployment
            that never finished starting rather than a missing setting -- or if
            multiple regional clients exist and the requested region is not
            among them.

    Both are deliberately loud: on a request path a missing client is a defect,
    and answering from another region instead would send a call somewhere the
    operator did not configure. A caller that must survive one -- a catalogue
    refresh, where an optional feature may not take the whole catalogue down --
    handles the ``KeyError`` itself and tells the two cases apart with
    :func:`pooled_clients`.
    """
    clients = _CLIENTS[service]
    try:
        return clients[region_name or SETTINGS.aws_bedrock_regions[0]]
    except KeyError:
        if len(clients) == 1:
            return next(iter(clients.values()))
        raise


#: Bedrock runtime operations run under the end user's own role session.
USER_ROLE_OPERATIONS: Final[frozenset[str]] = frozenset(
    {"Converse", "ConverseStream", "InvokeModel", "InvokeModelWithResponseStream"}
)


def signed_as_end_user(operation_name: str) -> bool:
    """Whether the current request signed *operation_name* as its own end user.

    Args:
        operation_name: AWS API operation the failing call invoked.

    Returns:
        True when the deployment attributes model invocations to end users and
        this request opened a session for one, so the identity AWS evaluated
        was the caller's rather than the server's.
    """
    if (
        SETTINGS.aws_bedrock_user_role_arn is None
        or operation_name not in USER_ROLE_OPERATIONS
    ):
        return False
    # Imported here: stdapi.monitoring transitively imports this module.
    from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

    log = REQUEST_LOG.get(None)
    return log is not None and bool(log.get("aws_role_session_name"))


#: Characters AWS STS rejects in a RoleSessionName (ASCII word characters, + = , . @ -).
_SESSION_NAME_INVALID_RE = re.compile(r"[^\w+=,.@-]", re.ASCII)

#: Characters AWS STS rejects in a session tag value (letters, digits, space, _ . : / = + - @).
_TAG_VALUE_INVALID_RE = re.compile(r"[^\w .:/=+\-@]")

#: Bytes of end user identifier digest appended to a session name and tag value.
_IDENTITY_DIGEST_SIZE: Final = 6

#: RoleSessionName length AWS STS accepts, minus the digest suffix and its separator.
_SESSION_NAME_PREFIX_MAX: Final = 64 - _IDENTITY_DIGEST_SIZE * 2 - 1

#: Session tag value length AWS STS accepts, minus the digest suffix and its separator.
_TAG_VALUE_PREFIX_MAX: Final = 256 - _IDENTITY_DIGEST_SIZE * 2 - 1

#: Role sessions kept per process, keyed by role session name (bounded LRU).
_USER_ROLE_CACHE: dict[str, _UserRoleCredentials] = {}

#: Maximum end users whose role session is cached; the least recently used is dropped.
_USER_ROLE_CACHE_MAX: int = 4096

#: Fraction of a role session's lifetime left when it is refreshed in the background.
_ADVISORY_REFRESH_RATIO: Final = 0.2

#: Fraction of a role session's lifetime left when a refresh becomes blocking.
_MANDATORY_REFRESH_RATIO: Final = 0.1

#: Attempts allowed to open the startup role session, whose trust policy may still propagate
_USER_ROLE_CHECK_ATTEMPTS: Final = 3

#: Delay between two startup role session attempts, in seconds
_USER_ROLE_CHECK_RETRY_DELAY: Final = 5.0

#: End user identifier of the startup check, which never invokes a model
_USER_ROLE_CHECK_IDENTITY: Final = "stdapi-ai-startup-check"

#: Client-facing message of a role session that could not be opened
_USER_ROLE_FAILURE_MESSAGE: Final = (
    "The request could not be processed for the end user it identifies. "
    "Retry the request; if the failure continues, contact the service operator."
)

#: Bidirectional service whose streams are model invocations, billed as such.
_BIDI_MODEL_SERVICE: Final = "bedrock-runtime"

#: Client-facing message of a real-time session no end user can be attributed with
_IDENTITY_UNATTRIBUTABLE_MESSAGE: Final = (
    "This service requires each request to identify the end user it is made for, "
    "which a real-time session cannot do. Use a non-realtime endpoint instead."
)

#: Client-facing message of a request that identifies no end user where one is required
_IDENTITY_REQUIRED_MESSAGE: Final = (
    "This service requires each request to identify the end user it is made for. "
    "Set 'safety_identifier' (or 'user') on the OpenAI-compatible APIs, or "
    "'metadata.user_id' on the Anthropic Messages API."
)


def user_role_session_identity(identity: str) -> tuple[str, str]:
    """Map an end user identifier to a role session name and session tag value.

    AWS STS accepts a narrower character set in a session name than in a tag
    value, and dropping the rejected characters alone would map two end users
    onto one session -- a mis-attribution in the Cost and Usage Report, and an
    authorization defect wherever a policy tests ``aws:PrincipalTag``. A digest
    of the identifier is therefore appended to both forms, which keeps them
    distinct, stable across servers, and readable for identifiers AWS accepts.

    Args:
        identity: End user identifier, of any length and character set.

    Returns:
        The RoleSessionName and the session tag value, both non-empty and
        within the limits AWS STS enforces.
    """
    digest = blake2b(identity.encode(), digest_size=_IDENTITY_DIGEST_SIZE).hexdigest()
    name = _SESSION_NAME_INVALID_RE.sub("-", identity)[:_SESSION_NAME_PREFIX_MAX].strip(
        "-"
    )
    value = _TAG_VALUE_INVALID_RE.sub("-", identity)[:_TAG_VALUE_PREFIX_MAX].strip(" -")
    return f"{name}-{digest}" if name else digest, (
        f"{value}-{digest}" if value else digest
    )


class _UserRoleCredentials(AioDeferredRefreshableCredentials):
    """An end user's role session, reopened before it expires.

    botocore reopens a session a fixed 15 minutes before it expires, which at
    the shortest session AWS STS grants means reopening one on every request.
    The windows here scale with the configured session lifetime instead.

    Attributes:
        session_name: RoleSessionName of the session, as AWS reports it.
    """

    def __init__(self, session_name: str, tag_value: str | None, duration: int) -> None:
        """Prepare an end user's role session, opened on first use.

        Args:
            session_name: RoleSessionName identifying the end user.
            tag_value: Session tag value identifying the end user, if tagged.
            duration: Session lifetime in seconds.
        """
        super().__init__(
            lambda: _assume_user_role(session_name, tag_value), "stdapi-user-role"
        )
        self.session_name = session_name
        self._advisory_refresh_timeout = int(duration * _ADVISORY_REFRESH_RATIO)
        self._mandatory_refresh_timeout = int(duration * _MANDATORY_REFRESH_RATIO)


async def _assume_user_role(session_name: str, tag_value: str | None) -> dict[str, Any]:
    """Open a role session for one end user.

    Args:
        session_name: RoleSessionName identifying the end user.
        tag_value: Session tag value identifying the end user, if tagged.

    Returns:
        The session credentials, in the form botocore refreshes from.
    """
    params: dict[str, Any] = {
        "RoleArn": SETTINGS.aws_bedrock_user_role_arn,
        "RoleSessionName": session_name,
        "DurationSeconds": SETTINGS.aws_bedrock_user_role_session_duration,
    }
    if tag_value is not None:
        params["Tags"] = [
            {"Key": SETTINGS.aws_bedrock_user_role_tag_key, "Value": tag_value}
        ]
    sts = get_client("sts", _STS_REGION)
    credentials = (await sts.assume_role(**params))["Credentials"]
    return {
        "access_key": credentials["AccessKeyId"],
        "secret_key": credentials["SecretAccessKey"],
        "token": credentials["SessionToken"],
        "expiry_time": credentials["Expiration"].isoformat(),
    }


async def user_role_credentials(identity: str) -> _UserRoleCredentials:
    """Return the credentials attributing AWS usage to *identity*.

    Sessions are cached per end user and reopened only as they approach expiry:
    AWS STS enforces an account-wide request quota, and AWS documents caching
    as a requirement of this design. A burst of first requests for one end user
    opens a single session.

    The cache is keyed by the session name rather than by the identifier it
    came from: an end user identifier is client-chosen and unbounded, so
    keeping one would bound the number of entries without bounding their size.

    Args:
        identity: End user identifier the usage is attributed to.

    Returns:
        The end user's role session credentials.

    Raises:
        ApiError: The session could not be opened, chained to nothing. The AWS
            failure reaches the request log and stops there: forwarded, its
            codes would be answered as the caller's own credentials being
            invalid, and the Bedrock region router -- which reads the cause of
            an ``ApiError`` -- would read a throttled AWS STS as a failed
            Region and re-attempt the request in every other one.
    """
    session_name, tag_value = user_role_session_identity(identity)
    credentials = _USER_ROLE_CACHE.pop(session_name, None)
    if credentials is None:
        if len(_USER_ROLE_CACHE) >= _USER_ROLE_CACHE_MAX:
            del _USER_ROLE_CACHE[next(iter(_USER_ROLE_CACHE))]
        credentials = _UserRoleCredentials(
            session_name,
            tag_value if SETTINGS.aws_bedrock_user_role_tag_key else None,
            SETTINGS.aws_bedrock_user_role_session_duration,
        )
    # Re-inserted last, so the least recently used end user is the one dropped.
    _USER_ROLE_CACHE[session_name] = credentials
    try:
        await credentials.get_frozen_credentials()
    except (BotoCoreError, ClientError, RuntimeError) as exception:
        # Dropped so the next request retries with a session of its own.
        _USER_ROLE_CACHE.pop(session_name, None)
        from stdapi.monitoring import REQUEST_LOG, log_error_details  # noqa: PLC0415

        if REQUEST_LOG.get(None) is not None:
            log_error_details(
                f"Opening the end user role session failed "
                f"({type(exception).__name__}: {exception})",
                level="error",
            )
        raise ApiError(_USER_ROLE_FAILURE_MESSAGE, status=503) from None
    return credentials


def clear_user_role_cache() -> None:
    """Drop every cached end user role session."""
    _USER_ROLE_CACHE.clear()


async def request_user_role_credentials() -> _UserRoleCredentials | None:
    """Return the credentials the current request's model invocations run under.

    Returns:
        The end user's role session credentials, or None to keep the server's
        own identity -- when the feature is disabled, when the call is the
        server's own rather than a client request, or when the request
        identifies no end user and none is required.

    Raises:
        ApiError: The request identifies no end user where the configuration
            requires one, or the end user's session could not be opened.
    """
    if SETTINGS.aws_bedrock_user_role_arn is None:
        return None
    # Imported here: stdapi.monitoring transitively imports this module.
    from stdapi.monitoring import REQUEST_LOG, resolve_request_identity  # noqa: PLC0415

    try:
        identity = resolve_request_identity()
    except LookupError:
        # Outside any request: the server's own call, attributed to itself.
        return None
    if not identity:
        if SETTINGS.aws_bedrock_user_role_require_identity:
            raise ApiError(_IDENTITY_REQUIRED_MESSAGE, status=400)
        return None
    credentials = await user_role_credentials(identity)
    REQUEST_LOG.get()["aws_role_session_name"] = credentials.session_name
    return credentials


def verify_user_role_identity() -> None:
    """Refuse an invocation that names no end user, off the botocore signing path.

    :func:`request_user_role_credentials` enforces the requirement from the
    botocore signing hook, which the transports of their own -- Bedrock Mantle,
    and an Amazon SageMaker AI endpoint -- never reach: they sign with the
    server's own credentials. The requirement is therefore checked here, so the
    documented ``400`` does not depend on which endpoint happens to serve the
    model. What it cannot restore is the per-end-user role itself: an
    identified request still runs under the server's identity there, which is
    why routing a dual-homed model to Mantle alongside this setting is refused
    at startup. A SageMaker AI endpoint has no such alternative to fall back
    to, so it is served under the server's identity instead.

    Raises:
        ApiError: The request identifies no end user where one is required.
    """
    if (
        SETTINGS.aws_bedrock_user_role_arn is None
        or not SETTINGS.aws_bedrock_user_role_require_identity
    ):
        return
    # Imported here: stdapi.monitoring transitively imports this module.
    from stdapi.monitoring import resolve_request_identity  # noqa: PLC0415

    try:
        identity = resolve_request_identity()
    except LookupError:
        # Outside any request: the server's own call, attributed to itself.
        return
    if not identity:
        raise ApiError(_IDENTITY_REQUIRED_MESSAGE, status=400)


def verify_bidi_user_role_policy(service: str) -> None:
    """Refuse a real-time model session the request's signing policy cannot cover.

    A bidirectional stream is signed as it opens, from the server's own
    credentials, and keeps that identity for its whole life: unlike a request,
    it has no point where another session can be substituted. A deployment
    that requires every model invocation to name its end user, or a tenant
    whose invocations must run under its own AWS account, is therefore served
    by refusing the session rather than by silently signing (and billing) it
    as the server. Streams of every other service are unaffected: they are not
    model invocations.

    Args:
        service: AWS service name of the stream about to open.

    Raises:
        ApiError: The deployment requires an end user identity, or the API key
            carries an AWS credential, that a real-time model session cannot
            be signed with.
    """
    if service != _BIDI_MODEL_SERVICE:
        return
    # Imported here: stdapi.monitoring transitively imports this module.
    from stdapi.monitoring import tenant_aws_credential  # noqa: PLC0415

    if tenant_aws_credential() is not None:
        raise ApiError(TENANT_REALTIME_MESSAGE, status=400)
    if (
        SETTINGS.aws_bedrock_user_role_arn is not None
        and SETTINGS.aws_bedrock_user_role_require_identity
    ):
        raise ApiError(_IDENTITY_UNATTRIBUTABLE_MESSAGE, status=400)


#: Lifetime of a tenant role session; the one-hour ceiling role chaining imposes.
_TENANT_SESSION_DURATION: Final = 3600

#: Tenant role sessions kept per process, keyed by a credential digest (bounded LRU).
_TENANT_ROLE_CACHE: dict[str, _TenantRoleCredentials] = {}

#: Maximum tenants whose role session is cached; the least recently used is dropped.
_TENANT_ROLE_CACHE_MAX: int = 4096

# Deliberate exception to the no-AWS-details register of the sibling messages
# above: these reach whoever holds a key that registered an AWS credential, and
# every detail they name -- the account, the role, its trust policy -- is the
# tenant's own resource, never this deployment's.
#: Client-facing message when the tenant's registered role cannot be assumed.
TENANT_CREDENTIAL_FAILURE_MESSAGE: Final = (
    "The AWS credential registered for this API key could not be used. "
    "Ask your administrator to verify the role and its trust policy."
)

#: Client-facing message when the tenant's own AWS account denies an invocation.
TENANT_ACCESS_DENIED_MESSAGE: Final = (
    "Your AWS account does not have access to this model. Grant the role "
    "registered for this API key access to it, or select another model."
)

#: Client-facing message when the tenant's role session could not be opened now.
TENANT_SESSION_UNAVAILABLE_MESSAGE: Final = (
    "The AWS credential registered for this API key could not be used right "
    "now. Retry the request; if the failure continues, contact the service "
    "operator."
)

#: AWS error codes naming the tenant's own role registration as the fault.
_TENANT_ROLE_FAULT_CODES: Final = frozenset(
    {
        "AccessDenied",
        "AccessDeniedException",
        "MalformedPolicyDocument",
        "MalformedPolicyDocumentException",
        "InvalidParameterValue",
        "NoSuchEntity",
        "ValidationError",
    }
)

#: Client-facing message of a real-time session under a tenant AWS credential.
TENANT_REALTIME_MESSAGE: Final = (
    "This API key carries an AWS credential of its own, which a real-time "
    "session cannot be signed with. Use a non-realtime endpoint instead."
)


class _TenantRoleCredentials(AioDeferredRefreshableCredentials):
    """A tenant's cross-account role session, reopened before it expires.

    Attributes:
        key_id: Tenant key the session belongs to, for targeted eviction.
    """

    def __init__(self, key_id: str, credential: TenantAwsCredential) -> None:
        """Prepare a tenant's role session, opened on first use.

        Args:
            key_id: Tenant key the credential is registered against.
            credential: The registered cross-account role and ExternalId.
        """
        super().__init__(
            lambda: _assume_tenant_role(key_id, credential), "stdapi-tenant-role"
        )
        self.key_id = key_id
        self._advisory_refresh_timeout = int(
            _TENANT_SESSION_DURATION * _ADVISORY_REFRESH_RATIO
        )
        self._mandatory_refresh_timeout = int(
            _TENANT_SESSION_DURATION * _MANDATORY_REFRESH_RATIO
        )


async def _assume_tenant_role(
    key_id: str, credential: TenantAwsCredential
) -> dict[str, Any]:
    """Open a role session in one tenant's own AWS account.

    The server-minted ExternalId is presented on every call: the tenant's
    trust policy requires it, which is what stops another customer who learned
    the role ARN from routing this deputy at it (AWS confused-deputy pattern).

    Args:
        key_id: Tenant key the credential is registered against.
        credential: The registered cross-account role and ExternalId.

    Returns:
        The session credentials, in the form botocore refreshes from.
    """
    sts = get_client("sts", _STS_REGION)
    credentials = (
        await sts.assume_role(
            RoleArn=credential.role_arn,
            RoleSessionName=f"stdapi-ai-tenant-{key_id}",
            ExternalId=credential.external_id,
            DurationSeconds=_TENANT_SESSION_DURATION,
        )
    )["Credentials"]
    return {
        "access_key": credentials["AccessKeyId"],
        "secret_key": credentials["SecretAccessKey"],
        "token": credentials["SessionToken"],
        "expiry_time": credentials["Expiration"].isoformat(),
    }


def _tenant_session_key(key_id: str, credential: TenantAwsCredential) -> str:
    """Return the cache key of one tenant credential's role session.

    Args:
        key_id: Tenant key the credential is registered against.
        credential: The registered cross-account role and ExternalId.

    Returns:
        A fixed-size digest, so a rotated role or ExternalId opens a fresh
        session instead of reusing the previous one's.
    """
    material = f"{key_id}\0{credential.role_arn}\0{credential.external_id}"
    return blake2b(material.encode(), digest_size=16).hexdigest()


async def tenant_role_credentials(
    key_id: str, credential: TenantAwsCredential
) -> _TenantRoleCredentials:
    """Return the session credentials of one tenant's registered role.

    Sessions are cached per credential and reopened as they approach expiry,
    so a tenant costs one AWS STS call per hour, not one per request.

    Args:
        key_id: Tenant key the credential is registered against.
        credential: The registered cross-account role and ExternalId.

    Returns:
        The tenant's role session credentials.

    Raises:
        TenantCredentialError: The registration itself is at fault -- trust
            revoked, ExternalId mismatch, role deleted. 403 with a fixed
            message the tenant can act on, naming none of the failure's AWS
            details.
        ApiError: AWS STS could not be reached, throttled the call, or this
            deployment's own session is gone. 503, because none of that is the
            tenant's registration and an SDK retries a 503 where it would give
            up on a 403.

    Both are chained to nothing, so no AWS code or ARN can leak and the Bedrock
    region router never reads the AWS STS failure as a failed region.
    """
    session_key = _tenant_session_key(key_id, credential)
    credentials = _TENANT_ROLE_CACHE.pop(session_key, None)
    if credentials is None:
        if len(_TENANT_ROLE_CACHE) >= _TENANT_ROLE_CACHE_MAX:
            del _TENANT_ROLE_CACHE[next(iter(_TENANT_ROLE_CACHE))]
        credentials = _TenantRoleCredentials(key_id, credential)
    # Re-inserted last, so the least recently used tenant is the one dropped.
    _TENANT_ROLE_CACHE[session_key] = credentials
    try:
        await credentials.get_frozen_credentials()
    except (BotoCoreError, ClientError, RuntimeError) as exception:
        # Dropped so the next request retries with a session of its own.
        _TENANT_ROLE_CACHE.pop(session_key, None)
        from stdapi.monitoring import REQUEST_LOG, log_error_details  # noqa: PLC0415

        # Only the codes AWS returns for the registration itself are the
        # tenant's fault. A throttled or unreachable AWS STS, and this
        # deployment's own session expiring, are this deployment's problem and
        # clear on their own: answering those with the tenant's 403 sends it
        # auditing a trust policy that is fine, and stops its SDK retrying.
        registration = (
            isinstance(exception, ClientError)
            and exception.response["Error"]["Code"] in _TENANT_ROLE_FAULT_CODES
        )
        if REQUEST_LOG.get(None) is not None:
            log_error_details(
                f"Opening the role session of tenant key '{key_id}' failed "
                f"({type(exception).__name__}: {exception})",
                level="warning" if registration else "error",
            )
        if registration:
            raise TenantCredentialError(TENANT_CREDENTIAL_FAILURE_MESSAGE) from None
        raise ApiError(TENANT_SESSION_UNAVAILABLE_MESSAGE, status=503) from None
    return credentials


def drop_tenant_sessions(key_id: str) -> None:
    """Drop one tenant's cached role sessions, so the next request reopens them.

    Args:
        key_id: Tenant key whose sessions must be dropped.
    """
    for session_key in [
        session_key
        for session_key, credentials in _TENANT_ROLE_CACHE.items()
        if credentials.key_id == key_id
    ]:
        _TENANT_ROLE_CACHE.pop(session_key, None)


def clear_tenant_role_cache() -> None:
    """Drop every cached tenant role session."""
    _TENANT_ROLE_CACHE.clear()


def signed_as_tenant(operation_name: str) -> bool:
    """Whether the current request signed *operation_name* with its tenant's credential.

    Args:
        operation_name: AWS API operation the failing call invoked.

    Returns:
        True when the request's tenant registered an AWS credential and this
        invocation was signed with it, so the identity AWS evaluated belongs
        to the tenant's own account rather than to this deployment.
    """
    if (
        not SETTINGS.tenant_aws_credentials
        or operation_name not in USER_ROLE_OPERATIONS
    ):
        return False
    # Imported here: stdapi.monitoring transitively imports this module.
    from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

    log = REQUEST_LOG.get(None)
    return log is not None and bool(log.get("aws_tenant_key_id"))


async def request_signing_credentials() -> AioDeferredRefreshableCredentials | None:
    """Return the credentials the current request's model invocations run under.

    The tenant's registered credential wins over the per-end-user role: the
    first moves the spend to another AWS account, the second only attributes
    it within this one.

    Returns:
        The tenant's or the end user's session credentials, or None to keep
        the server's own identity.

    Raises:
        ApiError: The applicable session could not be opened, or the request
            identifies no end user where the configuration requires one.
    """
    # Imported here: stdapi.monitoring transitively imports this module.
    from stdapi.monitoring import (  # noqa: PLC0415
        REQUEST_LOG,
        TENANT,
        tenant_aws_credential,
    )

    if (tenant := TENANT.get()) is not None and (
        credential := tenant_aws_credential()
    ) is not None:
        credentials = await tenant_role_credentials(tenant.key_id, credential)
        if (log := REQUEST_LOG.get(None)) is not None:
            log["aws_tenant_key_id"] = tenant.key_id
        return credentials
    return await request_user_role_credentials()


async def _set_request_credentials(context: dict[str, Any]) -> None:
    """Sign the request being built as its tenant or end user, if it has one.

    Args:
        context: The request context botocore signs from.
    """
    if (credentials := await request_signing_credentials()) is not None:
        context.setdefault("signing", {})["request_credentials"] = credentials


def _sign_model_invocation(
    model: OperationModel, context: dict[str, Any], **_kwargs: object
) -> Awaitable[None] | None:
    """Sign a Bedrock model invocation per request (``before-parameter-build``).

    Signs as the tenant's registered AWS credential when the request's API key
    carries one, else as the request's end user when the deployment attributes
    usage per end user. Registered once for the Bedrock runtime, so every
    other AWS service keeps the server's own identity by construction. Returns
    the coroutine doing the work instead of being one, so a deployment without
    either feature pays only this comparison.

    Args:
        model: Operation model of the call.
        context: Request context, which botocore later signs from.
        **_kwargs: Unused botocore event arguments.

    Returns:
        The awaitable resolving the request's credentials, or None to sign
        with the server's own.
    """
    if (
        SETTINGS.aws_bedrock_user_role_arn is None
        and not SETTINGS.tenant_aws_credentials
    ) or model.name not in USER_ROLE_OPERATIONS:
        return None
    return _set_request_credentials(context)


# On the shared session, so the ".no-retry" pool signs invocations the same way.
AWS_SESSION.register(
    "before-parameter-build.bedrock-runtime",
    _sign_model_invocation,
    unique_id="stdapi-user-role",
)


async def verify_user_role_access(start_event: EventLog) -> None:
    """Open a role session at startup, so a broken configuration surfaces there.

    A role created moments earlier stays unassumable for a few seconds, so the
    check is retried before it is believed. It is reported and not fatal: a
    server refusing to start would turn a slow IAM propagation into an outage,
    while every request that needs the session still fails closed on its own.

    Args:
        start_event: Startup log event a failure is reported on.
    """
    if SETTINGS.aws_bedrock_user_role_arn is None:
        return
    from stdapi.monitoring import add_server_warning  # noqa: PLC0415

    session_name, tag_value = user_role_session_identity(_USER_ROLE_CHECK_IDENTITY)
    failure: Exception | None = None
    for attempt in range(_USER_ROLE_CHECK_ATTEMPTS):
        if attempt:
            await sleep(_USER_ROLE_CHECK_RETRY_DELAY)
        try:
            await _assume_user_role(
                session_name,
                tag_value if SETTINGS.aws_bedrock_user_role_tag_key else None,
            )
        except (BotoCoreError, ClientError) as exception:
            failure = exception
        else:
            return
    add_server_warning(
        start_event,
        "Per-end-user cost attribution is configured but its role could not be "
        f"assumed ({type(failure).__name__}: {failure}): requests will fail until "
        "the role's trust policy allows this server to call sts:AssumeRole and "
        "sts:TagSession on it",
    )


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
    # No ``trust_env``: the endpoint is a link-local address on the task's own
    # host, so a configured proxy could never reach it.
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
        A warning message when the metadata endpoint could not be reached, or
        when reading it delayed startup; ``None`` otherwise.
    """
    try:
        metadata_path = environ["ECS_CONTAINER_METADATA_URI_V4"]
    except KeyError:
        await _set_account_id_from_sts()
        return None
    warning = None
    started = monotonic()
    for attempt in range(_ECS_METADATA_ATTEMPTS):
        if attempt:
            await sleep(_ECS_METADATA_RETRY_DELAY)
        try:
            await _set_account_info_from_ecs(metadata_path)
        except (HttpClientError, TimeoutError) as exception:
            warning = (
                "ECS container metadata endpoint unreachable after "
                f"{_ECS_METADATA_ATTEMPTS} attempts in "
                f"{monotonic() - started:.1f} s ({type(exception).__name__}): "
                "the server name does not identify the ECS task"
            )
        else:
            # A slow success is invisible otherwise: this runs alone before the
            # startup fan-out, so its whole duration is added to startup time.
            elapsed = monotonic() - started
            if attempt or elapsed >= _ECS_METADATA_SLOW_SECONDS:
                attempts = attempt + 1
                return (
                    f"ECS container metadata endpoint answered after {attempts} "
                    f"attempt{'s' if attempts > 1 else ''} in {elapsed:.1f} s: "
                    "server startup was delayed by that much"
                )
            return None
    await _set_account_id_from_sts()
    return warning
