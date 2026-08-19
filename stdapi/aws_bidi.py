"""Bidirectional AWS streaming: client pool, credentials, regions and failures.

Amazon Polly's speech synthesis stream and the Bedrock runtime's bidirectional
model stream are HTTP/2 *request* event streams, which botocore cannot drive.
AWS's generated async clients can, and everything the botocore path already
solved has to be solved again for them: credentials, endpoints, region choice
and failure translation. All of it lives here, so a route sees a session that
sends events, yields events, and raises ``ApiError``.

Every failure mode of these clients is a hang rather than an exception, so
nothing here ever waits on the SDK unbounded.
"""

from asyncio import CancelledError, Task, create_task, shield
from asyncio import timeout as async_timeout
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Final

from aws_sdk_bedrock_runtime.client import AsyncBedrockRuntimeClient
from aws_sdk_bedrock_runtime.config import Config as BedrockRuntimeConfig
from aws_sdk_polly.client import AsyncPollyClient
from aws_sdk_polly.config import Config as PollyConfig
from aws_sdk_transcribe_streaming.client import AsyncTranscribeStreamingClient
from aws_sdk_transcribe_streaming.config import Config as TranscribeStreamingConfig
from smithy_aws_core.identity import AWSCredentialsIdentity
from smithy_core.deserializers import DeserializeableShape
from smithy_core.serializers import SerializeableShape
from smithy_http.aio.crt import AWSCRTHTTPClient

from stdapi import server
from stdapi.api_errors import ApiError, FeatureUnavailableError
from stdapi.aws import (
    FAILOVER_ERROR_CODES,
    pooled_clients,
    verify_bidi_user_role_policy,
)
from stdapi.aws_bedrock import AWS_ERROR_MAP
from stdapi.cleanup import drain_tasks
from stdapi.config import AWS_SESSION, SETTINGS
from stdapi.exceptions import ServerError
from stdapi.monitoring import log_error_details

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from smithy_aws_core.identity import AWSIdentityProperties
    from smithy_core.aio.eventstream import DuplexEventStream
    from types_aiobotocore_bedrock.literals import RegionName

#: Services offering a bidirectional operation, with their generated client and config.
_BIDI_SERVICES: Final = {
    "polly": (AsyncPollyClient, PollyConfig),
    "bedrock-runtime": (AsyncBedrockRuntimeClient, BedrockRuntimeConfig),
    "transcribe": (AsyncTranscribeStreamingClient, TranscribeStreamingConfig),
}

#: Endpoint prefix of a bidirectional operation served apart from its own service.
_BIDI_ENDPOINT_PREFIX: Final[dict[str, str]] = {"transcribe": "transcribestreaming"}

#: Endpoint resolver, for an operation botocore hosts no client for.
_ENDPOINT_RESOLVER = AWS_SESSION.get_component("endpoint_resolver")

#: Bidirectional clients, keyed by service then region (populated at startup).
_BIDI_CLIENTS: dict[str, dict[RegionName, Any]] = {}

#: Transport every bidirectional client shares, built once at startup.
_TRANSPORT: _BidiTransport | None = None

#: Seconds allowed to close one half of a stream before it is abandoned.
_CLOSE_TIMEOUT: Final = 5.0

#: Strong references to detached close tasks, held until completion.
_CLOSE_TASKS: Final[set[Task[None]]] = set()

#: Statuses of modeled stream errors that ``AWS_ERROR_MAP`` does not name.
_STREAM_ERROR_STATUS: Final[dict[str, int]] = {
    "ConflictException": 409,
    "LexiconNotFoundException": 404,
    "ModelErrorException": 502,
    "ModelNotReadyException": 429,
    "ModelStreamErrorException": 502,
    "ModelTimeoutException": 503,
}

#: Client-facing message per status class; SDK error text is never forwarded.
_ERROR_MESSAGES: Final[dict[int, str]] = {
    400: "The request was rejected as invalid. Check the request parameters and retry.",
    401: "Unauthorized",
    403: "Forbidden",
    404: "The requested resource does not exist.",
    409: "The request conflicts with the current state of the session.",
    429: "Rate limit reached. Retry the request later.",
}

#: Message for any server-side or transport failure of a stream.
_SERVER_ERROR_MESSAGE: Final = "The request could not be completed. Retry the request."

#: Statuses meaning the gateway's own credential, not the caller's, was refused.
_DENIED_STATUSES: Final = frozenset({401, 403})

#: Feature name and stream permission per service, for a refused credential.
_BIDI_UNAVAILABLE: Final[dict[str, tuple[str, str]]] = {
    "bedrock-runtime": (
        "Live conversation",
        "bedrock:InvokeModelWithBidirectionalStream",
    ),
    "polly": ("Streaming speech synthesis", "polly:StartSpeechSynthesisStream"),
    "transcribe": ("Live transcription", "transcribe:StartStreamTranscription"),
}


class _BidiTransport(AWSCRTHTTPClient):
    """CRT transport whose per-operation copies share everything but connections.

    The SDK deep-copies the client config for every operation, and the base
    transport answers that copy by building a fresh TLS context: ~7 ms of
    blocking CPU per stream, on the event loop. Sharing one transport instead
    trades that for hung streams, because the connection pool is then shared
    too. Reusing the CRT state while isolating the pool keeps both properties.
    """

    def __deepcopy__(self, memo: Any) -> _BidiTransport:  # noqa: ANN401
        """Return the copy one operation gets.

        Args:
            memo: Objects already copied (unused: nothing here is copied deeply).

        Returns:
            A transport sharing this one's event loop, bootstrap, TLS context and
            socket options, with an empty connection pool of its own.
        """
        copy = object.__new__(type(self))
        copy.__dict__.update(
            self.__dict__, _config=deepcopy(self._config), _connections={}
        )
        return copy


class _CredentialsResolver:
    """Resolves the gateway's AWS credentials for the generated clients.

    One resolver serves the whole process: the credential chain is walked once
    and only re-frozen afterwards, which signing does per event frame.
    """

    __slots__ = ("_credentials",)

    def __init__(self) -> None:
        """Create a resolver with no credentials resolved yet."""
        self._credentials: Any = None

    async def get_identity(
        self,
        *,
        properties: AWSIdentityProperties,  # noqa: ARG002
    ) -> AWSCredentialsIdentity:
        """Return the current credentials as an SDK identity.

        Args:
            properties: Identity properties from the SDK (unused: the gateway has
                a single credential source).

        Returns:
            The identity SigV4 signs with.

        Raises:
            ApiError: No credentials could be resolved.
        """
        if (credentials := self._credentials) is None:
            credentials = self._credentials = await AWS_SESSION.get_credentials()
            if credentials is None:
                # The 401 matches a wrong API key; only the log tells the two apart.
                log_error_details(
                    "No AWS credentials could be resolved for the stream clients",
                    status=401,
                )
                msg = "Unauthorized"
                raise ApiError(msg, status=401)
        frozen = await credentials.get_frozen_credentials()
        # Only refreshable credentials expose an expiry, and only privately.
        return AWSCredentialsIdentity(
            access_key_id=frozen.access_key,
            secret_access_key=frozen.secret_key,
            session_token=frozen.token,
            account_id=frozen.account_id,
            expiration=getattr(credentials, "_expiry_time", None),
        )

    def __deepcopy__(self, memo: Any) -> _CredentialsResolver:  # noqa: ANN401
        """Return this resolver unchanged.

        Copying it succeeds but yields a second credentials object refreshing on
        its own schedule -- one silent credential chain per stream.

        Args:
            memo: Objects already copied (unused).

        Returns:
            This resolver.
        """
        return self


#: Credential bridge shared by every bidirectional client.
_CREDENTIALS_RESOLVER: Final = _CredentialsResolver()


def initialize_bidi_clients() -> None:
    """Create one bidirectional client per pooled service and region that has one.

    Called once at startup, from the lifespan, after the botocore pool is
    populated: constructing a client costs milliseconds and its transport a
    native thread, and each client targets the endpoint botocore already
    resolved for the same service and region. A region the bidirectional
    endpoint does not serve gets no client, so it is never a candidate.

    Raises:
        ServerError: A client config would carry no credentials resolver.
    """
    for service in _BIDI_SERVICES:
        clients = _BIDI_CLIENTS.setdefault(service, {})
        for region, pooled in pooled_clients(service).items():
            if region in clients:
                continue
            if (endpoint := _bidi_endpoint(service, region, pooled)) is not None:
                clients[region] = _create_client(service, region, endpoint)


def bidi_regions(service: str) -> list[RegionName]:
    """Return the regions whose bidirectional client exists, in priority order.

    Args:
        service: AWS service name (bidirectional client pool key).

    Returns:
        The regions, empty when the service has no bidirectional client at all.
    """
    return list(_BIDI_CLIENTS.get(service, {}))


def close_bidi_clients() -> None:
    """Drop every bidirectional client and the transport behind them."""
    global _TRANSPORT  # noqa: PLW0603
    _BIDI_CLIENTS.clear()
    _TRANSPORT = None


def get_bidi_client(service: str, region_name: RegionName | None = None) -> Any:  # noqa: ANN401
    """Get a bidirectional client from the pool.

    Args:
        service: AWS service name.
        region_name: Optional specific region, defaults to the first Bedrock region.

    Returns:
        The pooled client.

    Raises:
        KeyError: The service has no bidirectional client, or several regional
            clients exist and the requested region is not one of them.
    """
    clients = _BIDI_CLIENTS[service]
    try:
        return clients[region_name or SETTINGS.aws_bedrock_regions[0]]
    except KeyError:
        if len(clients) == 1:
            return next(iter(clients.values()))
        raise


def _bidi_endpoint(service: str, region: RegionName, pooled: Any) -> str | None:  # noqa: ANN401
    """Resolve the endpoint one bidirectional client targets.

    The SDK's own resolver hardcodes "amazonaws.com", absent in other
    partitions, so the endpoint comes from botocore instead: the pooled client's
    own for a service whose bidirectional operation it hosts, and botocore's
    endpoint data for one served on a separate hostname.

    Args:
        service: AWS service name.
        region: Region the client would serve.
        pooled: The botocore client of the same service and region.

    Returns:
        The endpoint URI, or None when the region does not serve the operation.
    """
    if (prefix := _BIDI_ENDPOINT_PREFIX.get(service)) is None:
        return pooled.meta.endpoint_url  # type: ignore[no-any-return]
    available = _ENDPOINT_RESOLVER.get_available_endpoints(
        prefix, partition_name=pooled.meta.partition
    )
    if region not in available:
        return None
    return (
        f"https://{_ENDPOINT_RESOLVER.construct_endpoint(prefix, region)['hostname']}"
    )


def _create_client(service: str, region: RegionName, endpoint: str) -> Any:  # noqa: ANN401
    """Build one bidirectional client.

    Args:
        service: AWS service name.
        region: Region the client serves.
        endpoint: Endpoint URI the client targets.

    Returns:
        The generated client.

    Raises:
        ServerError: The config carries no credentials resolver, which turns
            every call into a hang instead of an error.
    """
    global _TRANSPORT  # noqa: PLW0603
    if _TRANSPORT is None:
        _TRANSPORT = _BidiTransport()
    client_class, config_class = _BIDI_SERVICES[service]
    config = config_class(
        region=region,
        aws_credentials_identity_resolver=_CREDENTIALS_RESOLVER,
        endpoint_uri=endpoint,
        transport=_TRANSPORT,
        user_agent_extra=server.USER_AGENT,
    )
    if config.aws_credentials_identity_resolver is None:
        msg = f"No AWS credentials resolver for the {service} bidirectional client"
        raise ServerError(msg)
    return client_class(config=config)


def _stream_error_status(exception: BaseException) -> int:
    """Resolve the status a stream failure answers with.

    Modeled errors are mapped by class name, as botocore errors are by code;
    anything else is a transport failure.

    Args:
        exception: The failure raised by the SDK.

    Returns:
        The HTTP status the client receives for it.
    """
    if isinstance(exception, ApiError):
        return exception.status
    name = type(exception).__name__
    if (mapped := AWS_ERROR_MAP.get(name)) is not None:
        return mapped[0]
    if (modeled := _STREAM_ERROR_STATUS.get(name)) is not None:
        return modeled
    # The SDK blames the caller for its transport timeouts; they are transient.
    if getattr(exception, "is_timeout_error", False):
        return 503
    # An unmodelled failure is a transport one; a modeled one names the fault.
    return 400 if getattr(exception, "fault", None) == "client" else 503


def _stream_api_error(exception: BaseException, service: str) -> ApiError:
    """Translate a stream failure into the error a client receives.

    No SDK message is ever forwarded: they embed AWS request IDs and internal
    transport error names.

    Args:
        exception: The failure raised by the SDK.
        service: AWS service the stream was opened on.

    Returns:
        The equivalent API error.
    """
    if isinstance(exception, ApiError):
        return exception
    status = _stream_error_status(exception)
    if status in _DENIED_STATUSES and (denied := _BIDI_UNAVAILABLE.get(service)):
        # The caller was authenticated before this refusal: it is the deployment's.
        feature, permission = denied
        return FeatureUnavailableError(
            feature,
            f"AWS refused the {service} bidirectional stream "
            f"({type(exception).__name__}: {exception}); the server role needs "
            f"{permission}.",
        )
    # Withheld from the client, so the request log is the only diagnosis left.
    log_error_details(
        f"Bidirectional stream error ({type(exception).__name__}): {exception}",
        status=status,
    )
    return ApiError(_ERROR_MESSAGES.get(status, _SERVER_ERROR_MESSAGE), status=status)


def _is_stream_failover_error(exception: BaseException) -> bool:
    """Whether another region may still serve a stream this one refused.

    Args:
        exception: The failure raised while opening the stream.

    Returns:
        True for throttling, availability and transport failures, False for a
        caller error, which would be refused identically everywhere.
    """
    # These codes are regional even where their status is not, e.g. a job quota.
    if type(exception).__name__ in FAILOVER_ERROR_CODES:
        return True
    status = _stream_error_status(exception)
    return status >= 500 or status == 429


class BidiSession[IE: SerializeableShape, OE: DeserializeableShape]:
    """A live bidirectional stream: send events, iterate events, close once.

    Obtained from :func:`open_bidi_stream`, which only yields one whose output
    has resolved -- the single point at which the stream is known to be alive.
    """

    __slots__ = ("_service", "_stream", "region")

    def __init__(
        self, stream: DuplexEventStream[IE, OE, Any], region: RegionName, service: str
    ) -> None:
        """Wrap an opened SDK stream.

        Args:
            stream: The SDK's duplex event stream.
            region: Region serving it.
            service: AWS service the stream was opened on.
        """
        self._stream = stream
        self.region = region
        self._service = service

    async def await_open(self) -> None:
        """Wait for the service to answer, which is what proves the stream is alive.

        Raises:
            BaseException: Whatever the SDK reported instead of answering; the
                opening path translates it once it knows whether another region
                may still serve the request.
        """
        await self._stream.await_output()

    async def send(self, event: IE) -> None:
        """Send one event to the service.

        Args:
            event: The event to send.

        Raises:
            ApiError: The stream refused the event.
        """
        try:
            await self._stream.input_stream.send(event)
        except Exception as exception:
            raise _stream_api_error(exception, self._service) from exception

    async def close_input(self) -> None:
        """End the input half, leaving the output half streaming.

        A service waiting for the next input event has no other way to learn
        that there will not be one: Polly's own end-of-input event does not stop
        its idle timer, and the session dies with the audio still owed.

        Raises:
            ApiError: The input half could not be closed, in time or at all.
        """
        try:
            async with async_timeout(_CLOSE_TIMEOUT):
                await self._stream.input_stream.close()
        except Exception as exception:
            raise _stream_api_error(exception, self._service) from exception

    async def __aiter__(self) -> AsyncIterator[OE]:
        """Yield the service's events until the stream ends.

        Yields:
            Each event the service sends.

        Raises:
            ApiError: The stream failed; modeled errors are raised out of the
                iteration rather than yielded as events.
            ServerError: The session was never opened.
        """
        if (receiver := self._stream.output_stream) is None:
            msg = "Bidirectional stream iterated before its output was awaited"
            raise ServerError(msg)
        try:
            async for event in receiver:
                yield event
        except CancelledError:
            raise
        except Exception as exception:
            raise _stream_api_error(exception, self._service) from exception

    async def aclose(self) -> None:
        """Close both halves of the stream, bounded, ignoring their own errors.

        Never the SDK's ``close()``: it awaits the output first, so on the very
        path that needs closing most -- a stream that never answered -- it hangs.
        """
        for half in (self._stream.input_stream, self._stream.output_stream):
            if half is None:
                continue
            with suppress(Exception):
                async with async_timeout(_CLOSE_TIMEOUT):
                    await half.close()
        if self._stream.output_stream is None:
            await self._release_pending_output()

    async def _release_pending_output(self) -> None:
        """Release the output half of a stream that never resolved.

        The SDK gates the output on a task of its own, which a session closed
        before the service answered leaves running: it holds the request alive
        and its failure is reported as an unretrieved task exception, with the
        backend's own text, in the server log. Cancelling it also cancels the
        request task it awaits.
        """
        output = self._stream._output_future  # noqa: SLF001
        output.cancel()
        with suppress(Exception, CancelledError):
            async with async_timeout(_CLOSE_TIMEOUT):
                await output


@asynccontextmanager
async def open_bidi_stream[IE: SerializeableShape, OE: DeserializeableShape](
    service: str,
    regions: Sequence[RegionName],
    open_stream: Callable[[Any, RegionName], Awaitable[DuplexEventStream[IE, OE, Any]]],
    *,
    prime: Callable[[BidiSession[IE, OE]], Awaitable[None]] | None = None,
    open_timeout: float | None = None,
) -> AsyncIterator[BidiSession[IE, OE]]:
    """Open a bidirectional stream, failing over across regions, and close it.

    Args:
        service: AWS service name (bidirectional client pool key).
        regions: Candidate regions, in priority order (at least one).
        open_stream: Coroutine factory receiving the region's client and the
            region, returning the SDK's duplex stream.
        prime: Optional coroutine sending the handshake events a session needs
            before the service answers at all.
        open_timeout: Seconds allowed to open, prime and get the first answer.
            Defaults to the AWS connection timeout.

    Yields:
        The opened session, whose serving region is on ``region``.

    Raises:
        ApiError: No candidate region could open the stream, or the deployment
            requires an end user identity this kind of session cannot carry.
    """
    verify_bidi_user_role_policy(service)
    session = await _open_with_failover(
        service, regions, open_stream, prime, open_timeout
    )
    try:
        yield session
    finally:
        await _close_session(session)


async def _open_with_failover[IE: SerializeableShape, OE: DeserializeableShape](
    service: str,
    regions: Sequence[RegionName],
    open_stream: Callable[[Any, RegionName], Awaitable[DuplexEventStream[IE, OE, Any]]],
    prime: Callable[[BidiSession[IE, OE]], Awaitable[None]] | None,
    open_timeout: float | None,
) -> BidiSession[IE, OE]:
    """Open the stream on the first region that answers.

    Failover stops the moment the stream is alive: once the service answers, its
    region serves the whole session.

    Args:
        service: AWS service name.
        regions: Candidate regions, in priority order (at least one).
        open_stream: Coroutine factory receiving the region's client and the region.
        prime: Optional handshake coroutine.
        open_timeout: Seconds allowed per candidate region.

    Returns:
        The opened session.

    Raises:
        ApiError: Every candidate region failed, or one failed in a way no other
            region would serve differently.
    """
    *fallible, last_region = regions
    for region in fallible:
        try:
            return await _open_session(
                service, region, open_stream, prime, open_timeout
            )
        except Exception as exception:
            if not _is_stream_failover_error(exception):
                raise _stream_api_error(exception, service) from exception
            _log_region_failover(service, region, exception)
    try:
        return await _open_session(
            service, last_region, open_stream, prime, open_timeout
        )
    except Exception as exception:
        raise _stream_api_error(exception, service) from exception


async def _open_session[IE: SerializeableShape, OE: DeserializeableShape](
    service: str,
    region: RegionName,
    open_stream: Callable[[Any, RegionName], Awaitable[DuplexEventStream[IE, OE, Any]]],
    prime: Callable[[BidiSession[IE, OE]], Awaitable[None]] | None,
    open_timeout: float | None,
) -> BidiSession[IE, OE]:
    """Open, prime and gate one stream in one region.

    Args:
        service: AWS service name.
        region: Region to open the stream in.
        open_stream: Coroutine factory receiving the region's client and the region.
        prime: Optional handshake coroutine.
        open_timeout: Seconds allowed for the whole sequence.

    Returns:
        The opened session.

    Raises:
        BaseException: Whatever the SDK raised, with the half-open stream closed
            first; a stream that never answers raises ``TimeoutError``.
    """
    session: BidiSession[IE, OE] | None = None
    try:
        # Guarded: a region with no client is a reason to try the next, not a crash.
        client = get_bidi_client(service, region)
        async with async_timeout(
            SETTINGS.aws_connect_timeout if open_timeout is None else open_timeout
        ):
            session = BidiSession(await open_stream(client, region), region, service)
            if prime is not None:
                await prime(session)
            await session.await_open()
    except BaseException:
        if session is not None:
            await _close_session(session)
        raise
    return session


async def _close_session(session: BidiSession[Any, Any]) -> None:
    """Close a session so that a cancelled caller still releases the connection.

    The close runs in its own task and is awaited through a shield: an ``await``
    on a cancelled path is cancelled before it closes anything.

    Args:
        session: The session to close.
    """
    task = create_task(session.aclose())
    _CLOSE_TASKS.add(task)
    task.add_done_callback(_CLOSE_TASKS.discard)
    # Only this second delivery is dropped; the one that triggered it propagates.
    with suppress(CancelledError):
        await shield(task)


async def drain_stream_closes(timeout: float) -> int:  # noqa: ASYNC109 -- shared drain contract
    """Await the stream closes still releasing a connection after their caller left.

    Args:
        timeout: Seconds allowed before the unfinished closes are cancelled.

    Returns:
        Number of closes that had not finished at the deadline.
    """
    return await drain_tasks(_CLOSE_TASKS, timeout)


def _log_region_failover(
    service: str, region: RegionName, exception: BaseException
) -> None:
    """Record that a region could not open the stream, before trying the next.

    Args:
        service: AWS service name.
        region: The region that failed.
        exception: Its failure.
    """
    log_error_details(
        f"AWS {service} stream error in region {region} "
        f"({type(exception).__name__}: {exception}); "
        "failing over to the next region.",
        level="warning",
    )
