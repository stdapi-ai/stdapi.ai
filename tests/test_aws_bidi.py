"""Bidirectional AWS streams: client pool, credentials, region choice, failures.

Every failure mode of the generated async AWS clients is a *hang* rather than an
exception -- opening a stream succeeds against an unresolvable host, a rejected
request only surfaces when the output is awaited, and the SDK's own ``close()``
awaits that same output before closing anything. This module pins the bounds and
the translation that keep those out of a route, with a fake duplex stream that
the streaming audio features reuse instead of talking to AWS.

Ref: https://docs.aws.amazon.com/polly/latest/APIReference/API_StartSpeechSynthesisStream.html
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModelWithBidirectionalStream.html
     stdapi/aws_bidi.py
"""

from __future__ import annotations

from asyncio import CancelledError, Event, Future, Task, create_task, sleep, wait
from asyncio import timeout as async_timeout
from copy import deepcopy
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from aws_sdk_bedrock_runtime.models import (
    AccessDeniedException,
    ModelErrorException,
    ModelNotReadyException,
    ModelStreamErrorException,
    ModelTimeoutException,
    ThrottlingException,
    ValidationException,
)
from aws_sdk_polly.models import InvalidSsmlException, ServiceFailureException
from smithy_core.exceptions import CallError, ClientTimeoutError, SmithyError

import stdapi.aws
import stdapi.aws_bidi
from stdapi.api_errors import ApiError
from stdapi.aws_bidi import (
    BidiSession,
    close_bidi_clients,
    get_bidi_client,
    initialize_bidi_clients,
    open_bidi_stream,
)
from stdapi.config import SETTINGS
from stdapi.exceptions import ServerError
from stdapi.server import USER_AGENT

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from types_aiobotocore_bedrock.literals import RegionName

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Endpoint of a partition whose DNS suffix is not "amazonaws.com".
_EUSC_ENDPOINT = "https://polly.eusc-de-east-1.amazonaws.eu"

#: Message of the handshake failure the failover tests script.
_HANDSHAKE_ERROR = "The handshake failed."

#: Its region, which the Bedrock region literals do not name.
_EUSC_REGION: RegionName = "eusc-de-east-1"  # type: ignore[assignment]

#: What ``AWSEventPublisher`` raises on a send after its close.
_CLOSED_STREAM_ERROR = "Attempted to write to closed stream."

#: A per-end-user role, to put a deployment in scope of the attribution policy.
_USER_ROLE_ARN = "arn:aws:iam::123456789012:role/stdapi-end-user"

#: Event loop iterations a detached close is given to complete.
_DRAIN_ITERATIONS = 10

#: Bound the tests pass or patch in, instead of waiting out the real one.
_SHORT_TIMEOUT = 0.05

#: Seconds a test waits before calling a bounded operation unbounded.
_TEST_TIMEOUT = 5.0

#: A bound the test expects nothing to reach.
_UNREACHED_TIMEOUT = 30


class LimitExceededException(CallError):  # noqa: N818
    """Stand-in for a per-region quota error that no status table names.

    Named after the service's own error, because the class name is the key the
    translation looks up.

    The generated clients model one exception class per service error; this is
    the shape of the shared AWS codes that reach a stream, which the services
    report as the caller's fault even though another region may serve them.
    """


class FakeEventPublisher:
    """Input half of a fake duplex stream, recording what was sent.

    Closing follows ``AWSEventPublisher``: it is idempotent, and a send after it
    is an ``OSError``, which is how a driver racing one last event past the
    half-close fails in production.
    """

    def __init__(
        self, error: Exception | None = None, *, close_hangs: bool = False
    ) -> None:
        """Record sends, raising *error* on every one of them when given."""
        self.sent: list[Any] = []
        self.closed = 0
        self.release_close = Event()
        self._error = error
        self._close_hangs = close_hangs
        self._is_closed = False

    async def send(self, event: Any) -> None:  # noqa: ANN401
        """Record an event, or raise the scripted send error."""
        if self._is_closed:
            raise OSError(_CLOSED_STREAM_ERROR)
        if self._error is not None:
            raise self._error
        self.sent.append(event)

    async def close(self) -> None:
        """Count one close, ignoring every later one, or stall until released."""
        if self._is_closed:
            return
        if self._close_hangs:
            await self.release_close.wait()
        # The real close awaits a signed empty frame, so it is a suspension point.
        await sleep(0)
        self._is_closed = True
        self.closed += 1


class FakeEventReceiver:
    """Output half of a fake duplex stream, replaying scripted events."""

    def __init__(
        self,
        events: Sequence[Any] = (),
        error: BaseException | None = None,
        *,
        close_hangs: bool = False,
    ) -> None:
        """Replay *events*, then raise *error* once when given, then end."""
        self._events = list(events)
        self._error = error
        self._close_hangs = close_hangs
        self._is_closed = False
        self.release_close = Event()
        self.closed = 0

    async def receive(self) -> Any:  # noqa: ANN401
        """Return the next event, raise the scripted error, or end the stream."""
        if self._events:
            return self._events.pop(0)
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        return None

    async def __anext__(self) -> Any:  # noqa: ANN401
        """Yield the next event, closing and stopping when none is left."""
        if (event := await self.receive()) is None:
            await self.close()
            raise StopAsyncIteration
        return event

    def __aiter__(self) -> FakeEventReceiver:
        """Return self, as the SDK's receiver protocol does."""
        return self

    async def close(self) -> None:
        """Count one close, ignoring every later one, or stall until released."""
        if self._is_closed:
            return
        if self._close_hangs:
            await self.release_close.wait()
        self._is_closed = True
        self.closed += 1


class FakeDuplexStream:
    """Stand-in for the SDK's ``DuplexEventStream``.

    Reproduces the three behaviours the wrapper exists to contain: ``output_stream``
    stays ``None`` until ``await_output()`` resolves, the output is gated on a task
    of the SDK's own that outlives an abandoned open, and ``close()`` awaits that
    output first -- which is why calling it on a stream that never answered hangs
    forever.
    """

    def __init__(
        self,
        *,
        events: Sequence[Any] = (),
        receive_error: BaseException | None = None,
        send_error: Exception | None = None,
        open_error: Exception | None = None,
        never_answers: bool = False,
        close_hangs: bool = False,
    ) -> None:
        """Script one stream: what it sends back, and how it fails."""
        self.input_stream = FakeEventPublisher(send_error, close_hangs=close_hangs)
        self.output_stream: FakeEventReceiver | None = None
        self._receiver = FakeEventReceiver(
            events, receive_error, close_hangs=close_hangs
        )
        self._open_error = open_error
        self._never_answers = never_answers
        self.sdk_close_calls = 0
        self.output_task: Task[tuple[object, FakeEventReceiver]] | None = None

    @property
    def _output_future(self) -> Task[tuple[object, FakeEventReceiver]]:
        """The task the SDK gates the output on, as ``duplex_stream`` creates it.

        The real one exists from the moment the operation is opened; creating it
        on first use keeps the fake constructible outside a running loop.
        """
        if self.output_task is None:
            self.output_task = create_task(self._resolve_output())
        return self.output_task

    async def _resolve_output(self) -> tuple[object, FakeEventReceiver]:
        """Answer the initial response, raise the scripted error, or never answer."""
        if self._never_answers:
            # Never resolves, exactly as an unanswered stream does not.
            await Future()
        if self._open_error is not None:
            raise self._open_error
        return object(), self._receiver

    async def await_output(self) -> tuple[object, FakeEventReceiver]:
        """Resolve the initial response through the output task, as the SDK does."""
        output, self.output_stream = await self._output_future
        return output, self.output_stream

    async def close(self) -> None:
        """Reproduce the SDK's close: it awaits the output before closing."""
        self.sdk_close_calls += 1
        if self.output_stream is None:
            await self.await_output()
        await self.input_stream.close()
        await self._receiver.close()

    def release_closes(self) -> None:
        """Let a stalled close of either half finish.

        A close that stalls is only released by the transport, so a test that
        scripted one releases it before leaving the loop with it still pending.
        """
        self.input_stream.release_close.set()
        self._receiver.release_close.set()


class FakeCredentials:
    """Stand-in for an aiobotocore refreshable credentials object."""

    def __init__(self, expiry: datetime | None = None) -> None:
        """Hold a private expiry, as refreshable credentials do."""
        self.frozen_calls = 0
        self._expiry_time = expiry

    async def get_frozen_credentials(self) -> Any:  # noqa: ANN401
        """Return a frozen credentials tuple, as aiobotocore does -- awaitable."""
        from botocore.credentials import ReadOnlyCredentials  # noqa: PLC0415

        self.frozen_calls += 1
        return ReadOnlyCredentials("AKIAEXAMPLE", "secret", "token", "123456789012")


class FakeSession:
    """Stand-in for the shared aiobotocore session."""

    def __init__(self, credentials: FakeCredentials | None) -> None:
        """Answer ``get_credentials`` with *credentials*."""
        self.credentials = credentials
        self.calls = 0

    async def get_credentials(self) -> FakeCredentials | None:
        """Return the session credentials, as aiobotocore does -- awaitable."""
        self.calls += 1
        return self.credentials


class FakePooledClient:
    """Stand-in for a pooled botocore client, carrying only its endpoint."""

    class _Meta:
        """The ``meta`` a botocore client exposes."""

        def __init__(self, endpoint_url: str) -> None:
            """Carry the resolved endpoint."""
            self.endpoint_url = endpoint_url

    def __init__(self, endpoint_url: str) -> None:
        """Expose *endpoint_url* the way a pooled client does."""
        self.meta = self._Meta(endpoint_url)


@pytest.fixture
def botocore_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, dict[RegionName, FakePooledClient]]]:
    """Isolate the bidirectional pool and expose the botocore pool it is built from."""
    pool: dict[str, dict[RegionName, FakePooledClient]] = {}
    monkeypatch.setattr(stdapi.aws_bidi, "_BIDI_CLIENTS", {})
    monkeypatch.setattr(
        stdapi.aws_bidi, "pooled_clients", lambda service: pool.get(service, {})
    )
    yield pool
    close_bidi_clients()


def _pool(
    botocore_pool: dict[str, dict[RegionName, FakePooledClient]],
    service: str,
    *regions: RegionName,
    endpoint: str | None = None,
) -> None:
    """Declare *service* pooled in *regions*, as the startup pool would."""
    botocore_pool[service] = {
        region: FakePooledClient(
            endpoint or f"https://{service}.{region}.amazonaws.com"
        )
        for region in regions
    }


def _stream_opener(
    stream: FakeDuplexStream, opened: list[RegionName] | None = None
) -> Any:  # noqa: ANN401
    """Build an opener returning *stream* and recording the regions it was asked for."""

    async def _open(_client: Any, region: RegionName) -> FakeDuplexStream:  # noqa: ANN401
        if opened is not None:
            opened.append(region)
        return stream

    return _open


async def _drain_close_tasks() -> None:
    """Let the detached close tasks finish, as a real event loop would."""
    for _ in range(_DRAIN_ITERATIONS):
        if not stdapi.aws_bidi._CLOSE_TASKS:  # noqa: SLF001
            return
        await sleep(0)


def _per_region_opener(opened: list[FakeDuplexStream], **script: Any) -> Any:  # noqa: ANN401
    """Build an opener handing each region its own stream, collected in *opened*.

    Every region opens a stream of its own in production, so a test whose first
    region fails must not hand the second one the stream already abandoned.
    """

    async def _open(_client: Any, _region: RegionName) -> FakeDuplexStream:  # noqa: ANN401
        stream = FakeDuplexStream(**script)
        opened.append(stream)
        return stream

    return _open


def _failing_opener(errors: dict[str, Exception], stream: FakeDuplexStream) -> Any:  # noqa: ANN401
    """Build an opener raising the per-region error, else returning *stream*."""

    async def _open(_client: Any, region: RegionName) -> FakeDuplexStream:  # noqa: ANN401
        if (error := errors.get(region)) is not None:
            raise error
        return stream

    return _open


class TestTransportCopy:
    """The transport a bidirectional operation gets is a copy, and copies are not free.

    Every operation deep-copies the client config, and the SDK's own transport copy
    builds a fresh CRT TLS context -- measured at ~7 ms of blocking CPU per stream,
    on the event loop, which the performance rules forbid on a request path. Sharing
    one transport instead is worse: it shares the connection pool, and half of the
    concurrent streams then never answer. The copy must therefore share everything
    expensive and isolate the connection pool alone.

    Ref: stdapi/aws_bidi.py:_BidiTransport
    """

    def test_operation_copy_shares_the_expensive_crt_state(self) -> None:
        """The per-operation copy reuses the TLS context, event loop and bootstrap.

        Rebuilding any of them is what costs the blocking milliseconds; the TLS
        context alone is ~7 ms.
        """
        transport = stdapi.aws_bidi._BidiTransport()  # noqa: SLF001

        copy = deepcopy(transport)

        assert copy is not transport
        assert copy._tls_ctx is transport._tls_ctx  # noqa: SLF001
        assert copy._eventloop is transport._eventloop  # noqa: SLF001
        assert copy._client_bootstrap is transport._client_bootstrap  # noqa: SLF001
        assert copy._socket_options is transport._socket_options  # noqa: SLF001

    def test_operation_copy_gets_its_own_connection_pool(self) -> None:
        """Each copy starts with an empty, private connection dict.

        Sharing the pool is what makes concurrent bidirectional streams hang, so
        the copy must never see the prototype's connections.
        """
        transport = stdapi.aws_bidi._BidiTransport()  # noqa: SLF001
        connection: Any = object()
        transport._connections[  # noqa: SLF001
            ("https", "polly.us-east-1.amazonaws.com", 443)
        ] = connection

        copy = deepcopy(transport)

        assert copy._connections == {}  # noqa: SLF001
        assert copy._connections is not transport._connections  # noqa: SLF001


class TestClientConstruction:
    """Bidirectional clients are built at startup, from the pooled client's endpoint.

    The SDK's endpoint resolver hardcodes the ``amazonaws.com`` DNS suffix, which
    resolves to a nonexistent host in the European Sovereign Cloud and in China.
    Borrowing the endpoint botocore already resolved for the same service and
    region is what keeps every partition working.

    Ref: stdapi/aws_bidi.py:initialize_bidi_clients
         stdapi/aws.py:pooled_clients
    """

    def test_client_targets_the_pooled_botocore_endpoint(
        self, botocore_pool: dict[str, dict[RegionName, FakePooledClient]]
    ) -> None:
        """The pooled client's endpoint URL becomes the bidirectional client's endpoint.

        A sovereign-partition endpoint proves the SDK default was not used: its
        resolver would have produced ``polly.eusc-de-east-1.amazonaws.com``.
        """
        _pool(botocore_pool, "polly", _EUSC_REGION, endpoint=_EUSC_ENDPOINT)

        initialize_bidi_clients()

        config = get_bidi_client("polly", _EUSC_REGION)._config  # noqa: SLF001
        assert config.endpoint_uri == _EUSC_ENDPOINT
        assert config.region == _EUSC_REGION

    def test_client_carries_the_server_user_agent(
        self, botocore_pool: dict[str, dict[RegionName, FakePooledClient]]
    ) -> None:
        """The gateway's user agent rides along, partner tag included.

        The SDK appends it verbatim, so the AWS Partner Network tag that identifies
        this product on every other call is kept on this one.
        """
        _pool(botocore_pool, "polly", "us-east-1")

        initialize_bidi_clients()

        client = get_bidi_client("polly", "us-east-1")
        assert client._config.user_agent_extra == USER_AGENT  # noqa: SLF001

    def test_config_without_an_identity_resolver_is_refused(
        self,
        botocore_pool: dict[str, dict[RegionName, FakePooledClient]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A config missing the credentials resolver fails loudly at startup.

        Without it SigV4 has no credentials and the *call* hangs forever, with the
        identity error raised inside a task nobody awaits: a startup failure is the
        only way that becomes diagnosable.
        """
        _pool(botocore_pool, "polly", "us-east-1")
        monkeypatch.setattr(stdapi.aws_bidi, "_CREDENTIALS_RESOLVER", None)

        with pytest.raises(ServerError, match="credentials"):
            initialize_bidi_clients()

    def test_every_client_signs_with_the_one_process_resolver(
        self, botocore_pool: dict[str, dict[RegionName, FakePooledClient]]
    ) -> None:
        """All regional clients of both services carry the same credentials resolver.

        A resolver per client walks the credential chain on its own schedule, so
        two concurrent streams would sign with independently refreshed identities.
        """
        _pool(botocore_pool, "polly", "us-east-1", "eu-west-1")
        _pool(botocore_pool, "bedrock-runtime", "us-east-1", "eu-west-3")

        initialize_bidi_clients()

        pooled: tuple[tuple[str, RegionName], ...] = (
            ("polly", "us-east-1"),
            ("polly", "eu-west-1"),
            ("bedrock-runtime", "us-east-1"),
            ("bedrock-runtime", "eu-west-3"),
        )
        for service, region in pooled:
            config = get_bidi_client(service, region)._config  # noqa: SLF001
            assert (
                config.aws_credentials_identity_resolver
                is stdapi.aws_bidi._CREDENTIALS_RESOLVER  # noqa: SLF001
            )

    def test_the_per_operation_config_copy_keeps_both_singletons(
        self, botocore_pool: dict[str, dict[RegionName, FakePooledClient]]
    ) -> None:
        """The config the SDK deep-copies per operation forks neither shared object.

        This is the copy that actually happens on every stream: the resolver and
        the transport's CRT state must survive it, and only the connection pool
        must not.
        """
        _pool(botocore_pool, "polly", "us-east-1")
        initialize_bidi_clients()
        config = get_bidi_client("polly", "us-east-1")._config  # noqa: SLF001

        copy = deepcopy(config)

        assert (
            copy.aws_credentials_identity_resolver
            is stdapi.aws_bidi._CREDENTIALS_RESOLVER  # noqa: SLF001
        )
        assert copy.transport is not config.transport
        assert copy.transport._tls_ctx is config.transport._tls_ctx  # noqa: SLF001
        assert copy.transport._connections == {}  # noqa: SLF001

    def test_only_bidirectional_services_get_a_client(
        self, botocore_pool: dict[str, dict[RegionName, FakePooledClient]]
    ) -> None:
        """Pooled services with no bidirectional operation are skipped.

        The pool is walked service by service, and most pooled services have no
        duplex operation at all.
        """
        _pool(botocore_pool, "polly", "us-east-1")
        _pool(botocore_pool, "s3", "us-east-1")

        initialize_bidi_clients()

        assert get_bidi_client("polly", "us-east-1") is not None
        with pytest.raises(KeyError):
            get_bidi_client("s3", "us-east-1")

    def test_clients_are_reused_across_initializations(
        self, botocore_pool: dict[str, dict[RegionName, FakePooledClient]]
    ) -> None:
        """A second initialization keeps the client the first one built.

        Each client construction that built its own transport would cost a native
        thread; the pool must not grow one per call.
        """
        _pool(botocore_pool, "polly", "us-east-1")
        initialize_bidi_clients()
        first = get_bidi_client("polly", "us-east-1")

        initialize_bidi_clients()

        assert get_bidi_client("polly", "us-east-1") is first

    def test_every_client_shares_one_transport(
        self, botocore_pool: dict[str, dict[RegionName, FakePooledClient]]
    ) -> None:
        """All regional clients of both services share the transport prototype.

        Each ``AWSCRTHTTPClient`` spawns a native CRT thread, so one per client
        would multiply threads for nothing.
        """
        _pool(botocore_pool, "polly", "us-east-1", "eu-west-1")
        _pool(botocore_pool, "bedrock-runtime", "us-east-1", "eu-west-3")

        initialize_bidi_clients()

        pooled: tuple[tuple[str, RegionName], ...] = (
            ("polly", "us-east-1"),
            ("polly", "eu-west-1"),
            ("bedrock-runtime", "us-east-1"),
            ("bedrock-runtime", "eu-west-3"),
        )
        transports = {
            id(get_bidi_client(service, region)._config.transport)  # noqa: SLF001
            for service, region in pooled
        }
        assert len(transports) == 1

    def test_unknown_region_with_several_clients_is_an_error(
        self, botocore_pool: dict[str, dict[RegionName, FakePooledClient]]
    ) -> None:
        """With several pooled regions, an unpooled one is not silently substituted.

        Serving another region would send the request somewhere the caller did not
        choose, which for a region-bound feature is a wrong answer, not a fallback.
        """
        _pool(botocore_pool, "polly", "us-east-1", "eu-west-3")
        initialize_bidi_clients()

        with pytest.raises(KeyError):
            get_bidi_client("polly", "eu-west-1")

    def test_unknown_region_falls_back_to_the_only_client(
        self, botocore_pool: dict[str, dict[RegionName, FakePooledClient]]
    ) -> None:
        """With a single pooled region, any region asked for gets it.

        Mirrors ``get_client``: a deployment configured for one region answers
        every request from it.
        """
        _pool(botocore_pool, "polly", "eu-west-3")
        initialize_bidi_clients()

        assert get_bidi_client("polly", "us-east-1") is get_bidi_client(
            "polly", "eu-west-3"
        )

    def test_the_botocore_pool_is_where_the_endpoints_come_from(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``pooled_clients`` reads the startup pool, and answers nothing for an absent service.

        This is the real reader the stub above replaces: it is what makes the
        bidirectional clients follow the botocore pool region for region.

        Ref: stdapi/aws.py:pooled_clients
        """
        client = FakePooledClient(_EUSC_ENDPOINT)
        monkeypatch.setattr(stdapi.aws, "_CLIENTS", {"polly": {_EUSC_REGION: client}})

        assert stdapi.aws.pooled_clients("polly") == {_EUSC_REGION: client}
        assert stdapi.aws.pooled_clients("bedrock-runtime") == {}

    def test_closing_empties_the_pool(
        self, botocore_pool: dict[str, dict[RegionName, FakePooledClient]]
    ) -> None:
        """Shutdown drops every client and the transport behind them."""
        _pool(botocore_pool, "polly", "us-east-1")
        initialize_bidi_clients()

        close_bidi_clients()

        with pytest.raises(KeyError):
            get_bidi_client("polly", "us-east-1")


class TestCredentialsResolver:
    """One resolver bridges the gateway's credential chain to the SDK's SigV4 signer.

    aiobotocore's ``get_credentials`` and ``get_frozen_credentials`` are both
    coroutines, so the synchronous botocore spelling silently yields a coroutine
    object where a key is expected. The resolver is also deep-copied per operation,
    which would otherwise fork the credential chain into an independently
    refreshing copy per stream.

    Ref: stdapi/aws_bidi.py:_CredentialsResolver
    """

    @staticmethod
    def _resolver(monkeypatch: pytest.MonkeyPatch, session: FakeSession) -> Any:  # noqa: ANN401
        """Build a resolver reading from *session*."""
        monkeypatch.setattr(stdapi.aws_bidi, "AWS_SESSION", session)
        return stdapi.aws_bidi._CredentialsResolver()  # noqa: SLF001

    async def test_identity_carries_the_resolved_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Access key, secret, token and account ID reach the identity as strings.

        A string assertion is the point: an unawaited ``get_frozen_credentials``
        would put a coroutine object in each field and sign nothing.
        """
        expiry = datetime(2030, 1, 1, 12, tzinfo=UTC)
        session = FakeSession(FakeCredentials(expiry))
        resolver = self._resolver(monkeypatch, session)

        identity = await resolver.get_identity(properties={})

        assert identity.access_key_id == "AKIAEXAMPLE"
        assert identity.secret_access_key == "secret"  # noqa: S105
        assert identity.session_token == "token"  # noqa: S105
        assert identity.account_id == "123456789012"
        assert identity.expiration == expiry

    async def test_expiration_is_utc_aware(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A naive expiry becomes timezone-aware UTC, which the SDK requires."""
        session = FakeSession(FakeCredentials(datetime(2030, 1, 1, 12)))  # noqa: DTZ001
        resolver = self._resolver(monkeypatch, session)

        identity = await resolver.get_identity(properties={})

        assert identity.expiration is not None
        assert identity.expiration.tzinfo is not None

    async def test_non_refreshable_credentials_carry_no_expiration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Static credentials expose no expiry, and none is invented for them."""
        session = FakeSession(FakeCredentials())
        resolver = self._resolver(monkeypatch, session)

        identity = await resolver.get_identity(properties={})

        assert identity.expiration is None

    async def test_credentials_are_resolved_once_and_frozen_per_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The chain is walked once; every later call only re-freezes it.

        Signing happens per event frame, so resolution must stay a single cheap
        read of the already-resolved chain.
        """
        credentials = FakeCredentials()
        session = FakeSession(credentials)
        resolver = self._resolver(monkeypatch, session)

        await resolver.get_identity(properties={})
        await resolver.get_identity(properties={})

        assert session.calls == 1
        assert credentials.frozen_calls == 2

    async def test_missing_credentials_are_reported(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A deployment with no credential chain gets an error, not a hang.

        Its 401 is byte-identical to the one a wrong client API key gets, so the
        server log is the only place the two are told apart.
        """
        resolver = self._resolver(monkeypatch, FakeSession(None))

        with pytest.raises(ApiError) as exc_info:
            await resolver.get_identity(properties={})

        assert exc_info.value.status == 401
        assert any(
            "AWS credentials" in str(detail) for detail in request_log["error_detail"]
        )

    def test_deepcopy_returns_the_same_resolver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The per-operation config copy must not fork the credential chain.

        Deep-copying our resolver succeeds, producing a *different* credentials
        object that refreshes on its own schedule -- one silent credential chain
        per stream.
        """
        resolver = self._resolver(monkeypatch, FakeSession(FakeCredentials()))

        assert deepcopy(resolver) is resolver


class TestErrorTranslation:
    """Modeled stream errors become API errors, and their text never reaches a client.

    They are raised out of the ``async for``, not yielded as events, and their
    messages embed AWS request IDs and internal transport error names -- exactly
    what must not be forwarded.

    Ref: stdapi/aws_bidi.py:_stream_api_error
         stdapi/aws_bedrock.py:AWS_ERROR_MAP
    """

    @pytest.mark.parametrize(
        ("error", "status"),
        [
            (ValidationException(message="Invalid Engine parameter"), 400),
            (ThrottlingException(message="Too many requests"), 429),
            (ServiceFailureException(message="An unknown condition"), 503),
            (ModelStreamErrorException(message="Stream broke"), 502),
            (SmithyError("AWS_IO_DNS_INVALID_NAME: Host name was invalid"), 503),
            (OSError("Failed to read from stream."), 503),
            (TimeoutError(), 503),
            # Named by neither table: the fault decides, except a transport timeout.
            (InvalidSsmlException(message="Invalid SSML request"), 400),
            (CallError("The service failed", fault="server"), 503),
            (ClientTimeoutError("The request timed out"), 503),
        ],
    )
    def test_error_maps_to_its_status(self, error: Exception, status: int) -> None:
        """Each error class the stream can raise gets the status a client can act on.

        A transport timeout is the one the SDK misreports: it carries
        ``fault="client"``, so the fault fallback alone would tell the caller its
        parameters are wrong about a connection that simply stopped answering.
        """
        assert stdapi.aws_bidi._stream_api_error(error, "polly").status == status  # noqa: SLF001

    def test_an_already_translated_error_is_kept(self) -> None:
        """An error the gateway itself raised keeps its status and message.

        Anything a caller-supplied handshake raises has already been translated;
        re-translating it would flatten a precise 404 into a generic failure.
        """
        error = ApiError("The requested voice is unknown.", status=404)

        assert stdapi.aws_bidi._stream_api_error(error, "polly") is error  # noqa: SLF001

    def test_the_backend_message_is_kept_in_the_request_log(
        self, request_log: dict[str, Any]
    ) -> None:
        """The text withheld from the client is written to the request log.

        Withholding it from both would make a rejected stream undiagnosable.
        """
        error = ValidationException(message="RequestId=fa0d12c4 : Invalid Engine")

        api_error = stdapi.aws_bidi._stream_api_error(error, "polly")  # noqa: SLF001

        assert "RequestId=fa0d12c4" not in str(api_error)
        assert any(
            "RequestId=fa0d12c4" in str(detail)
            for detail in request_log["error_detail"]
        )

    @pytest.mark.parametrize(
        ("service", "permission"),
        [
            ("bedrock-runtime", "bedrock:InvokeModelWithBidirectionalStream"),
            ("polly", "polly:StartSpeechSynthesisStream"),
        ],
    )
    def test_a_denied_stream_is_the_deployment_not_the_caller(
        self, request_log: dict[str, Any], service: str, permission: str
    ) -> None:
        """A refused credential answers 503, and names the permission in the log.

        The caller's own credential was verified before the stream was opened,
        so blaming them for a missing IAM permission sends them to fix something
        they do not own -- and tells them which backend is behind the route.

        Ref: stdapi/aws_bidi.py:_stream_api_error
             stdapi/api_errors.py:FeatureUnavailableError
        """
        error = AccessDeniedException(message="User is not authorized to perform")

        api_error = stdapi.aws_bidi._stream_api_error(error, service)  # noqa: SLF001

        assert api_error.status == 503
        assert api_error.code == "feature_unavailable"
        assert "not available on the current server" in str(api_error)
        assert "contact the administrator" in str(api_error)
        assert permission not in str(api_error)
        assert any(permission in str(detail) for detail in request_log["error_detail"])
        assert request_log["level"] == "warning"

    @pytest.mark.parametrize(
        "error",
        [
            ValidationException(
                message="RequestId=fa0d12c4-1 : Error 1 : AudioOutputConfiguration must be set"
            ),
            SmithyError("AWS_IO_DNS_INVALID_NAME: Host name was invalid for dns"),
            OSError("Failed to write to stream."),
            ModelStreamErrorException(message="arn:aws:bedrock:us-east-1:1234:model/x"),
            InvalidSsmlException(message="RequestId=fa0d12c4 : Invalid SSML request"),
        ],
    )
    def test_no_backend_text_reaches_the_client(self, error: Exception) -> None:
        """No request ID, transport error name, or exception class name is forwarded."""
        message = str(stdapi.aws_bidi._stream_api_error(error, "polly"))  # noqa: SLF001

        for leaked in (
            "RequestId",
            "AWS_IO_",
            "Failed to write",
            "Failed to read",
            "arn:aws",
            type(error).__name__,
        ):
            assert leaked not in message


class TestFailoverEligibility:
    """Whether another region is tried follows the status the client would receive.

    Every modeled error carries ``fault="client"`` by default -- a throttle, a cold
    model and a backend timeout included -- so reading the fault alone declares
    region-level failures final and answers them from the first region that failed.

    Ref: stdapi/aws_bidi.py:_is_stream_failover_error
         stdapi/aws.py:FAILOVER_ERROR_CODES
    """

    @pytest.mark.parametrize(
        ("error", "eligible"),
        [
            (ModelNotReadyException(message="Model is warming up"), True),
            (ModelTimeoutException(message="The model timed out"), True),
            (
                ModelErrorException(
                    message="The model failed", original_status_code=424
                ),
                True,
            ),
            (ServiceFailureException(message="An unknown condition"), True),
            (ThrottlingException(message="Too many requests"), True),
            (ClientTimeoutError("The request timed out"), True),
            (TimeoutError(), True),
            (ApiError("Retry the request.", status=503), True),
            (ApiError("Rate limit reached.", status=429), True),
            (ValidationException(message="Invalid Engine parameter"), False),
            (InvalidSsmlException(message="Invalid SSML request"), False),
            (ApiError("The requested voice is unknown.", status=404), False),
        ],
    )
    def test_only_a_caller_error_stops_the_failover(
        self, error: BaseException, eligible: bool
    ) -> None:
        """A region-level failure is retried elsewhere; a caller error is not.

        Retrying a rejected request across five regions turns one bad request into
        five, and refusing to retry a cold model wastes the only region that could
        have served it.
        """
        assert stdapi.aws_bidi._is_stream_failover_error(error) is eligible  # noqa: SLF001

    def test_a_shared_failover_code_wins_over_its_own_status(self) -> None:
        """A per-region quota is worth another region even though it reads as a 4xx.

        Transcribe's ``LimitExceededException`` is the reason multi-region failover
        exists, and it is modeled as the caller's fault.
        """
        error = LimitExceededException("Too many concurrent jobs", fault="client")

        assert stdapi.aws_bidi._stream_error_status(error) == 400  # noqa: SLF001
        assert stdapi.aws_bidi._is_stream_failover_error(error) is True  # noqa: SLF001


class TestOpenAndFailover:
    """A stream is alive only once its output resolves, and that closes failover.

    Opening returns a stream object even when the host cannot be resolved, so a
    bounded wait on the output is the only liveness signal. A region-level failure
    before that point may be retried elsewhere; a caller error may not, and nothing
    may be retried once the first event has been handed to the client.

    Ref: stdapi/aws_bidi.py:open_bidi_stream
         stdapi/aws.py:call_with_region_failover
    """

    @pytest.fixture(autouse=True)
    def _pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Serve any region from a stub client, so no AWS client is built."""
        monkeypatch.setattr(
            stdapi.aws_bidi, "get_bidi_client", lambda _service, _region=None: object()
        )

    async def test_a_session_no_end_user_can_be_attributed_with_never_opens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The attribution policy is checked before a region is ever dialled.

        The policy function has its own tests; this one pins that opening a
        stream consults it at all, which is the wiring a refactor drops.

        Ref: stdapi/aws.py:verify_bidi_user_role_policy
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_arn", _USER_ROLE_ARN)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_user_role_require_identity", True)
        opened: list[FakeDuplexStream] = []

        async def _open(_client: Any, _region: RegionName) -> FakeDuplexStream:  # noqa: ANN401
            stream = FakeDuplexStream(events=["audio"])
            opened.append(stream)
            return stream

        opener: Any = _open
        with pytest.raises(ApiError) as raised:
            async with open_bidi_stream("bedrock-runtime", ["us-east-1"], opener):
                pass

        assert raised.value.status == 400
        assert not opened, "the stream was dialled before the policy was consulted"

    async def test_region_level_failure_falls_over(
        self, request_log: dict[str, Any]
    ) -> None:
        """A throttled region hands the stream to the next candidate, and says so.

        The abandoned region is only visible in the request log: the client is
        served normally and must learn nothing about the backend topology.
        """
        stream = FakeDuplexStream(events=["audio"])
        opener = _failing_opener(
            {"us-east-1": ThrottlingException(message="slow down")}, stream
        )

        async with open_bidi_stream(
            "polly", ["us-east-1", "eu-west-3"], opener
        ) as session:
            assert session.region == "eu-west-3"

        assert any(
            "failing over" in str(detail) for detail in request_log["error_detail"]
        )

    @staticmethod
    def _failing_prime(status: int) -> Any:  # noqa: ANN401
        """Build a handshake failing the first region with *status*."""
        attempts: list[str] = []

        async def _prime(_session: BidiSession[Any, Any]) -> None:
            attempts.append("attempt")
            if len(attempts) == 1:
                raise ApiError(_HANDSHAKE_ERROR, status=status)

        return _prime

    async def test_a_server_side_handshake_failure_falls_over(self) -> None:
        """A handshake failing server-side is worth trying in another region.

        The handshake runs our own code, so its failures arrive already
        translated, and their status is what decides.
        """
        opened: list[FakeDuplexStream] = []

        async with open_bidi_stream(
            "polly",
            ["us-east-1", "eu-west-3"],
            _per_region_opener(opened, events=["audio"]),
            prime=self._failing_prime(503),
        ) as session:
            assert session.region == "eu-west-3"

    async def test_a_rejected_handshake_keeps_its_own_error(self) -> None:
        """A handshake rejected as a caller error is final, and reaches the client as-is."""
        opened: list[FakeDuplexStream] = []

        with pytest.raises(ApiError) as exc_info:
            async with open_bidi_stream(
                "polly",
                ["us-east-1", "eu-west-3"],
                _per_region_opener(opened, events=["audio"]),
                prime=self._failing_prime(400),
            ):
                pass  # pragma: no cover

        assert exc_info.value.status == 400
        assert str(exc_info.value) == _HANDSHAKE_ERROR

    async def test_an_abandoned_open_releases_its_pending_output(self) -> None:
        """A session closed before it answered leaves no output task behind.

        The SDK gates the output on a task of its own: abandoning it holds the
        connection open and reports the backend's failure, verbatim, as an
        unretrieved task exception in the server log.
        """
        opened: list[FakeDuplexStream] = []

        with pytest.raises(ApiError):
            async with open_bidi_stream(
                "polly",
                ["us-east-1"],
                _per_region_opener(opened, events=["audio"]),
                prime=self._failing_prime(400),
            ):
                pass  # pragma: no cover

        assert len(opened) == 1
        task = opened[0].output_task
        assert task is not None, "the output task was never reached"
        assert task.done()

    async def test_caller_error_is_not_retried_elsewhere(self) -> None:
        """A rejected request fails immediately instead of failing over.

        The same request would be rejected identically in every region, so trying
        five of them turns one bad request into five.
        """
        stream = FakeDuplexStream()
        opener = _failing_opener(
            {"us-east-1": ValidationException(message="Invalid Engine parameter")},
            stream,
        )

        with pytest.raises(ApiError) as exc_info:
            async with open_bidi_stream("polly", ["us-east-1", "eu-west-3"], opener):
                pass  # pragma: no cover

        assert exc_info.value.status == 400

    async def test_rejection_surfacing_only_at_the_output_still_falls_over(
        self,
    ) -> None:
        """A stream that opens and then reports a region-level failure fails over.

        Opening succeeds even for an unusable stream, so this is where most
        failures actually appear.
        """
        opened: list[RegionName] = []
        broken = FakeDuplexStream(open_error=ServiceFailureException(message="down"))
        healthy = FakeDuplexStream(events=["audio"])

        async def _open(_client: Any, region: RegionName) -> FakeDuplexStream:  # noqa: ANN401
            opened.append(region)
            return broken if region == "us-east-1" else healthy

        opener: Any = _open
        async with open_bidi_stream(
            "polly", ["us-east-1", "eu-west-3"], opener
        ) as session:
            assert session.region == "eu-west-3"

        assert opened == ["us-east-1", "eu-west-3"]
        assert broken.input_stream.closed == 1

    async def test_a_region_with_no_client_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A region the pool never built a client for is one more failed candidate.

        A deployment whose pools disagree is a configuration fault, and answering
        it with a raw pool lookup error on the request path is worse than trying
        the next region and reporting a backend failure.
        """
        stream = FakeDuplexStream(events=["audio"])

        def _client(_service: str, region: RegionName | None = None) -> object:
            if region == "us-east-1":
                raise KeyError(region)
            return object()

        monkeypatch.setattr(stdapi.aws_bidi, "get_bidi_client", _client)

        async with open_bidi_stream(
            "polly", ["us-east-1", "eu-west-3"], _stream_opener(stream)
        ) as session:
            assert session.region == "eu-west-3"

    async def test_last_region_error_is_translated(self) -> None:
        """When every candidate fails, the last failure becomes the API error."""
        stream = FakeDuplexStream()
        opener = _failing_opener(
            {
                "us-east-1": ThrottlingException(message="slow down"),
                "eu-west-3": ThrottlingException(message="slow down"),
            },
            stream,
        )

        with pytest.raises(ApiError) as exc_info:
            async with open_bidi_stream("polly", ["us-east-1", "eu-west-3"], opener):
                pass  # pragma: no cover

        assert exc_info.value.status == 429

    async def test_priming_events_are_sent_before_the_output_is_awaited(self) -> None:
        """The handshake a session needs is sent before the liveness gate fires.

        A speech-to-speech session answers nothing until its configuration events
        are sent, so awaiting the output first would time out every time.
        """
        stream = FakeDuplexStream(events=["transcript"])
        order: list[str] = []

        async def _prime(session: BidiSession[Any, Any]) -> None:
            order.append("primed")
            await session.send("sessionStart")

        original = FakeDuplexStream.await_output

        async def _await_output(self: FakeDuplexStream) -> Any:  # noqa: ANN401
            order.append("awaited")
            return await original(self)

        FakeDuplexStream.await_output = _await_output  # type: ignore[method-assign]
        try:
            async with open_bidi_stream(
                "polly", ["us-east-1"], _stream_opener(stream), prime=_prime
            ):
                pass
        finally:
            FakeDuplexStream.await_output = original  # type: ignore[method-assign]

        assert order == ["primed", "awaited"]
        assert stream.input_stream.sent == ["sessionStart"]

    async def test_stream_that_never_answers_is_abandoned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unanswered output aborts the open instead of hanging the request.

        Both halves are closed directly: the SDK's own ``close()`` awaits the very
        output that never arrives.
        """
        monkeypatch.setattr(SETTINGS, "aws_connect_timeout", 1)
        stream = FakeDuplexStream(never_answers=True)

        with pytest.raises(ApiError) as exc_info:
            async with open_bidi_stream("polly", ["us-east-1"], _stream_opener(stream)):
                pass  # pragma: no cover

        assert exc_info.value.status == 503
        assert stream.sdk_close_calls == 0
        assert stream.input_stream.closed == 1

    async def test_an_explicit_open_timeout_replaces_the_connection_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller may bound the open itself, for a handshake worth more than a connection.

        Without it the whole sequence -- connect, handshake and first answer --
        rides the connection timeout, which a slower protocol cannot fit into.
        """
        monkeypatch.setattr(SETTINGS, "aws_connect_timeout", _UNREACHED_TIMEOUT)
        stream = FakeDuplexStream(never_answers=True)

        async with async_timeout(_TEST_TIMEOUT):
            with pytest.raises(ApiError) as exc_info:
                async with open_bidi_stream(
                    "polly",
                    ["us-east-1"],
                    _stream_opener(stream),
                    open_timeout=_SHORT_TIMEOUT,
                ):
                    pass  # pragma: no cover

        assert exc_info.value.status == 503


class TestSessionLifecycle:
    """A session sends, iterates and always closes -- including on cancellation.

    Ref: stdapi/aws_bidi.py:BidiSession
    """

    @pytest.fixture(autouse=True)
    def _pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Serve any region from a stub client, so no AWS client is built."""
        monkeypatch.setattr(
            stdapi.aws_bidi, "get_bidi_client", lambda _service, _region=None: object()
        )

    async def test_events_are_yielded_then_both_halves_close(self) -> None:
        """Scripted events reach the caller, and exiting closes input and output."""
        stream = FakeDuplexStream(events=["one", "two"])

        async with open_bidi_stream(
            "polly", ["us-east-1"], _stream_opener(stream)
        ) as session:
            await session.send("text")
            received = [event async for event in session]

        assert received == ["one", "two"]
        assert stream.input_stream.sent == ["text"]
        assert stream.input_stream.closed == 1
        assert stream.sdk_close_calls == 0

    async def test_modeled_error_from_the_iteration_becomes_an_api_error(self) -> None:
        """An error raised out of the ``async for`` is translated, not propagated raw."""
        stream = FakeDuplexStream(
            events=["one"], receive_error=ThrottlingException(message="slow down")
        )

        async def _consume() -> None:
            async with open_bidi_stream(
                "polly", ["us-east-1"], _stream_opener(stream)
            ) as session:
                async for _event in session:
                    pass

        with pytest.raises(ApiError) as exc_info:
            await _consume()

        assert exc_info.value.status == 429

    async def test_a_cancelled_read_stays_a_cancellation(self) -> None:
        """Cancellation is not a backend failure and must not become one.

        Translating it would turn a client disconnect into a logged 503 and swallow
        the cancellation the caller's task group is waiting for.
        """
        stream = FakeDuplexStream(receive_error=CancelledError())

        async def _consume() -> None:
            async with open_bidi_stream(
                "polly", ["us-east-1"], _stream_opener(stream)
            ) as session:
                async for _event in session:
                    pass  # pragma: no cover

        with pytest.raises(CancelledError):
            await _consume()

    async def test_send_failure_becomes_an_api_error(self) -> None:
        """A write to a broken stream is translated the same way a read is."""
        stream = FakeDuplexStream(send_error=OSError("Failed to write to stream."))

        with pytest.raises(ApiError) as exc_info:
            async with open_bidi_stream(
                "polly", ["us-east-1"], _stream_opener(stream)
            ) as session:
                await session.send("text")

        assert exc_info.value.status == 503

    async def test_cancellation_still_closes_the_stream_once(self) -> None:
        """A client disconnect closes the session instead of leaking the connection.

        The close is shielded because closing suspends: a task group that cancels
        its members more than once -- the normal shape of a client disconnect --
        cancels the cleanup mid-close, and an unshielded ``await`` there closes
        nothing at all.
        """
        stream = FakeDuplexStream(events=["one"])
        started = Event()

        async def _consume() -> None:
            async with open_bidi_stream(
                "polly", ["us-east-1"], _stream_opener(stream)
            ) as session:
                started.set()
                async for _event in session:
                    await sleep(10)

        task = create_task(_consume())
        await started.wait()
        task.cancel()
        # The second delivery lands while the cleanup is closing the stream.
        await sleep(0)
        task.cancel()
        with pytest.raises(CancelledError):
            await task
        await _drain_close_tasks()

        assert stream.input_stream.closed == 1
        assert stream.sdk_close_calls == 0
        assert not stdapi.aws_bidi._CLOSE_TASKS  # noqa: SLF001

    async def test_iterating_an_unopened_session_is_refused(self) -> None:
        """A session whose output never resolved cannot be iterated.

        Nothing reaches this from the public entry point, which always gates on the
        output first; it exists so a future caller gets an error and not a ``None``.
        """
        session: BidiSession[Any, Any] = BidiSession(
            FakeDuplexStream(never_answers=True),  # type: ignore[arg-type]
            "us-east-1",
            "polly",
        )

        with pytest.raises(ServerError):
            async for _event in session:
                pass  # pragma: no cover

    async def test_the_input_half_closes_without_ending_the_output(self) -> None:
        """Half-closing tells the service the input is complete, nothing more.

        Amazon Polly keeps its idle timer running until the request body ends,
        so a session that only sends its own end-of-input event is dropped with
        the audio still owed.
        """
        stream = FakeDuplexStream(events=["audio"])

        async with open_bidi_stream(
            "polly", ["us-east-1"], _stream_opener(stream)
        ) as session:
            await session.close_input()
            received = [event async for event in session]

        assert received == ["audio"], "the output half must survive the half-close"
        assert stream.input_stream.closed == 1, "closing is idempotent, as the SDK's is"

    async def test_a_send_after_the_half_close_is_an_api_error(self) -> None:
        """An event racing past the half-close is translated, not raised raw.

        A driver sending from its own task alongside the reader can always lose
        that race, and the publisher answers a late write with an ``OSError``.
        """
        stream = FakeDuplexStream(events=["audio"])

        async with open_bidi_stream(
            "polly", ["us-east-1"], _stream_opener(stream)
        ) as session:
            await session.close_input()
            with pytest.raises(ApiError) as exc_info:
                await session.send("late")

        assert exc_info.value.status == 503
        assert _CLOSED_STREAM_ERROR not in str(exc_info.value)
        assert stream.input_stream.sent == []

    async def test_a_session_survives_a_send_still_in_flight_at_teardown(self) -> None:
        """A session sent from one task and iterated in another closes exactly once.

        The speech driver sends its text from a task of its own while the request
        task iterates, and cancels that task in its ``finally`` -- so the wrapper
        always meets a send that has not finished on the way out.
        """
        stream = FakeDuplexStream(events=["one", "two"])
        unblocked = Event()

        async def _send(session: BidiSession[Any, Any]) -> None:
            await session.send("text")
            # Held open on purpose: this is the send still in flight at teardown.
            await unblocked.wait()
            await session.send("never")

        async with open_bidi_stream(
            "polly", ["us-east-1"], _stream_opener(stream)
        ) as session:
            sender = create_task(_send(session))
            received = [event async for event in session]
            assert not sender.done(), "the sender must still be running at teardown"
        sender.cancel()
        with pytest.raises(CancelledError):
            await sender
        await _drain_close_tasks()

        assert received == ["one", "two"]
        assert stream.input_stream.sent == ["text"]
        assert stream.input_stream.closed == 1
        assert not stdapi.aws_bidi._CLOSE_TASKS  # noqa: SLF001

    async def test_a_failed_input_close_is_translated(self) -> None:
        """A half-close the transport refuses is an API error, not an OSError."""
        stream = FakeDuplexStream()

        async def _refuse() -> None:
            """Fail the way a broken transport does."""
            msg = "broken pipe"
            raise OSError(msg)

        async with open_bidi_stream(
            "polly", ["us-east-1"], _stream_opener(stream)
        ) as session:
            stream.input_stream.close = _refuse  # type: ignore[method-assign]
            with pytest.raises(ApiError) as excinfo:
                await session.close_input()

        assert excinfo.value.status == 503
        assert "broken pipe" not in str(excinfo.value)


class TestBoundedClose:
    """Closing a half is bounded, because closing one can block on the wire.

    The real publisher's ``close()`` writes a signed empty frame and awaits it, so
    a stalled transport holds the request worker for as long as it is allowed to.
    The close also runs detached, and the task is held until it completes.

    Ref: stdapi/aws_bidi.py:_CLOSE_TIMEOUT
         stdapi/aws_bidi.py:_close_session
    """

    @pytest.fixture(autouse=True)
    def _pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Serve any region from a stub client, so no AWS client is built."""
        monkeypatch.setattr(
            stdapi.aws_bidi, "get_bidi_client", lambda _service, _region=None: object()
        )

    @pytest.fixture(autouse=True)
    def _short_close_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shorten the bound, which is what the test would otherwise wait out."""
        monkeypatch.setattr(stdapi.aws_bidi, "_CLOSE_TIMEOUT", _SHORT_TIMEOUT)

    async def test_a_half_close_that_never_returns_is_an_api_error(self) -> None:
        """A stalled half-close fails the request instead of hanging it.

        Watched from outside rather than cancelled from a timeout: an unbounded
        close holds the request exactly by outliving the cancellation sent to it,
        so this test has to observe the hang instead of provoking one.
        """
        stream = FakeDuplexStream(events=["audio"], close_hangs=True)
        statuses: list[int] = []

        async def _half_close() -> None:
            async with open_bidi_stream(
                "polly", ["us-east-1"], _stream_opener(stream)
            ) as session:
                try:
                    await session.close_input()
                except ApiError as error:
                    statuses.append(error.status)

        task = create_task(_half_close())
        done, _pending = await wait([task], timeout=_TEST_TIMEOUT)
        stream.release_closes()
        await wait([task], timeout=_TEST_TIMEOUT)

        assert task in done, "the half-close was not bounded"
        assert statuses == [503]
        assert stream.input_stream.closed == 0

    async def test_a_stalled_close_still_ends_the_request(self) -> None:
        """Leaving the session returns even when neither half can be closed."""
        stream = FakeDuplexStream(events=["audio"], close_hangs=True)

        async with async_timeout(_TEST_TIMEOUT):
            async with open_bidi_stream("polly", ["us-east-1"], _stream_opener(stream)):
                pass
        await _drain_close_tasks()
        stream.release_closes()

        assert stream.sdk_close_calls == 0
        assert not stdapi.aws_bidi._CLOSE_TASKS  # noqa: SLF001

    async def test_the_detached_close_is_held_until_it_completes(self) -> None:
        """The close task is strongly referenced while it runs, and dropped after.

        A detached task nobody holds can be collected mid-close, which leaks the
        connection it was closing.
        """
        stream = FakeDuplexStream(close_hangs=True)
        session: BidiSession[Any, Any] = BidiSession(stream, "us-east-1", "polly")  # type: ignore[arg-type]

        closing = create_task(stdapi.aws_bidi._close_session(session))  # noqa: SLF001
        await sleep(0)
        held = set(stdapi.aws_bidi._CLOSE_TASKS)  # noqa: SLF001
        async with async_timeout(_TEST_TIMEOUT):
            await closing
        stream.release_closes()

        assert len(held) == 1
        assert not stdapi.aws_bidi._CLOSE_TASKS  # noqa: SLF001
