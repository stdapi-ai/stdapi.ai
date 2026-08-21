"""Application assembly and exception handling in :mod:`stdapi.main`.

Covers the ASGI app's exception handlers (botocore errors, TaskGroup-wrapped
exception groups), the request-scoped input-file tracking the middleware resets,
and router discovery when two providers share a routes prefix.

Ref: stdapi/main.py
"""

from __future__ import annotations

from asyncio import CancelledError, Event, create_task, sleep, wait_for
from contextlib import asynccontextmanager, contextmanager
from inspect import signature
from io import BytesIO
from json import loads
from os import environ
from subprocess import run
from sys import executable
from typing import TYPE_CHECKING, Any, Self

import pytest
from botocore.exceptions import ClientError
from httpx import ASGITransport, AsyncClient
from starlette.datastructures import UploadFile
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import StreamingResponse

from stdapi import main as stdapi_main
from stdapi import metering, monitoring
from stdapi.cleanup import (
    _DETACHED_TASKS,
    CLEANUPS,
    run_cleanups_detached,
    schedule_cleanup,
)
from stdapi.config import AWS_REGION, SETTINGS
from stdapi.exceptions import (
    InvalidProductError,
    NotEntitledError,
    ServerError,
    UnsupportedPlatformError,
)
from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile
from stdapi.main import handle_exception_group
from stdapi.routes import core_root, openai_chat_completions
from stdapi.server import SERVER_ID
from stdapi.types.openai_chat_completions import ChatCompletion
from tests._helpers import make_client_error, make_event_log, make_model_details
from tests.conftest import REPO_ROOT

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, MutableMapping

    from fastapi.testclient import TestClient
    from starlette.responses import Response

#: All tests in this module exercise the local implementation in-process, and log
#: outside request scope, so they need the shared request-log context.
pytestmark = [pytest.mark.local, pytest.mark.usefixtures("request_log")]


#: Starlette's own GZipMiddleware compression threshold, used when gzip is disabled.
_GZIP_DEFAULT_MINIMUM_SIZE: int = (
    signature(GZipMiddleware.__init__).parameters["minimum_size"].default
)


def _gzip_minimum_size() -> int:
    """Return the body size from which the app would compress a response.

    Returns:
        The ``minimum_size`` configured on the app's GZipMiddleware, or
        Starlette's default (the stricter bound) when gzip is disabled.
    """
    for middleware in stdapi_main.app.user_middleware:
        # Starlette types ``cls`` as a generic factory and ``kwargs`` as ParamSpec
        # keywords, so neither narrows to the middleware actually installed.
        if middleware.cls is GZipMiddleware:  # type: ignore[comparison-overlap]
            return int(  # type: ignore[no-any-return,call-overload]
                middleware.kwargs.get("minimum_size", _GZIP_DEFAULT_MINIMUM_SIZE)
            )
    return _GZIP_DEFAULT_MINIMUM_SIZE


def _request() -> Request:
    """Return a minimal HTTP request for exception handlers.

    Returns:
        Starlette request with an empty GET scope.
    """
    return Request(
        {"type": "http", "method": "GET", "path": "/v1/models", "headers": []}
    )


class _ValidationException(ClientError):  # noqa: N818 - mirrors the AWS exception name
    """Modeled botocore subclass, as raised by aiobotocore service clients."""


class TestHandleExceptionGroup:
    """Handler resolution for TaskGroup-wrapped exceptions.

    Ref: stdapi/main.py:handle_exception_group
    """

    async def test_client_error_subclass_resolves_to_botocore_handler(self) -> None:
        """A ClientError subclass inside an ExceptionGroup is handled, not turned into a 500.

        aiobotocore raises modeled subclasses such as ``ValidationException``,
        which an exact-class lookup in ``_EXCEPTION_HANDLERS`` would miss; the
        handler must match by ``isinstance``. The AWS message reaching the body
        proves the botocore handler ran rather than a generic 400.

        Ref: stdapi/main.py:handle_botocore_client_error
        """
        error = _ValidationException(
            {"Error": {"Code": "ValidationException", "Message": "bad input"}},
            "Converse",
        )
        group = ExceptionGroup("task group", [error])
        response = await handle_exception_group(_request(), group)
        assert response.status_code == 400
        assert loads(bytes(response.body)) == {"error": "bad input"}

    async def test_unhandled_exception_reraises_the_group(self) -> None:
        """Groups holding unhandled exception types propagate unchanged.

        Re-raising the original group (rather than a substitute) is what routes
        the failure to the default 500 path with critical logging and keeps the
        sub-exception tracebacks intact.
        """
        error = RuntimeError("boom")
        group = ExceptionGroup("task group", [error])
        with pytest.raises(ExceptionGroup) as excinfo:
            await handle_exception_group(_request(), group)
        assert excinfo.value is group
        assert excinfo.value.exceptions == (error,)


async def _validate_model(model_id: str, *_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
    """Return canned model details without hitting AWS."""
    return make_model_details(model_id)


class _RecordingChatBackend:
    """Stub chat backend that prefetches input files like the default backend."""

    def __init__(self) -> None:
        self.seen_input_files: list[InputFile] | None = None

    def native_store_supported(self) -> bool:
        """Local stub: no native storage."""
        return False

    async def create_completion(
        self,
        request: Any,  # noqa: ANN401
        completion_id: str,
        created: int,
    ) -> ChatCompletion:
        """Prefetch tracked input files, record them, return a canned completion."""
        from stdapi.input_file import prefetch_all_content_types  # noqa: PLC0415

        await prefetch_all_content_types()
        self.seen_input_files = list(_CURRENT_INPUT_FILES.get(()))
        return ChatCompletion.model_validate(
            {
                "id": completion_id,
                "created": created,
                "model": request.model,
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )


class TestRequestScopedInputFiles:
    """Input-file tracking is scoped to each request context.

    Ref: stdapi/input_file.py:reset_current_input_files
         stdapi/main.py:_middleware
    """

    async def test_stale_input_files_do_not_leak_into_requests(
        self, monkeypatch: pytest.MonkeyPatch, api_key: str
    ) -> None:
        """A stale InputFile bound to an outer context does not reach a request.

        Without the reset, an ``InputFile`` tracked in a longer-lived context
        (here the test task; in production a keep-alive connection task) whose
        backing file is already closed 500s every later request during
        content-type prefetch. The backend records what it saw, so an empty
        list is the proof the reset happened.
        """
        from stdapi.main import app  # noqa: PLC0415

        backend = _RecordingChatBackend()
        monkeypatch.setattr(openai_chat_completions, "validate_model", _validate_model)
        monkeypatch.setattr(
            openai_chat_completions, "get_chat_model", lambda _model_id: backend
        )

        # Bind a stale, closed upload in the outer context the request inherits.
        stale_file = BytesIO(b"stale upload content")
        stale = InputFile(UploadFile(file=stale_file, filename="stale.bin"))
        stale_file.close()
        assert stale in _CURRENT_INPUT_FILES.get(())

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "amazon.nova-micro-v1:0",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
        assert response.status_code == 200, response.text
        assert backend.seen_input_files == []


class TestBotocoreConnectionErrorHandler:
    """The 503 connection-error handler hides internal endpoint detail.

    Ref: stdapi/main.py:handle_botocore_connection_error
    """

    async def test_generic_body_while_endpoint_stays_in_logs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clients see a fixed generic 503; the endpoint URL is logged only.

        The endpoint is needed for operators, so it must appear in the structured
        log while being absent from the response body. No AWS service name or
        endpoint host is ever echoed back, regardless of which service failed.
        """
        from botocore.exceptions import EndpointConnectionError  # noqa: PLC0415

        from stdapi import main  # noqa: PLC0415

        endpoint = "https://bedrock-runtime.us-east-1.amazonaws.com"
        logged: list[str] = []
        monkeypatch.setattr(
            main,
            "log_error_details",
            lambda message, **_kwargs: logged.append(str(message)),
        )

        response = await main.handle_botocore_connection_error(
            _request(), EndpointConnectionError(endpoint_url=endpoint)
        )

        assert response.status_code == 503
        body = bytes(response.body).decode()
        assert "The service is temporarily unavailable." in body
        assert endpoint not in body
        assert "Bedrock" not in body
        assert any(endpoint in entry for entry in logged)


class TestSharedPrefixRouterOptOut:
    """Anthropic routers opt out when their prefix equals the OpenAI prefix.

    ``/v1/files`` and ``/v1/models`` exist on both surfaces, so mounting both at
    the same base path would register duplicate path/method pairs.

    Ref: stdapi/routes/__init__.py:discover_routers
         stdapi/config.py:_validate_unique_routes_prefixes
    """

    def test_colliding_routers_disabled_and_discovery_skips_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a shared base path, colliding Anthropic routers are disabled.

        Their modules expose ``router = None``, discovery skips them instead of
        raising, and the resulting app has no duplicate path/method pairs.
        """
        from collections import Counter  # noqa: PLC0415
        from importlib import reload  # noqa: PLC0415

        from fastapi import FastAPI  # noqa: PLC0415
        from fastapi.routing import iter_route_contexts  # noqa: PLC0415

        from stdapi.config import SETTINGS  # noqa: PLC0415
        from stdapi.routes import (  # noqa: PLC0415
            anthropic_files,
            anthropic_models,
            discover_routers,
        )

        monkeypatch.setattr(SETTINGS, "openai_routes_prefix", "")
        monkeypatch.setattr(SETTINGS, "anthropic_routes_prefix", "")
        try:
            assert reload(anthropic_files).router is None
            assert reload(anthropic_models).router is None
            app = FastAPI()
            discover_routers(app)
            # Routes are enumerated through the router wrappers FastAPI mounts:
            # app.routes holds those wrappers, never the routes themselves.
            methods_seen = Counter(
                (context.path, method)
                for context in iter_route_contexts(app.routes)
                for method in context.methods or ()
            )
            assert methods_seen, "no route was enumerated, so nothing was checked"
            assert not [key for key, count in methods_seen.items() if count > 1]
        finally:
            monkeypatch.undo()
            reload(anthropic_files)
            reload(anthropic_models)
        assert anthropic_files.router is not None
        assert anthropic_models.router is not None


class TestMiddlewareCleanupDrain:
    """Scheduled cleanups run however the request ends.

    Whether no response reaches the client at all, or one whose body is still
    being produced when the endpoint returns.

    Ref: stdapi/main.py:_middleware
         stdapi/cleanup.py:run_cleanups_detached
         stdapi/cleanup.py:run_scheduled_cleanups
    """

    async def test_unhandled_error_still_runs_scheduled_cleanups(self) -> None:
        """An exception no handler converts must not drop the scheduled cleanups.

        Handled exceptions become responses inside ``call_next`` and drain via
        the post-response background task; an unhandled one escapes it, so
        without the detached drain a store-enabled route's deferred discard
        would leak its Bedrock stored session on every such 500.
        """
        ran = Event()

        async def cleanup() -> None:
            ran.set()

        async def call_next(_request: Request) -> Response:
            schedule_cleanup(cleanup())
            msg = "no handler registered for this type"
            raise ZeroDivisionError(msg)

        with pytest.raises(ZeroDivisionError):
            await stdapi_main._middleware(_request(), call_next)  # noqa: SLF001
        await wait_for(ran.wait(), timeout=5)

    async def test_client_disconnect_answers_499_and_runs_cleanups(self) -> None:
        """The 499 disconnect short-circuit also drains scheduled cleanups."""
        ran = Event()

        async def cleanup() -> None:
            ran.set()

        async def call_next(_request: Request) -> Response:
            schedule_cleanup(cleanup())
            msg = "No response returned."
            raise RuntimeError(msg)

        async def receive() -> dict[str, Any]:
            return {"type": "http.disconnect"}

        request = Request(
            {"type": "http", "method": "GET", "path": "/v1/models", "headers": []},
            receive,
        )
        response = await stdapi_main._middleware(request, call_next)  # noqa: SLF001
        assert response.status_code == 499
        await wait_for(ran.wait(), timeout=5)

    async def test_runtime_error_without_a_disconnect_is_not_masked_as_499(
        self,
    ) -> None:
        """A RuntimeError from a still-connected client propagates instead of becoming 499.

        The 499 short-circuit exists for Starlette's "No response returned."
        after the peer went away; a genuine ``RuntimeError`` raised while the
        connection is alive is a server fault, and answering it as a client
        disconnect would hide the 500 from the logs and from the client.
        """
        ran = Event()

        async def cleanup() -> None:
            ran.set()

        async def call_next(_request: Request) -> Response:
            schedule_cleanup(cleanup())
            msg = "backend seam misconfigured"
            raise RuntimeError(msg)

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "more_body": False}

        request = Request(
            {"type": "http", "method": "GET", "path": "/v1/models", "headers": []},
            receive,
        )
        with pytest.raises(RuntimeError, match=r"^backend seam misconfigured$"):
            await stdapi_main._middleware(request, call_next)  # noqa: SLF001
        await wait_for(ran.wait(), timeout=5)

    async def test_a_cleanup_scheduled_while_streaming_still_runs(self) -> None:
        """Work deferred by a streamed response must not be dropped.

        A streamed endpoint returns before its body has produced anything, so
        everything it defers is registered after the middleware has already
        built the response.  Dropping those is silent and expensive: a vector
        store queried only through streamed answers never has its activity
        refreshed, and can expire while it is still in use.
        """
        ran = Event()

        async def cleanup() -> None:
            ran.set()

        async def body() -> AsyncGenerator[bytes]:
            yield b"first"
            schedule_cleanup(cleanup())
            yield b"last"

        async def call_next(_request: Request) -> Response:
            return StreamingResponse(body())

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "more_body": False}

        sent: list[MutableMapping[str, Any]] = []

        async def send(message: MutableMapping[str, Any]) -> None:
            sent.append(message)

        response = await stdapi_main._middleware(_request(), call_next)  # noqa: SLF001
        # ASGI 2.4 sends the body without the concurrent disconnect watcher.
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

        streamed = b"".join(
            message.get("body", b"")
            for message in sent
            if message["type"] == "http.response.body"
        )
        assert streamed == b"firstlast", "the client still receives the whole body"
        await wait_for(ran.wait(), timeout=5)

    async def test_cancellation_still_runs_scheduled_cleanups(self) -> None:
        """A cancelled request scope drains cleanups in a detached task.

        The drain must survive the cancellation that killed the request scope,
        so the middleware hands it to the event loop instead of awaiting it
        inside the cancelled scope.
        """
        ran = Event()

        async def cleanup() -> None:
            ran.set()

        async def call_next(_request: Request) -> Response:
            schedule_cleanup(cleanup())
            raise CancelledError

        with pytest.raises(CancelledError):
            await stdapi_main._middleware(_request(), call_next)  # noqa: SLF001
        await wait_for(ran.wait(), timeout=5)


class TestSharedResponsesStayBelowTheGzipThreshold:
    """Pre-rendered singleton responses must never reach the gzip minimum size.

    Ref: https://www.starlette.io/middleware/#gzipmiddleware
         stdapi/routes/core_root.py
    """

    def test_cached_response_bodies_stay_uncompressible(self) -> None:
        """Each shared response body stays under the configured gzip threshold.

        Starlette hands ``raw_headers`` to the ASGI message by reference and
        GZipMiddleware mutates that list in place, so the first gzip-accepting
        request against a shared response whose body reaches the threshold
        stamps ``content-encoding: gzip`` and a compressed ``content-length``
        onto the singleton for the process's lifetime, breaking every later
        client that did not ask for gzip.
        """
        minimum_size = _gzip_minimum_size()

        for path, response in (
            ("/", core_root._ROOT_RESPONSE),  # noqa: SLF001
            ("/health", core_root._HEALTH_RESPONSE),  # noqa: SLF001
            ("/ping", core_root._PING_RESPONSE),  # noqa: SLF001
            ("/.well-known/api-catalog", core_root._API_CATALOG_RESPONSE),  # noqa: SLF001
        ):
            assert len(response.body) < minimum_size, (
                f"The cached {path} response body reached {minimum_size} bytes: "
                "GZipMiddleware would rewrite the shared response headers in "
                "place, permanently serving gzip to clients that did not ask "
                "for it. Render this response per request instead of caching it."
            )


#: Application built in a subprocess, so the proxy-header settings apply at import.
_FORWARDED_CLIENT_SCRIPT = """
import os
os.environ.update({
    "AWS_BEDROCK_REGIONS": "us-east-1",
    "ENABLE_PROXY_HEADERS": "true",
    "LOG_CLIENT_IP": "true",
    "PROXY_TRUSTED_HOSTS": '["10.0.0.0/8"]',
})
import asyncio, contextlib, io, json
import stdapi.main
from httpx import ASGITransport, AsyncClient


async def request():
    transport = ASGITransport(app=stdapi.main.app, client=("10.0.0.5", 1234))
    async with AsyncClient(transport=transport, base_url="http://gateway") as client:
        await client.get("/v1/no-such-route", headers={"x-forwarded-for": "203.0.113.42"})


captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    asyncio.run(request())
for line in captured.getvalue().splitlines():
    try:
        event = json.loads(line)
    except ValueError:
        continue
    if event.get("type") == "request":
        print(event.get("client_ip"))
"""


class TestForwardedClientAddress:
    """The recorded client address is the forwarded one, not the proxy's.

    ``ProxyHeadersMiddleware`` rewrites ``scope["client"]``, and ``_middleware``
    reads it the moment it opens the request log, so the proxy shim only has any
    effect while it stays registered *after* ``_middleware`` — ``add_middleware``
    inserts at the head, making the last registration the outermost. Registered
    the other way round, every deployment behind a load balancer silently records
    the balancer's own address as the client, in logs and in OpenTelemetry spans.

    The settings are read at import, so the application is built in a subprocess
    rather than reconfigured in place.

    Ref: https://www.uvicorn.org/deployment/#proxies-and-forwarded-headers
         stdapi/main.py:_middleware
         stdapi/monitoring.py:log_request_event
    """

    def test_forwarded_client_ip_reaches_the_request_log(self) -> None:
        """A trusted proxy's ``X-Forwarded-For`` wins over the peer address."""
        environment = {
            k: v
            for k, v in environ.items()
            if not k.startswith(("ENABLE_", "LOG_", "PROXY_", "AWS_BEDROCK_"))
        }
        environment.update(
            AWS_EC2_METADATA_DISABLED="true",
            AWS_CONFIG_FILE="/dev/null",
            AWS_SHARED_CREDENTIALS_FILE="/dev/null",
            AWS_REGION="us-east-1",
            AWS_DEFAULT_REGION="us-east-1",
        )
        result = run(  # noqa: S603
            [executable, "-c", _FORWARDED_CLIENT_SCRIPT],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=environment,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip().splitlines()[-1] == "203.0.113.42", (
            "The request log recorded the immediate peer instead of the forwarded "
            "client. ProxyHeadersMiddleware must be registered after _middleware "
            f"so that it wraps it.\nstdout: {result.stdout!r}"
        )


#: Stand-in AWS Marketplace product code for the Enterprise build's registration.
_FAKE_PRODUCT_CODE = "test-product-code"


class _FakeMeteringClient:
    """Marketplace metering client returning a canned answer or raising *error*."""

    def __init__(self, error: ClientError | None = None) -> None:
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def register_usage(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the call and answer it, or fail it with the configured error."""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"Signature": "sig", "PublicKeyRotationTimestamp": "2026-01-01"}


class TestMarketplaceRegistration:
    """AWS Marketplace RegisterUsage runs only on the metered build, and maps its errors.

    The call gates startup on the hourly-billed Enterprise build: a subscription
    that is missing, a platform that cannot meter, or a mistyped product code
    must each surface as the specific exception the lifespan reports, not as a
    raw botocore error. The Community build must never reach the API at all.

    Ref: https://docs.aws.amazon.com/marketplace/latest/APIReference/API_RegisterUsage.html
         stdapi/metering.py:register
    """

    @staticmethod
    def _install(
        monkeypatch: pytest.MonkeyPatch,
        client: _FakeMeteringClient,
        *,
        product_code: str = _FAKE_PRODUCT_CODE,
    ) -> list[tuple[str, str | None]]:
        """Pin the product code and hand *client* to ``register``.

        Args:
            monkeypatch: Fixture scoping the module-level patches.
            client: Stub returned by the patched ``create_client``.
            product_code: Value for the build's ``PRODUCT_CODE`` constant.

        Returns:
            The (service, region) pairs ``create_client`` was asked for.
        """
        opened: list[tuple[str, str | None]] = []

        @asynccontextmanager
        async def create_client(
            service: str,
            **kwargs: Any,  # noqa: ANN401
        ) -> AsyncGenerator[_FakeMeteringClient]:
            """Yield the stub instead of a real meteringmarketplace client."""
            opened.append((service, kwargs.get("region_name")))
            yield client

        session = type("_Session", (), {"create_client": staticmethod(create_client)})()
        monkeypatch.setattr(metering, "PRODUCT_CODE", product_code)
        monkeypatch.setattr(metering, "AWS_SESSION", session)
        return opened

    async def test_successful_registration_is_recorded_on_the_start_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The response lands on the start event, and the nonce is this server's ID.

        The nonce is what makes a restart idempotent for Marketplace, so it must
        be the per-process server ID rather than a fresh value per call.
        """
        client = _FakeMeteringClient()
        opened = self._install(monkeypatch, client)
        start_event = make_event_log(type="start")

        await metering.register(start_event)

        assert opened == [("meteringmarketplace", AWS_REGION)]
        assert client.calls == [
            {
                "ProductCode": _FAKE_PRODUCT_CODE,
                "PublicKeyVersion": 1,
                "Nonce": SERVER_ID,
            }
        ]
        assert start_event["register_usage_response"]["Signature"] == "sig"

    async def test_community_build_never_calls_the_metering_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no product code, no client is opened and no field is added.

        The unmetered build ships to accounts with no Marketplace subscription,
        where the call would fail startup outright.
        """
        client = _FakeMeteringClient()
        opened = self._install(monkeypatch, client, product_code="")
        start_event = make_event_log(type="start")

        await metering.register(start_event)

        assert opened == []
        assert client.calls == []
        assert "register_usage_response" not in start_event

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("CustomerNotEntitledException", NotEntitledError),
            ("PlatformNotSupportedException", UnsupportedPlatformError),
            ("DisabledApiException", UnsupportedPlatformError),
            ("InvalidProductCodeException", InvalidProductError),
            ("InvalidPublicKeyVersionException", InvalidProductError),
        ],
    )
    async def test_each_registration_failure_maps_to_its_own_exception(
        self, monkeypatch: pytest.MonkeyPatch, code: str, expected: type[Exception]
    ) -> None:
        """Every documented RegisterUsage error becomes the matching startup exception.

        Each one also has to say what to do about it: the operator sees only
        this message when the server refuses to start.
        """
        client = _FakeMeteringClient(make_client_error(code, "RegisterUsage"))
        self._install(monkeypatch, client)

        with pytest.raises(expected) as excinfo:
            await metering.register(make_event_log(type="start"))

        assert str(excinfo.value).strip(), "the startup failure carries no message"


class TestValidationErrorSelection:
    """The reported validation error is the most specific one, not the first.

    A field typed as a union reports one error per branch, and pydantic lists
    the shallowest first: it blames the whole field for being the wrong type
    when the real fault is one item inside the value the client did send.
    Naming that item is what makes the 400 actionable.

    Ref: https://docs.pydantic.dev/latest/errors/validation_errors/
         stdapi/main.py:handle_validation_exception
    """

    def test_the_deepest_location_is_the_one_reported(
        self, app_client: TestClient
    ) -> None:
        """A malformed item inside a union-typed field is named at its own path.

        ``input`` accepts a string or a list of items, so a bad item yields one
        error per branch; reporting the first would answer "input should be a
        valid string" about a list the client deliberately sent. The path also
        has to stay readable: Pydantic names every union it descended through,
        which would bury the field under a list of 30-odd member names.
        """
        response = app_client.post(
            "/v1/responses",
            json={
                "model": "amazon.nova-lite-v1:0",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text"}],
                    }
                ],
            },
        )

        assert response.status_code == 400, response.text
        message = response.json()["error"]["message"]
        assert message == (
            "Validation error at body.input.0.EasyInputMessage.content.0"
            ".input_text.text: Field required"
        ), message
        assert "[" not in message, "the union wrappers Pydantic walked leaked out"


class TestRequestSetupErrors:
    """A header rejected before routing still answers with the API error envelope.

    The per-request setup runs in the outermost middleware, above the
    ``ExceptionMiddleware`` that serves the exception handlers and before the
    route's ``authenticate`` dependency: an ``ApiError`` escaping it reaches
    ``ServerErrorMiddleware``, which answers any caller, credentials or not, a
    bare ``500 Internal Server Error`` in ``text/plain`` and logs the request as
    critical.

    Ref: stdapi/main.py:_middleware
         stdapi/aws_bedrock_mantle.py:set_mantle_project
    """

    @pytest.mark.parametrize("header", ["OpenAI-Project", "anthropic-workspace"])
    def test_malformed_project_header_is_a_logged_400(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch, header: str
    ) -> None:
        """Either Mantle project header, when malformed, answers 400 and logs a warning.

        Both header names feed the same setup call, and a request-supplied
        header is honored on every default deployment (no configured project),
        so either one decides the request's status, body and log level.
        """
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_project", None)
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        response = app_client.get("/v1/models", headers={header: "not a project!"})

        assert response.status_code == 400, response.text
        assert response.json()["error"]["type"] == "invalid_request_error"
        assert response.headers["x-request-id"]
        assert [event["level"] for event in written] == ["warning"]

    def test_an_unauthenticated_caller_cannot_force_a_server_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without credentials the same header is a client error, not a 500.

        The setup runs before the route's authentication dependency, so any
        anonymous caller reaches it: it must not be a way to have every request
        answered as a server fault and logged critical.
        """
        from fastapi.testclient import TestClient  # noqa: PLC0415

        written: list[dict[str, Any]] = []
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_project", None)
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        response = TestClient(stdapi_main.app).get(
            "/v1/models", headers={"OpenAI-Project": "not a project!"}
        )

        assert response.status_code < 500, response.text
        assert [event["level"] for event in written] == ["warning"]

    def test_the_error_envelope_follows_the_target_route(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An Anthropic route answers its own error shape, not the default envelope.

        The request has not been routed when the setup fails, so the envelope
        only matches the API the client called if the target route is resolved
        before the error is formatted.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_project", None)

        response = anthropic_app_client.post(
            f"{SETTINGS.anthropic_routes_prefix}/v1/messages",
            headers={"anthropic-workspace": "not a project!"},
            json={},
        )

        assert response.status_code == 400, response.text
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert response.headers["request-id"]


class TestWarmedS3Clients:
    """Startup warms an S3 client for every region a configured bucket lives in.

    ``get_client`` only falls back to the single pooled client when exactly one
    exists, so a bucket whose region was never warmed raises ``KeyError`` on a
    multi-region deployment and turns an accepted input file into a 500.

    Ref: stdapi/main.py:lifespan
         stdapi/aws.py:get_client
         https://stdapi.ai/operations_configuration/
    """

    async def test_accepted_bucket_regions_are_warmed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The region declared for an accepted bucket gets its own S3 client.

        ``AWS_S3_ACCEPTED_BUCKETS`` declares buckets the gateway reads but does
        not own, in regions no other setting mentions.
        """
        warmed: list[tuple[str, str | None]] = []

        class _RecordingConnectionManager:
            """Records the requested clients, then aborts the startup."""

            def __init__(self, *clients: tuple[str, str | None]) -> None:
                warmed.extend(clients)

            async def __aenter__(self) -> Self:
                """Stop the startup once the client list is known."""
                msg = "startup aborted after the client list was built"
                raise ServerError(msg)

            async def __aexit__(self, *_exc: object) -> None:
                """Never reached: ``__aenter__`` always raises."""

        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1", "eu-west-1"])
        monkeypatch.setattr(SETTINGS, "aws_s3_regional_buckets", {"eu-west-1": "b-eu"})
        monkeypatch.setattr(SETTINGS, "aws_s3_accepted_buckets", {"data": "eu-west-3"})
        monkeypatch.setattr(
            stdapi_main, "AWSConnectionManager", _RecordingConnectionManager
        )
        monkeypatch.setattr(stdapi_main, "write_log_event", lambda _event: None)

        with pytest.raises(ServerError):
            async with stdapi_main.lifespan(stdapi_main.app):
                pass  # pragma: no cover - the manager aborts the startup

        assert ("s3", "eu-west-3") in warmed, (
            "the accepted bucket's region has no S3 client to reach it with"
        )


#: Loop iterations a started drain is given to finish, if it were not awaiting.
_DRAIN_YIELDS: int = 20


class _NullConnectionManager:
    """AWS client pool that warms nothing, so the lifespan runs without AWS."""

    def __init__(self, *_clients: tuple[str, str | None]) -> None:
        """Accept and ignore the requested clients."""

    async def __aenter__(self) -> Self:
        """Enter without opening anything.

        Returns:
            This manager.
        """
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Leave without closing anything."""


class _RecordingOtel:
    """Telemetry manager recording only when it is asked to flush."""

    def __init__(self, calls: list[str]) -> None:
        """Record into the shared call log.

        Args:
            calls: Ordered log of the shutdown steps observed.
        """
        self._calls = calls

    def start_span(self, _name: str, attributes: object = None) -> None:
        """Return no span context.

        Args:
            _name: Span name (unused).
            attributes: Span attributes (unused).
        """

    @contextmanager
    def use_span(self, _context: object) -> Generator[None]:
        """Enter no span.

        Args:
            _context: Span context (unused).

        Yields:
            None.
        """
        yield

    def flush(self) -> None:
        """Record the telemetry flush."""
        self._calls.append("flush")


class TestShutdownDrain:
    """Shutdown waits for the background work requests deliberately left running.

    A stop signal is routine — a deployment, a scale-in, a Spot interruption —
    and without the drain every task detached from its request is dropped when
    the process exits, silently. The wait is best effort: it is bounded, and
    whatever it cannot finish is cancelled and counted rather than abandoned.

    Ref: stdapi/main.py:drain_background_tasks
         stdapi/cleanup.py:drain_tasks
         https://stdapi.ai/operations_configuration/#shutdown-drain-timeout
    """

    @staticmethod
    def _detach_cleanup(release: Event) -> Event:
        """Detach one cleanup task that blocks until ``release`` is set.

        The blocking is explicit rather than merely slow, so "the drain waited"
        cannot be confused with "the work happened to finish first".

        Args:
            release: Event the cleanup waits on before completing.

        Returns:
            Event set once the cleanup has run to completion.
        """
        finished = Event()

        async def cleanup() -> None:
            await release.wait()
            finished.set()

        CLEANUPS.set([])
        schedule_cleanup(cleanup())
        run_cleanups_detached("shutdown-drain-test")
        return finished

    @staticmethod
    async def _flush_registries() -> None:
        """Drain what earlier tests detached, so the counts are this test's own."""
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(SETTINGS, "shutdown_drain_timeout", 5.0)
            assert await stdapi_main.drain_background_tasks() == {}

    async def test_drain_awaits_work_that_is_still_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The drain must not return while a detached task is still running.

        Without it the process exits here, and the temporary object the cleanup
        was going to delete is simply left behind.
        """
        await self._flush_registries()
        monkeypatch.setattr(SETTINGS, "shutdown_drain_timeout", 30.0)
        release = Event()
        finished = self._detach_cleanup(release)
        drain = create_task(stdapi_main.drain_background_tasks())
        try:
            for _ in range(_DRAIN_YIELDS):
                await sleep(0)
            assert not drain.done(), "the drain returned without awaiting the cleanup"
            release.set()
            assert await wait_for(drain, timeout=5) == {}
        finally:
            release.set()
        assert finished.is_set(), "the cleanup did not run to completion"

    async def test_drain_returns_within_its_timeout_and_settles_the_rest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Work that never finishes is cancelled at the deadline, and counted.

        The deadline is what has to hold: a container runtime kills the process
        a fixed delay after the stop signal, so a drain that waited for this
        cleanup would be killed mid-wait instead of reporting the loss.
        """
        await self._flush_registries()
        monkeypatch.setattr(SETTINGS, "shutdown_drain_timeout", 0.05)
        release = Event()
        finished = self._detach_cleanup(release)
        detached = next(iter(_DETACHED_TASKS))
        assert await wait_for(stdapi_main.drain_background_tasks(), timeout=5) == {
            "cleanups": 1
        }
        assert detached.cancelled(), "the unfinished cleanup was abandoned, not settled"
        assert not finished.is_set()
        release.set()

    async def test_a_zero_timeout_cancels_immediately_instead_of_waiting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``0`` is the operator's opt-out: stop as fast as the platform allows."""
        await self._flush_registries()
        monkeypatch.setattr(SETTINGS, "shutdown_drain_timeout", 0.0)
        release = Event()
        self._detach_cleanup(release)
        assert await wait_for(stdapi_main.drain_background_tasks(), timeout=5) == {
            "cleanups": 1
        }
        release.set()

    async def test_lifespan_drains_after_the_sessions_close_and_before_the_flush(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The drain sits where its inputs exist and its output can still be exported.

        Closing the realtime sessions is what enqueues their teardown, so the
        drain follows it; the AWS clients and the price catalog the pending work
        calls are torn down after it; and the telemetry flush comes last, so the
        stop event reporting the loss is exported rather than dropped.
        """
        await self._flush_registries()
        calls: list[str] = []
        events: list[Any] = []

        async def _none(*_args: object, **_kwargs: object) -> None:
            pass

        real_drain = stdapi_main.drain_background_tasks

        async def recording_drain() -> dict[str, int]:
            calls.append("drain")
            return await real_drain()

        async def recording_stop_price_catalog() -> None:
            calls.append("stop_price_catalog")

        def recording_close_sessions() -> None:
            calls.append("close_realtime_sessions")

        def recording_write_log_event(event: Any) -> None:  # noqa: ANN401
            calls.append(f"log:{event['type']}")
            events.append(event)

        for name in (
            "initialize_authentication",
            "initialize_aws_account_info",
            "initialize_bedrock_models",
            "initialize_moderation_models",
            "initialize_polly_models",
            "initialize_transcribe_models",
            "measure_region_latencies",
            "register",
            "verify_knowledge_bases",
            "verify_user_role_access",
        ):
            monkeypatch.setattr(stdapi_main, name, _none)
        for name in (
            "initialize_bidi_clients",
            "open_realtime_sessions",
            "start_price_catalog",
            "update_unified_models_collections",
        ):
            monkeypatch.setattr(stdapi_main, name, lambda: None)
        monkeypatch.setattr(stdapi_main, "AWSConnectionManager", _NullConnectionManager)
        monkeypatch.setattr(stdapi_main, "otel_manager", _RecordingOtel(calls))
        monkeypatch.setattr(stdapi_main, "drain_background_tasks", recording_drain)
        monkeypatch.setattr(
            stdapi_main, "stop_price_catalog", recording_stop_price_catalog
        )
        monkeypatch.setattr(
            stdapi_main, "close_realtime_sessions", recording_close_sessions
        )
        monkeypatch.setattr(stdapi_main, "write_log_event", recording_write_log_event)
        monkeypatch.setattr(SETTINGS, "shutdown_drain_timeout", 0.05)

        release = Event()
        async with stdapi_main.lifespan(stdapi_main.app):
            self._detach_cleanup(release)
        release.set()

        assert calls == [
            "log:start",
            "close_realtime_sessions",
            "drain",
            "stop_price_catalog",
            "log:stop",
            "flush",
        ]
        stop_event = events[-1]
        assert stop_event["abandoned_background_tasks"] == {"cleanups": 1}
        assert stop_event["level"] == "warning", (
            "a silent loss is the failure this reports"
        )
