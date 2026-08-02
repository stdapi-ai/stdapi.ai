"""Application assembly and exception handling in :mod:`stdapi.main`.

Covers the ASGI app's exception handlers (botocore errors, TaskGroup-wrapped
exception groups), the request-scoped input-file tracking the middleware resets,
and router discovery when two providers share a routes prefix.

Ref: stdapi/main.py
"""

from __future__ import annotations

from asyncio import CancelledError, Event, wait_for
from contextlib import asynccontextmanager
from inspect import signature
from io import BytesIO
from json import loads
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError
from httpx import ASGITransport, AsyncClient
from starlette.datastructures import UploadFile
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request

from stdapi import main as stdapi_main
from stdapi import metering
from stdapi.cleanup import schedule_cleanup
from stdapi.config import AWS_REGION
from stdapi.exceptions import (
    InvalidProductError,
    NotEntitledError,
    UnsupportedPlatformError,
)
from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile
from stdapi.main import handle_exception_group
from stdapi.routes import core_root, openai_chat_completions
from stdapi.server import SERVER_ID
from stdapi.types.openai_chat_completions import ChatCompletion
from tests._helpers import make_client_error, make_event_log, make_model_details

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

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
        if middleware.cls is GZipMiddleware:
            return int(
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

        Reproduces the cross-request leak: an ``InputFile`` tracked in a
        longer-lived context (here, the test task; in production, a keep-alive
        connection task) whose backing file is already closed used to make
        every later request 500 during content-type prefetch. The backend records
        what it saw, so an empty list is the proof the reset happened.
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
        from fastapi.routing import APIRoute  # noqa: PLC0415

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
            methods_seen = Counter(
                (route.path, method)
                for route in app.routes
                if isinstance(route, APIRoute)
                for method in route.methods or ()
            )
            assert not [key for key, count in methods_seen.items() if count > 1]
        finally:
            monkeypatch.undo()
            reload(anthropic_files)
            reload(anthropic_models)
        assert anthropic_files.router is not None
        assert anthropic_models.router is not None


class TestMiddlewareCleanupDrain:
    """Scheduled cleanups still run when no response reaches the client.

    Ref: stdapi/main.py:_middleware
         stdapi/cleanup.py:run_cleanups_detached
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
