"""Unit tests for application-level helpers (:mod:`stdapi.main`)."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError
from httpx import ASGITransport, AsyncClient
from starlette.datastructures import UploadFile
from starlette.requests import Request

from stdapi.input_file import _CURRENT_INPUT_FILES, InputFile
from stdapi.main import _upstream_service_name, handle_exception_group
from stdapi.models import ModelDetails
from stdapi.routes import openai_chat_completions
from stdapi.types.openai_chat_completions import ChatCompletion

if TYPE_CHECKING:
    from collections.abc import Iterator

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _request_log_context() -> Iterator[None]:
    """Provide the request-log context that logging outside request scope needs."""
    from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

    token = REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
    yield
    REQUEST_LOG.reset(token)


class _FakeConnectionError:
    """Minimal stand-in for a botocore connection error carrying ``kwargs``."""

    def __init__(self, endpoint_url: str) -> None:
        self.kwargs = {"endpoint_url": endpoint_url}


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://bedrock-runtime.us-east-1.amazonaws.com", "Bedrock"),
        ("https://s3.eu-west-1.amazonaws.com", "S3"),
        ("https://polly.us-east-1.amazonaws.com", "Polly"),
    ],
)
def test_upstream_service_name_maps_known_services(
    endpoint: str, expected: str
) -> None:
    """A recognised AWS endpoint host yields its friendly service name."""
    assert _upstream_service_name(_FakeConnectionError(endpoint)) == expected  # type: ignore[arg-type]


def test_upstream_service_name_hides_unknown_endpoint() -> None:
    """A custom/VPC endpoint host is not disclosed; a generic name is returned."""
    exc = _FakeConnectionError("https://vpce-0abc123def.execute-api.example")
    assert _upstream_service_name(exc) == "The upstream service"  # type: ignore[arg-type]


def test_upstream_service_name_handles_missing_endpoint() -> None:
    """An error without an endpoint URL yields the generic name."""
    assert _upstream_service_name(_FakeConnectionError("")) == "The upstream service"  # type: ignore[arg-type]


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
    """Handler resolution for TaskGroup-wrapped exceptions."""

    async def test_client_error_subclass_resolves_to_botocore_handler(self) -> None:
        """A ClientError subclass inside an ExceptionGroup must not 500."""
        error = _ValidationException(
            {"Error": {"Code": "ValidationException", "Message": "bad input"}},
            "Converse",
        )
        group = ExceptionGroup("task group", [error])
        response = await handle_exception_group(_request(), group)
        assert response.status_code == 400

    async def test_unhandled_exception_reraises_the_group(self) -> None:
        """Groups holding unhandled exception types propagate unchanged."""
        group = ExceptionGroup("task group", [RuntimeError("boom")])
        with pytest.raises(ExceptionGroup):
            await handle_exception_group(_request(), group)


async def _validate_model(model_id: str, *_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
    """Return canned model details without hitting AWS."""
    return ModelDetails(
        id=model_id,
        name=model_id,
        provider="Vendor",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=["us-east-1"],
    )


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
    """Input-file tracking must be scoped to each request context."""

    async def test_stale_input_files_do_not_leak_into_requests(
        self, monkeypatch: pytest.MonkeyPatch, api_key: str
    ) -> None:
        """A stale InputFile bound to an outer context must not reach a request.

        Reproduces the cross-request leak: an ``InputFile`` tracked in a
        longer-lived context (here, the test task; in production, a keep-alive
        connection task) whose backing file is already closed used to make
        every later request 500 during content-type prefetch.
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
    """The 503 connection-error handler hides internal endpoint detail."""

    async def test_generic_body_while_endpoint_stays_in_logs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clients see a generic per-service 503; the endpoint URL is logged only."""
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
        assert "Bedrock is temporarily unavailable." in body
        assert endpoint not in body
        assert any(endpoint in entry for entry in logged)

    async def test_unknown_endpoint_is_not_disclosed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A custom/VPC endpoint is never named or disclosed in the 503 body."""
        from botocore.exceptions import EndpointConnectionError  # noqa: PLC0415

        from stdapi import main  # noqa: PLC0415

        endpoint = "https://vpce-0abc123.execute-api.internal"
        monkeypatch.setattr(main, "log_error_details", lambda *_a, **_k: None)

        response = await main.handle_botocore_connection_error(
            _request(), EndpointConnectionError(endpoint_url=endpoint)
        )

        assert response.status_code == 503
        body = bytes(response.body).decode()
        assert "The upstream service is temporarily unavailable." in body
        assert "vpce-0abc123" not in body
