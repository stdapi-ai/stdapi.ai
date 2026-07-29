"""Integration tests for error payload structure on OpenAI, Anthropic and Cohere routes.

Verifies that error responses match the official API envelope formats:
- OpenAI: ``{"error": {"message", "type", "param", "code"}}``
- Anthropic: ``{"type": "error", "error": {"type", "message"}, "request_id": <str>}``
- Cohere: ``{"message": <str>, "id": <str>}``
"""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError
from starlette.requests import Request
from starlette.responses import Response

from stdapi.api_providers.anthropic import TAG_ANTHROPIC
from stdapi.api_providers.anthropic import _format_error as anthropic_format_error
from stdapi.api_providers.cohere import TAG_COHERE
from stdapi.api_providers.cohere import _format_error as cohere_format_error
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.api_providers.openai import _format_error as openai_format_error
from stdapi.config import SETTINGS
from stdapi.main import handle_botocore_client_error, set_retry_after_header
from stdapi.monitoring import REQUEST
from stdapi.region_routing import RegionRouter, quota_retry_after

if TYPE_CHECKING:
    from collections.abc import Iterator

    from starlette.testclient import TestClient

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


def _openai_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}


def _cohere_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_openai_error_shape(body: dict[str, Any]) -> dict[str, Any]:
    """Assert the response body matches the OpenAI error envelope and return the inner error.

    The expected shape is::

        {
            "error": {
                "message": <str>,
                "type": <str>,
                "param": <str | None>,
                "code": <str | None>,
            }
        }

    Args:
        body: Parsed JSON response body.

    Returns:
        The inner ``error`` dict for further assertions.
    """
    assert set(body.keys()) == {"error"}, f"Unexpected top-level keys: {body.keys()}"
    err = body["error"]
    assert set(err.keys()) == {"message", "type", "param", "code"}, (
        f"Unexpected error keys: {err.keys()}"
    )
    assert isinstance(err["message"], str)
    assert isinstance(err["type"], str)
    assert err["param"] is None or isinstance(err["param"], str)
    assert err["code"] is None or isinstance(err["code"], str)
    return err  # type: ignore[no-any-return]


def _assert_anthropic_error_shape(body: dict[str, Any]) -> dict[str, Any]:
    """Assert the response body matches the Anthropic error envelope and return the inner error.

    The expected shape is::

        {
            "type": "error",
            "error": {
                "type": <str>,
                "message": <str>,
            },
            "request_id": <str>,
        }

    Args:
        body: Parsed JSON response body.

    Returns:
        The inner ``error`` dict for further assertions.
    """
    assert set(body.keys()) == {"type", "error", "request_id"}, (
        f"Unexpected top-level keys: {body.keys()}"
    )
    assert body["type"] == "error"
    assert isinstance(body["request_id"], str)
    err = body["error"]
    assert set(err.keys()) == {"type", "message"}, (
        f"Unexpected error keys: {err.keys()}"
    )
    assert isinstance(err["type"], str)
    assert isinstance(err["message"], str)
    return err  # type: ignore[no-any-return]


def _assert_cohere_error_shape(body: dict[str, Any]) -> str:
    """Assert the response body matches the Cohere error envelope and return the message.

    The expected shape is::

        {"message": <str>, "id": <str>}

    Args:
        body: Parsed JSON response body.

    Returns:
        The ``message`` string for further assertions.
    """
    assert set(body.keys()) == {"message", "id"}, (
        f"Unexpected top-level keys: {body.keys()}"
    )
    assert isinstance(body["message"], str)
    assert isinstance(body["id"], str)
    return body["message"]


class TestFormatErrorFunctions:
    """Pure unit tests for the provider `_format_error` envelope builders.

    No server or test client involved — these exercise the status-to-type
    mapping functions directly.
    """

    @pytest.mark.parametrize(
        ("status", "expected_type"),
        [
            (500, "server_error"),
            (502, "server_error"),
            (503, "server_error"),
            (529, "server_error"),
            (404, "invalid_request_error"),
            (429, "rate_limit_error"),
        ],
    )
    def test_openai_format_error_maps_status_to_type(
        self, status: int, expected_type: str
    ) -> None:
        """`_format_error` maps each status code to the expected OpenAI error type."""
        body, returned_status = openai_format_error(status, "boom")
        err = _assert_openai_error_shape(body)
        assert err["type"] == expected_type
        assert returned_status == status

    @pytest.mark.parametrize(
        ("status", "expected_type", "expected_status"),
        [
            (402, "billing_error", 402),
            (409, "conflict_error", 409),
            (413, "request_too_large", 413),
            (500, "api_error", 500),
            (502, "api_error", 502),
            (503, "overloaded_error", 529),
            (504, "timeout_error", 504),
            (529, "overloaded_error", 529),
        ],
    )
    def test_anthropic_format_error_maps_status_to_type(
        self, status: int, expected_type: str, expected_status: int
    ) -> None:
        """`_format_error` maps each status code to type, remapping 503 to 529."""
        body, returned_status = anthropic_format_error(status, "boom")
        err = _assert_anthropic_error_shape(body)
        assert err["type"] == expected_type
        assert returned_status == expected_status

    @pytest.mark.parametrize("status", [400, 401, 404, 429, 500])
    def test_cohere_format_error_returns_message_envelope(self, status: int) -> None:
        """`_format_error` returns the flat Cohere envelope with the status unchanged."""
        body, returned_status = cohere_format_error(status, "boom")
        assert _assert_cohere_error_shape(body) == "boom"
        assert returned_status == status

    def test_anthropic_format_error_includes_body_request_id(self) -> None:
        """`_format_error` echoes the current request ID as a top-level ``request_id`` field."""
        from stdapi.monitoring import REQUEST_ID  # noqa: PLC0415

        token = REQUEST_ID.set("req_test123")
        try:
            body, _ = anthropic_format_error(404, "boom")
        finally:
            REQUEST_ID.reset(token)
        assert body["request_id"] == "req_test123"

    def test_cohere_format_error_includes_id(self) -> None:
        """`_format_error` echoes the current request ID as the ``id`` field."""
        from stdapi.monitoring import REQUEST_ID  # noqa: PLC0415

        token = REQUEST_ID.set("req_test456")
        try:
            body, _ = cohere_format_error(404, "boom")
        finally:
            REQUEST_ID.reset(token)
        assert body["id"] == "req_test456"


class TestOpenaiErrorPayloads:
    """Verify OpenAI routes return the correct error envelope structure."""

    @pytest.fixture(autouse=True)
    def _skip_non_local(self, test_client: TestClient) -> None:
        """Skip web search tests when running against the official Anthropic API."""
        if not test_client:
            pytest.skip("Unittest only for local tests.")

    def test_invalid_model_returns_openai_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """A non-existent model on an OpenAI route must return 404 with the OpenAI error shape.

        Validates:
            - HTTP 404 status code.
            - ``{"error": {"message", "type", "param", "code"}}`` envelope.
            - ``type`` is ``"invalid_request_error"`` and ``code`` is ``"model_not_found"``.
        """
        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model-xyz",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=_openai_headers(api_key),
        )
        assert resp.status_code == 404
        err = _assert_openai_error_shape(resp.json())
        assert err["type"] == "invalid_request_error"
        assert err["code"] == "model_not_found"
        assert "model" in err["message"].lower()

    def test_validation_error_returns_openai_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """A Pydantic validation error on an OpenAI route must return 400 with the OpenAI shape.

        Validates:
            - HTTP 400 status code.
            - Correct OpenAI error envelope structure.
            - ``type`` is ``"invalid_request_error"``.
        """
        resp = test_client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": "not-a-list"},
            headers=_openai_headers(api_key),
        )
        assert resp.status_code == 400
        err = _assert_openai_error_shape(resp.json())
        assert err["type"] == "invalid_request_error"

    def test_auth_error_returns_openai_envelope(self, test_client: TestClient) -> None:
        """A missing/invalid API key on an OpenAI route must return 401 with the OpenAI shape.

        Validates:
            - HTTP 401 status code.
            - Correct OpenAI error envelope structure.
            - ``type`` is ``"authentication_error"``.
        """
        resp = test_client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401
        err = _assert_openai_error_shape(resp.json())
        assert err["type"] == "authentication_error"


class TestAnthropicErrorPayloads:
    """Verify Anthropic routes return the correct error envelope structure."""

    @pytest.fixture(autouse=True)
    def _skip_non_local(self, test_client: TestClient) -> None:
        """Skip web search tests when running against the official Anthropic API."""
        if not test_client:
            pytest.skip("Unittest only for local tests.")

    def test_invalid_model_returns_anthropic_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """A non-existent model on an Anthropic route must return 404 with the Anthropic error shape.

        The official Anthropic API returns 404 ``not_found_error`` for
        unknown models.

        Validates:
            - HTTP 404 status code.
            - ``{"type": "error", "error": {"type", "message"}}`` envelope.
            - Inner ``type`` is ``"not_found_error"``.
        """
        resp = test_client.post(
            "/anthropic/v1/messages",
            json={
                "model": "nonexistent-model-xyz",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=_anthropic_headers(api_key),
        )
        assert resp.status_code == 404
        err = _assert_anthropic_error_shape(resp.json())
        assert err["type"] == "not_found_error"
        assert "model" in err["message"].lower()

    def test_validation_error_returns_anthropic_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """A Pydantic validation error on an Anthropic route must return 400 with the Anthropic shape.

        Validates:
            - HTTP 400 status code.
            - Correct Anthropic error envelope structure.
            - Inner ``type`` is ``"invalid_request_error"``.
        """
        resp = test_client.post(
            "/anthropic/v1/messages",
            json={"model": "x", "max_tokens": 100, "messages": "not-a-list"},
            headers=_anthropic_headers(api_key),
        )
        assert resp.status_code == 400
        err = _assert_anthropic_error_shape(resp.json())
        assert err["type"] == "invalid_request_error"

    def test_auth_error_returns_anthropic_envelope(
        self, test_client: TestClient
    ) -> None:
        """A missing/invalid API key on an Anthropic route must return 401 with the Anthropic shape.

        Validates:
            - HTTP 401 status code.
            - Correct Anthropic error envelope structure.
            - Inner ``type`` is ``"authentication_error"``.
        """
        resp = test_client.post(
            "/anthropic/v1/messages",
            json={
                "model": "x",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"x-api-key": "wrong-key", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 401
        err = _assert_anthropic_error_shape(resp.json())
        assert err["type"] == "authentication_error"


class TestCohereErrorPayloads:
    """Verify Cohere routes return the correct error envelope structure."""

    @pytest.fixture(autouse=True)
    def _skip_non_local(self, test_client: TestClient) -> None:
        """Skip web search tests when running against the official Anthropic API."""
        if not test_client:
            pytest.skip("Unittest only for local tests.")

    def test_invalid_model_returns_cohere_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """A non-existent model on a Cohere route must return 404 with the Cohere error shape.

        Validates:
            - HTTP 404 status code.
            - ``{"message": <str>}`` envelope.
            - The message mentions the model.
        """
        resp = test_client.post(
            "/cohere/v2/rerank",
            json={"model": "nonexistent-model-xyz", "query": "q", "documents": ["a"]},
            headers=_cohere_headers(api_key),
        )
        assert resp.status_code == 404
        message = _assert_cohere_error_shape(resp.json())
        assert "model" in message.lower()

    def test_validation_error_returns_cohere_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """A Pydantic validation error on a Cohere route must return 400 with the Cohere shape.

        Validates:
            - HTTP 400 status code.
            - Correct Cohere error envelope structure.
        """
        resp = test_client.post(
            "/cohere/v2/rerank",
            json={"model": "x", "query": "q", "documents": "not-a-list"},
            headers=_cohere_headers(api_key),
        )
        assert resp.status_code == 400
        _assert_cohere_error_shape(resp.json())

    def test_auth_error_returns_cohere_envelope(self, test_client: TestClient) -> None:
        """A missing/invalid API key on a Cohere route must return 401 with the Cohere shape.

        Validates:
            - HTTP 401 status code.
            - Correct Cohere error envelope structure.
        """
        resp = test_client.post(
            "/cohere/v2/rerank",
            json={"model": "x", "query": "q", "documents": ["a"]},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401
        _assert_cohere_error_shape(resp.json())

    def test_embed_auth_error_returns_cohere_envelope(
        self, test_client: TestClient
    ) -> None:
        """A missing/invalid API key on the embed route must return 401 with the Cohere shape.

        Validates:
            - HTTP 401 status code.
            - Correct Cohere error envelope structure.
        """
        resp = test_client.post(
            "/cohere/v2/embed",
            json={"model": "x", "texts": ["a"], "input_type": "search_document"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401
        _assert_cohere_error_shape(resp.json())


class TestCrossRouteConsistency:
    """Verify the same logical error produces different envelopes per route type."""

    @pytest.fixture(autouse=True)
    def _skip_non_local(self, test_client: TestClient) -> None:
        """Skip web search tests when running against the official Anthropic API."""
        if not test_client:
            pytest.skip("Unittest only for local tests.")

    def test_same_invalid_model_different_envelopes(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """The same non-existent model must produce OpenAI envelope on /v1 and Anthropic envelope on /anthropic/v1.

        Validates:
            - Both routes return 404 (matching official APIs).
            - OpenAI route has ``"error"`` top-level key only.
            - Anthropic route has ``"type": "error"`` top-level key.
            - Neither envelope leaks fields from the other format.
        """
        openai_resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model-xyz",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=_openai_headers(api_key),
        )
        anthropic_resp = test_client.post(
            "/anthropic/v1/messages",
            json={
                "model": "nonexistent-model-xyz",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=_anthropic_headers(api_key),
        )

        assert openai_resp.status_code == 404
        assert anthropic_resp.status_code == 404

        openai_body = openai_resp.json()
        anthropic_body = anthropic_resp.json()

        # OpenAI envelope must NOT have top-level "type"
        assert "type" not in openai_body
        _assert_openai_error_shape(openai_body)

        # Anthropic envelope must NOT have top-level "error.param" or "error.code"
        _assert_anthropic_error_shape(anthropic_body)
        assert "param" not in anthropic_body["error"]
        assert "code" not in anthropic_body["error"]


class TestRoutingErrorPayloads:
    """Verify Starlette-level routing failures (404/405) use an error envelope.

    Regression coverage for BUG-1: the base ``starlette.exceptions.HTTPException``
    raised by the router for no-route/method-not-allowed must not bypass the
    custom exception handler and fall back to Starlette's default ``{"detail": ...}``.
    """

    @pytest.fixture(autouse=True)
    def _skip_non_local(self, test_client: TestClient) -> None:
        """Skip when running against a remote/official API (no in-process router)."""
        if not test_client:
            pytest.skip("Unittest only for local tests.")

    def test_wrong_method_returns_openai_envelope(
        self, test_client: TestClient
    ) -> None:
        """GET on the POST-only OpenAI embeddings route returns a 405 OpenAI envelope."""
        resp = test_client.get("/v1/embeddings")
        assert resp.status_code == 405
        assert "detail" not in resp.json()
        err = _assert_openai_error_shape(resp.json())
        assert err["type"] == "invalid_request_error"

    def test_unknown_path_returns_error_envelope(self, test_client: TestClient) -> None:
        """A request to an undefined path returns a 404 error envelope, not Starlette's default."""
        resp = test_client.get("/v1/nonexistent")
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" not in body
        assert "error" in body


def _openai_request(
    method: str = "POST", path: str = "/v1/chat/completions"
) -> Request:
    """Return a request scoped to an OpenAI-tagged route.

    Args:
        method: HTTP method for the request scope.
        path: URL path for the request scope.

    Returns:
        Starlette request whose resolved route carries the OpenAI tag, so
        ``format_http_error`` dispatches to the OpenAI envelope formatter.
    """
    return Request(
        {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],
            "route": SimpleNamespace(tags=[TAG_OPENAI]),
        }
    )


def _client_error(code: str, message: str = "boom") -> ClientError:
    """Build a botocore ClientError carrying the given AWS error code.

    Args:
        code: AWS error code (e.g. ``"ThrottlingException"``).
        message: Error message.

    Returns:
        A ``ClientError`` mimicking one raised by an AWS SDK call.
    """
    return ClientError({"Error": {"Code": code, "Message": message}}, "SomeOperation")


class TestBotocoreClientErrorEnvelope:
    """Verify ``handle_botocore_client_error`` builds a correct OpenAI envelope.

    Regression coverage for BUG-2 (bogus ``param``) and BUG-3 (S3 multipart
    errors falling through to 502).
    """

    @pytest.fixture(autouse=True)
    def _request_log_context(self) -> Iterator[None]:
        """Provide the request-log context that logging outside request scope needs."""
        from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

        token = REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
        yield
        REQUEST_LOG.reset(token)

    async def test_client_error_param_is_not_the_internal_error_type(self) -> None:
        """The envelope's ``param`` must stay None; it must not leak the AWS_ERROR_MAP entry."""
        response = await handle_botocore_client_error(
            _openai_request(), _client_error("ThrottlingException")
        )
        err = _assert_openai_error_shape(json.loads(bytes(response.body)))
        assert response.status_code == 429
        assert err["param"] is None
        assert err["code"] == "ThrottlingException"

    @pytest.mark.parametrize("code", ["EntityTooSmall", "InvalidPart"])
    async def test_s3_multipart_client_errors_map_to_400(self, code: str) -> None:
        """S3 multipart-upload validation errors map to 400 invalid_request_error, not 502."""
        response = await handle_botocore_client_error(
            _openai_request(), _client_error(code)
        )
        err = _assert_openai_error_shape(json.loads(bytes(response.body)))
        assert response.status_code == 400
        assert err["type"] == "invalid_request_error"
        assert err["param"] is None
        assert err["code"] == code


def _tagged_request(tag: str) -> Request:
    """Build a request whose resolved route carries the given API provider tag.

    Args:
        tag: Route tag identifying the API provider.

    Returns:
        Starlette request usable by the response-header stage.
    """
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/",
            "headers": [],
            "route": SimpleNamespace(tags=[tag]),
        }
    )


class TestRetryAfterHeader:
    """Verify 429 responses advertise the region router's quota backoff.

    ``retry-after`` is a standard HTTP header shared by all three provider
    envelopes, so it is emitted by the common response-header stage.
    """

    @pytest.fixture(autouse=True)
    def _request_log_context(self) -> Iterator[None]:
        """Provide the request-log context that ``mark_error`` logging needs."""
        from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

        token = REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
        yield
        REQUEST_LOG.reset(token)

    @staticmethod
    def _mark_error(
        request: Request,
        code: str,
        region: str = "us-east-1",
        router: RegionRouter | None = None,
    ) -> None:
        """Record *code* against *region* while *request* is the active request."""
        token = REQUEST.set(request)
        try:
            (router or RegionRouter()).mark_error(
                "amazon.nova-micro-v1:0", region, code
            )
        finally:
            REQUEST.reset(token)

    @pytest.mark.parametrize("tag", [TAG_OPENAI, TAG_ANTHROPIC, TAG_COHERE])
    def test_quota_backoff_is_advertised_on_429(self, tag: str) -> None:
        """Every provider envelope gets the router backoff as ``retry-after`` on a 429."""
        request = _tagged_request(tag)
        self._mark_error(request, "ThrottlingException")
        response = Response(status_code=429)
        set_retry_after_header(request, response)
        assert response.headers["retry-after"] == str(
            SETTINGS.aws_bedrock_region_routing_quota_backoff_seconds
        )

    def test_no_header_when_no_backoff_was_applied(self) -> None:
        """Without a recorded quota backoff the header is omitted, not invented."""
        response = Response(status_code=429)
        set_retry_after_header(_tagged_request(TAG_OPENAI), response)
        assert "retry-after" not in response.headers

    def test_unavailability_backoff_is_not_advertised(self) -> None:
        """Unavailability backoffs drive 503/529 responses and carry no ``retry-after``."""
        request = _tagged_request(TAG_OPENAI)
        self._mark_error(request, "ServiceUnavailableException")
        assert quota_retry_after(request) is None

    def test_header_only_on_rate_limited_responses(self) -> None:
        """A successful response never carries ``retry-after``."""
        request = _tagged_request(TAG_OPENAI)
        self._mark_error(request, "ThrottlingException")
        response = Response(status_code=200)
        set_retry_after_header(request, response)
        assert "retry-after" not in response.headers

    def test_fractional_backoff_is_rounded_up(self) -> None:
        """Sub-second precision rounds up so the client never retries too early."""
        request = _tagged_request(TAG_OPENAI)
        request.scope["state"] = {"stdapi_quota_backoff_seconds": 12.1}
        assert quota_retry_after(request) == 13

    def test_smallest_backoff_across_regions_wins(self) -> None:
        """With several regions blocked, the earliest one to recover sets the delay."""
        base = SETTINGS.aws_bedrock_region_routing_quota_backoff_seconds
        router = RegionRouter()
        self._mark_error(
            _tagged_request(TAG_OPENAI), "ThrottlingException", router=router
        )

        request = _tagged_request(TAG_OPENAI)
        # us-east-1 escalates to twice the base backoff; us-west-2 is still fresh.
        self._mark_error(request, "ThrottlingException", router=router)
        assert quota_retry_after(request) == 2 * base
        self._mark_error(
            request, "ThrottlingException", region="us-west-2", router=router
        )
        assert quota_retry_after(request) == base
