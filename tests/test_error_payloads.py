"""Error envelopes emitted by the OpenAI, Anthropic and Cohere routes.

Each provider surface carries its own wire shape, selected from the matched
route's tag:

- OpenAI: ``{"error": {"message", "type", "param", "code"}}``
- Anthropic: ``{"type": "error", "error": {"type", "message"}, "request_id": <str>}``
- Cohere: ``{"message": <str>, "id": <str>}``
- No matched route: the minimal ``{"error": <message>}`` fallback.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://platform.claude.com/docs/en/api/errors
     https://docs.cohere.com/reference/errors
     stdapi/api_providers/__init__.py:format_http_error
"""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
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
from tests._helpers import make_client_error

if TYPE_CHECKING:
    from starlette.testclient import TestClient

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


def _bearer_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_openai_error_shape(body: dict[str, Any]) -> dict[str, Any]:
    """Assert the response body matches the OpenAI error envelope and return the inner error.

    The gateway always emits all four inner keys (``param``/``code`` possibly
    null), matching the OpenAPI spec's ``required: [type, message, param, code]``
    rather than the sparser bodies the live OpenAI API sometimes returns::

        {
            "error": {
                "message": <str>,
                "type": <str>,
                "param": <str | None>,
                "code": <str | None>,
            }
        }

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

    Cohere's own reference never specifies the error JSON, so the flat
    ``{"message", "id"}`` shape is the gateway's contract::

        {"message": <str>, "id": <str>}

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
    """The per-provider ``_format_error`` builders map an HTTP status to an error type.

    These call the envelope builders directly: no server, no test client.

    Ref: stdapi/api_providers/openai.py:_format_error
         stdapi/api_providers/anthropic.py:_format_error
         stdapi/api_providers/cohere.py:_format_error
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
        """Each status code maps to its OpenAI ``error.type``, and the status is unchanged.

        529 is not an OpenAI status: the gateway maps it to ``server_error`` so
        an Anthropic-style overload surfaced on an OpenAI route stays typed.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        body, returned_status = openai_format_error(status, "boom")
        err = _assert_openai_error_shape(body)
        assert err["type"] == expected_type
        assert err["message"] == "boom"
        assert err["param"] is None
        assert err["code"] is None
        assert returned_status == status

    @pytest.mark.parametrize(
        ("status", "expected_type", "expected_status"),
        [
            (402, "billing_error", 402),
            (409, "invalid_request_error", 409),
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
        """Each status maps to its Anthropic ``error.type``, and 503 is re-emitted as 529.

        Anthropic's documented table has no 503 entry: the gateway rewrites an
        upstream 503 to ``overloaded_error`` with HTTP 529, the status the
        Anthropic SDKs treat as retryable overload. 502 likewise collapses to
        ``api_error``. Neither remap is documented upstream. 409 has no
        Anthropic error type at all (the SDK union has no ``conflict_error``),
        so it degrades to the default ``invalid_request_error``.

        Ref: https://platform.claude.com/docs/en/api/errors
        """
        body, returned_status = anthropic_format_error(status, "boom")
        err = _assert_anthropic_error_shape(body)
        assert err["type"] == expected_type
        assert err["message"] == "boom"
        assert returned_status == expected_status

    @pytest.mark.parametrize("status", [400, 401, 404, 429, 500])
    def test_cohere_format_error_returns_message_envelope(self, status: int) -> None:
        """The Cohere envelope carries the message verbatim and never rewrites the status.

        Unlike the other two providers there is no status-to-type table: the
        status alone conveys the error class.

        Ref: https://docs.cohere.com/reference/errors
        """
        body, returned_status = cohere_format_error(status, "boom")
        assert _assert_cohere_error_shape(body) == "boom"
        assert returned_status == status

    def test_anthropic_format_error_includes_body_request_id(self) -> None:
        """The active request ID is echoed as the top-level ``request_id`` field.

        Anthropic's error object carries ``request_id`` in the body, not only in
        the ``request-id`` header, so SDK users can quote it from a caught error.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/monitoring.py:REQUEST_ID
        """
        from stdapi.monitoring import REQUEST_ID  # noqa: PLC0415

        token = REQUEST_ID.set("req_test123")
        try:
            body, _ = anthropic_format_error(404, "boom")
        finally:
            REQUEST_ID.reset(token)
        assert body["request_id"] == "req_test123"
        assert _assert_anthropic_error_shape(body)["type"] == "not_found_error"

    def test_cohere_format_error_includes_id(self) -> None:
        """The active request ID is echoed as the Cohere envelope's ``id`` field.

        Ref: stdapi/api_providers/cohere.py:_format_error
        """
        from stdapi.monitoring import REQUEST_ID  # noqa: PLC0415

        token = REQUEST_ID.set("req_test456")
        try:
            body, _ = cohere_format_error(404, "boom")
        finally:
            REQUEST_ID.reset(token)
        assert body["id"] == "req_test456"
        assert _assert_cohere_error_shape(body) == "boom"


class TestOpenaiErrorPayloads:
    """OpenAI-tagged routes return the OpenAI error envelope for every failure class.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/api_providers/openai.py:_format_error
    """

    def test_invalid_model_returns_openai_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """An unknown model yields 404 ``invalid_request_error`` / ``model_not_found``.

        404 has no entry in the OpenAI status table, so it falls back to
        ``invalid_request_error``; the machine-readable discriminator is
        ``code``, carried from ``UnsupportedModelError.code``.

        Ref: stdapi/api_errors.py:UnsupportedModelError
        """
        resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model-xyz",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=_bearer_headers(api_key),
        )
        assert resp.status_code == 404
        err = _assert_openai_error_shape(resp.json())
        assert err["type"] == "invalid_request_error"
        assert err["code"] == "model_not_found"
        assert "nonexistent-model-xyz" in err["message"]
        assert "does not exist" in err["message"]

    def test_validation_error_returns_openai_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """A request-body validation failure yields 400 ``invalid_request_error``.

        The handler flattens the first Pydantic error into a single
        ``"Validation error at <loc>: <msg>"`` sentence naming the offending
        field, and leaves ``param``/``code`` null.

        Ref: stdapi/main.py:handle_validation_exception
        """
        resp = test_client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": "not-a-list"},
            headers=_bearer_headers(api_key),
        )
        assert resp.status_code == 400
        err = _assert_openai_error_shape(resp.json())
        assert err["type"] == "invalid_request_error"
        assert err["message"].startswith("Validation error")
        assert "messages" in err["message"]
        assert err["param"] is None
        assert err["code"] is None

    def test_auth_error_returns_openai_envelope(self, test_client: TestClient) -> None:
        """A wrong API key yields 401 ``authentication_error`` with a detail-free message.

        The message is fixed to ``"Unauthorized"``: ``hide_security_details``
        strips every 401/403 detail so a rejected credential never leaks why.

        Ref: stdapi/utils.py:hide_security_details
        """
        resp = test_client.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
            headers=_bearer_headers("wrong-key"),
        )
        assert resp.status_code == 401
        err = _assert_openai_error_shape(resp.json())
        assert err["type"] == "authentication_error"
        assert err["message"] == "Unauthorized"


class TestAnthropicErrorPayloads:
    """Anthropic-tagged routes return the Anthropic error envelope for every failure class.

    Ref: https://platform.claude.com/docs/en/api/errors
         stdapi/api_providers/anthropic.py:_format_error
    """

    def test_invalid_model_returns_anthropic_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """An unknown model yields 404 ``not_found_error`` in the Anthropic envelope.

        Anthropic maps 404 to ``not_found_error``, so the same gateway
        ``UnsupportedModelError`` that OpenAI routes report as
        ``invalid_request_error``/``model_not_found`` is typed differently here.

        Ref: stdapi/api_errors.py:UnsupportedModelError
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
        assert "nonexistent-model-xyz" in err["message"]
        assert "does not exist" in err["message"]

    def test_validation_error_returns_anthropic_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """A request-body validation failure yields 400 ``invalid_request_error``.

        400 has no entry in the Anthropic status table either, so it resolves
        through the same default as OpenAI's — but wrapped in Anthropic's
        ``{"type": "error", ...}`` envelope.

        Ref: stdapi/main.py:handle_validation_exception
        """
        resp = test_client.post(
            "/anthropic/v1/messages",
            json={"model": "x", "max_tokens": 100, "messages": "not-a-list"},
            headers=_anthropic_headers(api_key),
        )
        assert resp.status_code == 400
        err = _assert_anthropic_error_shape(resp.json())
        assert err["type"] == "invalid_request_error"
        assert err["message"].startswith("Validation error")
        assert "messages" in err["message"]

    def test_auth_error_returns_anthropic_envelope(
        self, test_client: TestClient
    ) -> None:
        """A wrong ``x-api-key`` yields 401 ``authentication_error``, detail-free.

        Ref: stdapi/utils.py:hide_security_details
        """
        resp = test_client.post(
            "/anthropic/v1/messages",
            json={
                "model": "x",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=_anthropic_headers("wrong-key"),
        )
        assert resp.status_code == 401
        err = _assert_anthropic_error_shape(resp.json())
        assert err["type"] == "authentication_error"
        assert err["message"] == "Unauthorized"


class TestCohereErrorPayloads:
    """Cohere-tagged routes return the flat Cohere error envelope for every failure class.

    The envelope has no ``type`` discriminator, so the HTTP status is the only
    error classifier and the request ID travels in ``id``.

    Ref: https://docs.cohere.com/reference/errors
         stdapi/api_providers/cohere.py:_format_error
    """

    def test_invalid_model_returns_cohere_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """An unknown rerank model yields 404 with the requested ID named in ``message``.

        Ref: stdapi/api_errors.py:UnsupportedModelError
        """
        resp = test_client.post(
            "/cohere/v2/rerank",
            json={"model": "nonexistent-model-xyz", "query": "q", "documents": ["a"]},
            headers=_bearer_headers(api_key),
        )
        assert resp.status_code == 404
        message = _assert_cohere_error_shape(resp.json())
        assert "nonexistent-model-xyz" in message
        assert "does not exist" in message

    def test_validation_error_returns_cohere_envelope(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """A request-body validation failure yields 400 with the flattened Pydantic message.

        Ref: stdapi/main.py:handle_validation_exception
        """
        resp = test_client.post(
            "/cohere/v2/rerank",
            json={"model": "x", "query": "q", "documents": "not-a-list"},
            headers=_bearer_headers(api_key),
        )
        assert resp.status_code == 400
        message = _assert_cohere_error_shape(resp.json())
        assert message.startswith("Validation error")
        assert "documents" in message

    def test_auth_error_returns_cohere_envelope(self, test_client: TestClient) -> None:
        """A wrong API key on the rerank route yields 401 with a detail-free message.

        Ref: stdapi/utils.py:hide_security_details
        """
        resp = test_client.post(
            "/cohere/v2/rerank",
            json={"model": "x", "query": "q", "documents": ["a"]},
            headers=_bearer_headers("wrong-key"),
        )
        assert resp.status_code == 401
        assert _assert_cohere_error_shape(resp.json()) == "Unauthorized"

    def test_embed_auth_error_returns_cohere_envelope(
        self, test_client: TestClient
    ) -> None:
        """A wrong API key on the embed route yields the same 401 Cohere envelope.

        Embed is mounted from a different router than rerank, so it needs its own
        coverage that the Cohere tag (and therefore the envelope) is attached.

        Ref: stdapi/routes/cohere_embed.py
        """
        resp = test_client.post(
            "/cohere/v2/embed",
            json={"model": "x", "texts": ["a"], "input_type": "search_document"},
            headers=_bearer_headers("wrong-key"),
        )
        assert resp.status_code == 401
        assert _assert_cohere_error_shape(resp.json()) == "Unauthorized"


class TestCrossRouteConsistency:
    """One gateway error is rendered in whichever envelope the matched route's tag selects.

    Ref: stdapi/api_providers/__init__.py:format_http_error
    """

    def test_same_invalid_model_different_envelopes(
        self, test_client: TestClient, api_key: str
    ) -> None:
        """One unknown model yields an OpenAI envelope on /v1 and an Anthropic one on /anthropic/v1.

        Both surfaces report 404 for the same ``UnsupportedModelError``, but
        neither envelope may leak the other's fields: the OpenAI body has no
        top-level ``type``, and the Anthropic inner error has no
        ``param``/``code``.

        Ref: stdapi/api_errors.py:UnsupportedModelError
        """
        openai_resp = test_client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model-xyz",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=_bearer_headers(api_key),
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
        openai_err = _assert_openai_error_shape(openai_body)
        assert openai_err["code"] == "model_not_found"

        # Anthropic envelope must NOT have top-level "error.param" or "error.code"
        anthropic_err = _assert_anthropic_error_shape(anthropic_body)
        assert "param" not in anthropic_body["error"]
        assert "code" not in anthropic_body["error"]
        assert anthropic_err["type"] == "not_found_error"

        # Same underlying error: both messages name the model that was rejected.
        assert "nonexistent-model-xyz" in openai_err["message"]
        assert "nonexistent-model-xyz" in anthropic_err["message"]


class TestRoutingErrorPayloads:
    """Starlette-level routing failures (404/405) are rendered as an error envelope.

    The base ``starlette.exceptions.HTTPException`` the router raises for
    no-route/method-not-allowed must not bypass the custom exception handler and
    fall back to Starlette's default ``{"detail": ...}``.

    Ref: stdapi/main.py:handle_http_exception
         stdapi/api_providers/__init__.py:format_http_error
    """

    def test_wrong_method_returns_openai_envelope(
        self, test_client: TestClient
    ) -> None:
        """GET on the POST-only OpenAI embeddings route returns a 405 OpenAI envelope.

        The path matches partially, so the resolved route still carries the
        OpenAI tag and the provider formatter is selected rather than the
        route-less fallback.
        """
        resp = test_client.get("/v1/embeddings")
        assert resp.status_code == 405
        assert "detail" not in resp.json()
        err = _assert_openai_error_shape(resp.json())
        assert err["type"] == "invalid_request_error"
        assert "Method Not Allowed" in err["message"]

    def test_unknown_path_returns_error_envelope(self, test_client: TestClient) -> None:
        """An undefined path returns the route-less ``{"error": <message>}`` fallback.

        No route matches, so no provider tag is available and
        ``_default_formatter`` emits the minimal envelope — still not
        Starlette's ``{"detail": ...}``.

        Ref: stdapi/api_providers/__init__.py:_default_formatter
        """
        resp = test_client.get("/v1/nonexistent")
        assert resp.status_code == 404
        body = resp.json()
        assert "detail" not in body
        assert set(body) == {"error"}
        assert "Not Found" in body["error"]


def _openai_request(
    method: str = "POST", path: str = "/v1/chat/completions"
) -> Request:
    """Return a request scoped to an OpenAI-tagged route.

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


@pytest.mark.usefixtures("request_log")
class TestBotocoreClientErrorEnvelope:
    """``handle_botocore_client_error`` maps an AWS error code onto the OpenAI envelope.

    The raw AWS error code is never surfaced to the client, neither as ``param``
    (which OpenAI clients read as a request-field name) nor as ``code`` (an
    AWS-internal identifier upstream never emits there); ``code`` stays null.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
         stdapi/main.py:handle_botocore_client_error
         stdapi/aws_bedrock.py:AWS_ERROR_MAP
    """

    async def test_client_error_param_is_not_the_internal_error_type(self) -> None:
        """``ThrottlingException`` becomes a 429 whose ``param``/``code`` stay null.

        The second element of the ``AWS_ERROR_MAP`` entry is the gateway's
        internal error type, not a request field; leaking it as ``param`` would
        make OpenAI clients report a non-existent parameter. The AWS exception
        code itself must not reach ``code`` either.
        """
        response = await handle_botocore_client_error(
            _openai_request(), make_client_error("ThrottlingException", message="boom")
        )
        err = _assert_openai_error_shape(json.loads(bytes(response.body)))
        assert response.status_code == 429
        assert err["type"] == "rate_limit_error"
        assert err["param"] is None
        assert err["code"] is None

    @pytest.mark.parametrize("code", ["EntityTooSmall", "InvalidPart"])
    async def test_s3_multipart_client_errors_map_to_400(self, code: str) -> None:
        """S3 multipart-upload validation errors map to 400 invalid_request_error, not 502.

        These codes mean the *client* sent bad part data, so they must not fall
        through to the ``(502, "server_error")`` default for unmapped AWS codes.
        The raw S3 code is never surfaced as ``code``.

        Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/qfacts.html
        """
        response = await handle_botocore_client_error(
            _openai_request(), make_client_error(code, message="boom")
        )
        err = _assert_openai_error_shape(json.loads(bytes(response.body)))
        assert response.status_code == 400
        assert err["type"] == "invalid_request_error"
        assert err["param"] is None
        assert err["code"] is None


def _tagged_request(tag: str) -> Request:
    """Build a request whose resolved route carries the given API provider tag.

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


@pytest.mark.usefixtures("request_log")
class TestRetryAfterHeader:
    """429 responses advertise the region router's own quota backoff as ``retry-after``.

    ``retry-after`` is a standard HTTP header shared by all three provider
    envelopes, so it is emitted by the common response-header stage rather than
    by a per-provider formatter. The value is the router's remaining backoff, so
    SDKs wait the server-driven delay instead of guessing.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
         stdapi/main.py:set_retry_after_header
         stdapi/region_routing.py:quota_retry_after
    """

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
        """Every provider envelope gets the router backoff as ``retry-after`` on a 429.

        The header is provider-independent, so the same value is emitted whether
        the failing route was tagged OpenAI, Anthropic or Cohere.
        """
        request = _tagged_request(tag)
        self._mark_error(request, "ThrottlingException")
        response = Response(status_code=429)
        set_retry_after_header(request, response)
        assert response.headers["retry-after"] == str(
            SETTINGS.aws_bedrock_region_routing_quota_backoff_seconds
        )

    def test_no_header_when_no_backoff_was_applied(self) -> None:
        """Without a recorded quota backoff the header is omitted, not invented."""
        request = _tagged_request(TAG_OPENAI)
        assert quota_retry_after(request) is None
        response = Response(status_code=429)
        set_retry_after_header(request, response)
        assert "retry-after" not in response.headers

    def test_unavailability_backoff_is_not_advertised(self) -> None:
        """Unavailability backoffs drive 503/529 responses and carry no ``retry-after``.

        Only quota (throttling) errors produce a delay a client can honour; an
        unavailable region says nothing about when *this* request may be retried,
        so no header is emitted even on a 429.
        """
        request = _tagged_request(TAG_OPENAI)
        self._mark_error(request, "ServiceUnavailableException")
        assert quota_retry_after(request) is None
        response = Response(status_code=429)
        set_retry_after_header(request, response)
        assert "retry-after" not in response.headers

    def test_header_only_on_rate_limited_responses(self) -> None:
        """A successful response never carries ``retry-after``, even with a live backoff."""
        request = _tagged_request(TAG_OPENAI)
        self._mark_error(request, "ThrottlingException")
        assert quota_retry_after(request) is not None, (
            "precondition: a quota backoff must be recorded for this to be meaningful"
        )
        response = Response(status_code=200)
        set_retry_after_header(request, response)
        assert "retry-after" not in response.headers

    def test_fractional_backoff_is_rounded_up(self) -> None:
        """Sub-second precision rounds up so the client never retries too early.

        12.1 s becomes 13, not 12: ``retry-after`` only accepts whole seconds and
        truncation would let the client retry while the region is still blocked.
        """
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
