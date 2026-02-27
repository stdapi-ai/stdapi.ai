"""Integration tests for error payload structure on OpenAI and Anthropic routes.

Verifies that error responses match the official API envelope formats:
- OpenAI: ``{"error": {"message", "type", "param", "code"}}``
- Anthropic: ``{"type": "error", "error": {"type", "message"}}``
"""

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from starlette.testclient import TestClient


def _openai_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}


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
            }
        }

    Args:
        body: Parsed JSON response body.

    Returns:
        The inner ``error`` dict for further assertions.
    """
    assert set(body.keys()) == {"type", "error"}, (
        f"Unexpected top-level keys: {body.keys()}"
    )
    assert body["type"] == "error"
    err = body["error"]
    assert set(err.keys()) == {"type", "message"}, (
        f"Unexpected error keys: {err.keys()}"
    )
    assert isinstance(err["type"], str)
    assert isinstance(err["message"], str)
    return err  # type: ignore[no-any-return]


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
        """A non-existent model on an Anthropic route must return 400 with the Anthropic error shape.

        The official Anthropic API returns 400 ``invalid_request_error`` for
        unknown models (not 404).

        Validates:
            - HTTP 400 status code.
            - ``{"type": "error", "error": {"type", "message"}}`` envelope.
            - Inner ``type`` is ``"invalid_request_error"``.
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
        assert resp.status_code == 400
        err = _assert_anthropic_error_shape(resp.json())
        assert err["type"] == "invalid_request_error"
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
            - OpenAI route returns 404, Anthropic route returns 400 (matching official APIs).
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
        assert anthropic_resp.status_code == 400

        openai_body = openai_resp.json()
        anthropic_body = anthropic_resp.json()

        # OpenAI envelope must NOT have top-level "type"
        assert "type" not in openai_body
        _assert_openai_error_shape(openai_body)

        # Anthropic envelope must NOT have top-level "error.param" or "error.code"
        _assert_anthropic_error_shape(anthropic_body)
        assert "param" not in anthropic_body["error"]
        assert "code" not in anthropic_body["error"]
