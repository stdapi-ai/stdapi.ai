"""Tests for the synchronous-failure HTTP contract of POST /v1/responses (unit).

Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
     stdapi/routes/openai_responses.py:_failed_response_error
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from stdapi.routes import openai_responses
from stdapi.types.openai_responses import Response, ResponseCreateParams, ResponseError
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from sse_starlette import EventSourceResponse
    from starlette.testclient import TestClient

    from stdapi.models import ModelDetails

pytestmark = pytest.mark.local


class _StubFailedChatModel:
    """Stub chat backend returning a terminal ``status='failed'`` Response."""

    def native_store_supported(self) -> bool:
        """Report Converse-style (non-Mantle) storage."""
        return False

    async def create_response(
        self,
        request: ResponseCreateParams,
        response_id: str,
        created_at: float,
        moderation_builder: object = None,
    ) -> Response | EventSourceResponse:
        """Return a failed Response, mirroring a malformed_model_output stop reason."""
        return Response(
            id=response_id,
            created_at=int(created_at),
            model=request.model,
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            status="failed",
            error=ResponseError(
                code="server_error", message="The model failed to generate output."
            ),
            background=request.background,
        )


@pytest.fixture
def failed_chat_backend(monkeypatch: pytest.MonkeyPatch) -> _StubFailedChatModel:
    """Stub model validation and the chat generation backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        return make_model_details(model_id)

    stub = _StubFailedChatModel()
    monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
    monkeypatch.setattr(openai_responses, "get_chat_model", lambda _model_id: stub)
    return stub


@pytest.mark.usefixtures("failed_chat_backend")
def test_synchronous_failed_response_returns_502(app_client: TestClient) -> None:
    """A synchronous ``status="failed"`` Response is surfaced as a 502 error envelope.

    A failed Response carries the failure in ``error`` and no usable output, so
    returning it as a 200 body would hide the failure from clients that only
    read ``output_text``. The gateway re-raises it as a 502, which the OpenAI
    error formatter types as ``server_error``.

    Ref: stdapi/api_providers/openai.py:_format_error
    """
    response = app_client.post(
        "/v1/responses", json={"model": "amazon.nova-pro-v1:0", "input": "hi"}
    )
    assert response.status_code == 502
    error = response.json()["error"]
    assert "failed to generate" in error["message"]
    assert error["type"] == "server_error"


@pytest.mark.usefixtures("failed_chat_backend")
def test_background_failed_response_stays_200(app_client: TestClient) -> None:
    """A background request returns the failed terminal state as a 200 Response.

    ``background`` responses are polled, so the terminal ``failed`` state and
    its ``error`` object must stay readable on the Response object instead of
    being raised as an HTTP error.

    Ref: https://developers.openai.com/api/docs/guides/background
    """
    response = app_client.post(
        "/v1/responses",
        json={"model": "amazon.nova-pro-v1:0", "input": "hi", "background": True},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "failed"
    assert body["background"] is True
    assert body["error"] == {
        "code": "server_error",
        "message": "The model failed to generate output.",
    }
    assert body["output"] == []
