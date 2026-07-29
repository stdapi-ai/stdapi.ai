"""Tests for the synchronous-failure HTTP contract of POST /v1/responses (unit)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from starlette.testclient import TestClient

from stdapi.models import ModelDetails
from stdapi.routes import openai_responses
from stdapi.types.openai_responses import Response, ResponseCreateParams, ResponseError

if TYPE_CHECKING:
    from sse_starlette import EventSourceResponse

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
def client(api_key: str) -> TestClient:
    """Test client without lifespan (no AWS startup), pre-authenticated."""
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})


@pytest.fixture
def failed_chat_backend(monkeypatch: pytest.MonkeyPatch) -> _StubFailedChatModel:
    """Stub model validation and the chat generation backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        return ModelDetails(
            id=model_id,
            name=model_id,
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )

    stub = _StubFailedChatModel()
    monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
    monkeypatch.setattr(openai_responses, "get_chat_model", lambda _model_id: stub)
    return stub


@pytest.mark.usefixtures("failed_chat_backend")
def test_synchronous_failed_response_returns_502(client: TestClient) -> None:
    """A synchronous request must surface a failed status as a 502, not a 200 body."""
    response = client.post(
        "/v1/responses", json={"model": "amazon.nova-pro-v1:0", "input": "hi"}
    )
    assert response.status_code == 502
    assert "failed to generate" in response.json()["error"]["message"]


@pytest.mark.usefixtures("failed_chat_backend")
def test_background_failed_response_stays_200(client: TestClient) -> None:
    """A background request keeps the failed terminal state for polling instead of raising."""
    response = client.post(
        "/v1/responses",
        json={"model": "amazon.nova-pro-v1:0", "input": "hi", "background": True},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "failed"
