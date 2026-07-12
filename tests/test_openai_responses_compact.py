"""Tests for the OpenAI-compatible POST /v1/responses/compact route (unit)."""

from base64 import urlsafe_b64decode
from typing import TYPE_CHECKING

import pytest
from starlette.testclient import TestClient

from stdapi.api_errors import ApiError

if TYPE_CHECKING:
    from openai import OpenAI
from stdapi.models import ModelDetails
from stdapi.models.chat._adapters._openai_responses import (
    encode_compaction_content,
    map_input,
)
from stdapi.routes import openai_responses
from stdapi.types.openai_responses import (
    CompactionItemParam,
    InputTokensDetails,
    OutputTokensDetails,
    Response,
    ResponseCreateParams,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)


def _usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=11,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=7,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=18,
    )


class _StubChatModel:
    """Stub chat backend recording the generation request."""

    def __init__(self) -> None:
        self.requests: list[ResponseCreateParams] = []

    async def create_response(
        self, request: ResponseCreateParams, response_id: str, created_at: float
    ) -> Response:
        """Record the request and return a canned summary response."""
        self.requests.append(request)
        return Response(
            id=response_id,
            created_at=created_at,
            model=request.model,
            object="response",
            output=[
                ResponseOutputMessage(
                    id="msg-1",
                    content=[
                        ResponseOutputText(
                            annotations=[], text="THE SUMMARY", type="output_text"
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
            usage=_usage(),
        )


@pytest.fixture
def client(api_key: str) -> TestClient:
    """Test client without lifespan (no AWS startup), pre-authenticated."""
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})


@pytest.fixture
def chat_backend(monkeypatch: pytest.MonkeyPatch) -> _StubChatModel:
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

    stub = _StubChatModel()
    monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
    monkeypatch.setattr(openai_responses, "get_chat_model", lambda _model_id: stub)
    return stub


class TestResponsesCompactRoute:
    """POST /v1/responses/compact: response shape and generation request."""

    def test_compact_returns_compaction_item(
        self, client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """The summary is wrapped in an opaque compaction item with usage."""
        response = client.post(
            "/v1/responses/compact",
            json={"model": "amazon.nova-pro-v1:0", "input": "a long conversation"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "response.compaction"
        (item,) = body["output"]
        assert item["type"] == "compaction"
        assert urlsafe_b64decode(item["encrypted_content"]) == b"THE SUMMARY"
        assert body["usage"]["total_tokens"] == 18

    def test_compact_appends_summarization_directive(
        self, client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """The generation request keeps the input and appends the directive."""
        client.post(
            "/v1/responses/compact",
            json={
                "model": "amazon.nova-pro-v1:0",
                "input": [{"role": "user", "content": "hello"}],
                "instructions": "be nice",
            },
        )
        (request,) = chat_backend.requests
        assert request.instructions == "be nice"
        assert isinstance(request.input, list)
        assert request.input[0].content == "hello"
        assert "Summarize the conversation" in str(request.input[-1].content)

    def test_previous_response_id_is_rejected(
        self, client: TestClient, chat_backend: _StubChatModel
    ) -> None:
        """previous_response_id is rejected as unsupported."""
        response = client.post(
            "/v1/responses/compact",
            json={"model": "m", "input": "x", "previous_response_id": "resp-1"},
        )
        assert response.status_code == 400
        assert "previous_response_id" in response.json()["error"]["message"]
        assert not chat_backend.requests


class TestCompactionItemRoundTrip:
    """Compaction items round-trip through the Responses input mapping."""

    async def test_compaction_item_maps_to_user_summary_message(self) -> None:
        """The decoded summary is injected as a user message."""
        item = CompactionItemParam(
            encrypted_content=encode_compaction_content("hello world"),
            type="compaction",
        )
        messages, system = await map_input([item], None)
        assert system == []
        (message,) = messages
        assert message["role"] == "user"
        assert "hello world" in message["content"][0]["text"]

    async def test_invalid_compaction_content_is_rejected(self) -> None:
        """Undecodable compaction content raises a 400 error."""
        item = CompactionItemParam(encrypted_content="!!!", type="compaction")
        with pytest.raises(ApiError, match="compaction"):
            await map_input([item], None)


class TestResponsesCompactLive:
    """Live conversation compaction and round-trip continuation."""

    def test_compact_and_continue(
        self, openai_client: OpenAI, responses_model: str, use_official_api: bool
    ) -> None:
        """A compacted conversation carries its facts into the next turn."""
        if use_official_api:
            pytest.skip("compaction is model-restricted on the official API")
        compacted = openai_client.responses.compact(
            model=responses_model,
            input=[
                {"role": "user", "content": "My favorite color is teal."},
                {
                    "role": "assistant",
                    "content": "Understood, your favorite color is teal.",
                },
            ],
        )
        assert compacted.object == "response.compaction"
        assert compacted.usage.total_tokens > 0
        (item,) = compacted.output
        assert item.type == "compaction"
        assert item.encrypted_content

        follow = openai_client.responses.create(
            model=responses_model,
            input=[
                {
                    "type": "compaction",
                    "id": item.id,
                    "encrypted_content": item.encrypted_content,
                },
                {
                    "role": "user",
                    "content": "What is my favorite color? Reply with one word.",
                },
            ],
        )
        assert "teal" in follow.output_text.lower()
